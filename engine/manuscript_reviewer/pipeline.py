"""Full audit pipeline: media verification, frame ledger, shot truth, artifacts.

Deterministic, local-only, no network:

    VIDEO → MEDIA VERIFICATION → FRAME LEDGER → CROSS-CHECK
          → SHOT TRUTH (metrics → candidates → adversarial verification →
            shot proposals → evidence)
          → VALIDATION → QC + MANIFEST ARTIFACTS
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from . import __version__
from .artifacts import writer
from .audio.asr.runtime import ASRConfig
from .audio.engine import run_audio_analysis
from .caption_brain import CaptionBrainError, CaptionBrainOutput, finalize_run
from .media import frames as frames_mod
from .media import probe as probe_mod
from .media.ffmpeg_tools import FFmpegNotFoundError, ToolExecutionError, find_tool, tool_version
from .models.audio import AudioQCResult
from .models.frame import FrameLedger
from .models.media import MediaInfo
from .models.review_intelligence import VisualIntelligenceResult
from .models.run import RunManifest
from .models.shot_truth import ShotTruthResult
from .models.validation import (
    FrameCountSignal,
    QCReport,
    RunStatus,
    Severity,
    ValidatorIssue,
)
from .rules.loader import load_rules
from .shots.engine import run_shot_analysis
from .validation.ledger_validator import (
    compute_run_status,
    cross_check_frame_count,
    validate_ledger,
)
from .validation.media_validator import validate_media
from .visual_intelligence import run_visual_intelligence

logger = logging.getLogger(__name__)


def _vi_attr(result: AuditResult, name: str) -> str | None:
    """A string provenance field off the visual-intelligence result, or None."""
    vi = result.visual_intelligence
    return getattr(vi, name) if vi is not None else None


@dataclass
class AuditResult:
    """Everything the CLI needs to report one audit run."""

    run_id: str
    run_dir: Path
    status: RunStatus
    media: MediaInfo | None = None
    ledger: FrameLedger | None = None
    qc: QCReport | None = None
    manifest: RunManifest | None = None
    fatal_error: str | None = None
    stage_timings: dict[str, float] = field(default_factory=dict)
    shot_truth: ShotTruthResult | None = None
    audio_truth: AudioQCResult | None = None
    visual_intelligence: VisualIntelligenceResult | None = None
    caption_brain: CaptionBrainOutput | None = None


class _StageTimer:
    def __init__(self, sink: dict[str, float], name: str) -> None:
        self._sink = sink
        self._name = name

    def __enter__(self) -> None:
        self._start = time.perf_counter()

    def __exit__(self, *exc_info: object) -> None:
        self._sink[self._name] = round(time.perf_counter() - self._start, 4)


def run_audit(
    video_path: Path,
    artifacts_root: Path,
    seed_path: Path | None = None,
    extract_frames: bool = False,
    shot_analysis: bool = False,
    shot_sensitivity: str = "normal",
    extract_shot_evidence: bool = True,
    use_scdet: bool = True,
    audio_analysis: bool = False,
    asr_enabled: bool = True,
    asr_config: ASRConfig | None = None,
    visual_intelligence: bool = True,
    feedback_path: Path | None = None,
    visual_anchors_path: Path | None = None,
    review_decisions_path: Path | None = None,
    ocr_enabled: bool = True,
    extract_visual_evidence: bool = False,
    caption_brain: bool = False,
    human_facts_path: Path | None = None,
    final_review_path: Path | None = None,
) -> AuditResult:
    """Run the full audit (media verification, frame ledger, optional shot
    truth) and write all artifacts.

    Never raises for media-level problems — failures are recorded in qc.json
    and the returned status. Raises only for environment problems (missing
    ffmpeg, unwritable artifact root).
    """
    started_at = datetime.now(UTC)
    wall_start = time.perf_counter()
    run_id = writer.new_run_id()
    run_dir = writer.create_run_dir(artifacts_root, video_path, run_id)
    result = AuditResult(run_id=run_id, run_dir=run_dir, status=RunStatus.FAILED)
    timings = result.stage_timings

    log_path = run_dir / "run.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root_logger = logging.getLogger("manuscript_reviewer")
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.DEBUG)

    issues: list[ValidatorIssue] = []
    checks_run: list[str] = []
    signals: list[FrameCountSignal] = []
    artifact_paths: list[Path] = []
    media: MediaInfo | None = None
    ledger: FrameLedger | None = None
    rules = load_rules()

    try:
        ffprobe_version = tool_version(find_tool("ffprobe"))
        ffmpeg_version = tool_version(find_tool("ffmpeg"))

        # Hash the source BEFORE analysis; re-hash after to detect mid-run change.
        with _StageTimer(timings, "hash_source"):
            source_hash_before = writer.sha256_file(video_path)
        seed_hash: str | None = None
        if seed_path is not None:
            seed_hash = writer.sha256_file(seed_path)
            shutil.copy2(seed_path, run_dir / f"seed{seed_path.suffix or '.dat'}")
            logger.info("Seed copied and hashed: %s", seed_hash)
        feedback_hash: str | None = None
        if feedback_path is not None:
            feedback_hash = writer.sha256_file(feedback_path)
        review_decisions_hash: str | None = None
        if review_decisions_path is not None:
            review_decisions_hash = writer.sha256_file(review_decisions_path)
        human_facts_hash: str | None = None
        if human_facts_path is not None:
            human_facts_hash = writer.sha256_file(human_facts_path)
        final_review_hash: str | None = None
        if final_review_path is not None:
            final_review_hash = writer.sha256_file(final_review_path)
        # Snapshot + hash the visual anchors so tracks are never claimed
        # reproducible without the exact anchor bytes they were seeded from.
        visual_anchors_hash: str | None = None
        if visual_anchors_path is not None:
            visual_anchors_hash = writer.sha256_file(visual_anchors_path)
            shutil.copy2(
                visual_anchors_path,
                run_dir / f"visual_anchors{visual_anchors_path.suffix or '.json'}",
            )

        # --- MEDIA VERIFICATION ---
        try:
            with _StageTimer(timings, "probe"):
                media, raw_probe = probe_mod.probe_media(video_path)
            result.media = media
            checks_run.append("probe")
            artifact_paths.append(writer.write_media_json(run_dir, media, raw_probe))
        except (probe_mod.ProbeError, ToolExecutionError) as exc:
            issues.append(
                ValidatorIssue(
                    rule_id="P1-MEDIA-000",
                    severity=Severity.FAIL,
                    location=video_path.name,
                    message=f"Video cannot be probed: {exc}",
                )
            )
            media = None

        if media is not None:
            issues.extend(validate_media(media))
            checks_run.append("media_validator")

        # --- FRAME LEDGER ---
        if media is not None and media.video_streams:
            video_stream = media.video_streams[0]
            try:
                with _StageTimer(timings, "enumerate_frames"):
                    ledger = frames_mod.enumerate_frames(
                        video_path, video_stream.time_base, stream_index=0
                    )
                result.ledger = ledger
                checks_run.append("frame_enumeration")
            except (frames_mod.FrameEnumerationError, ToolExecutionError) as exc:
                issues.append(
                    ValidatorIssue(
                        rule_id="P1-LEDGER-000",
                        severity=Severity.FAIL,
                        location=video_path.name,
                        message=f"Frame enumeration failed: {exc}",
                    )
                )

        if ledger is not None:
            issues.extend(validate_ledger(ledger))
            checks_run.append("ledger_validator")

            # --- INDEPENDENT CROSS-CHECK ---
            assert media is not None
            with _StageTimer(timings, "count_frames_crosscheck"):
                decoded_count = probe_mod.count_decoded_frames(video_path, stream_index=0)
            signals, count_issues = cross_check_frame_count(ledger, media, decoded_count)
            issues.extend(count_issues)
            checks_run.append("frame_count_cross_check")

            artifact_paths.append(writer.write_frames_csv(run_dir, ledger))
            artifact_paths.append(writer.write_frames_jsonl(run_dir, ledger))

            if extract_frames:
                with _StageTimer(timings, "extract_frames"):
                    extracted = frames_mod.extract_evidence_frames(
                        video_path, ledger, run_dir / "frames"
                    )
                logger.info("Extracted %d evidence frames", len(extracted))
                checks_run.append("frame_extraction")

        # --- SHOT TRUTH (Phase 2) ---
        if shot_analysis and ledger is not None and media is not None:
            with _StageTimer(timings, "shot_analysis_total"):
                shot_output = run_shot_analysis(
                    video_path,
                    run_dir,
                    media,
                    ledger,
                    sensitivity_name=shot_sensitivity,
                    extract_evidence=extract_shot_evidence,
                    use_scdet=use_scdet,
                )
            timings.update(shot_output.stage_timings)
            issues.extend(shot_output.issues)
            artifact_paths.extend(shot_output.artifact_paths)
            result.shot_truth = shot_output.result
            checks_run.append("shot_truth")

        # --- AUDIO TRUTH (Phase 3) ---
        if audio_analysis and ledger is not None and media is not None:
            with _StageTimer(timings, "audio_analysis_total"):
                audio_output = run_audio_analysis(
                    video_path,
                    run_dir,
                    media,
                    ledger,
                    result.shot_truth,
                    (
                        result.shot_truth.annotation_endpoint_exact
                        if result.shot_truth is not None
                        else None
                    ),
                    asr_enabled=asr_enabled,
                    asr_config=asr_config,
                )
            timings.update(audio_output.stage_timings)
            issues.extend(audio_output.issues)
            artifact_paths.extend(audio_output.artifact_paths)
            result.audio_truth = audio_output.result
            checks_run.append("audio_truth")

        # --- VISUAL REVIEW INTELLIGENCE (Phase 4) ---
        if visual_intelligence:
            with _StageTimer(timings, "visual_intelligence_total"):
                vi_output = run_visual_intelligence(
                    run_dir,
                    media,
                    ledger,
                    result.shot_truth,
                    result.audio_truth,
                    video_path=video_path,
                    seed_path=seed_path,
                    feedback_path=feedback_path,
                    visual_anchors_path=visual_anchors_path,
                    review_decisions_path=review_decisions_path,
                    video_sha256=source_hash_before,
                    rules_version=rules.version,
                    ocr_enabled=ocr_enabled,
                    extract_visual_evidence=extract_visual_evidence,
                )
            timings.update(vi_output.stage_timings)
            issues.extend(vi_output.issues)
            artifact_paths.extend(vi_output.artifact_paths)
            result.visual_intelligence = vi_output.result
            checks_run.append("visual_intelligence")

        # --- CAPTION BRAIN (Phase 5) ---
        # Runs from the evidence artifacts already written; never re-decodes
        # media. Unresolved review never crashes the pipeline — it produces
        # REVIEW_REQUIRED plus the reviewer packet and a draft when safe (§95).
        if caption_brain and result.shot_truth is not None:
            with _StageTimer(timings, "caption_brain_total"):
                try:
                    cb_output = finalize_run(
                        run_dir,
                        review_decisions_path=review_decisions_path,
                        human_facts_path=human_facts_path,
                        final_review_path=final_review_path,
                        video_sha256=source_hash_before,
                        # Canonical identity comes from the actual media file,
                        # never from the seed (§5.1-2).
                        canonical_video_id=video_path.stem,
                    )
                except CaptionBrainError as exc:
                    cb_output = None
                    issues.append(
                        ValidatorIssue(
                            rule_id="M2-MEDIA-003",
                            severity=Severity.FAIL,
                            location="caption_brain",
                            message=f"Caption Brain could not run: {exc}",
                        )
                    )
            if cb_output is not None:
                timings.update(cb_output.stage_timings)
                artifact_paths.extend(cb_output.artifact_paths)
                result.caption_brain = cb_output
                checks_run.append("caption_brain")

        # --- SOURCE STABILITY ---
        with _StageTimer(timings, "rehash_source"):
            source_hash_after = writer.sha256_file(video_path)
        checks_run.append("source_stability")
        if source_hash_after != source_hash_before:
            issues.append(
                ValidatorIssue(
                    rule_id="P1-SOURCE-001",
                    severity=Severity.FAIL,
                    location=video_path.name,
                    message="Source video changed during processing (SHA-256 mismatch).",
                )
            )

        # --- QC REPORT ---
        # Shot-truth uncertainty propagates to the top-level status: the audit
        # never reports PASS while an unresolved possible real cut remains.
        status = compute_run_status(issues, ledger)
        if result.shot_truth is not None:
            if result.shot_truth.overall_status == "FAILED":
                status = RunStatus.FAILED
            elif (
                result.shot_truth.overall_status == "REVIEW_REQUIRED"
                and status == RunStatus.PASS
            ):
                status = RunStatus.REVIEW_REQUIRED
        if result.audio_truth is not None:
            if result.audio_truth.overall_status == "FAILED":
                status = RunStatus.FAILED
            elif (
                result.audio_truth.overall_status == "REVIEW_REQUIRED"
                and status == RunStatus.PASS
            ):
                status = RunStatus.REVIEW_REQUIRED
        # Phase 4 is a review-preparation stage: it never upgrades to PASS, and
        # an unresolved review queue keeps the run in REVIEW_REQUIRED.
        if result.visual_intelligence is not None:
            if result.visual_intelligence.overall_status == "FAILED":
                status = RunStatus.FAILED
            elif (
                result.visual_intelligence.overall_status == "REVIEW_REQUIRED"
                and status == RunStatus.PASS
            ):
                status = RunStatus.REVIEW_REQUIRED
        # Phase 5: a caption that is not READY_TO_ENTER keeps the run in
        # review — the audit never claims PASS over an unready caption.
        if (
            result.caption_brain is not None
            and result.caption_brain.result.readiness.value
            in ("BLOCKED", "REVIEW_REQUIRED")
            and status == RunStatus.PASS
        ):
            status = RunStatus.REVIEW_REQUIRED
        qc = QCReport(
            status=status,
            issues=issues,
            frame_count_signals=signals,
            checks_run=checks_run,
        )
        result.qc = qc
        result.status = status
        artifact_paths.append(writer.write_qc_json(run_dir, qc))

        # --- MANIFEST ---
        ended_at = datetime.now(UTC)
        entries = writer.collect_artifact_entries(run_dir, artifact_paths)
        manifest = RunManifest(
            run_id=run_id,
            source_video_path=str(video_path.resolve()),
            source_video_sha256=source_hash_before,
            source_seed_path=str(seed_path.resolve()) if seed_path else None,
            source_seed_sha256=seed_hash,
            source_feedback_path=str(feedback_path.resolve()) if feedback_path else None,
            source_feedback_sha256=feedback_hash,
            review_decisions_path=(
                str(review_decisions_path.resolve()) if review_decisions_path else None
            ),
            review_decisions_sha256=review_decisions_hash,
            source_visual_anchors_path=(
                str(visual_anchors_path.resolve()) if visual_anchors_path else None
            ),
            source_visual_anchors_sha256=visual_anchors_hash,
            human_facts_path=(
                str(human_facts_path.resolve()) if human_facts_path else None
            ),
            human_facts_sha256=human_facts_hash,
            final_review_path=(
                str(final_review_path.resolve()) if final_review_path else None
            ),
            final_review_sha256=final_review_hash,
            caption_brain_version=(
                result.caption_brain.result.caption_brain_version
                if result.caption_brain is not None
                else None
            ),
            caption_final_status=(
                result.caption_brain.result.readiness.value
                if result.caption_brain is not None
                else None
            ),
            visual_intelligence_version=(
                result.visual_intelligence.visual_intelligence_version
                if result.visual_intelligence is not None
                else None
            ),
            ocr_status=(
                result.visual_intelligence.ocr_status.value
                if result.visual_intelligence is not None
                else None
            ),
            ocr_engine=_vi_attr(result, "ocr_engine"),
            ocr_version=_vi_attr(result, "ocr_version"),
            ocr_language=_vi_attr(result, "ocr_language"),
            tracking_version=_vi_attr(result, "tracking_version"),
            tracking_config=(
                result.visual_intelligence.tracking_config
                if result.visual_intelligence is not None
                else {}
            ),
            app_version=__version__,
            rules_version=rules.version,
            ffmpeg_version=ffmpeg_version,
            ffprobe_version=ffprobe_version,
            started_at_utc=started_at.isoformat(),
            ended_at_utc=ended_at.isoformat(),
            analysis_duration_seconds=round(time.perf_counter() - wall_start, 4),
            stage_timings_seconds=timings,
            artifacts=entries,
            validation_status=status,
        )
        result.manifest = manifest
        writer.write_manifest_json(run_dir, manifest.model_dump(mode="json"))
        logger.info("Audit %s finished with status %s", run_id, status)
        return result

    except (FFmpegNotFoundError, writer.ArtifactWriteError) as exc:
        result.fatal_error = str(exc)
        result.status = RunStatus.FAILED
        logger.error("Fatal: %s", exc)
        return result
    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()
