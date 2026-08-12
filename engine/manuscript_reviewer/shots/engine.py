"""Shot Truth Engine orchestration: metrics → candidates → verification →
shots → evidence → validation. Called by the main pipeline as one stage."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..artifacts import shot_writer
from ..media.clock import AnnotationClock
from ..media.endpoint import compute_annotation_endpoint
from ..models.frame import FrameLedger
from ..models.media import MediaInfo
from ..models.shot_truth import (
    BoundaryCandidate,
    CandidateStatus,
    PairMetrics,
    ShotProposal,
    ShotTruthResult,
)
from ..models.validation import Severity, ValidatorIssue
from ..validation import shot_validator
from .baseline import compute_local_baselines
from .builder import build_shot_proposals
from .candidates import Sensitivity, generate_candidates
from .decode import MetricDecodeError, decode_metric_frames
from .evidence import render_candidate_evidence
from .metrics import compute_frame_stats, compute_pair_metrics
from .regions import detect_blends, detect_fades, detect_flash_regions
from .scdet import scdet_scores
from .verifier import VerifierContext, verify_all

logger = logging.getLogger(__name__)


@dataclass
class ShotAnalysisOutput:
    result: ShotTruthResult | None
    issues: list[ValidatorIssue]
    artifact_paths: list[Path]
    stage_timings: dict[str, float] = field(default_factory=dict)


def _timed(sink: dict[str, float], name: str, start: float) -> None:
    sink[name] = round(time.perf_counter() - start, 4)


def run_shot_analysis(
    video_path: Path,
    run_dir: Path,
    media: MediaInfo,
    ledger: FrameLedger,
    sensitivity_name: str = "normal",
    extract_evidence: bool = True,
    use_scdet: bool = True,
) -> ShotAnalysisOutput:
    """Run the full Phase 2 stage and write its artifacts into ``run_dir``."""
    issues: list[ValidatorIssue] = []
    artifact_paths: list[Path] = []
    timings: dict[str, float] = {}
    sensitivity = Sensitivity.from_name(sensitivity_name)
    has_audio = bool(media.audio_streams)
    clock = AnnotationClock.from_ledger(ledger)

    # --- metric decode ---
    start = time.perf_counter()
    try:
        frames = decode_metric_frames(video_path, ledger.frame_count)
    except MetricDecodeError as exc:
        issues.append(
            ValidatorIssue(
                rule_id="P2-DECODE-001",
                severity=Severity.FAIL,
                location=video_path.name,
                message=f"Metric decode failed: {exc}",
            )
        )
        return ShotAnalysisOutput(
            result=None, issues=issues, artifact_paths=[], stage_timings=timings
        )
    _timed(timings, "shot_metric_decode", start)

    # --- per-frame stats + pair metrics ---
    start = time.perf_counter()
    stats = compute_frame_stats(frames)
    pairs: list[PairMetrics] = compute_pair_metrics(frames, ledger)
    _timed(timings, "shot_pair_metrics", start)

    # --- scdet evidence (optional independent signal) ---
    if use_scdet:
        start = time.perf_counter()
        scores = scdet_scores(video_path)
        for pair in pairs:
            entry = scores.get(pair.right_frame_index)
            if entry is not None:
                pair.scdet_score, pair.scdet_mafd = entry
        _timed(timings, "shot_scdet", start)

    # --- local baselines ---
    start = time.perf_counter()
    baselines = compute_local_baselines(pairs)
    _timed(timings, "shot_baselines", start)

    # --- regions: flash / fade / blend ---
    start = time.perf_counter()
    flash_regions = detect_flash_regions(stats)
    fades = detect_fades(stats, ledger)
    blends = detect_blends(pairs, [b.diff_median for b in baselines], ledger)
    _timed(timings, "shot_regions", start)

    # --- candidate generation (recall-first) ---
    start = time.perf_counter()
    candidates, raw_count = generate_candidates(
        pairs, baselines, flash_regions, ledger, sensitivity, clock
    )
    _timed(timings, "shot_candidates", start)

    # --- adversarial verification ---
    start = time.perf_counter()
    ctx = VerifierContext(
        frames=frames,
        ledger=ledger,
        pairs=pairs,
        baselines=baselines,
        flash_regions=flash_regions,
        fades=fades,
        blends=blends,
        has_audio=has_audio,
        clock=clock,
    )
    verified: list[BoundaryCandidate] = verify_all(ctx, candidates)
    _timed(timings, "shot_verification", start)

    # --- evidence bundles for supported + review candidates ---
    start = time.perf_counter()
    if extract_evidence:
        interesting = [
            c
            for c in verified
            if c.status in (CandidateStatus.SUPPORTED, CandidateStatus.REVIEW_REQUIRED)
        ]
        if interesting:
            refs = render_candidate_evidence(
                video_path, ledger, interesting, run_dir / "shot_evidence"
            )
            verified = [
                c.model_copy(update={"evidence_refs": refs.get(c.candidate_id, [])})
                if c.candidate_id in refs
                else c
                for c in verified
            ]
            for ref_list in refs.values():
                artifact_paths.extend(run_dir / ref for ref in ref_list)
    _timed(timings, "shot_evidence_render", start)

    # --- canonical annotation endpoint + shot proposals ---
    start = time.perf_counter()
    endpoint = compute_annotation_endpoint(media, ledger, video_path.stem, clock)
    if endpoint.conflict:
        issues.append(
            ValidatorIssue(
                rule_id="P2-END-001",
                severity=Severity.WARN,
                location=video_path.name,
                message=(
                    "Annotation endpoint evidence conflicts or is unverified: "
                    + "; ".join(endpoint.notes)
                ),
            )
        )
    shots: list[ShotProposal] = build_shot_proposals(
        ledger, verified, endpoint.endpoint, clock
    )
    _timed(timings, "shot_proposals", start)

    # --- validation ---
    start = time.perf_counter()
    issues.extend(shot_validator.validate_pairs(ledger, pairs))
    issues.extend(shot_validator.validate_candidates(ledger, verified))
    issues.extend(shot_validator.validate_shots(ledger, shots, verified))
    issues.extend(
        shot_validator.validate_shot_timeline(
            ledger, shots, verified, endpoint.annotation_endpoint, clock
        )
    )
    if extract_evidence:
        issues.extend(shot_validator.validate_evidence(verified))
    had_failure = any(i.severity == Severity.FAIL for i in issues)
    overall = shot_validator.compute_shot_status(
        verified, had_failure, endpoint_conflict=endpoint.conflict
    )
    _timed(timings, "shot_validation", start)

    result = ShotTruthResult(
        frame_count=ledger.frame_count,
        adjacent_pair_count=len(pairs),
        raw_candidate_count=raw_count,
        merged_candidate_count=len(candidates),
        supported_count=sum(1 for c in verified if c.status == CandidateStatus.SUPPORTED),
        rejected_count=sum(1 for c in verified if c.status == CandidateStatus.REJECTED),
        review_required_count=sum(
            1 for c in verified if c.status == CandidateStatus.REVIEW_REQUIRED
        ),
        proposed_shot_count=len(shots),
        overall_status=overall,
        annotation_timeline_origin=clock.origin,
        annotation_endpoint_exact=endpoint.annotation_endpoint,
        annotation_endpoint_method=endpoint.method,
        annotation_endpoint_conflict=endpoint.conflict,
        candidates=verified,
        fades=fades,
        blends=blends,
        shots=shots,
    )

    # --- artifacts ---
    start = time.perf_counter()
    artifact_paths.append(shot_writer.write_adjacent_metrics_csv(run_dir, pairs))
    artifact_paths.append(shot_writer.write_adjacent_metrics_jsonl(run_dir, pairs))
    artifact_paths.append(shot_writer.write_cut_candidates(run_dir, candidates))
    artifact_paths.append(shot_writer.write_boundary_evidence(run_dir, verified))
    artifact_paths.append(shot_writer.write_shots_proposed(run_dir, shots))
    artifact_paths.append(
        shot_writer.write_transition_evidence(run_dir, verified, fades, blends)
    )
    artifact_paths.append(shot_writer.write_shot_qc(run_dir, result))
    _timed(timings, "shot_artifacts", start)

    return ShotAnalysisOutput(
        result=result, issues=issues, artifact_paths=artifact_paths, stage_timings=timings
    )
