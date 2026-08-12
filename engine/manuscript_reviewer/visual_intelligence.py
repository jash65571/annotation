"""Phase 4 — Visual Review Intelligence orchestrator.

Runs after media / shots / audio. Turns deterministic Phase 1-3 evidence plus
the seed and task feedback into structured reviewer intelligence:

    SEED SNAPSHOT -> PARSE -> ATOMIC CLAIMS -> STRUCTURAL COMPARISON ->
    CLAIM<->EVIDENCE MATRIX -> KEEP/FIX/REDO PROPOSALS -> REVIEW QUEUE ->
    SEED TRIAGE -> VISUAL QC

Later slices (frame observations, tracking, OCR, camera, actions, speed) plug
into the same tested harness. Phase 4 never writes final caption prose and never
performs a cloud/media upload; the default runtime is fully local.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .actions.boundaries import build_action_candidates
from .actions.contacts import build_contact_events
from .actions.final_state import build_final_state_checks
from .artifacts import review_writer, visual_writer
from .camera.segmentation import analyze_camera_motion
from .media.clock import AnnotationClock
from .models.audio import AudioQCResult
from .models.frame import FrameLedger
from .models.media import MediaInfo
from .models.review_intelligence import (
    CameraMotionCandidate,
    EntityTrack,
    FrameObservation,
    OCRStatus,
    TextTrack,
    VisualIntelligenceResult,
)
from .models.shot_truth import ShotTruthResult
from .models.validation import Severity, ValidatorIssue
from .ocr.pipeline import run_ocr
from .ocr.tesseract_adapter import TesseractAdapter
from .review.decisions import (
    DecisionLoadError,
    apply_decisions_to_claims,
    load_decisions,
)
from .review.evidence_bundles import extract_evidence_bundles
from .review.proposals import build_proposals, count_by_outcome
from .review.queue import build_review_queue, build_triage
from .seed import feedback as feedback_mod
from .seed import snapshot as snapshot_mod
from .seed.claims import collect_seed_diagnostics, extract_claims
from .seed.comparison import build_rows, compare_seed
from .seed.parser import parse_seed_text
from .shots.decode import MetricDecodeError
from .speed.evidence import build_playback_speed_evidence
from .tracking.anchors import AnchorLoadError, load_anchors
from .tracking.continuity import build_continuity
from .tracking.tracker import track_anchor
from .validation import review_intelligence_validator as ri_validator
from .validation import seed_validator, visual_validator
from .visual.decode import FrameCache
from .visual.observations import build_frame_observations

if TYPE_CHECKING:
    from collections.abc import Callable
    from fractions import Fraction

    import numpy as np

logger = logging.getLogger(__name__)

VISUAL_INTELLIGENCE_VERSION = "0.4.0"


@dataclass
class VisualIntelligenceOutput:
    result: VisualIntelligenceResult | None
    issues: list[ValidatorIssue]
    artifact_paths: list[Path]
    stage_timings: dict[str, float] = field(default_factory=dict)


def _timed(sink: dict[str, float], name: str, start: float) -> None:
    sink[name] = round(time.perf_counter() - start, 4)


def run_visual_intelligence(
    run_dir: Path,
    media: MediaInfo | None,
    ledger: FrameLedger | None,
    shot_truth: ShotTruthResult | None,
    audio_truth: AudioQCResult | None,
    video_path: Path | None = None,
    seed_path: Path | None = None,
    feedback_path: Path | None = None,
    visual_anchors_path: Path | None = None,
    review_decisions_path: Path | None = None,
    video_sha256: str | None = None,
    rules_version: str | None = None,
    ocr_enabled: bool = True,
    extract_visual_evidence: bool = False,
) -> VisualIntelligenceOutput:
    """Run the Phase 4 seed-comparison slice and write artifacts into run_dir."""
    issues: list[ValidatorIssue] = []
    artifacts: list[Path] = []
    timings: dict[str, float] = {}
    stage_start = time.perf_counter()

    result = VisualIntelligenceResult(
        visual_intelligence_version=VISUAL_INTELLIGENCE_VERSION,
        seed_present=seed_path is not None,
    )

    # --- visual frame observations + OCR (media-driven; run with or without a seed) ---
    ocr_text_tracks: list[TextTrack] = []
    camera_candidates: list[CameraMotionCandidate] = []
    entity_tracks: list[EntityTrack] = []
    cache: FrameCache | None = None
    if ledger is not None and media is not None and video_path is not None:
        cache = FrameCache(video_path, ledger)
        clock = AnnotationClock.from_ledger(ledger)
        start = time.perf_counter()
        observations = _build_observations(
            cache, ledger, clock, shot_truth, run_dir, artifacts, issues
        )
        result.frame_observation_count = len(observations)
        _timed(timings, "frame_observations", start)
        start = time.perf_counter()
        camera_candidates = _run_camera_stage(
            cache, ledger, clock, shot_truth, run_dir, artifacts, issues, result
        )
        _timed(timings, "camera_motion", start)
        if shot_truth is not None:
            start = time.perf_counter()
            speed_evidence = build_playback_speed_evidence(cache.gray_frames(), shot_truth)
            issues.extend(visual_validator.validate_speed_evidence(speed_evidence))
            artifacts.append(
                visual_writer.write_playback_speed_evidence(
                    run_dir / "visual" / "speed", speed_evidence
                )
            )
            _timed(timings, "playback_speed", start)
        if visual_anchors_path is not None:
            start = time.perf_counter()
            entity_tracks = _run_tracking_stage(
                cache, ledger, clock, media, shot_truth, visual_anchors_path,
                run_dir, artifacts, issues, result,
            )
            _timed(timings, "tracking", start)
        if ocr_enabled:
            start = time.perf_counter()
            ocr_text_tracks = _run_ocr_stage(
                cache, ledger, clock, run_dir, artifacts, issues, result
            )
            _timed(timings, "ocr", start)

    # No seed: Phase 4 has no claims to compare, so it produces no findings and
    # does not downgrade the run. Emit a real (not fake) visual QC reflecting the
    # observations that did run, so the manifest still records the stage.
    if seed_path is None:
        result.overall_status = ri_validator.compute_overall_status([], issues)
        review_dir = run_dir / "review"
        artifacts.append(review_writer.write_visual_qc(review_dir, result))
        _timed(timings, "visual_intelligence_total", stage_start)
        return VisualIntelligenceOutput(result, issues, artifacts, timings)

    seed_dir = run_dir / "seed"

    # --- immutable snapshot ---
    start = time.perf_counter()
    snapshot = snapshot_mod.snapshot_seed(seed_path, seed_dir)
    artifacts.append(seed_dir / "seed_original.txt")
    artifacts.append(seed_dir / "seed_sha256.txt")
    _timed(timings, "seed_snapshot", start)

    # --- parse (from a decoded copy of the immutable bytes) ---
    start = time.perf_counter()
    seed_text = (seed_dir / "seed_original.txt").read_text(encoding="utf-8", errors="replace")
    doc = parse_seed_text(seed_text, snapshot)
    claims = extract_claims(doc)
    # Platform-semantic atomicity diagnostics (seed-review leads, INFO).
    doc.issues.extend(collect_seed_diagnostics(doc))
    _timed(timings, "seed_parse", start)
    result.seed_parsed = True
    result.seed_claim_count = len(claims)
    artifacts.append(review_writer.write_seed_parse(seed_dir, doc))
    artifacts.append(review_writer.write_seed_parse_issues(seed_dir, doc))

    issues.extend(seed_validator.validate_seed_snapshot(seed_dir, doc))
    issues.extend(seed_validator.validate_seed_document(doc))
    issues.extend(seed_validator.validate_claims_source_links(claims))

    # --- task feedback ---
    feedback_directives = []
    if feedback_path is not None:
        feedback_dir = run_dir / "feedback"
        fb_snapshot = snapshot_mod.snapshot_feedback(feedback_path, feedback_dir)
        fb_text = (feedback_dir / "feedback_original.txt").read_text(
            encoding="utf-8", errors="replace"
        )
        fb_doc = feedback_mod.parse_feedback_text(fb_text, fb_snapshot)
        feedback_directives = fb_doc.directives
        artifacts.append(feedback_dir / "feedback_original.txt")
        artifacts.append(review_writer.write_feedback_directives(feedback_dir, fb_doc))

    # --- structural comparison (zero new CV) ---
    start = time.perf_counter()
    comparison = compare_seed(
        doc, claims, media, shot_truth, ocr_text_tracks, camera_candidates
    )
    _timed(timings, "seed_comparison", start)

    # --- human decisions: apply as an evidence-override layer, THEN recompute (F) ---
    if review_decisions_path is not None and video_sha256 is not None:
        try:
            decisions = load_decisions(review_decisions_path)
            applications = apply_decisions_to_claims(
                decisions, comparison.claims, video_sha256, rules_version or "unknown"
            )
            comparison.rows = build_rows(comparison.claims)  # reflect overrides
            review_dir = run_dir / "review"
            artifacts.append(review_writer.write_review_decisions(review_dir, decisions))
            artifacts.append(
                review_writer.write_decision_applications(review_dir, applications)
            )
        except DecisionLoadError as exc:
            issues.append(
                ValidatorIssue(
                    rule_id="P4-REVIEW-001",
                    severity=Severity.FAIL,
                    location=str(review_decisions_path.name),
                    message=f"Review decisions rejected: {exc}",
                )
            )

    # Artifacts reflect the applied (post-override) claim state.
    artifacts.append(review_writer.write_seed_claims(seed_dir, comparison.claims))

    # --- proposals + queue + triage (recomputed after overrides) ---
    start = time.perf_counter()
    frame_for_time = (
        _make_time_to_frame(ledger, AnnotationClock.from_ledger(ledger))
        if ledger is not None
        else None
    )
    proposals = build_proposals(comparison, feedback_directives)
    queue_items = build_review_queue(comparison, feedback_directives, frame_for_time)
    triage = build_triage(comparison, time.perf_counter() - stage_start)
    _timed(timings, "review_aggregation", start)

    # Link claim-level proposal outcomes back into the matrix rows.
    claim_outcomes = {
        p.subject_id: p.outcome for p in proposals if p.level == "claim"
    }
    for row in comparison.rows:
        row.review_proposal = claim_outcomes.get(row.claim_id)

    # --- high-risk visual evidence bundles (Z; opt-in) ---
    if extract_visual_evidence and cache is not None and ledger is not None:
        try:
            bundles = extract_evidence_bundles(
                queue_items,
                cache,
                ledger.frame_count,
                _frame_to_time(ledger, AnnotationClock.from_ledger(ledger)),
                run_dir / "visual_evidence",
            )
            artifacts.extend(bundles)
        except (MetricDecodeError, ValueError) as exc:
            issues.append(
                ValidatorIssue(
                    rule_id="P4-EVIDENCE-000",
                    severity=Severity.WARN,
                    location="visual_evidence",
                    message=f"Evidence bundle extraction failed: {exc}",
                )
            )

    # --- validation ---
    issues.extend(ri_validator.validate_matrix(comparison.rows))
    issues.extend(ri_validator.validate_proposals(proposals, comparison.rows))
    # Compute status AFTER all P4 validators so a P4 FAIL forces FAILED (G).
    overall = ri_validator.compute_overall_status(queue_items, issues)
    issues.extend(ri_validator.validate_qc_gate(overall, queue_items))

    # --- populate result summary ---
    result.foundation_status = comparison.foundation_status
    result.seed_shot_count = comparison.seed_shot_count
    result.verified_shot_count = comparison.verified_shot_count
    result.proposal_counts = count_by_outcome(proposals)
    result.critical_review_item_count = sum(
        1 for i in queue_items if i.priority.value == "CRITICAL"
    )
    result.review_item_count = len(queue_items)
    result.overall_status = overall
    result.triage = triage
    result.foundation_checks = comparison.foundation_checks
    result.proposals = proposals

    # --- artifacts ---
    review_dir = run_dir / "review"
    artifacts.append(review_writer.write_seed_triage(seed_dir, triage))
    artifacts.append(review_writer.write_claim_evidence_matrix_csv(review_dir, comparison.rows))
    artifacts.append(review_writer.write_claim_evidence_matrix_json(review_dir, comparison.rows))
    artifacts.append(review_writer.write_review_proposals(review_dir, proposals))
    artifacts.append(review_writer.write_visual_review_queue(review_dir, queue_items))
    artifacts.append(review_writer.write_visual_qc(review_dir, result))

    _timed(timings, "visual_intelligence_total", stage_start)
    # entity_tracks are consumed inside the tracking stage; audio_truth is
    # reserved for a future audio-linkage slice.
    _ = (audio_truth, entity_tracks)
    return VisualIntelligenceOutput(result, issues, artifacts, timings)


def _build_observations(
    cache: FrameCache,
    ledger: FrameLedger,
    clock: AnnotationClock,
    shot_truth: ShotTruthResult | None,
    run_dir: Path,
    artifacts: list[Path],
    issues: list[ValidatorIssue],
) -> list[FrameObservation]:
    """Decode once (shared cache) and build the per-frame observation ledger."""
    visual_dir = run_dir / "visual"
    try:
        observations = build_frame_observations(cache, ledger, clock, shot_truth)
    except (MetricDecodeError, ValueError) as exc:
        issues.append(
            ValidatorIssue(
                rule_id="P4-OBS-000",
                severity=Severity.WARN,
                location="frame_observations",
                message=f"Frame observation pass failed: {exc}",
            )
        )
        return []
    issues.extend(visual_validator.validate_frame_observations(observations, ledger))
    artifacts.append(visual_writer.write_frame_observations_csv(visual_dir, observations))
    artifacts.append(visual_writer.write_frame_observations_jsonl(visual_dir, observations))
    artifacts.append(visual_writer.write_enriched_frame_ledger(visual_dir, ledger, observations))
    return observations


def _run_camera_stage(
    cache: FrameCache,
    ledger: FrameLedger,
    clock: AnnotationClock,
    shot_truth: ShotTruthResult | None,
    run_dir: Path,
    artifacts: list[Path],
    issues: list[ValidatorIssue],
    result: VisualIntelligenceResult,
) -> list[CameraMotionCandidate]:
    """Per-shot global camera-motion phases from the shared gray grid.

    Returns the smoothed phases so seed CAMERA_MOVEMENT claims can be compared (R).
    """
    camera_dir = run_dir / "visual" / "camera"
    try:
        gray = cache.gray_frames()
        candidates, raw_pairs = analyze_camera_motion(gray, ledger, clock, shot_truth)
    except (MetricDecodeError, ValueError) as exc:
        issues.append(
            ValidatorIssue(
                rule_id="P4-CAMERA-000",
                severity=Severity.WARN,
                location="camera_motion",
                message=f"Camera-motion pass failed: {exc}",
            )
        )
        return []
    issues.extend(
        visual_validator.validate_camera_candidates(candidates, shot_truth, ledger.frame_count)
    )
    result.camera_phase_count = len(candidates)
    result.camera_direction_reversals = visual_validator.count_direction_reversals(candidates)
    artifacts.append(visual_writer.write_camera_events(camera_dir, candidates))
    artifacts.append(visual_writer.write_camera_pair_metrics(camera_dir, raw_pairs))
    return candidates


def _run_tracking_stage(
    cache: FrameCache,
    ledger: FrameLedger,
    clock: AnnotationClock,
    media: MediaInfo,
    shot_truth: ShotTruthResult | None,
    anchors_path: Path,
    run_dir: Path,
    artifacts: list[Path],
    issues: list[ValidatorIssue],
    result: VisualIntelligenceResult,
) -> list[EntityTrack]:
    """Anchor-seeded local tracking + continuity hypotheses (S/T/U)."""
    entities_dir = run_dir / "visual" / "entities"
    stream = media.video_streams[0]
    full_w, full_h = stream.width, stream.height
    try:
        anchors = load_anchors(anchors_path)
        gray = cache.gray_frames()
        tracks = [track_anchor(gray, ledger, clock, a, full_w, full_h) for a in anchors]
    except (AnchorLoadError, MetricDecodeError, ValueError) as exc:
        issues.append(
            ValidatorIssue(
                rule_id="P4-ENTITY-000",
                severity=Severity.WARN,
                location="tracking",
                message=f"Tracking pass failed: {exc}",
            )
        )
        return []
    characters, objects, links = build_continuity(tracks)
    issues.extend(visual_validator.validate_tracks(tracks, ledger.frame_count))
    issues.extend(visual_validator.validate_hypotheses(characters, objects))
    result.character_hypothesis_count = len(characters)
    result.object_hypothesis_count = len(objects)
    artifacts.append(visual_writer.write_entity_tracks(entities_dir, tracks))
    artifacts.append(visual_writer.write_character_hypotheses(entities_dir, characters))
    artifacts.append(visual_writer.write_object_hypotheses(entities_dir, objects))
    artifacts.append(visual_writer.write_continuity_links(entities_dir, links))

    char_tracks = [t for t in tracks if t.entity_type.upper() == "CHARACTER"]
    obj_tracks = [t for t in tracks if t.entity_type.upper() == "OBJECT"]

    # V: ownership/contact events.
    contact_events = build_contact_events(char_tracks, obj_tracks)
    issues.extend(visual_validator.validate_contact_events(contact_events))
    # W: final object-state checks (removal never inferred from a shot ending).
    final_checks = build_final_state_checks(obj_tracks, shot_truth, contact_events)
    issues.extend(visual_validator.validate_final_states(final_checks, shot_truth))
    artifacts.append(visual_writer.write_final_state_checks(entities_dir, final_checks))
    # X: atomic action-boundary candidates (semantic label stays None).
    frame_time = _frame_to_time(ledger, clock)
    action_candidates = build_action_candidates(
        char_tracks, obj_tracks, contact_events, frame_time
    )
    issues.extend(
        visual_validator.validate_action_candidates(action_candidates, ledger.frame_count)
    )
    result.action_candidate_count = len(action_candidates)
    result.semantic_review_required_count = sum(
        1 for a in action_candidates if a.semantic_label is None
    )
    actions_dir = run_dir / "visual" / "actions"
    artifacts.append(visual_writer.write_contact_events(actions_dir, contact_events))
    artifacts.append(visual_writer.write_action_candidates(actions_dir, action_candidates))
    return tracks


def _run_ocr_stage(
    cache: FrameCache,
    ledger: FrameLedger,
    clock: AnnotationClock,
    run_dir: Path,
    artifacts: list[Path],
    issues: list[ValidatorIssue],
    result: VisualIntelligenceResult,
) -> list[TextTrack]:
    """Optional local OCR. Degrades safely: if Tesseract is unavailable only
    ocr_status.json is written (no fake artifacts) and OCR status is UNAVAILABLE.
    Returns the text tracks so seed ON_SCREEN_TEXT claims can be compared (N)."""
    ocr_dir = run_dir / "visual" / "ocr"
    adapter = TesseractAdapter()
    info = adapter.engine_info()
    result.ocr_status = info.status
    artifacts.append(visual_writer.write_ocr_status(ocr_dir, info))
    if info.status == OCRStatus.UNAVAILABLE:
        return []
    try:
        # One shared sequential color decode feeds OCR (H); candidate text
        # regions focus recognition with a whole-frame fallback (I).
        ocr_result = run_ocr(
            cache.iter_color_frames(),
            ledger,
            clock,
            adapter,
            total_frames=ledger.frame_count,
            region_detector=_region_boxes,
        )
    except (MetricDecodeError, ValueError) as exc:
        issues.append(
            ValidatorIssue(
                rule_id="P4-OCR-000",
                severity=Severity.WARN,
                location="ocr",
                message=f"OCR pass failed: {exc}",
            )
        )
        return []
    # Re-write the (possibly DEGRADED) engine info from the pass (M).
    result.ocr_status = ocr_result.engine_info.status
    artifacts.append(visual_writer.write_ocr_status(ocr_dir, ocr_result.engine_info))
    issues.extend(ri_validator.validate_text_tracks(ocr_result.text_tracks))
    result.ocr_track_count = len(ocr_result.text_tracks)
    result.source_verified_ocr_count = sum(
        1 for t in ocr_result.text_tracks if t.caption_text_eligible
    )
    artifacts.append(visual_writer.write_ocr_observations(ocr_dir, ocr_result.observations))
    artifacts.append(visual_writer.write_text_tracks(ocr_dir, ocr_result.text_tracks))
    artifacts.append(
        visual_writer.write_watermark_candidates(ocr_dir, ocr_result.watermark_candidates)
    )
    return ocr_result.text_tracks


def _region_boxes(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    from .ocr.regions import detect_text_regions

    return [(r.x, r.y, r.width, r.height) for r in detect_text_regions(gray)]


def _frame_to_time(
    ledger: FrameLedger, clock: AnnotationClock
) -> Callable[[int], Fraction | None]:
    """Resolve a ledger frame index to its exact annotation time."""

    def resolve(frame_index: int) -> Fraction | None:
        if 0 <= frame_index < ledger.frame_count:
            src = ledger.frames[frame_index].pts_time_seconds
            if src is not None:
                return clock.to_annotation(src)
        return None

    return resolve


def _make_time_to_frame(
    ledger: FrameLedger, clock: AnnotationClock
) -> Callable[[Fraction], int | None]:
    """Resolve an annotation time to the containing ledger frame index (AA)."""
    anns: list[tuple[Fraction, int]] = []
    for record in ledger.frames:
        if record.pts_time_seconds is not None:
            anns.append((clock.to_annotation(record.pts_time_seconds), record.frame_index))
    anns.sort()

    def resolve(t: Fraction) -> int | None:
        chosen: int | None = None
        for ann_time, index in anns:
            if ann_time <= t:
                chosen = index
            else:
                break
        return chosen if chosen is not None else (anns[0][1] if anns else None)

    return resolve
