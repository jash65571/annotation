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

from .artifacts import review_writer
from .models.audio import AudioQCResult
from .models.frame import FrameLedger
from .models.media import MediaInfo
from .models.review_intelligence import VisualIntelligenceResult
from .models.shot_truth import ShotTruthResult
from .models.validation import Severity, ValidatorIssue
from .review.decisions import DecisionLoadError, apply_decisions, load_decisions
from .review.proposals import build_proposals, count_by_outcome
from .review.queue import build_review_queue, build_triage
from .seed import feedback as feedback_mod
from .seed import snapshot as snapshot_mod
from .seed.claims import extract_claims
from .seed.comparison import compare_seed
from .seed.parser import parse_seed_text
from .validation import review_intelligence_validator as ri_validator
from .validation import seed_validator

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

    # No seed: Phase 4 has no claims to compare in this slice, so it produces no
    # findings and does not downgrade the run. Emit a real (not fake) visual QC
    # reflecting that, so the manifest still records the stage.
    if seed_path is None:
        result.overall_status = ri_validator.compute_overall_status([])
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
    comparison = compare_seed(doc, claims, media, shot_truth)
    _timed(timings, "seed_comparison", start)
    artifacts.append(review_writer.write_seed_claims(seed_dir, comparison.claims))

    # --- proposals + queue + triage ---
    start = time.perf_counter()
    proposals = build_proposals(comparison, feedback_directives)
    queue_items = build_review_queue(comparison, feedback_directives)
    triage = build_triage(comparison, time.perf_counter() - stage_start)
    _timed(timings, "review_aggregation", start)

    # Link claim-level proposal outcomes back into the matrix rows.
    claim_outcomes = {
        p.subject_id: p.outcome for p in proposals if p.level == "claim"
    }
    for row in comparison.rows:
        row.review_proposal = claim_outcomes.get(row.claim_id)

    # --- optional human decisions (bound to media/rules) ---
    if review_decisions_path is not None and video_sha256 is not None:
        try:
            decisions = load_decisions(review_decisions_path)
            applications = apply_decisions(
                decisions, video_sha256, rules_version or "unknown"
            )
            stale = [a for a in applications if a.stale]
            if stale:
                logger.warning("%d stale human decision(s) skipped", len(stale))
        except DecisionLoadError as exc:
            issues.append(
                ValidatorIssue(
                    rule_id="P4-REVIEW-001",
                    severity=Severity.FAIL,
                    location=str(review_decisions_path.name),
                    message=f"Review decisions rejected: {exc}",
                )
            )

    # --- validation ---
    issues.extend(ri_validator.validate_matrix(comparison.rows))
    issues.extend(ri_validator.validate_proposals(proposals, comparison.rows))
    overall = ri_validator.compute_overall_status(queue_items)
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
    # Reserved for later slices (frame observations, tracking, OCR, anchors).
    _ = (ledger, audio_truth, visual_anchors_path, ocr_enabled, extract_visual_evidence)
    return VisualIntelligenceOutput(result, issues, artifacts, timings)
