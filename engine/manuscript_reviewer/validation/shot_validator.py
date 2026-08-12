"""Phase 2 validators: pair integrity, candidate integrity, shot coverage,
transition legality, evidence completeness, and the no-hidden-uncertainty rule."""

from __future__ import annotations

import itertools
from fractions import Fraction

from ..media.clock import AnnotationClock
from ..models.frame import FrameLedger
from ..models.shot_truth import (
    BoundaryCandidate,
    CandidateStatus,
    PairMetrics,
    ShotProposal,
    TransitionStatus,
)
from ..models.validation import Severity, ValidatorIssue
from ..rules.loader import load_rules


def _issue(rule_id: str, severity: Severity, location: str, message: str) -> ValidatorIssue:
    return ValidatorIssue(rule_id=rule_id, severity=severity, location=location, message=message)


def validate_pairs(ledger: FrameLedger, pairs: list[PairMetrics]) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    expected = ledger.frame_count - 1
    if len(pairs) != expected:
        issues.append(
            _issue(
                "P2-PAIR-001",
                Severity.FAIL,
                "adjacent_metrics",
                f"Adjacent metric row count ({len(pairs)}) != frame_count - 1 ({expected}).",
            )
        )
    for i, pair in enumerate(pairs):
        if pair.left_frame_index != i or pair.right_frame_index != i + 1:
            issues.append(
                _issue(
                    "P2-PAIR-002",
                    Severity.FAIL,
                    f"pair {i}",
                    f"Pair {i} references frames {pair.left_frame_index}→"
                    f"{pair.right_frame_index}; must be exactly consecutive ledger frames.",
                )
            )
            break
    for pair in pairs:
        if (
            pair.left_pts is not None
            and pair.right_pts is not None
            and pair.right_pts <= pair.left_pts
        ):
            issues.append(
                _issue(
                    "P2-PAIR-003",
                    Severity.FAIL,
                    f"pair {pair.left_frame_index}",
                    f"Pair timestamps go backward: {pair.left_pts} → {pair.right_pts}.",
                )
            )
            break
    return issues


def validate_candidates(
    ledger: FrameLedger, candidates: list[BoundaryCandidate]
) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    n = ledger.frame_count
    for candidate in candidates:
        if not (
            0 <= candidate.left_frame_index < n
            and candidate.right_frame_index == candidate.left_frame_index + 1
            and candidate.right_frame_index < n
        ):
            issues.append(
                _issue(
                    "P2-CAND-001",
                    Severity.FAIL,
                    candidate.candidate_id,
                    f"Candidate references invalid adjacent frames "
                    f"{candidate.left_frame_index}→{candidate.right_frame_index}.",
                )
            )
            continue
        right_time = ledger.frames[candidate.right_frame_index].pts_time_seconds
        if candidate.boundary_time_exact != right_time:
            issues.append(
                _issue(
                    "P2-CAND-002",
                    Severity.FAIL,
                    candidate.candidate_id,
                    "Candidate boundary exact time does not equal the incoming/right "
                    "frame PTS time.",
                )
            )
    return issues


def validate_shot_timeline(
    ledger: FrameLedger,
    shots: list[ShotProposal],
    candidates: list[BoundaryCandidate],
    annotation_endpoint: object,
    clock: AnnotationClock,
) -> list[ValidatorIssue]:
    """Continuous annotation-interval validation (separate from frame-range
    ownership validation): the timeline must be gapless, overlap-free, anchored
    to boundary exact times, and end at the canonical annotation endpoint.
    All comparisons use the ANNOTATION clock."""
    issues: list[ValidatorIssue] = []
    if not shots or not ledger.frames:
        return issues

    if (
        ledger.frames[0].pts_time_seconds is not None
        and shots[0].start_exact != Fraction(0)
    ):
        issues.append(
            _issue(
                "P2-TIME-001",
                Severity.FAIL,
                "shot 1",
                "Shot 1 start_exact does not equal the annotation timeline start (0).",
            )
        )

    for prev, current in itertools.pairwise(shots):
        if prev.end_exact != current.start_exact:
            issues.append(
                _issue(
                    "P2-TIME-002",
                    Severity.FAIL,
                    f"shot {current.shot_index}",
                    f"Temporal gap/overlap: shot {prev.shot_index} end_exact "
                    f"({prev.end_exact}) != shot {current.shot_index} start_exact "
                    f"({current.start_exact}).",
                )
            )

    by_id = {c.candidate_id: c for c in candidates}
    for prev, current in itertools.pairwise(shots):
        boundary = by_id.get(current.supporting_boundary_id or "")
        if boundary is None:
            continue
        source_boundary = ledger.frames[boundary.right_frame_index].pts_time_seconds
        boundary_time = (
            clock.to_annotation(source_boundary) if source_boundary is not None else None
        )
        if prev.end_exact != boundary_time:
            issues.append(
                _issue(
                    "P2-TIME-003",
                    Severity.FAIL,
                    f"shot {prev.shot_index}",
                    "Non-final shot end_exact does not equal its supporting "
                    "boundary exact time.",
                )
            )
        if current.start_exact != boundary_time:
            issues.append(
                _issue(
                    "P2-TIME-004",
                    Severity.FAIL,
                    f"shot {current.shot_index}",
                    "Incoming shot start_exact does not equal its supporting "
                    "boundary exact time.",
                )
            )

    if annotation_endpoint is not None and shots[-1].end_exact != annotation_endpoint:
        issues.append(
            _issue(
                "P2-TIME-005",
                Severity.FAIL,
                f"shot {shots[-1].shot_index}",
                "Final shot end_exact does not equal the canonical annotation "
                "endpoint.",
            )
        )
    return issues


def validate_shots(
    ledger: FrameLedger,
    shots: list[ShotProposal],
    candidates: list[BoundaryCandidate],
) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    if not shots:
        return issues
    n = ledger.frame_count
    rules = load_rules()
    allowed_transitions = set(rules.get("shots.allowed_transition_types", []))

    if shots[0].start_frame_index != 0:
        issues.append(
            _issue("P2-SHOT-001", Severity.FAIL, "shot 1", "Shot 1 does not begin at media start.")
        )
    if shots[-1].end_frame_index != n - 1:
        issues.append(
            _issue(
                "P2-SHOT-002",
                Severity.FAIL,
                f"shot {shots[-1].shot_index}",
                "Final shot does not end at media end.",
            )
        )
    for prev, current in itertools.pairwise(shots):
        if current.start_frame_index != prev.end_frame_index + 1:
            rule = (
                "P2-SHOT-003"
                if current.start_frame_index > prev.end_frame_index + 1
                else "P2-SHOT-004"
            )
            message = (
                "Gap between shot proposals."
                if rule == "P2-SHOT-003"
                else "Shot proposals overlap."
            )
            issues.append(
                _issue(rule, Severity.FAIL, f"shot {current.shot_index}", message)
            )

    supported_ids = [
        c.candidate_id for c in candidates if c.status == CandidateStatus.SUPPORTED
    ]
    used_ids = [s.supporting_boundary_id for s in shots if s.supporting_boundary_id]
    if sorted(supported_ids) != sorted(used_ids):
        issues.append(
            _issue(
                "P2-SHOT-005",
                Severity.FAIL,
                "shots",
                f"Supported boundaries {supported_ids} do not map 1:1 onto shot "
                f"boundaries {used_ids}.",
            )
        )

    opening = rules.get("shots.shot_one_transition", "Opening shot")
    if shots[0].transition_into_shot != opening:
        issues.append(
            _issue(
                "P2-SHOT-006",
                Severity.FAIL,
                "shot 1",
                f"Shot 1 transition must be {opening!r}.",
            )
        )
    for shot in shots[1:]:
        if shot.transition_into_shot == opening:
            issues.append(
                _issue(
                    "P2-SHOT-007",
                    Severity.FAIL,
                    f"shot {shot.shot_index}",
                    "Later shots cannot be marked Opening shot.",
                )
            )

    for shot in shots:
        transition = shot.transition_into_shot
        if transition is not None and transition not in allowed_transitions:
            issues.append(
                _issue(
                    "P2-TRANS-001",
                    Severity.FAIL,
                    f"shot {shot.shot_index}",
                    f"Transition {transition!r} is not in the allowed Manuscript menu.",
                )
            )
        if transition in ("L-cut", "J-cut"):
            issues.append(
                _issue(
                    "P2-TRANS-002",
                    Severity.FAIL,
                    f"shot {shot.shot_index}",
                    "L-cut/J-cut cannot be auto-finalized without audio evidence "
                    "(Phase 3).",
                )
            )
    return issues


def validate_evidence(
    candidates: list[BoundaryCandidate],
) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    for candidate in candidates:
        if candidate.status == CandidateStatus.SUPPORTED and not candidate.evidence_refs:
            issues.append(
                _issue(
                    "P2-EVID-001",
                    Severity.FAIL,
                    candidate.candidate_id,
                    "Supported boundary has no adjacent-frame evidence artifacts.",
                )
            )
        if candidate.status == CandidateStatus.REVIEW_REQUIRED and not candidate.evidence_refs:
            issues.append(
                _issue(
                    "P2-EVID-002",
                    Severity.FAIL,
                    candidate.candidate_id,
                    "REVIEW_REQUIRED candidate has no evidence artifacts.",
                )
            )
    return issues


def compute_shot_status(
    candidates: list[BoundaryCandidate],
    had_failure: bool,
    endpoint_conflict: bool = False,
) -> str:
    """Overall shot status. PASS is forbidden while unresolved potential real
    cuts remain (P2-QC-001) or while the annotation endpoint is unverified."""
    if had_failure:
        return "FAILED"
    if endpoint_conflict:
        return "REVIEW_REQUIRED"
    review = any(c.status == CandidateStatus.REVIEW_REQUIRED for c in candidates)
    unresolved_transition = any(
        c.status == CandidateStatus.SUPPORTED
        and c.transition is not None
        and c.transition.status
        in (TransitionStatus.UNRESOLVED, TransitionStatus.REVIEW_REQUIRED)
        for c in candidates
    )
    if review or unresolved_transition:
        return "REVIEW_REQUIRED"
    return "PASS"
