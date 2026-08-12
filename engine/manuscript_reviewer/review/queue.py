"""Visual review queue + early seed triage.

The queue never hides low-priority items. Every item states what needs review,
why, the exact shot/time/frame range, supporting AND contradicting evidence, and
a recommended reviewer action.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

from ..models.caption import SeedClaim
from ..models.evidence import EvidenceReference, EvidenceType
from ..models.review_intelligence import (
    ClaimImportance,
    EvidenceStatus,
    FeedbackDirective,
    FoundationStatus,
    ProposalReasonCode,
    ReviewerAction,
    ReviewPriority,
    ReviewQueueItem,
    SeedClaimType,
    SeedTriage,
    TriageStrategy,
)
from ..seed.comparison import ComparisonResult


class _Counter:
    def __init__(self) -> None:
        self.value = 0

    def next(self) -> str:
        self.value += 1
        return f"RQ-{self.value:04d}"


#: Resolve an annotation time to the containing ledger frame index (AA).
FrameResolver = Callable[[Fraction], int | None]

#: An unresolved camera phase is only "material" without a seed once it spans at
#: least this many frames (brief unresolved phases are not flagged on clean clips).
_MATERIAL_CAMERA_FRAMES = 15


def build_review_queue(
    comparison: ComparisonResult,
    feedback: list[FeedbackDirective] | None = None,
    frame_for_time: FrameResolver | None = None,
) -> list[ReviewQueueItem]:
    counter = _Counter()
    items: list[ReviewQueueItem] = []

    # Foundation-level critical items.
    for check in comparison.foundation_checks:
        if check.status != FoundationStatus.CONTRADICTED:
            continue
        items.append(
            ReviewQueueItem(
                item_id=counter.next(),
                priority=ReviewPriority.CRITICAL,
                title=f"Foundation contradicted: {check.subject}",
                reason=check.detail or "Structural foundation contradiction.",
                supporting_evidence_refs=[],
                contradicting_evidence_refs=list(check.evidence_refs),
                recommended_action=ReviewerAction.REBUILD_SECTION,
            )
        )

    # Claim-level items.
    for claim in comparison.claims:
        item = _claim_item(counter, claim)
        if item is not None:
            _fill_frame_range(item, frame_for_time)
            items.append(item)

    # Task feedback (HIGH: requirement unresolved).
    for directive in feedback or []:
        items.append(
            ReviewQueueItem(
                item_id=counter.next(),
                priority=ReviewPriority.HIGH,
                title=f"Task feedback requirement: {directive.directive_id}",
                reason=directive.raw_text.strip(),
                recommended_action=ReviewerAction.VERIFY,
            )
        )
    return items


def _claim_item(counter: _Counter, claim: SeedClaim) -> ReviewQueueItem | None:
    status = claim.evidence_status or EvidenceStatus.UNRESOLVED
    foundational = claim.importance == ClaimImportance.FOUNDATIONAL
    time_range = claim.seed_time_range
    start = time_range.start_seconds if time_range else None
    end = time_range.end_seconds if time_range else None

    if status == EvidenceStatus.SUPPORTED:
        return None  # nothing to review

    if status == EvidenceStatus.CONTRADICTED:
        priority = ReviewPriority.CRITICAL if foundational else ReviewPriority.NORMAL
        action = ReviewerAction.REJECT_CLAIM if foundational else ReviewerAction.CORRECT_BOUNDARY
        title = f"Contradicted claim: {claim.claim_type.value if claim.claim_type else '?'}"
    elif status == EvidenceStatus.PARTIALLY_SUPPORTED:
        priority = ReviewPriority.NORMAL
        action = ReviewerAction.CORRECT_BOUNDARY
        title = f"Partially supported: {claim.claim_type.value if claim.claim_type else '?'}"
    else:  # UNRESOLVED
        priority = _unresolved_priority(claim)
        action = ReviewerAction.VERIFY
        title = f"Unresolved claim: {claim.claim_type.value if claim.claim_type else '?'}"

    contradicting = (
        list(claim.evidence) if status == EvidenceStatus.CONTRADICTED else []
    )
    supporting = (
        list(claim.evidence) if status == EvidenceStatus.PARTIALLY_SUPPORTED else []
    )
    return ReviewQueueItem(
        item_id=counter.next(),
        priority=priority,
        title=title,
        reason=(claim.text or "")[:200],
        shot_number=claim.shot_number,
        start_exact=start,
        end_exact=end,
        supporting_evidence_refs=supporting,
        contradicting_evidence_refs=contradicting,
        recommended_action=action,
        related_claim_ids=[claim.claim_id],
    )


def _fill_frame_range(item: ReviewQueueItem, frame_for_time: FrameResolver | None) -> None:
    """AA: resolve exact frame range from the item's seed time via the annotation
    clock instead of leaving it empty."""
    if frame_for_time is None:
        return
    if item.start_exact is not None and item.start_frame is None:
        item.start_frame = frame_for_time(item.start_exact)
    if item.end_exact is not None and item.end_frame is None:
        item.end_frame = frame_for_time(item.end_exact)


def _unresolved_priority(claim: SeedClaim) -> ReviewPriority:
    if claim.claim_type == SeedClaimType.PROTECTED_TRAIT:
        return ReviewPriority.HIGH
    if claim.importance == ClaimImportance.FOUNDATIONAL:
        return ReviewPriority.HIGH
    if claim.claim_type in (SeedClaimType.ON_SCREEN_TEXT, SeedClaimType.PLAYBACK_SPEED):
        return ReviewPriority.NORMAL
    return ReviewPriority.LOW


def build_machine_review_items(
    evidence: object, ocr_status: str | None = None
) -> list[ReviewQueueItem]:
    """Review items for unresolved MACHINE visual evidence (item 8). These affect
    Phase 4 status even when the seed is clean or absent."""
    from ..seed.comparison import VisualEvidence

    assert isinstance(evidence, VisualEvidence)
    counter = _Counter()
    items: list[ReviewQueueItem] = []

    # CRITICAL: one seed/anchor id spanning two disjoint tracks (identity collision).
    for seed_id, tracks in evidence.entity_by_seed_id.items():
        if len(tracks) >= 2 and any(
            a.last_frame_index < b.first_frame_index or b.last_frame_index < a.first_frame_index
            for i, a in enumerate(tracks)
            for b in tracks[i + 1 :]
        ):
            # One graded frame-range ref per colliding track so the evidence bundle
            # renders BOTH spans the reviewer must disambiguate (item 19).
            refs = [
                _frame_ref(f"EV-IDENT-{seed_id}-{t.track_id}", t.first_frame_index,
                           t.last_frame_index, f"{t.track_id} span")
                for t in tracks
            ]
            lo = min(t.first_frame_index for t in tracks)
            hi = max(t.last_frame_index for t in tracks)
            items.append(_mitem(
                counter, ReviewPriority.CRITICAL,
                f"Identity collision: {seed_id}",
                f"{seed_id} spans two distinct tracks", ReviewerAction.CHOOSE_IDENTITY,
                lo, hi, refs=refs))

    # HIGH: reacquired identities, unresolved final states, unresolved camera, OCR degraded.
    for track in evidence.entity_tracks:
        if track.reacquired:
            ref = _frame_ref(f"EV-REACQ-{track.track_id}", track.first_frame_index,
                             track.last_frame_index, "reacquired track span")
            items.append(_mitem(counter, ReviewPriority.HIGH,
                                f"Reacquired identity: {track.track_id}",
                                "track was lost and reacquired; identity not proven",
                                ReviewerAction.VERIFY,
                                track.first_frame_index, track.last_frame_index, refs=[ref]))
    for check in evidence.final_state_checks:
        if not check.resolved:
            refs = []
            if check.final_visible_frame is not None:
                refs = [_frame_ref(f"EV-FINAL-{check.entity_id}-{check.shot_number}",
                                   check.final_visible_frame, check.final_visible_frame,
                                   "last trusted-visible frame")]
            items.append(_mitem(
                counter, ReviewPriority.HIGH,
                f"Unresolved final state: {check.entity_id}@shot{check.shot_number}",
                check.review_reason or "final object state unresolved",
                ReviewerAction.VERIFY, check.final_visible_frame, refs=refs))
    # Camera-unresolved is surfaced only when MATERIAL by magnitude (a long
    # unresolved phase), so a clean clip's brief unresolved phases stay silent
    # while a genuinely ambiguous sweep is flagged even without a seed.
    for cand in evidence.camera_candidates:
        span = cand.last_supporting_frame - cand.start_frame + 1
        if cand.review_required and cand.motion_class.value == "UNRESOLVED" \
                and span >= _MATERIAL_CAMERA_FRAMES:
            items.append(_mitem(
                counter, ReviewPriority.HIGH,
                f"Unresolved camera motion: {cand.candidate_id}",
                f"global-motion model did not resolve a {span}-frame phase",
                ReviewerAction.VERIFY, cand.start_frame, cand.last_supporting_frame,
                cand.shot_number))
    if ocr_status in ("DEGRADED",):
        items.append(_mitem(counter, ReviewPriority.HIGH, "OCR degraded",
                            "a meaningful fraction of frames failed OCR", ReviewerAction.VERIFY))

    # NORMAL: speed candidates and a per-shot semantic-action summary.
    for ev_speed in evidence.speed_evidence:
        if ev_speed.review_required:
            items.append(_mitem(counter, ReviewPriority.NORMAL,
                                f"Playback speed candidate: shot {ev_speed.shot_number}",
                                f"speed evidence: {ev_speed.conclusion.value}",
                                ReviewerAction.VERIFY, shot=ev_speed.shot_number))
    actions_by_shot: dict[int | None, int] = {}
    for a in evidence.action_candidates:
        if a.review_required:  # a human-resolved action no longer needs review
            actions_by_shot[a.shot_number] = actions_by_shot.get(a.shot_number, 0) + 1
    for shot, n in sorted(actions_by_shot.items(), key=lambda kv: (kv[0] is None, kv[0])):
        items.append(_mitem(counter, ReviewPriority.NORMAL,
                            f"Semantic action review: shot {shot}",
                            f"{n} atomic action candidate(s) need semantic review",
                            ReviewerAction.VERIFY, shot=shot))
    return items


def _frame_ref(evidence_id: str, start: int, end: int, note: str) -> EvidenceReference:
    """A graded FRAME_RANGE evidence pointer (where to look — not a semantic claim)."""
    return EvidenceReference(
        evidence_id=evidence_id, evidence_type=EvidenceType.FRAME_RANGE,
        start_frame=start, end_frame=end, source="tracking", notes=note,
    )


def _mitem(
    counter: _Counter, priority: ReviewPriority, title: str, reason: str,
    action: ReviewerAction, start_frame: int | None = None, end_frame: int | None = None,
    shot: int | None = None, refs: list[EvidenceReference] | None = None,
) -> ReviewQueueItem:
    return ReviewQueueItem(
        item_id=counter.next(), priority=priority, title=title, reason=reason,
        supporting_evidence_refs=refs or [],
        shot_number=shot, start_frame=start_frame, end_frame=end_frame,
        recommended_action=action,
    )


def build_triage(comparison: ComparisonResult, runtime_seconds: float) -> SeedTriage:
    """The early patch-vs-rebuild recommendation ("minute-8 triage"). An early
    recommendation only — never prevents later correction."""
    status = comparison.foundation_status
    reasons: list[ProposalReasonCode] = []
    if status == FoundationStatus.CONTRADICTED:
        strategy = TriageStrategy.REBUILD
        if comparison.seed_shot_count != comparison.verified_shot_count:
            reasons.append(ProposalReasonCode.SHOT_COUNT_CONTRADICTION)
        else:
            reasons.append(ProposalReasonCode.SHOT_BOUNDARY_FOUNDATION_INVALID)
    elif status == FoundationStatus.SUPPORTED:
        strategy = TriageStrategy.PATCH
        reasons.append(ProposalReasonCode.FOUNDATION_SUPPORTED)
    else:
        strategy = TriageStrategy.REVIEW_REQUIRED
        reasons.append(ProposalReasonCode.EVIDENCE_CANNOT_DECIDE)
    return SeedTriage(
        triage_runtime_seconds=round(runtime_seconds, 4),
        foundation_status=status,
        foundational_conflicts=list(comparison.foundational_conflicts),
        suggested_strategy=strategy,
        reason_codes=reasons,
    )
