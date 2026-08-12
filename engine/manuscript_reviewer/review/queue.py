"""Visual review queue + early seed triage.

The queue never hides low-priority items. Every item states what needs review,
why, the exact shot/time/frame range, supporting AND contradicting evidence, and
a recommended reviewer action.
"""

from __future__ import annotations

from ..models.caption import SeedClaim
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


def build_review_queue(
    comparison: ComparisonResult,
    feedback: list[FeedbackDirective] | None = None,
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


def _unresolved_priority(claim: SeedClaim) -> ReviewPriority:
    if claim.claim_type == SeedClaimType.PROTECTED_TRAIT:
        return ReviewPriority.HIGH
    if claim.importance == ClaimImportance.FOUNDATIONAL:
        return ReviewPriority.HIGH
    if claim.claim_type in (SeedClaimType.ON_SCREEN_TEXT, SeedClaimType.PLAYBACK_SPEED):
        return ReviewPriority.NORMAL
    return ReviewPriority.LOW


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
