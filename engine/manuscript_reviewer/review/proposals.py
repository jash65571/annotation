"""KEEP / FIX_ENRICH / REDO_REBUILD / HUMAN_DECISION_REQUIRED proposal builder.

Machine output is a PROPOSAL, never a human decision. Proposals are built at
several levels (claim, shot, overview, seed). Reasoning is recorded as
structured reason codes, never hidden in prose.

Gates (mandatory):

* KEEP only when the foundation is supported, no material contradiction exists,
  no high-risk unresolved claim exists, and no required evidence is missing.
* REDO_REBUILD must carry a foundational/structural reason code.
* A disproven foundation is never proposed KEEP.
"""

from __future__ import annotations

from ..models.caption import SeedClaim
from ..models.review_intelligence import (
    ClaimImportance,
    EvidenceStatus,
    FeedbackDirective,
    FoundationStatus,
    ProposalReasonCode,
    ReviewProposal,
    ReviewProposalOutcome,
    SeedClaimType,
)
from ..seed.comparison import ComparisonResult

_STRUCTURAL_REASONS = frozenset(
    {
        ProposalReasonCode.SHOT_COUNT_CONTRADICTION,
        ProposalReasonCode.SHOT_BOUNDARY_FOUNDATION_INVALID,
        ProposalReasonCode.MEDIA_IDENTITY_MISMATCH,
        ProposalReasonCode.TIMELINE_FOUNDATION_INVALID,
        ProposalReasonCode.CHARACTER_IDENTITY_COLLISION,
        ProposalReasonCode.CHARACTER_TRACK_SPLIT,
        ProposalReasonCode.OBJECT_IDENTITY_COLLISION,
        ProposalReasonCode.OBJECT_DUPLICATE_ID,
        ProposalReasonCode.ENTITY_NOT_VISIBLE_AT_CLAIM_TIME,
    }
)


class _Counter:
    def __init__(self) -> None:
        self.value = 0

    def next(self) -> str:
        self.value += 1
        return f"RP-{self.value:04d}"


def build_proposals(
    comparison: ComparisonResult,
    feedback: list[FeedbackDirective] | None = None,
) -> list[ReviewProposal]:
    counter = _Counter()
    claims = comparison.claims
    proposals: list[ReviewProposal] = []

    # --- claim-level ---
    for claim in claims:
        proposals.append(_claim_proposal(counter, claim))

    # --- shot-level ---
    shot_numbers = sorted({c.shot_number for c in claims if c.shot_number is not None})
    shot_foundation_broken = comparison.seed_shot_count != comparison.verified_shot_count and (
        comparison.seed_shot_count is not None and comparison.verified_shot_count is not None
    )
    for shot in shot_numbers:
        proposals.append(
            _shot_proposal(counter, shot, claims, shot_foundation_broken)
        )

    # --- overview-level ---
    proposals.append(_overview_proposal(counter, claims))

    # --- feedback (task feedback outranks seed content; always human-confirmed) ---
    for directive in feedback or []:
        proposals.append(
            ReviewProposal(
                proposal_id=counter.next(),
                level="feedback",
                subject_id=directive.directive_id,
                outcome=ReviewProposalOutcome.HUMAN_DECISION_REQUIRED,
                reason_codes=[ProposalReasonCode.TASK_FEEDBACK_UNRESOLVED],
                detail=directive.raw_text.strip(),
            )
        )

    # --- whole-seed ---
    proposals.append(_seed_proposal(counter, comparison, proposals))
    return proposals


def _claim_proposal(counter: _Counter, claim: SeedClaim) -> ReviewProposal:
    status = claim.evidence_status or EvidenceStatus.UNRESOLVED
    foundational = claim.importance == ClaimImportance.FOUNDATIONAL
    reasons: list[ProposalReasonCode] = []

    if status == EvidenceStatus.SUPPORTED:
        outcome = ReviewProposalOutcome.KEEP
        reasons.append(ProposalReasonCode.NO_MATERIAL_CONTRADICTION)
    elif status == EvidenceStatus.CONTRADICTED:
        if foundational:
            outcome = ReviewProposalOutcome.REDO_REBUILD
            reasons.append(_structural_reason_for(claim))
        else:
            outcome = ReviewProposalOutcome.FIX_ENRICH
            reasons.append(ProposalReasonCode.LOCAL_FACT_CONTRADICTED)
    elif status == EvidenceStatus.PARTIALLY_SUPPORTED:
        outcome = ReviewProposalOutcome.FIX_ENRICH
        reasons.append(ProposalReasonCode.LOCAL_FACT_WEAK)
    else:  # UNRESOLVED / NOT_APPLICABLE
        if claim.claim_type == SeedClaimType.PROTECTED_TRAIT:
            outcome = ReviewProposalOutcome.HUMAN_DECISION_REQUIRED
            reasons.append(ProposalReasonCode.EVIDENCE_CANNOT_DECIDE)
        elif foundational:
            outcome = ReviewProposalOutcome.HUMAN_DECISION_REQUIRED
            reasons.append(ProposalReasonCode.UNRESOLVED_HIGH_RISK)
        else:
            outcome = ReviewProposalOutcome.HUMAN_DECISION_REQUIRED
            reasons.append(ProposalReasonCode.EVIDENCE_CANNOT_DECIDE)

    return ReviewProposal(
        proposal_id=counter.next(),
        level="claim",
        subject_id=claim.claim_id,
        outcome=outcome,
        reason_codes=reasons,
        foundational=foundational,
        claim_ids=[claim.claim_id],
        evidence_refs=list(claim.evidence),
        detail=(claim.text or "")[:200],
    )


def _structural_reason_for(claim: SeedClaim) -> ProposalReasonCode:
    if claim.claim_type == SeedClaimType.SHOT_COUNT:
        return ProposalReasonCode.SHOT_COUNT_CONTRADICTION
    if claim.claim_type == SeedClaimType.SHOT_BOUNDARY:
        return ProposalReasonCode.SHOT_BOUNDARY_FOUNDATION_INVALID
    if claim.claim_type == SeedClaimType.MEDIA_ID:
        return ProposalReasonCode.MEDIA_IDENTITY_MISMATCH
    if claim.claim_type == SeedClaimType.TRANSITION:
        return ProposalReasonCode.SHOT_BOUNDARY_FOUNDATION_INVALID
    return ProposalReasonCode.TIMELINE_FOUNDATION_INVALID


def _shot_proposal(
    counter: _Counter,
    shot: int,
    claims: list[SeedClaim],
    shot_foundation_broken: bool,
) -> ReviewProposal:
    shot_claims = [c for c in claims if c.shot_number == shot]
    foundational_contradicted = [
        c
        for c in shot_claims
        if c.importance == ClaimImportance.FOUNDATIONAL
        and c.evidence_status == EvidenceStatus.CONTRADICTED
    ]
    local_issues = [
        c
        for c in shot_claims
        if c.importance == ClaimImportance.LOCAL
        and c.evidence_status in (EvidenceStatus.CONTRADICTED, EvidenceStatus.PARTIALLY_SUPPORTED)
    ]
    unresolved = [c for c in shot_claims if c.evidence_status == EvidenceStatus.UNRESOLVED]
    reasons: list[ProposalReasonCode] = []

    if shot_foundation_broken or foundational_contradicted:
        outcome = ReviewProposalOutcome.REDO_REBUILD
        if shot_foundation_broken:
            reasons.append(ProposalReasonCode.SHOT_COUNT_CONTRADICTION)
        if foundational_contradicted:
            reasons.append(ProposalReasonCode.SHOT_BOUNDARY_FOUNDATION_INVALID)
    elif local_issues:
        outcome = ReviewProposalOutcome.FIX_ENRICH
        reasons.append(ProposalReasonCode.LOCAL_FACT_CONTRADICTED)
    elif unresolved:
        outcome = ReviewProposalOutcome.HUMAN_DECISION_REQUIRED
        reasons.append(ProposalReasonCode.UNRESOLVED_HIGH_RISK)
    else:
        outcome = ReviewProposalOutcome.KEEP
        reasons.append(ProposalReasonCode.NO_MATERIAL_CONTRADICTION)

    return ReviewProposal(
        proposal_id=counter.next(),
        level="shot",
        subject_id=f"shot_{shot}",
        outcome=outcome,
        reason_codes=_dedup_reasons(reasons),
        foundational=bool(shot_foundation_broken or foundational_contradicted),
        claim_ids=[c.claim_id for c in shot_claims],
    )


def _overview_proposal(counter: _Counter, claims: list[SeedClaim]) -> ReviewProposal:
    overview_types = {
        SeedClaimType.CHARACTER_EXISTS,
        SeedClaimType.CHARACTER_TRAIT,
        SeedClaimType.OBJECT_EXISTS,
        SeedClaimType.OBJECT_IDENTITY,
        SeedClaimType.SCENE_STATE,
        SeedClaimType.STYLE_STATE,
        SeedClaimType.SOUND,
        SeedClaimType.VISUAL_CONCERN,
        SeedClaimType.AUDIO_CONCERN,
        SeedClaimType.PROTECTED_TRAIT,
    }
    overview_claims = [
        c for c in claims if c.claim_type in overview_types and c.shot_number is None
    ]
    contradicted = [c for c in overview_claims if c.evidence_status == EvidenceStatus.CONTRADICTED]
    unresolved = [c for c in overview_claims if c.evidence_status == EvidenceStatus.UNRESOLVED]
    foundational_contradicted = [
        c for c in contradicted if c.importance == ClaimImportance.FOUNDATIONAL
    ]
    reasons: list[ProposalReasonCode] = []
    if foundational_contradicted:
        outcome = ReviewProposalOutcome.REDO_REBUILD
        reasons.append(ProposalReasonCode.CHARACTER_IDENTITY_COLLISION)
    elif contradicted:
        outcome = ReviewProposalOutcome.FIX_ENRICH
        reasons.append(ProposalReasonCode.LOCAL_FACT_CONTRADICTED)
    elif unresolved:
        outcome = ReviewProposalOutcome.HUMAN_DECISION_REQUIRED
        reasons.append(ProposalReasonCode.EVIDENCE_CANNOT_DECIDE)
    else:
        outcome = ReviewProposalOutcome.KEEP
        reasons.append(ProposalReasonCode.NO_MATERIAL_CONTRADICTION)
    return ReviewProposal(
        proposal_id=counter.next(),
        level="overview",
        subject_id="overview",
        outcome=outcome,
        reason_codes=reasons,
        foundational=bool(foundational_contradicted),
        claim_ids=[c.claim_id for c in overview_claims],
    )


def _seed_proposal(
    counter: _Counter,
    comparison: ComparisonResult,
    proposals: list[ReviewProposal],
) -> ReviewProposal:
    reasons: list[ProposalReasonCode] = []
    any_redo = any(p.outcome == ReviewProposalOutcome.REDO_REBUILD for p in proposals)
    any_human = any(
        p.outcome == ReviewProposalOutcome.HUMAN_DECISION_REQUIRED for p in proposals
    )
    any_fix = any(p.outcome == ReviewProposalOutcome.FIX_ENRICH for p in proposals)

    if comparison.foundation_status == FoundationStatus.CONTRADICTED or any_redo:
        outcome = ReviewProposalOutcome.REDO_REBUILD
        reasons.append(ProposalReasonCode.SHOT_BOUNDARY_FOUNDATION_INVALID)
    elif any_human:
        outcome = ReviewProposalOutcome.HUMAN_DECISION_REQUIRED
        reasons.append(ProposalReasonCode.EVIDENCE_CANNOT_DECIDE)
    elif any_fix:
        outcome = ReviewProposalOutcome.FIX_ENRICH
        reasons.append(ProposalReasonCode.ENRICHMENT_NEEDED)
    else:
        outcome = ReviewProposalOutcome.KEEP
        reasons.append(ProposalReasonCode.NO_MATERIAL_CONTRADICTION)
    return ReviewProposal(
        proposal_id=counter.next(),
        level="seed",
        subject_id="whole_seed",
        outcome=outcome,
        reason_codes=reasons,
        foundational=comparison.foundation_status == FoundationStatus.CONTRADICTED,
    )


def _dedup_reasons(reasons: list[ProposalReasonCode]) -> list[ProposalReasonCode]:
    seen: set[ProposalReasonCode] = set()
    result: list[ProposalReasonCode] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            result.append(reason)
    return result


def count_by_outcome(proposals: list[ReviewProposal]) -> dict[str, int]:
    """Count *seed-facing* proposals (claim + section + seed levels)."""
    counts = {outcome.value: 0 for outcome in ReviewProposalOutcome}
    for proposal in proposals:
        if proposal.level == "claim":
            counts[proposal.outcome.value] += 1
    return counts
