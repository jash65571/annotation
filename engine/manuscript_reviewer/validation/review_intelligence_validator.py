"""Phase 4 review-intelligence validators (P4-CLAIM/REVIEW/OCR/TEXT/QC/PRIVACY).

These encode the invariants Phase 4 exists to protect. They run every slice so
new subsystems can never silently regress a safety property.
"""

from __future__ import annotations

from ..models.evidence import EvidenceReference
from ..models.review_intelligence import (
    ClaimEvidenceRow,
    EvidenceStatus,
    ProposalReasonCode,
    ReviewPriority,
    ReviewProposal,
    ReviewProposalOutcome,
    ReviewQueueItem,
    TextTrack,
)
from ..models.validation import Severity, ValidatorIssue

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
        ProposalReasonCode.OBJECT_OWNERSHIP_CONTRADICTION,
        ProposalReasonCode.ENTITY_NOT_VISIBLE_AT_CLAIM_TIME,
    }
)


def _is_graded(ref: EvidenceReference) -> bool:
    """A reference can back a SUPPORTED claim only when it anchors to concrete
    media (exact frames, PTS, an artifact) or is human verification — never a
    prose-only note (AB)."""
    return (
        ref.start_frame is not None
        or ref.start_pts is not None
        or bool(ref.artifact_paths)
        or ref.is_factual
    )


def validate_matrix(rows: list[ClaimEvidenceRow]) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    for row in rows:
        # P4-CLAIM-001: SUPPORTED claim must have evidence.
        if (
            row.evidence_status in (EvidenceStatus.SUPPORTED, EvidenceStatus.PARTIALLY_SUPPORTED)
            and not row.supporting_evidence_refs
        ):
            issues.append(
                ValidatorIssue(
                    rule_id="P4-CLAIM-001",
                    severity=Severity.FAIL,
                    location=f"claim {row.claim_id}",
                    message="Claim marked SUPPORTED with no supporting evidence.",
                )
            )
        # AB: a fully SUPPORTED claim needs at least one *graded* reference; a
        # prose-only structural note is not enough to call a claim factual.
        elif row.evidence_status == EvidenceStatus.SUPPORTED and not any(
            _is_graded(e) for e in row.supporting_evidence_refs
        ):
            issues.append(
                ValidatorIssue(
                    rule_id="P4-CLAIM-004",
                    severity=Severity.FAIL,
                    location=f"claim {row.claim_id}",
                    message="SUPPORTED claim has only prose evidence (no frame/artifact anchor).",
                )
            )
        # P4-CLAIM-002: CONTRADICTED claim must cite contradiction evidence.
        if (
            row.evidence_status == EvidenceStatus.CONTRADICTED
            and not row.contradicting_evidence_refs
        ):
            issues.append(
                ValidatorIssue(
                    rule_id="P4-CLAIM-002",
                    severity=Severity.WARN,
                    location=f"claim {row.claim_id}",
                    message="Claim marked CONTRADICTED without cited contradiction evidence.",
                )
            )
    return issues


def validate_proposals(
    proposals: list[ReviewProposal], rows: list[ClaimEvidenceRow]
) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    status_by_claim = {r.claim_id: r for r in rows}

    for proposal in proposals:
        # P4-REVIEW-001: a proposal is always machine, never a human decision.
        if proposal.proposed_by != "machine":
            issues.append(
                ValidatorIssue(
                    rule_id="P4-REVIEW-001",
                    severity=Severity.FAIL,
                    location=f"proposal {proposal.proposal_id}",
                    message=f"Machine proposal has proposed_by={proposal.proposed_by!r}.",
                )
            )
        # P4-REVIEW-002: REDO_REBUILD must carry a structural reason code.
        if proposal.outcome == ReviewProposalOutcome.REDO_REBUILD and not (
            set(proposal.reason_codes) & _STRUCTURAL_REASONS
        ):
            issues.append(
                ValidatorIssue(
                    rule_id="P4-REVIEW-002",
                    severity=Severity.FAIL,
                    location=f"proposal {proposal.proposal_id}",
                    message="REDO_REBUILD proposal lacks a foundational reason code.",
                )
            )
        # P4-REVIEW-003 / P4-CLAIM-003: KEEP forbidden while a material
        # contradiction or foundational-unresolved claim exists in scope.
        if proposal.outcome == ReviewProposalOutcome.KEEP and proposal.claim_ids:
            for claim_id in proposal.claim_ids:
                row = status_by_claim.get(claim_id)
                if row is None:
                    continue
                if row.evidence_status == EvidenceStatus.CONTRADICTED:
                    issues.append(
                        ValidatorIssue(
                            rule_id="P4-REVIEW-003",
                            severity=Severity.FAIL,
                            location=f"proposal {proposal.proposal_id}",
                            message=f"KEEP proposed while claim {claim_id} is CONTRADICTED.",
                        )
                    )
                elif row.foundational and row.evidence_status == EvidenceStatus.UNRESOLVED:
                    issues.append(
                        ValidatorIssue(
                            rule_id="P4-CLAIM-003",
                            severity=Severity.FAIL,
                            location=f"proposal {proposal.proposal_id}",
                            message=(
                                f"KEEP proposed while foundational claim {claim_id} is "
                                "UNRESOLVED."
                            ),
                        )
                    )
    return issues


def validate_text_tracks(tracks: list[TextTrack]) -> list[ValidatorIssue]:
    """P4-OCR-001 / P4-TEXT-001: machine OCR text is never caption-eligible
    without human source verification."""
    issues: list[ValidatorIssue] = []
    for track in tracks:
        if track.caption_text_eligible and track.verification_status.value in (
            "UNVERIFIED",
            "REJECTED",
        ):
            issues.append(
                ValidatorIssue(
                    rule_id="P4-OCR-001",
                    severity=Severity.FAIL,
                    location=f"text track {track.track_id}",
                    message="OCR text marked caption-eligible without source verification.",
                )
            )
    return issues


def validate_qc_gate(
    overall_status: str, queue_items: list[ReviewQueueItem]
) -> list[ValidatorIssue]:
    """P4-QC-001: top-level PASS forbidden while a CRITICAL review item remains."""
    issues: list[ValidatorIssue] = []
    has_critical = any(i.priority == ReviewPriority.CRITICAL for i in queue_items)
    if overall_status == "PASS" and has_critical:
        issues.append(
            ValidatorIssue(
                rule_id="P4-QC-001",
                severity=Severity.FAIL,
                location="visual_qc",
                message="Top-level PASS reported while a CRITICAL review item is unresolved.",
            )
        )
    return issues


def compute_overall_status(
    queue_items: list[ReviewQueueItem], issues: list[ValidatorIssue] | None = None
) -> str:
    """Phase 4 QC status (G):

    * any P4 validator FAIL  -> FAILED (visual_qc and the audit must agree);
    * otherwise any review item -> REVIEW_REQUIRED (never PASS with work left);
    * a clean completed stage  -> PASS.
    """
    if issues and any(
        i.severity == Severity.FAIL and i.rule_id.startswith("P4") for i in issues
    ):
        return "FAILED"
    if queue_items:
        return "REVIEW_REQUIRED"
    return "PASS"
