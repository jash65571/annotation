"""Structural seed-vs-media comparison (the high-value, zero-CV Phase 4 wins).

Verifies seed claims against deterministic Phase 1-3 truth: shot count, shot
boundaries, transitions, timestamp containment, media identity, and seed-internal
C/O reference integrity. Sets each structural claim's :class:`EvidenceStatus`
and produces :class:`FoundationCheck` records and the claim<->evidence matrix
rows.

Semantic claim types (actions, traits, ownership, on-screen text wording, ...)
are deliberately left ``UNRESOLVED`` here — they require later visual slices or
human verification and are never inferred structurally. Protected traits are
always ``UNRESOLVED`` (never visually inferable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from ..models.caption import SeedClaim
from ..models.evidence import EvidenceReference, EvidenceType
from ..models.media import MediaInfo
from ..models.review_intelligence import (
    ClaimEvidenceRow,
    ClaimImportance,
    EvidenceStatus,
    FoundationCheck,
    FoundationStatus,
    ProposalReasonCode,
    SeedClaimType,
    SeedDocument,
)
from ..models.shot_truth import ShotProposal, ShotTruthResult

_TENTH = Fraction(1, 10)


def _round_tenth(value: Fraction) -> Fraction:
    return Fraction(round(value * 10), 10)


def _same_tenth(a: Fraction | None, b: Fraction | None) -> bool | None:
    if a is None or b is None:
        return None
    return _round_tenth(a) == _round_tenth(b)


@dataclass
class ComparisonResult:
    claims: list[SeedClaim]
    rows: list[ClaimEvidenceRow]
    foundation_checks: list[FoundationCheck]
    foundation_status: FoundationStatus
    seed_shot_count: int | None
    verified_shot_count: int | None
    foundational_conflicts: list[str] = field(default_factory=list)


def _struct_ref(ref_id: str, source: str, notes: str) -> EvidenceReference:
    return EvidenceReference(
        evidence_id=ref_id,
        evidence_type=EvidenceType.STRUCTURAL_CHECK,
        source=source,
        notes=notes,
    )


def compare_seed(
    doc: SeedDocument,
    claims: list[SeedClaim],
    media: MediaInfo | None,
    shot_truth: ShotTruthResult | None,
) -> ComparisonResult:
    checks: list[FoundationCheck] = []
    conflicts: list[str] = []

    verified_shot_count = shot_truth.proposed_shot_count if shot_truth is not None else None
    seed_shot_count = doc.seed_shot_count
    shots_by_index: dict[int, ShotProposal] = {}
    if shot_truth is not None:
        shots_by_index = {s.shot_index: s for s in shot_truth.shots}

    shot_foundation = _verify_shot_count(
        claims, seed_shot_count, verified_shot_count, shot_truth, checks, conflicts
    )

    _verify_media_identity(claims, doc, media, checks)
    _verify_shot_boundaries(claims, shots_by_index, shot_foundation, checks, conflicts)
    _verify_transitions(claims, shots_by_index, checks)
    _verify_timestamp_containment(claims, shots_by_index)
    _verify_entity_references(doc, claims, checks, conflicts)
    _mark_semantic_unresolved(claims)

    foundation_status = _overall_foundation(checks)
    rows = [_row(claim) for claim in claims]
    return ComparisonResult(
        claims=claims,
        rows=rows,
        foundation_checks=checks,
        foundation_status=foundation_status,
        seed_shot_count=seed_shot_count,
        verified_shot_count=verified_shot_count,
        foundational_conflicts=conflicts,
    )


def _verify_shot_count(
    claims: list[SeedClaim],
    seed_count: int | None,
    verified_count: int | None,
    shot_truth: ShotTruthResult | None,
    checks: list[FoundationCheck],
    conflicts: list[str],
) -> FoundationStatus:
    claim = next((c for c in claims if c.claim_type == SeedClaimType.SHOT_COUNT), None)
    if seed_count is None or verified_count is None or shot_truth is None:
        status = FoundationStatus.NOT_EVALUATED
        if claim is not None:
            claim.evidence_status = EvidenceStatus.UNRESOLVED
        checks.append(
            FoundationCheck(
                check_id="FC-SHOTS",
                subject="shot_structure",
                status=status,
                seed_value=str(seed_count) if seed_count is not None else None,
                verified_value=str(verified_count) if verified_count is not None else None,
                detail="Shot Truth unavailable; shot count not evaluated.",
            )
        )
        return status

    ref = _struct_ref(
        "EV-SHOTCOUNT", "shot_truth", f"verified proposed_shot_count={verified_count}"
    )
    if seed_count == verified_count:
        status = FoundationStatus.SUPPORTED
        reasons = [ProposalReasonCode.FOUNDATION_SUPPORTED]
        if claim is not None:
            claim.evidence_status = EvidenceStatus.SUPPORTED
            claim.evidence.append(ref)
    else:
        status = FoundationStatus.CONTRADICTED
        reasons = [ProposalReasonCode.SHOT_COUNT_CONTRADICTION]
        conflicts.append(f"Seed shot count {seed_count} != verified {verified_count}")
        if claim is not None:
            claim.evidence_status = EvidenceStatus.CONTRADICTED
            claim.evidence.append(ref)
    checks.append(
        FoundationCheck(
            check_id="FC-SHOTS",
            subject="shot_structure",
            status=status,
            seed_value=str(seed_count),
            verified_value=str(verified_count),
            reason_codes=reasons,
            evidence_refs=[ref],
        )
    )
    return status


def _verify_media_identity(
    claims: list[SeedClaim],
    doc: SeedDocument,
    media: MediaInfo | None,
    checks: list[FoundationCheck],
) -> None:
    claim = next((c for c in claims if c.claim_type == SeedClaimType.MEDIA_ID), None)
    if claim is None or doc.video_id is None:
        return
    filename = media.file_name if media is not None else None
    if filename is None:
        claim.evidence_status = EvidenceStatus.UNRESOLVED
        checks.append(
            FoundationCheck(
                check_id="FC-MEDIA",
                subject="media_identity",
                status=FoundationStatus.NOT_EVALUATED,
                seed_value=doc.video_id,
                detail="No media info to compare seed video id against.",
            )
        )
        return
    matches = doc.video_id.lower() in filename.lower()
    ref = _struct_ref("EV-MEDIAID", "media", f"source_path={filename}")
    if matches:
        claim.evidence_status = EvidenceStatus.SUPPORTED
        claim.evidence.append(ref)
        status = FoundationStatus.SUPPORTED
    else:
        # Cannot confirm identity from the filename alone; never assert a
        # mismatch as fact — leave it for review.
        claim.evidence_status = EvidenceStatus.UNRESOLVED
        status = FoundationStatus.UNRESOLVED
    checks.append(
        FoundationCheck(
            check_id="FC-MEDIA",
            subject="media_identity",
            status=status,
            seed_value=doc.video_id,
            verified_value=filename,
            evidence_refs=[ref],
            detail=None if matches else "Seed video id not found in source filename; review.",
        )
    )


def _verify_shot_boundaries(
    claims: list[SeedClaim],
    shots_by_index: dict[int, ShotProposal],
    shot_foundation: FoundationStatus,
    checks: list[FoundationCheck],
    conflicts: list[str],
) -> None:
    for claim in claims:
        if claim.claim_type != SeedClaimType.SHOT_BOUNDARY:
            continue
        shot = claim.shot_number
        if shot is None or not shots_by_index:
            claim.evidence_status = EvidenceStatus.UNRESOLVED
            continue
        proposal = shots_by_index.get(shot)
        if proposal is None:
            # Seed references a shot that verified structure does not contain.
            ref = _struct_ref(
                f"EV-BOUND-{shot}",
                "shot_truth",
                f"verified structure contains no shot {shot}",
            )
            claim.evidence_status = EvidenceStatus.CONTRADICTED
            claim.evidence.append(ref)
            conflicts.append(f"Seed shot {shot} has no verified counterpart")
            checks.append(
                FoundationCheck(
                    check_id=f"FC-BOUND-{shot}",
                    subject=f"shot_{shot}_boundary",
                    status=FoundationStatus.CONTRADICTED,
                    reason_codes=[ProposalReasonCode.SHOT_BOUNDARY_FOUNDATION_INVALID],
                    evidence_refs=[ref],
                    detail=f"No verified shot {shot} exists.",
                )
            )
            continue
        seed_range = claim.seed_time_range
        if seed_range is None:
            claim.evidence_status = EvidenceStatus.UNRESOLVED
            continue
        start_ok = _same_tenth(seed_range.start_seconds, proposal.start_exact)
        end_ok = _same_tenth(seed_range.end_seconds, proposal.end_exact)
        ref = _struct_ref(
            f"EV-BOUND-{shot}",
            "shot_truth",
            f"verified [{_fmt(proposal.start_exact)}, {_fmt(proposal.end_exact)}]",
        )
        if start_ok and end_ok:
            claim.evidence_status = EvidenceStatus.SUPPORTED
            claim.evidence.append(ref)
            status = FoundationStatus.SUPPORTED
        elif start_ok or end_ok:
            claim.evidence_status = EvidenceStatus.PARTIALLY_SUPPORTED
            claim.evidence.append(ref)
            status = FoundationStatus.UNRESOLVED
        else:
            claim.evidence_status = EvidenceStatus.CONTRADICTED
            claim.evidence.append(ref)
            status = FoundationStatus.CONTRADICTED
            conflicts.append(f"Seed shot {shot} boundary contradicts verified interval")
        checks.append(
            FoundationCheck(
                check_id=f"FC-BOUND-{shot}",
                subject=f"shot_{shot}_boundary",
                status=status,
                seed_value=f"[{_fmt(seed_range.start_seconds)}, {_fmt(seed_range.end_seconds)}]",
                verified_value=f"[{_fmt(proposal.start_exact)}, {_fmt(proposal.end_exact)}]",
                evidence_refs=[ref],
            )
        )


def _verify_transitions(
    claims: list[SeedClaim],
    shots_by_index: dict[int, ShotProposal],
    checks: list[FoundationCheck],
) -> None:
    for claim in claims:
        if claim.claim_type != SeedClaimType.TRANSITION:
            continue
        shot = claim.shot_number
        seed_text = (claim.text or "").strip()
        seed_type = _extract_transition(seed_text)
        if shot == 1:
            # Shot 1 must be "Opening shot" (rule shots.shot_one_transition).
            ref1 = _struct_ref(
                "EV-TRANS-1", "rules", "shot 1 transition is 'Opening shot' by rule"
            )
            if seed_type is not None and seed_type.lower() == "opening shot":
                claim.evidence_status = EvidenceStatus.SUPPORTED
                claim.evidence.append(ref1)
            elif seed_type is not None:
                claim.evidence_status = EvidenceStatus.CONTRADICTED
                claim.evidence.append(ref1)
                checks.append(
                    FoundationCheck(
                        check_id="FC-TRANS-1",
                        subject="shot_1_transition",
                        status=FoundationStatus.CONTRADICTED,
                        seed_value=seed_type,
                        verified_value="Opening shot",
                        evidence_refs=[ref1],
                        detail="Shot 1 transition must be 'Opening shot'.",
                    )
                )
            else:
                claim.evidence_status = EvidenceStatus.UNRESOLVED
            continue
        proposal = shots_by_index.get(shot) if shot is not None else None
        verified = proposal.transition_into_shot if proposal is not None else None
        if verified is None:
            # Unresolved transition is NOT Hard cut; cannot confirm.
            claim.evidence_status = EvidenceStatus.UNRESOLVED
            continue
        ref = _struct_ref(f"EV-TRANS-{shot}", "shot_truth", f"verified={verified}")
        if seed_type is not None and seed_type.lower() == verified.lower():
            claim.evidence_status = EvidenceStatus.SUPPORTED
            claim.evidence.append(ref)
        elif seed_type is not None:
            claim.evidence_status = EvidenceStatus.CONTRADICTED
            claim.evidence.append(ref)
        else:
            claim.evidence_status = EvidenceStatus.UNRESOLVED


def _verify_timestamp_containment(
    claims: list[SeedClaim], shots_by_index: dict[int, ShotProposal]
) -> None:
    """Flag shot-scoped claims whose seed time falls outside the verified shot
    interval. Consistency here is necessary but not sufficient for support, so
    it never upgrades a semantic claim to SUPPORTED — it can only CONTRADICT."""
    for claim in claims:
        seed_range = claim.seed_time_range
        shot = claim.shot_number
        if seed_range is None or shot is None or claim.claim_type == SeedClaimType.SHOT_BOUNDARY:
            continue
        proposal = shots_by_index.get(shot)
        if proposal is None or proposal.start_exact is None or proposal.end_exact is None:
            continue
        start = seed_range.start_seconds
        end = seed_range.end_seconds
        outside = start < proposal.start_exact - _TENTH or end > proposal.end_exact + _TENTH
        if outside:
            claim.evidence_status = EvidenceStatus.CONTRADICTED
            claim.evidence.append(
                _struct_ref(
                    f"EV-CONTAIN-{claim.claim_id}",
                    "shot_truth",
                    f"seed time [{_fmt(start)},{_fmt(end)}] outside verified shot {shot} "
                    f"[{_fmt(proposal.start_exact)},{_fmt(proposal.end_exact)}]",
                )
            )


def _verify_entity_references(
    doc: SeedDocument,
    claims: list[SeedClaim],
    checks: list[FoundationCheck],
    conflicts: list[str],
) -> None:
    defined_c = {
        cid
        for claim in claims
        if claim.claim_type == SeedClaimType.CHARACTER_EXISTS
        for cid in claim.subject_ids
    }
    defined_o = {
        oid
        for claim in claims
        if claim.claim_type == SeedClaimType.OBJECT_EXISTS
        for oid in claim.object_ids
    }
    referenced_c: set[str] = set()
    referenced_o: set[str] = set()
    for section in doc.sections:
        for entry in section.entries:
            if entry.field_label == "Shot header":
                continue
            referenced_c.update(entry.referenced_character_ids)
            referenced_o.update(entry.referenced_object_ids)

    undefined_c = sorted(referenced_c - defined_c)
    undefined_o = sorted(referenced_o - defined_o)
    ghost_c = sorted(defined_c - referenced_c)
    ghost_o = sorted(defined_o - referenced_o)

    for cid in undefined_c:
        conflicts.append(f"Character {cid} referenced but never defined")
        checks.append(
            FoundationCheck(
                check_id=f"FC-UNDEF-{cid}",
                subject=cid,
                status=FoundationStatus.CONTRADICTED,
                detail=f"{cid} referenced in a shot but not defined in Characters.",
            )
        )
    for oid in undefined_o:
        conflicts.append(f"Object {oid} referenced but never defined")
        checks.append(
            FoundationCheck(
                check_id=f"FC-UNDEF-{oid}",
                subject=oid,
                status=FoundationStatus.CONTRADICTED,
                detail=f"{oid} referenced in a shot but not defined in Objects.",
            )
        )
    for gid in ghost_c + ghost_o:
        checks.append(
            FoundationCheck(
                check_id=f"FC-GHOST-{gid}",
                subject=gid,
                status=FoundationStatus.UNRESOLVED,
                detail=f"{gid} defined but never referenced (ghost id).",
            )
        )


def _mark_semantic_unresolved(claims: list[SeedClaim]) -> None:
    semantic = {
        SeedClaimType.CHARACTER_EXISTS,
        SeedClaimType.CHARACTER_TRAIT,
        SeedClaimType.CHARACTER_VISIBILITY,
        SeedClaimType.CHARACTER_POSITION,
        SeedClaimType.OBJECT_EXISTS,
        SeedClaimType.OBJECT_IDENTITY,
        SeedClaimType.OBJECT_OWNERSHIP,
        SeedClaimType.OBJECT_CONTACT,
        SeedClaimType.SCENE_STATE,
        SeedClaimType.STYLE_STATE,
        SeedClaimType.CAMERA_FRAMING,
        SeedClaimType.CAMERA_MOVEMENT,
        SeedClaimType.ACTION,
        SeedClaimType.SPEECH,
        SeedClaimType.SOUND,
        SeedClaimType.ON_SCREEN_TEXT,
        SeedClaimType.PLAYBACK_SPEED,
        SeedClaimType.VISUAL_CONCERN,
        SeedClaimType.AUDIO_CONCERN,
    }
    for claim in claims:
        if claim.evidence_status is not None:
            continue  # already resolved (e.g. contradicted by containment)
        if claim.claim_type == SeedClaimType.PROTECTED_TRAIT or claim.claim_type in semantic:
            claim.evidence_status = EvidenceStatus.UNRESOLVED


def _overall_foundation(checks: list[FoundationCheck]) -> FoundationStatus:
    statuses = [c.status for c in checks]
    if not statuses:
        return FoundationStatus.NOT_EVALUATED
    if FoundationStatus.CONTRADICTED in statuses:
        return FoundationStatus.CONTRADICTED
    if FoundationStatus.UNRESOLVED in statuses:
        return FoundationStatus.UNRESOLVED
    only_supported_or_na = all(
        s in (FoundationStatus.SUPPORTED, FoundationStatus.NOT_EVALUATED) for s in statuses
    )
    if only_supported_or_na and FoundationStatus.SUPPORTED in statuses:
        return FoundationStatus.SUPPORTED
    return FoundationStatus.NOT_EVALUATED


def _row(claim: SeedClaim) -> ClaimEvidenceRow:
    status = claim.evidence_status or EvidenceStatus.UNRESOLVED
    supporting = [e for e in claim.evidence if e.notes and "outside" not in (e.notes or "")]
    contradicting = [e for e in claim.evidence if e.notes and "outside" in (e.notes or "")]
    if status == EvidenceStatus.CONTRADICTED:
        contradicting = list(claim.evidence)
        supporting = []
    elif status in (EvidenceStatus.SUPPORTED, EvidenceStatus.PARTIALLY_SUPPORTED):
        supporting = list(claim.evidence)
        contradicting = []
    seed_range = claim.seed_time_range
    reasons: list[str] = []
    if status == EvidenceStatus.UNRESOLVED:
        reasons.append("Requires later visual/audio evidence or human verification.")
    return ClaimEvidenceRow(
        claim_id=claim.claim_id,
        claim_type=claim.claim_type or SeedClaimType.UNCLASSIFIED,
        seed_text=claim.text,
        seed_shot=claim.shot_number,
        seed_source_line=claim.seed_source_line,
        seed_start_exact=seed_range.start_seconds if seed_range else None,
        seed_end_exact=seed_range.end_seconds if seed_range else None,
        evidence_status=status,
        importance=claim.importance or ClaimImportance.LOCAL,
        foundational=claim.importance == ClaimImportance.FOUNDATIONAL,
        supporting_evidence_refs=supporting,
        contradicting_evidence_refs=contradicting,
        unresolved_reasons=reasons,
    )


# --- helpers --------------------------------------------------------------

_ALLOWED_TRANSITIONS = [
    "Opening shot",
    "Hard cut",
    "Cross dissolve",
    "Fade in",
    "Fade out",
    "Match cut",
    "Jump cut",
    "Smash cut",
    "Wipe",
    "Iris",
    "L-cut",
    "J-cut",
    "Whip pan",
    "Swish pan",
]


def _extract_transition(text: str) -> str | None:
    lowered = text.lower()
    for name in _ALLOWED_TRANSITIONS:
        if name.lower() in lowered:
            return name
    return None


def _fmt(value: Fraction | None) -> str:
    if value is None:
        return "?"
    return f"{float(value):.3f}"
