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

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction

from ..media.timestamps import to_manuscript_display
from ..models.caption import SeedClaim
from ..models.evidence import EvidenceReference, EvidenceType
from ..models.media import MediaInfo
from ..models.review_intelligence import (
    ActionCandidate,
    CameraMotionCandidate,
    CameraMotionClass,
    CharacterHypothesis,
    ClaimEvidenceRow,
    ClaimImportance,
    ClaimReviewStatus,
    ContactEvent,
    ContactEventKind,
    EntityTrack,
    EvidenceStatus,
    FinalStateCheck,
    FoundationCheck,
    FoundationStatus,
    ObjectHypothesis,
    PlaybackSpeedEvidence,
    ProposalReasonCode,
    SeedClaimType,
    SeedDocument,
    SpeedConclusion,
    TextTrack,
    TrackStatus,
)
from ..models.shot_truth import ShotProposal, ShotTruthResult
from ..rules.loader import load_rules


@dataclass
class VisualEvidence:
    """All machine visual/CV evidence + resolvers, threaded into reconciliation.

    Machine evidence is never made factual: it can only PARTIALLY_SUPPORT (with a
    graded reference), CONTRADICT where deterministic, or stay UNRESOLVED/review.
    """

    text_tracks: list[TextTrack] = field(default_factory=list)
    camera_candidates: list[CameraMotionCandidate] = field(default_factory=list)
    character_hypotheses: list[CharacterHypothesis] = field(default_factory=list)
    object_hypotheses: list[ObjectHypothesis] = field(default_factory=list)
    entity_tracks: list[EntityTrack] = field(default_factory=list)
    contact_events: list[ContactEvent] = field(default_factory=list)
    final_state_checks: list[FinalStateCheck] = field(default_factory=list)
    action_candidates: list[ActionCandidate] = field(default_factory=list)
    speed_evidence: list[PlaybackSpeedEvidence] = field(default_factory=list)
    time_to_frame: Callable[[Fraction], int | None] | None = None
    frame_to_shot: Callable[[int], int | None] | None = None
    shots_by_index: dict[int, ShotProposal] = field(default_factory=dict)
    #: Seed C/O id -> tracks, only when anchors are labelled with seed ids
    #: (item 2 identity checks are only computable for anchored entities).
    entity_by_seed_id: dict[str, list[EntityTrack]] = field(default_factory=dict)


def _display(value: Fraction | None) -> Decimal | None:
    """Manuscript 0.1 s display projection (ROUND_HALF_UP) — the ONE rounding."""
    return to_manuscript_display(value) if value is not None else None


def _same_display(a: Fraction | None, b: Fraction | None) -> bool | None:
    da, db = _display(a), _display(b)
    if da is None or db is None:
        return None
    return da == db


def _disp(value: Fraction | None) -> str:
    d = _display(value)
    return f"{d}s" if d is not None else "?"


@dataclass
class ComparisonResult:
    claims: list[SeedClaim]
    rows: list[ClaimEvidenceRow]
    foundation_checks: list[FoundationCheck]
    foundation_status: FoundationStatus
    seed_shot_count: int | None
    verified_shot_count: int | None
    foundational_conflicts: list[str] = field(default_factory=list)


def _struct_ref(
    ref_id: str,
    source: str,
    notes: str,
    start_frame: int | None = None,
    end_frame: int | None = None,
    artifact_paths: list[str] | None = None,
    evidence_type: EvidenceType = EvidenceType.STRUCTURAL_CHECK,
) -> EvidenceReference:
    return EvidenceReference(
        evidence_id=ref_id,
        evidence_type=evidence_type,
        start_frame=start_frame,
        end_frame=end_frame,
        artifact_paths=artifact_paths or [],
        source=source,
        notes=notes,
    )


def is_graded(ref: EvidenceReference) -> bool:
    """A reference is *graded* (can back a SUPPORTED claim) only when it anchors
    to concrete media — exact frames, an artifact, PTS, or human verification.
    Prose-only references never grade a claim as SUPPORTED (validator AB)."""
    return (
        ref.start_frame is not None
        or ref.start_pts is not None
        or bool(ref.artifact_paths)
        or ref.is_factual
    )


def compare_seed(
    doc: SeedDocument,
    claims: list[SeedClaim],
    media: MediaInfo | None,
    shot_truth: ShotTruthResult | None,
    evidence: VisualEvidence | None = None,
) -> ComparisonResult:
    ev = evidence or VisualEvidence()
    checks: list[FoundationCheck] = []
    conflicts: list[str] = []

    verified_shot_count = shot_truth.proposed_shot_count if shot_truth is not None else None
    seed_shot_count = doc.seed_shot_count
    shots_by_index: dict[int, ShotProposal] = {}
    if shot_truth is not None:
        shots_by_index = {s.shot_index: s for s in shot_truth.shots}
    if not ev.shots_by_index:
        ev.shots_by_index = shots_by_index

    shot_foundation = _verify_shot_count(
        claims, seed_shot_count, verified_shot_count, shot_truth, checks, conflicts
    )

    _verify_media_identity(claims, doc, media, checks)
    _verify_shot_boundaries(claims, shots_by_index, shot_foundation, checks, conflicts)
    _verify_transitions(claims, shots_by_index, checks)
    _verify_timestamp_containment(claims, shots_by_index)
    _verify_entity_references(doc, claims, checks, conflicts)
    _verify_on_screen_text(claims, ev)  # item 13 (shot+time scoped)
    _verify_camera_movement(claims, ev)  # item 12 (shot+time+direction)
    _verify_playback_speed(claims, ev)  # item 9 integration
    _verify_entity_foundation(claims, ev, checks, conflicts)  # item 2
    _verify_object_states(claims, ev)  # item 1 (ownership/contact)
    _mark_semantic_unresolved(claims)

    foundation_status = _overall_foundation(checks)
    rows = build_rows(claims)
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
        "EV-SHOTCOUNT",
        "shot_truth",
        f"verified proposed_shot_count={verified_count}",
        artifact_paths=["shot_qc.json", "shots_proposed.json"],
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
    ref = _struct_ref(
        "EV-MEDIAID", "media", f"source file_name={filename}", artifact_paths=["media.json"]
    )
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
        start_ok = _same_display(seed_range.start_seconds, proposal.start_exact)
        end_ok = _same_display(seed_range.end_seconds, proposal.end_exact)
        artifacts = ["shots_proposed.json"]
        if proposal.supporting_boundary_id is not None:
            artifacts.append(f"boundary:{proposal.supporting_boundary_id}")
        ref = _struct_ref(
            f"EV-BOUND-{shot}",
            "shot_truth",
            f"verified shot {shot} interval "
            f"[{_disp(proposal.start_exact)}, {_disp(proposal.end_exact)}] "
            f"frames {proposal.start_frame_index}-{proposal.end_frame_index}",
            start_frame=proposal.start_frame_index,
            end_frame=proposal.end_frame_index,
            artifact_paths=artifacts,
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
                seed_value=f"[{_disp(seed_range.start_seconds)}, {_disp(seed_range.end_seconds)}]",
                verified_value=f"[{_disp(proposal.start_exact)}, {_disp(proposal.end_exact)}]",
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
            # Shot 1 must be the rule-defined opening transition.
            opening = shot_one_transition()
            ref1 = _struct_ref(
                "EV-TRANS-1",
                "rules",
                f"shot 1 transition is '{opening}' by rule shots.shot_one_transition",
                artifact_paths=["shots_proposed.json"],
            )
            if seed_type is not None and seed_type.lower() == opening.lower():
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
                        verified_value=opening,
                        evidence_refs=[ref1],
                        detail=f"Shot 1 transition must be '{opening}'.",
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
        boundary_artifacts = ["shots_proposed.json"]
        if proposal is not None and proposal.supporting_boundary_id is not None:
            boundary_artifacts.append(f"boundary:{proposal.supporting_boundary_id}")
        ref = _struct_ref(
            f"EV-TRANS-{shot}",
            "shot_truth",
            f"verified transition into shot {shot} = {verified}",
            start_frame=proposal.start_frame_index if proposal is not None else None,
            artifact_paths=boundary_artifacts,
        )
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
        # Compare on the Manuscript 0.1 s display grid (the projection the seed
        # times were themselves written at); no extra tolerance is added.
        # Boundaries are inclusive: seed_end == shot_end is inside; +0.1 is out.
        seed_start_d = _display(seed_range.start_seconds)
        seed_end_d = _display(seed_range.end_seconds)
        shot_start_d = _display(proposal.start_exact)
        shot_end_d = _display(proposal.end_exact)
        if None in (seed_start_d, seed_end_d, shot_start_d, shot_end_d):
            continue
        assert seed_start_d is not None and seed_end_d is not None
        assert shot_start_d is not None and shot_end_d is not None
        outside = seed_start_d < shot_start_d or seed_end_d > shot_end_d
        if outside:
            claim.evidence_status = EvidenceStatus.CONTRADICTED
            claim.evidence.append(
                _struct_ref(
                    f"EV-CONTAIN-{claim.claim_id}",
                    "shot_truth",
                    f"seed time [{seed_start_d}s,{seed_end_d}s] outside verified shot {shot} "
                    f"[{shot_start_d}s,{shot_end_d}s]",
                    start_frame=proposal.start_frame_index,
                    end_frame=proposal.end_frame_index,
                    artifact_paths=["shots_proposed.json"],
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


# --- scope helpers shared by OCR (13) and camera (12) matching ---------------


def _seed_frame_window(claim: SeedClaim, ev: VisualEvidence) -> tuple[int, int] | None:
    sr = claim.seed_time_range
    if sr is None or ev.time_to_frame is None:
        return None
    sf = ev.time_to_frame(sr.start_seconds)
    efr = ev.time_to_frame(sr.end_seconds)
    if sf is None or efr is None:
        return None
    return min(sf, efr), max(sf, efr)


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _in_scope(
    claim: SeedClaim, item_shot: int | None, item_start: int, item_end: int, ev: VisualEvidence
) -> bool:
    """An item is in a claim's scope only if it is in the seed's asserted shot
    AND overlaps the seed's time window (items 12/13)."""
    if claim.shot_number is not None and item_shot is not None and item_shot != claim.shot_number:
        return False
    window = _seed_frame_window(claim, ev)
    if window is not None:
        return _overlaps(window, (item_start, item_end))
    if claim.shot_number is not None:
        prop = ev.shots_by_index.get(claim.shot_number)
        if prop is not None:
            return _overlaps(
                (prop.start_frame_index, prop.end_frame_index), (item_start, item_end)
            )
    return True


def _verify_on_screen_text(claims: list[SeedClaim], ev: VisualEvidence) -> None:
    """Compare seed ON_SCREEN_TEXT claims with OCR text tracks IN SCOPE (item 13).

    A candidate must be in the seed's shot AND overlap the seed's time window
    before text is compared ("GO" in shot 1 never supports a "GO" claim in shot
    3). Machine OCR is only ever PARTIALLY_SUPPORTED + review, never final."""
    tracks = [t for t in ev.text_tracks if t.consensus is not None]
    if not tracks:
        return
    from ..ocr.timing import text_similarity

    for claim in claims:
        if claim.claim_type != SeedClaimType.ON_SCREEN_TEXT:
            continue
        seed_text = claim.quoted_text or claim.text
        in_scope = [
            t for t in tracks
            if _in_scope(claim, None, t.first_candidate_frame,
                         t.last_stable_frame or t.first_candidate_frame, ev)
        ]
        best: TextTrack | None = None
        best_sim = 0.0
        for track in in_scope:
            assert track.consensus is not None
            sim = text_similarity(seed_text, track.consensus.consensus_text)
            if sim > best_sim:
                best_sim = sim
                best = track
        if best is None:
            continue
        assert best.consensus is not None
        claim.evidence.append(
            _struct_ref(
                f"EV-OCR-{claim.claim_id}",
                "ocr",
                f"machine OCR consensus '{best.consensus.consensus_text}' "
                f"similarity {best_sim:.2f} (UNVERIFIED machine evidence)",
                start_frame=best.first_candidate_frame,
                end_frame=best.last_stable_frame,
                artifact_paths=["visual/ocr/text_tracks.json"],
                evidence_type=EvidenceType.OCR_TRACK,
            )
        )
        claim.evidence_status = (
            EvidenceStatus.PARTIALLY_SUPPORTED if best_sim >= 0.6 else EvidenceStatus.UNRESOLVED
        )
        claim.review_status = ClaimReviewStatus.REVIEW_REQUIRED


#: Seed movement words that 2D global motion can NEVER prove (need depth/3D).
_CAMERA_3D_TERMS = re.compile(
    r"\b(dolly|track(?:ing)?|push(?:-?in)?|pull(?:-?back)?|truck)\b", re.I
)
_SCALE_TERMS = re.compile(r"\b(zoom|scale)\b", re.I)
_HORIZONTAL_TERMS = re.compile(r"\bpan|screen-(?:left|right)\b", re.I)
_VERTICAL_TERMS = re.compile(r"\btilt|upward|downward\b", re.I)


def _seed_camera_direction(text: str) -> str | None:
    lowered = text.lower()
    if "screen-left" in lowered:
        return "screen-left"
    if "screen-right" in lowered:
        return "screen-right"
    if "upward" in lowered or "tilt up" in lowered:
        return "up"
    if "downward" in lowered or "tilt down" in lowered:
        return "down"
    return None


def _verify_camera_movement(claims: list[SeedClaim], ev: VisualEvidence) -> None:
    """Compare seed CAMERA_MOVEMENT claims with camera evidence IN SCOPE, matching
    direction (item 12). A screen-right phase never supports a screen-left claim;
    2D motion never proves dolly/track/push/pull; scale never proves zoom
    direction. Machine support is only ever PARTIAL + review."""
    if not ev.camera_candidates:
        return
    for claim in claims:
        if claim.claim_type != SeedClaimType.CAMERA_MOVEMENT:
            continue
        text = claim.text or ""
        if _CAMERA_3D_TERMS.search(text):
            claim.evidence_status = EvidenceStatus.UNRESOLVED
            claim.review_status = ClaimReviewStatus.REVIEW_REQUIRED
            claim.evidence.append(
                _struct_ref(
                    f"EV-CAM-{claim.claim_id}", "camera",
                    "seed states a 3D move (dolly/track/push/pull); 2D global "
                    "motion cannot prove it",
                )
            )
            continue
        wanted: CameraMotionClass | None = None
        if _HORIZONTAL_TERMS.search(text):
            wanted = CameraMotionClass.HORIZONTAL_GLOBAL_MOTION
        elif _VERTICAL_TERMS.search(text):
            wanted = CameraMotionClass.VERTICAL_GLOBAL_MOTION
        elif _SCALE_TERMS.search(text):
            wanted = CameraMotionClass.SCALE_INCREASE
        if wanted is None:
            continue
        wanted_dir = _seed_camera_direction(text)
        scoped = [
            c for c in ev.camera_candidates
            if _in_scope(claim, c.shot_number, c.start_frame, c.last_supporting_frame, ev)
        ]
        match = _find_camera_match(scoped, wanted, wanted_dir)
        if match is not None:
            claim.evidence.append(
                _struct_ref(
                    f"EV-CAM-{claim.claim_id}", "camera",
                    f"matching {match.motion_class.value} "
                    f"({match.direction}) phase frames "
                    f"{match.start_frame}-{match.last_supporting_frame}",
                    start_frame=match.start_frame,
                    end_frame=match.last_supporting_frame,
                    artifact_paths=["visual/camera/camera_events.json"],
                )
            )
            claim.evidence_status = EvidenceStatus.PARTIALLY_SUPPORTED
            claim.review_status = ClaimReviewStatus.REVIEW_REQUIRED


def _find_camera_match(
    candidates: list[CameraMotionCandidate],
    wanted: CameraMotionClass,
    wanted_dir: str | None,
) -> CameraMotionCandidate | None:
    scale_classes = {CameraMotionClass.SCALE_INCREASE, CameraMotionClass.SCALE_DECREASE}
    for cand in candidates:
        same_family = cand.motion_class == wanted or (
            wanted in scale_classes and cand.motion_class in scale_classes
        )
        if not same_family:
            continue
        # Direction must match when the seed states one (item 12).
        if wanted_dir is not None and cand.direction != wanted_dir:
            continue
        return cand
    return None


_SEED_SPEED = re.compile(r"\b(slow[\s-]?motion|slow|accelerated|sped[\s-]?up|regular)\b", re.I)


def _verify_playback_speed(claims: list[SeedClaim], ev: VisualEvidence) -> None:
    """Compare seed PLAYBACK_SPEED claims with per-shot speed evidence (item 9)."""
    if not ev.speed_evidence:
        return
    by_shot = {e.shot_number: e for e in ev.speed_evidence}
    for claim in claims:
        if claim.claim_type != SeedClaimType.PLAYBACK_SPEED or claim.shot_number is None:
            continue
        evidence = by_shot.get(claim.shot_number)
        if evidence is None:
            continue
        m = _SEED_SPEED.search(claim.text or "")
        seed_speed = m.group(1).lower().replace(" ", "").replace("-", "") if m else None
        ref = _struct_ref(
            f"EV-SPEED-{claim.claim_id}", "speed",
            f"shot {claim.shot_number} speed evidence: {evidence.conclusion.value}",
            artifact_paths=["visual/speed/playback_speed_evidence.json"],
        )
        claim.evidence.append(ref)
        claim.review_status = ClaimReviewStatus.REVIEW_REQUIRED
        if seed_speed == "regular" and evidence.conclusion == SpeedConclusion.REGULAR_SUPPORTED:
            claim.evidence_status = EvidenceStatus.PARTIALLY_SUPPORTED
        elif (
            seed_speed in ("slowmotion", "slow")
            and evidence.conclusion == SpeedConclusion.REGULAR_SUPPORTED
        ):
            # Strong regular cadence contradicts a slow-motion seed claim.
            claim.evidence_status = EvidenceStatus.CONTRADICTED
        else:
            # Seed asserts a speed but evidence is unresolved/ambiguous -> review.
            claim.evidence_status = EvidenceStatus.UNRESOLVED


_CO_ID = re.compile(r"^[CO]\d+$", re.IGNORECASE)


def _verify_entity_foundation(
    claims: list[SeedClaim],
    ev: VisualEvidence,
    checks: list[FoundationCheck],
    conflicts: list[str],
) -> None:
    """Seed-vs-continuity identity checks (item 2). Only computable for anchored
    entities (a seed id is linked to a track when an anchor is labelled with it).
    Similar-but-unproven tracks are NEVER auto-merged — those are review links."""
    if not ev.entity_by_seed_id:
        return
    for seed_id, tracks in ev.entity_by_seed_id.items():
        # COLLISION: one seed id spans two temporally-disjoint tracks.
        if len(tracks) >= 2:
            disjoint = any(
                a.last_frame_index < b.first_frame_index or b.last_frame_index < a.first_frame_index
                for i, a in enumerate(tracks)
                for b in tracks[i + 1 :]
            )
            if disjoint:
                is_char = seed_id.upper().startswith("C")
                reason = (
                    ProposalReasonCode.CHARACTER_IDENTITY_COLLISION
                    if is_char
                    else ProposalReasonCode.OBJECT_IDENTITY_COLLISION
                )
                conflicts.append(f"Seed {seed_id} combines two distinct tracks")
                checks.append(
                    FoundationCheck(
                        check_id=f"FC-IDENT-{seed_id}",
                        subject=seed_id,
                        status=FoundationStatus.CONTRADICTED,
                        reason_codes=[reason],
                        detail=f"Seed {seed_id} appears to combine two distinct tracked entities.",
                    )
                )
                for claim in claims:
                    ident_hit = seed_id in claim.subject_ids or seed_id in claim.object_ids
                    if ident_hit and claim.importance == ClaimImportance.FOUNDATIONAL:
                        claim.evidence_status = EvidenceStatus.CONTRADICTED
                        claim.evidence.append(
                            _struct_ref(
                                f"EV-IDENT-{claim.claim_id}", "tracking",
                                f"{seed_id} spans two distinct tracks",
                                artifact_paths=["visual/entities/tracks.json"],
                            )
                        )

    # ENTITY_NOT_VISIBLE_AT_CLAIM_TIME: a claim references an id at a time whose
    # frame has no visible (TRACKED) observation of that id's track.
    if ev.time_to_frame is None:
        return
    for claim in claims:
        window = _seed_frame_window(claim, ev)
        if window is None:
            continue
        for seed_id in list(claim.subject_ids) + list(claim.object_ids):
            tracks = ev.entity_by_seed_id.get(seed_id, [])
            if not tracks:
                continue
            visible = any(
                o.status == TrackStatus.TRACKED and window[0] <= o.frame_index <= window[1]
                for t in tracks
                for o in t.observations
            )
            if not visible:
                claim.evidence_status = EvidenceStatus.CONTRADICTED
                claim.review_status = ClaimReviewStatus.REVIEW_REQUIRED
                claim.evidence.append(
                    _struct_ref(
                        f"EV-NOTVIS-{claim.claim_id}", "tracking",
                        f"{seed_id} not visibly tracked in frames {window[0]}-{window[1]}",
                        start_frame=window[0], end_frame=window[1],
                        artifact_paths=["visual/entities/tracks.json"],
                    )
                )
                conflicts.append(f"{seed_id} not visible at claim {claim.claim_id} time")


def _verify_object_states(claims: list[SeedClaim], ev: VisualEvidence) -> None:
    """Object ownership/contact claims vs tracked contact continuity (item 1)."""
    if not ev.contact_events:
        return
    held_ids = {
        e.object_track_id
        for e in ev.contact_events
        if e.kind in (ContactEventKind.HELD_STATE_BEGINS, ContactEventKind.OBJECT_PICKUP_CANDIDATE)
    }
    for claim in claims:
        if claim.claim_type not in (SeedClaimType.OBJECT_OWNERSHIP, SeedClaimType.OBJECT_CONTACT):
            continue
        # Link seed O-ids to tracks via anchor labels where available.
        linked = [
            t.track_id for oid in claim.object_ids for t in ev.entity_by_seed_id.get(oid, [])
        ]
        if any(tid in held_ids for tid in linked):
            claim.evidence.append(
                _struct_ref(
                    f"EV-OWN-{claim.claim_id}", "contacts",
                    "tracked HELD/contact continuity supports ownership/contact",
                    artifact_paths=["visual/actions/contact_events.json"],
                )
            )
            claim.evidence_status = EvidenceStatus.PARTIALLY_SUPPORTED
            claim.review_status = ClaimReviewStatus.REVIEW_REQUIRED


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


def build_rows(claims: list[SeedClaim]) -> list[ClaimEvidenceRow]:
    """Build claim<->evidence matrix rows (recomputed after decision overrides)."""
    return [_row(claim) for claim in claims]


def _row(claim: SeedClaim) -> ClaimEvidenceRow:
    status = claim.evidence_status or EvidenceStatus.UNRESOLVED
    supporting: list[EvidenceReference] = []
    contradicting: list[EvidenceReference] = []
    if status == EvidenceStatus.CONTRADICTED:
        contradicting = list(claim.evidence)
    elif status in (EvidenceStatus.SUPPORTED, EvidenceStatus.PARTIALLY_SUPPORTED):
        supporting = list(claim.evidence)
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


def allowed_transitions() -> list[str]:
    """The single controlling transition vocabulary — from the rule file, never
    a hardcoded copy (rules ``shots.allowed_transition_types``)."""
    values = load_rules().get("shots.allowed_transition_types", [])
    return [str(v) for v in values] if isinstance(values, list) else []


def shot_one_transition() -> str:
    """The rule-defined shot-1 transition (rules ``shots.shot_one_transition``)."""
    return str(load_rules().get("shots.shot_one_transition", "Opening shot"))


def _extract_transition(text: str) -> str | None:
    lowered = text.lower()
    # Longest names first so "Whip pan" is not shadowed by a shorter substring.
    for name in sorted(allowed_transitions(), key=len, reverse=True):
        if name.lower() in lowered:
            return name
    return None
