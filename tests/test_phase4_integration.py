"""Phase 4 evidence->claim integration tests (items 1, 2, 8, 12, 13)."""

from __future__ import annotations

from fractions import Fraction

from manuscript_reviewer.models.review_intelligence import (
    CameraMotionCandidate,
    CameraMotionClass,
    EntityTrack,
    EvidenceStatus,
    FinalStateCheck,
    ObjectStateKind,
    ReviewPriority,
    SeedClaimType,
    TextConsensus,
    TextTrack,
    TrackObservation,
    TrackStatus,
)
from manuscript_reviewer.models.shot_truth import (
    CandidateStatus,
    ShotProposal,
    ShotTruthResult,
    TransitionStatus,
)
from manuscript_reviewer.review.queue import build_machine_review_items
from manuscript_reviewer.seed.claims import extract_claims
from manuscript_reviewer.seed.comparison import VisualEvidence, compare_seed
from manuscript_reviewer.seed.parser import parse_seed_text
from manuscript_reviewer.validation.review_intelligence_validator import compute_overall_status


def _shots(ranges: list[tuple[int, int, int]], starts: list[Fraction], ends: list[Fraction]):  # type: ignore[no-untyped-def]
    props = [
        ShotProposal(
            shot_index=i, start_frame_index=lo, end_frame_index=hi,
            start_exact=s, end_exact=e, last_owned_frame_start_exact=s,
            start_manuscript=None, end_manuscript=None, transition_into_shot=None,
            transition_status=TransitionStatus.PROPOSED, supporting_boundary_id=None,
            review_status=CandidateStatus.SUPPORTED,
        )
        for (i, lo, hi), s, e in zip(ranges, starts, ends, strict=True)
    ]
    return ShotTruthResult(
        frame_count=200, adjacent_pair_count=0, raw_candidate_count=0, merged_candidate_count=0,
        supported_count=0, rejected_count=0, review_required_count=0,
        proposed_shot_count=len(props), overall_status="PASS", candidates=[], shots=props,
    )


def _t10(t: Fraction) -> int | None:
    return int(t * 10)


def _track(track_id: str, etype: str, first: int, last: int, reacq: bool = False) -> EntityTrack:
    obs = [
        TrackObservation(frame_index=f, x=0, y=0, width=10, height=10, status=TrackStatus.TRACKED)
        for f in range(first, last + 1)
    ]
    return EntityTrack(
        track_id=track_id, entity_type=etype, first_frame_index=first, last_frame_index=last,
        observations=obs, status=TrackStatus.REVIEW_REQUIRED if reacq else TrackStatus.TRACKED,
        reacquired=reacq,
    )


# --------------------------------------------------------------------------
# Item 8: machine-evidence review items drive status
# --------------------------------------------------------------------------


def test_reacquired_identity_creates_review_item() -> None:
    ev = VisualEvidence(entity_tracks=[_track("TRK-a", "CHARACTER", 0, 20, reacq=True)])
    items = build_machine_review_items(ev)
    assert any(i.priority == ReviewPriority.HIGH for i in items)
    assert compute_overall_status(items, []) == "REVIEW_REQUIRED"


def test_unresolved_final_state_creates_review_item() -> None:
    fs = FinalStateCheck(shot_number=1, entity_id="O", final_state=ObjectStateKind.REVIEW_REQUIRED,
                         resolved=False)
    ev = VisualEvidence(final_state_checks=[fs])
    items = build_machine_review_items(ev)
    assert items and compute_overall_status(items, []) == "REVIEW_REQUIRED"


def test_no_seed_identity_collision_is_critical_review() -> None:
    a = _track("TRK-a", "CHARACTER", 0, 5)
    b = _track("TRK-b", "CHARACTER", 40, 60)  # disjoint from a
    ev = VisualEvidence(entity_by_seed_id={"C1": [a, b]})
    items = build_machine_review_items(ev)
    assert any(i.priority == ReviewPriority.CRITICAL for i in items)
    assert compute_overall_status(items, []) == "REVIEW_REQUIRED"


# --------------------------------------------------------------------------
# Item 19: high-risk items carry graded frame-range evidence -> bundle range
# --------------------------------------------------------------------------


def test_collision_item_carries_both_track_frame_ranges() -> None:
    from manuscript_reviewer.review.evidence_bundles import _bundle_range

    a = _track("TRK-a", "CHARACTER", 0, 5)
    b = _track("TRK-b", "CHARACTER", 40, 60)
    item = next(i for i in build_machine_review_items(VisualEvidence(
        entity_by_seed_id={"C1": [a, b]})) if i.priority == ReviewPriority.CRITICAL)
    # A graded FRAME_RANGE ref per colliding track, each is_factual as a pointer.
    spans = {(r.start_frame, r.end_frame) for r in item.supporting_evidence_refs}
    assert spans == {(0, 5), (40, 60)}
    assert all(r.is_factual for r in item.supporting_evidence_refs)
    # The bundle spans both tracks the reviewer must disambiguate.
    assert _bundle_range(item) == (0, 60)


def test_bundle_range_falls_back_to_evidence_refs() -> None:
    from manuscript_reviewer.models.evidence import EvidenceReference, EvidenceType
    from manuscript_reviewer.models.review_intelligence import (
        ReviewerAction,
        ReviewQueueItem,
    )
    from manuscript_reviewer.review.evidence_bundles import _bundle_range

    item = ReviewQueueItem(
        item_id="RQ-1", priority=ReviewPriority.HIGH, title="t", reason="r",
        recommended_action=ReviewerAction.VERIFY,
        supporting_evidence_refs=[EvidenceReference(
            evidence_id="EV-1", evidence_type=EvidenceType.FRAME_RANGE,
            start_frame=12, end_frame=20)],
    )
    assert item.start_frame is None
    assert _bundle_range(item) == (12, 20)  # derived from the ref, not the (absent) anchor


# --------------------------------------------------------------------------
# Item 2: entity foundation checks
# --------------------------------------------------------------------------


def test_entity_not_visible_at_claim_time_contradicts() -> None:
    seed = "[Shot 1: 0.0-10.0]\nCut: Opening shot\nAction & Audio: 5.0-6.0: C1 waves.\n"
    doc = parse_seed_text(seed)
    claims = extract_claims(doc)
    # C1 tracked only at frames 0-2, but the claim references C1 at 5.0s (frame 50).
    track = _track("TRK-c1", "CHARACTER", 0, 2)
    ev = VisualEvidence(
        entity_by_seed_id={"C1": [track]}, time_to_frame=_t10,
    )
    res = compare_seed(doc, claims, None,
                       _shots([(1, 0, 100)], [Fraction(0)], [Fraction(10)]), ev)
    action = next(c for c in res.claims if c.claim_type == SeedClaimType.ACTION)
    assert action.evidence_status == EvidenceStatus.CONTRADICTED


def test_seed_id_spanning_two_tracks_is_identity_collision() -> None:
    seed = "Characters\nC1: A man.\n[Shot 1: 0.0-6.0]\nCut: Opening shot\n"
    doc = parse_seed_text(seed)
    claims = extract_claims(doc)
    a = _track("TRK-a", "CHARACTER", 0, 5)
    b = _track("TRK-b", "CHARACTER", 40, 60)
    ev = VisualEvidence(entity_by_seed_id={"C1": [a, b]}, time_to_frame=_t10)
    res = compare_seed(doc, claims, None,
                       _shots([(1, 0, 100)], [Fraction(0)], [Fraction(6)]), ev)
    assert any("combines two distinct tracks" in c for c in res.foundational_conflicts)


# --------------------------------------------------------------------------
# Items 12 / 13: shot + time + direction scoping
# --------------------------------------------------------------------------


def test_camera_screen_right_does_not_support_screen_left_claim() -> None:
    seed = (
        "[Shot 1: 0.0-5.0]\nCut: Opening shot\n"
        "Camera Movements: 0.0-1.0: Camera pans screen-left.\n"
    )
    doc = parse_seed_text(seed)
    claims = extract_claims(doc)
    cand = CameraMotionCandidate(
        candidate_id="CAM-1", shot_number=1, start_frame=0, last_supporting_frame=10, end_frame=11,
        motion_class=CameraMotionClass.HORIZONTAL_GLOBAL_MOTION, direction="screen-right",
        strength=1.0, inlier_ratio=0.9,
    )
    ev = VisualEvidence(camera_candidates=[cand], time_to_frame=_t10)
    res = compare_seed(doc, claims, None,
                       _shots([(1, 0, 50)], [Fraction(0)], [Fraction(5)]), ev)
    cm = next(c for c in res.claims if c.claim_type == SeedClaimType.CAMERA_MOVEMENT)
    # Opposite direction -> not partially supported.
    assert cm.evidence_status != EvidenceStatus.PARTIALLY_SUPPORTED


def test_ocr_in_wrong_shot_does_not_support_claim() -> None:
    seed = (
        "[Shot 1: 0.0-3.0]\nCut: Opening shot\n"
        "[Shot 2: 3.0-6.0]\nCut: Hard cut\n"
        '[Shot 3: 6.0-9.0]\nCut: Hard cut\nAction & Audio: 7.0-8.0: On-screen text reads "GO".\n'
    )
    doc = parse_seed_text(seed)
    claims = extract_claims(doc)
    # OCR "GO" track lives in shot 1 (frames 0-20), not shot 3.
    track = TextTrack(
        track_id="TT-1", first_candidate_frame=0, first_stable_frame=0, last_stable_frame=20,
        consensus=TextConsensus(consensus_text="GO", support_frames=20, confidence=0.9),
    )
    shots = _shots([(1, 0, 29), (2, 30, 59), (3, 60, 89)],
                   [Fraction(0), Fraction(3), Fraction(6)],
                   [Fraction(3), Fraction(6), Fraction(9)])
    ev = VisualEvidence(text_tracks=[track], time_to_frame=_t10)
    res = compare_seed(doc, claims, None, shots, ev)
    ost = next(c for c in res.claims if c.claim_type == SeedClaimType.ON_SCREEN_TEXT)
    # "GO" in shot 1 must not support the shot-3 claim.
    assert ost.evidence_status != EvidenceStatus.PARTIALLY_SUPPORTED
