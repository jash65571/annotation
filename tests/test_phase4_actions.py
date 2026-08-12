"""Phase 4 tests for contacts (V), final-state (W), and action boundaries (X).

Pure-logic over constructed tracks (deterministic, no ffmpeg)."""

from __future__ import annotations

from manuscript_reviewer.actions.boundaries import build_action_candidates
from manuscript_reviewer.actions.contacts import build_contact_events
from manuscript_reviewer.actions.final_state import build_final_state_checks
from manuscript_reviewer.models.review_intelligence import (
    ActionStateClass,
    ContactEventKind,
    EntityTrack,
    ObjectStateKind,
    TrackObservation,
    TrackStatus,
)
from manuscript_reviewer.models.shot_truth import (
    CandidateStatus,
    ShotProposal,
    ShotTruthResult,
    TransitionStatus,
)


def _track(track_id: str, entity_type: str, boxes: dict[int, tuple[int, int]],
           status: TrackStatus = TrackStatus.TRACKED) -> EntityTrack:
    obs = [
        TrackObservation(frame_index=f, x=x, y=y, width=20, height=20, status=status)
        for f, (x, y) in sorted(boxes.items())
    ]
    return EntityTrack(
        track_id=track_id, entity_type=entity_type,
        first_frame_index=obs[0].frame_index, last_frame_index=obs[-1].frame_index,
        observations=obs, status=status,
    )


def _shot_truth(shots: list[tuple[int, int, int]]) -> ShotTruthResult:
    props = [
        ShotProposal(
            shot_index=i, start_frame_index=lo, end_frame_index=hi,
            start_exact=None, end_exact=None, last_owned_frame_start_exact=None,
            start_manuscript=None, end_manuscript=None, transition_into_shot=None,
            transition_status=TransitionStatus.PROPOSED, supporting_boundary_id=None,
            review_status=CandidateStatus.SUPPORTED,
        )
        for i, lo, hi in shots
    ]
    return ShotTruthResult(
        frame_count=100, adjacent_pair_count=0, raw_candidate_count=0, merged_candidate_count=0,
        supported_count=0, rejected_count=0, review_required_count=0,
        proposed_shot_count=len(props), overall_status="PASS", candidates=[], shots=props,
    )


# --------------------------------------------------------------------------
# V: contacts / ownership
# --------------------------------------------------------------------------


def test_object_pickup_candidate_from_state_sequence() -> None:
    # Object static & separate, then character reaches it and it moves together.
    char = _track("C", "CHARACTER", {f: (100 - f * 5, 50) for f in range(0, 12)})
    obj = _track(
        "O", "OBJECT",
        {**dict.fromkeys(range(0, 5), (40, 50)),  # static & separate
         **{f: (100 - f * 5 + 2, 50) for f in range(5, 12)}},  # moves with character
    )
    events = build_contact_events([char], [obj])
    kinds = {e.kind for e in events}
    assert ContactEventKind.CONTACT_BEGINS in kinds
    assert ContactEventKind.HELD_STATE_BEGINS in kinds
    assert ContactEventKind.OBJECT_PICKUP_CANDIDATE in kinds


def test_no_pickup_without_supporting_sequence() -> None:
    # Two objects that never come into contact -> no held/pickup.
    char = _track("C", "CHARACTER", dict.fromkeys(range(0, 8), (10, 10)))
    obj = _track("O", "OBJECT", dict.fromkeys(range(0, 8), (200, 200)))
    events = build_contact_events([char], [obj])
    assert not any(e.kind == ContactEventKind.OBJECT_PICKUP_CANDIDATE for e in events)


# --------------------------------------------------------------------------
# W: final-state
# --------------------------------------------------------------------------


def test_final_state_persists_to_shot_end() -> None:
    obj = _track("O", "OBJECT", dict.fromkeys(range(0, 11), (40, 50)))
    checks = build_final_state_checks([obj], _shot_truth([(1, 0, 10)]))
    assert len(checks) == 1
    assert checks[0].resolved is True
    assert checks[0].final_visible_frame == 10


def test_final_state_not_inferred_as_removed_when_track_ends_early() -> None:
    # Track ends at frame 5 but shot runs to 10 -> removal is NOT inferred.
    obj = _track("O", "OBJECT", dict.fromkeys(range(0, 6), (40, 50)))
    checks = build_final_state_checks([obj], _shot_truth([(1, 0, 10)]))
    assert checks[0].resolved is False
    assert checks[0].final_state == ObjectStateKind.REVIEW_REQUIRED
    assert "not inferred" in (checks[0].review_reason or "")


# --------------------------------------------------------------------------
# X: action boundaries
# --------------------------------------------------------------------------


def test_action_candidates_have_no_semantic_label() -> None:
    boxes = dict.fromkeys(range(0, 4), (40, 50))
    boxes.update({f: (40 + (f - 3) * 6, 50) for f in range(4, 10)})
    obj = _track("O", "OBJECT", boxes)
    candidates = build_action_candidates([], [obj])
    assert candidates
    # Semantic labels are never forced from generic motion; every candidate is
    # review-required and carries evidence.
    assert all(c.semantic_label is None for c in candidates)
    assert all(c.review_required for c in candidates)
    assert all(c.evidence_refs for c in candidates)
    classes = {c.action_class for c in candidates}
    assert ActionStateClass.MOTION_BEGINS in classes
    # Tracker start is NOT proof of appearance (item 7): no ENTITY_APPEARS.
    assert ActionStateClass.ENTITY_APPEARS not in classes


def test_no_action_from_occluded_or_gapped_observations() -> None:
    # A track that is OCCLUDED in the middle must not produce motion across the gap.
    obs = [
        TrackObservation(frame_index=0, x=10, y=10, width=20, height=20,
                         status=TrackStatus.TRACKED),
        TrackObservation(frame_index=1, x=10, y=10, width=20, height=20,
                         status=TrackStatus.OCCLUDED),
        TrackObservation(frame_index=2, x=80, y=10, width=20, height=20,
                         status=TrackStatus.TRACKED),
    ]
    track = EntityTrack(track_id="O", entity_type="OBJECT", first_frame_index=0,
                        last_frame_index=2, observations=obs, status=TrackStatus.REVIEW_REQUIRED)
    candidates = build_action_candidates([], [track])
    # No MOTION_BEGINS across the occluded frame (0->2 jump is not consecutive).
    assert not any(c.action_class == ActionStateClass.MOTION_BEGINS for c in candidates)
    # Occlusion boundaries ARE defensible.
    classes = {c.action_class for c in candidates}
    assert ActionStateClass.OCCLUSION_BEGINS in classes
    assert ActionStateClass.OCCLUSION_ENDS in classes


def test_contact_ignores_occluded_observations() -> None:
    # Character and object overlap, but the object is OCCLUDED there -> no contact.
    char = _track("C", "CHARACTER", dict.fromkeys(range(0, 6), (40, 50)))
    obj_obs = [
        TrackObservation(frame_index=f, x=40, y=50, width=20, height=20,
                         status=TrackStatus.OCCLUDED)
        for f in range(0, 6)
    ]
    obj = EntityTrack(track_id="O", entity_type="OBJECT", first_frame_index=0,
                      last_frame_index=5, observations=obj_obs, status=TrackStatus.REVIEW_REQUIRED)
    events = build_contact_events([char], [obj])
    assert not events  # untrusted observations never infer contact


def test_missing_final_state_check_fails_validator() -> None:
    from manuscript_reviewer.validation.visual_validator import validate_final_states

    obj = _track("O", "OBJECT", dict.fromkeys(range(0, 11), (40, 50)))
    truth = _shot_truth([(1, 0, 10)])
    # No checks supplied though the object appears in shot 1 -> coverage FAIL.
    issues = validate_final_states([], [obj], truth)
    assert any(i.rule_id == "P4-FINAL-001" for i in issues)


def test_action_candidate_boundaries_are_ordered() -> None:
    obj = _track("O", "OBJECT", {f: (40 + f * 6, 50) for f in range(0, 8)})
    for c in build_action_candidates([], [obj]):
        assert c.start_frame <= c.end_frame
