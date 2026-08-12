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
    # Semantic labels are never forced from generic motion.
    assert all(c.semantic_label is None for c in candidates)
    assert all(c.review_required for c in candidates)
    classes = {c.action_class for c in candidates}
    assert ActionStateClass.ENTITY_APPEARS in classes
    assert ActionStateClass.MOTION_BEGINS in classes


def test_action_candidate_boundaries_are_ordered() -> None:
    obj = _track("O", "OBJECT", {f: (40 + f * 6, 50) for f in range(0, 8)})
    for c in build_action_candidates([], [obj]):
        assert c.start_frame <= c.end_frame
