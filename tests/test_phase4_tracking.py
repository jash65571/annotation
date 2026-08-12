"""Phase 4 tracking tests (S/T/U): anchor-seeded local tracking, continuity, and
the identity defenses (two-similar-{person,object}, reacquisition)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from manuscript_reviewer.media import frames as frames_mod
from manuscript_reviewer.media import probe as probe_mod
from manuscript_reviewer.media.clock import AnnotationClock
from manuscript_reviewer.models.frame import FrameLedger
from manuscript_reviewer.models.review_intelligence import (
    EntityTrack,
    TrackObservation,
    TrackStatus,
    VisualAnchor,
)
from manuscript_reviewer.tracking.continuity import build_continuity
from manuscript_reviewer.tracking.tracker import track_anchor
from manuscript_reviewer.validation.visual_validator import validate_tracks
from manuscript_reviewer.visual.decode import FrameCache

from .conftest import requires_ffmpeg, synth_clip

_W, _H = 320, 180


def _bg() -> np.ndarray:
    rng = np.random.default_rng(3)
    g = rng.integers(0, 120, size=(_H, _W), dtype=np.uint8)
    return np.stack([g, g, g], axis=2)


#: A distinctive TEXTURED square (uniform patches degenerate template matching).
_SQUARE = np.random.default_rng(99).integers(140, 256, size=(30, 30), dtype=np.uint8)


def _place_square(frame: np.ndarray, x: int, y: int = 70) -> None:
    patch = np.stack([_SQUARE, _SQUARE, _SQUARE], axis=2)
    frame[y : y + 30, x : x + 30] = patch


def _build_ledger(path: Path) -> FrameLedger:
    media, _ = probe_mod.probe_media(path)
    stream = media.video_streams[0]
    return frames_mod.enumerate_frames(path, stream.time_base, stream_index=0)


# --------------------------------------------------------------------------
# Pure continuity / defenses
# --------------------------------------------------------------------------


def _track(track_id: str, entity_type: str, first: int, last: int,
           reacquired: bool = False) -> EntityTrack:
    obs = [
        TrackObservation(frame_index=f, x=0, y=0, width=10, height=10)
        for f in range(first, last + 1)
    ]
    return EntityTrack(
        track_id=track_id,
        entity_type=entity_type,
        first_frame_index=first,
        last_frame_index=last,
        observations=obs,
        status=TrackStatus.REVIEW_REQUIRED if reacquired else TrackStatus.TRACKED,
        reacquired=reacquired,
    )


def test_two_similar_people_are_not_merged() -> None:
    a = _track("TRK-a", "CHARACTER", 0, 10)
    b = _track("TRK-b", "CHARACTER", 0, 10)  # visually identical, same time
    chars, _objs, _links = build_continuity([a, b])
    assert len(chars) == 2  # never merged into one
    assert {c.proposed_label for c in chars} == {"C1", "C2"}
    assert all(len(c.track_ids) == 1 for c in chars)


def test_two_similar_objects_are_not_merged() -> None:
    a = _track("TRK-a", "OBJECT", 0, 5)
    b = _track("TRK-b", "OBJECT", 0, 5)
    _chars, objs, _links = build_continuity([a, b])
    assert len(objs) == 2
    assert {o.proposed_label for o in objs} == {"O1", "O2"}


def test_labels_in_first_appearance_order() -> None:
    late = _track("TRK-late", "CHARACTER", 20, 30)
    early = _track("TRK-early", "CHARACTER", 2, 10)
    chars, _o, _l = build_continuity([late, early])
    # C1 is the earliest first appearance regardless of input order.
    c1 = next(c for c in chars if c.proposed_label == "C1")
    assert c1.track_ids == ["TRK-early"]


def test_reacquired_track_cannot_be_silently_verified() -> None:
    bad = _track("TRK-x", "OBJECT", 0, 20, reacquired=True)
    bad.status = TrackStatus.TRACKED  # illegal: reacquired must stay REVIEW
    issues = validate_tracks([bad], frame_count=30)
    assert any(i.rule_id == "P4-ENTITY-003" for i in issues)


def _two_shots():  # type: ignore[no-untyped-def]
    from fractions import Fraction

    from manuscript_reviewer.models.shot_truth import (
        CandidateStatus,
        ShotProposal,
        ShotTruthResult,
        TransitionStatus,
    )

    def prop(i: int, lo: int, hi: int) -> ShotProposal:
        return ShotProposal(
            shot_index=i, start_frame_index=lo, end_frame_index=hi,
            start_exact=Fraction(lo, 10), end_exact=Fraction(hi, 10),
            last_owned_frame_start_exact=Fraction(lo, 10),
            start_manuscript=None, end_manuscript=None, transition_into_shot=None,
            transition_status=TransitionStatus.PROPOSED, supporting_boundary_id=None,
            review_status=CandidateStatus.SUPPORTED,
        )
    return ShotTruthResult(
        frame_count=40, adjacent_pair_count=0, raw_candidate_count=0, merged_candidate_count=0,
        supported_count=0, rejected_count=0, review_required_count=0, proposed_shot_count=2,
        overall_status="PASS", candidates=[], shots=[prop(1, 0, 19), prop(2, 20, 39)],
    )


def test_track_spanning_two_shots_is_flagged() -> None:
    # A single track with observations in shot 1 (0-19) AND shot 2 (20-30).
    crossing = _track("TRK-cross", "CHARACTER", 10, 30)
    issues = validate_tracks([crossing], frame_count=40, shot_result=_two_shots())
    assert any(i.rule_id == "P4-ENTITY-005" for i in issues)
    # A shot-bounded track raises nothing.
    bounded = _track("TRK-ok", "CHARACTER", 0, 15)
    assert not any(
        i.rule_id == "P4-ENTITY-005"
        for i in validate_tracks([bounded], frame_count=40, shot_result=_two_shots())
    )


def test_same_entity_candidate_link_must_stay_review_required() -> None:
    from manuscript_reviewer.models.review_intelligence import ContinuityLink
    from manuscript_reviewer.validation.visual_validator import validate_continuity_links

    bad = ContinuityLink(
        link_id="L1", from_track_id="TRK-a", to_track_id="TRK-b",
        relationship="SAME_ENTITY_CANDIDATE", review_required=False,  # illegal auto-merge
    )
    assert any(i.rule_id == "P4-ENTITY-006" for i in validate_continuity_links([bad]))
    ok = ContinuityLink(
        link_id="L2", from_track_id="TRK-a", to_track_id="TRK-b",
        relationship="SAME_ENTITY_CANDIDATE", review_required=True,
    )
    assert not validate_continuity_links([ok])


# --------------------------------------------------------------------------
# ffmpeg: real tracking on a synthetic clip
# --------------------------------------------------------------------------


@requires_ffmpeg
def test_track_follows_moving_template(tmp_path: Path) -> None:
    frames = []
    for i in range(16):
        f = _bg().copy()
        x = 20 + i * 8
        _place_square(f, x)  # textured square moving right
        frames.append(f)
    clip = synth_clip(tmp_path / "moving.mp4", frames)
    ledger = _build_ledger(clip)
    cache = FrameCache(clip, ledger)
    clock = AnnotationClock.from_ledger(ledger)
    anchor = VisualAnchor(
        anchor_id="a1", frame_index=0, x=20, y=70, width=30, height=30,
        entity_type="OBJECT", temporary_label="square",
    )
    track = track_anchor(cache.gray_frames(), ledger, clock, anchor, _W, _H)
    xs = [o.x for o in track.observations]
    # The tracked box follows the square rightward across the clip.
    assert xs[-1] > xs[0]
    assert len(track.observations) >= 10


def _grid_ledger(n: int) -> FrameLedger:
    from fractions import Fraction

    from manuscript_reviewer.models.frame import FrameRecord
    frames = [
        FrameRecord(frame_index=i, pts=i, pts_time_seconds=Fraction(i, 24),
                    key_frame=(i == 0), width=160, height=90)
        for i in range(n)
    ]
    return FrameLedger(stream_index=0, time_base=Fraction(1, 24), frames=frames)


def _grid_clip_gray(n: int = 12) -> np.ndarray:
    rng = np.random.default_rng(5)
    bg = rng.integers(0, 120, size=(90, 160), dtype=np.uint8)
    patch = np.random.default_rng(9).integers(160, 256, size=(12, 12), dtype=np.uint8)
    frames = []
    for i in range(n):
        f = bg.copy()
        x = 10 + i * 4
        f[40:52, x : x + 12] = patch
        frames.append(f)
    return np.stack(frames, axis=0)


def test_tracking_is_bounded_to_shot() -> None:
    # The patch is trackable across all 12 frames, but the anchor's shot is [0,5]:
    # tracking must not cross the cut into later frames (item 3).
    from manuscript_reviewer.media.clock import AnnotationClock

    gray = _grid_clip_gray(12)
    ledger = _grid_ledger(12)
    clock = AnnotationClock.from_ledger(ledger)
    anchor = VisualAnchor(
        anchor_id="a", frame_index=0, x=10, y=40, width=12, height=12, entity_type="OBJECT",
    )
    track = track_anchor(gray, ledger, clock, anchor, 160, 90, shot_bounds=(0, 5))
    assert max(o.frame_index for o in track.observations) <= 5
    assert min(o.frame_index for o in track.observations) >= 0


@requires_ffmpeg
def test_occlusion_then_reacquire_is_review_required(tmp_path: Path) -> None:
    frames = []
    for i in range(18):
        f = _bg().copy()
        x = 20 + i * 6
        _place_square(f, x)
        if 7 <= i <= 11:
            f[:, :] = 0  # whole frame occluded (black) -> template lost
        frames.append(f)
    clip = synth_clip(tmp_path / "occlude.mp4", frames)
    ledger = _build_ledger(clip)
    cache = FrameCache(clip, ledger)
    clock = AnnotationClock.from_ledger(ledger)
    anchor = VisualAnchor(
        anchor_id="a1", frame_index=0, x=20, y=70, width=30, height=30,
        entity_type="OBJECT",
    )
    track = track_anchor(cache.gray_frames(), ledger, clock, anchor, _W, _H)
    # A loss/reacquisition is never silently verified.
    assert track.status == TrackStatus.REVIEW_REQUIRED
    assert not validate_tracks([track], frame_count=ledger.frame_count)
    # OCCLUDED observations carry uncertainty and never the anchor's score of 1.0.
    occluded = [o for o in track.observations if o.status == TrackStatus.OCCLUDED]
    assert occluded
    assert all(o.appearance_similarity is None for o in occluded)
    assert all(o.notes for o in occluded)
