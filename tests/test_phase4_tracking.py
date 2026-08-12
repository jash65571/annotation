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
