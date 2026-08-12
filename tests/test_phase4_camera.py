"""Phase 4 camera-motion tests on deterministic synthetic clips.

Proves the camera-vs-subject defense and phase segmentation:
- global background shift  -> camera-motion candidate
- moving object, static bg -> NOT a pan
- direction reversal       -> two separate phases
- uniform scale            -> scale-change candidate (never a pan)
- static                   -> STATIC only

Assertions are on the derived motion class / direction / exact phase frames —
never on raw float magnitudes — so they survive cross-platform FFT differences.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from manuscript_reviewer.camera.segmentation import analyze_camera_motion
from manuscript_reviewer.media import frames as frames_mod
from manuscript_reviewer.media import probe as probe_mod
from manuscript_reviewer.media.clock import AnnotationClock
from manuscript_reviewer.models.frame import FrameLedger
from manuscript_reviewer.models.review_intelligence import CameraMotionCandidate, CameraMotionClass
from manuscript_reviewer.validation.visual_validator import count_direction_reversals
from manuscript_reviewer.visual.decode import FrameCache

from .conftest import requires_ffmpeg, synth_clip

_W, _H = 320, 180
_MOVEMENT = {
    CameraMotionClass.HORIZONTAL_GLOBAL_MOTION,
    CameraMotionClass.VERTICAL_GLOBAL_MOTION,
    CameraMotionClass.DIAGONAL_GLOBAL_MOTION,
}


def _texture() -> np.ndarray:
    rng = np.random.default_rng(7)
    gray = rng.integers(0, 256, size=(_H, _W), dtype=np.uint8)
    return np.stack([gray, gray, gray], axis=2)


def _to_bgr(gray2d: np.ndarray) -> np.ndarray:
    return np.stack([gray2d, gray2d, gray2d], axis=2)


def _build_ledger(path: Path) -> FrameLedger:
    media, _ = probe_mod.probe_media(path)
    stream = media.video_streams[0]
    return frames_mod.enumerate_frames(path, stream.time_base, stream_index=0)


def _analyze(path: Path) -> list[CameraMotionCandidate]:
    ledger = _build_ledger(path)
    cache = FrameCache(path, ledger)
    clock = AnnotationClock.from_ledger(ledger)
    return analyze_camera_motion(cache.gray_frames(), ledger, clock, None)


@requires_ffmpeg
def test_global_horizontal_shift_is_camera_motion(tmp_path: Path) -> None:
    base = _texture()[:, :, 0]
    frames = [_to_bgr(np.roll(base, shift=i * 3, axis=1)) for i in range(18)]
    clip = synth_clip(tmp_path / "hshift.mp4", frames)
    candidates = _analyze(clip)
    horiz = [c for c in candidates if c.motion_class == CameraMotionClass.HORIZONTAL_GLOBAL_MOTION]
    assert horiz, [c.motion_class.value for c in candidates]
    # A movement phase must be supported by a real global correlation.
    assert all(c.inlier_ratio > 0 for c in horiz)


@requires_ffmpeg
def test_moving_object_static_background_is_not_a_pan(tmp_path: Path) -> None:
    base = _texture()[:, :, 0]
    frames = []
    for i in range(18):
        frame = base.copy()
        x = 10 + i * 12
        frame[70:110, x : x + 30] = 255  # small bright object crossing frame
        frames.append(_to_bgr(frame))
    clip = synth_clip(tmp_path / "object.mp4", frames)
    candidates = _analyze(clip)
    movement = [c for c in candidates if c.motion_class in _MOVEMENT]
    assert not movement, [(c.motion_class.value, c.direction) for c in candidates]


@requires_ffmpeg
def test_direction_reversal_splits_into_two_phases(tmp_path: Path) -> None:
    base = _texture()[:, :, 0]
    frames = []
    for i in range(20):
        shift = i * 3 if i < 10 else (10 * 3 - (i - 9) * 3)
        frames.append(_to_bgr(np.roll(base, shift=shift, axis=1)))
    clip = synth_clip(tmp_path / "reversal.mp4", frames)
    candidates = _analyze(clip)
    horiz = [c for c in candidates if c.motion_class == CameraMotionClass.HORIZONTAL_GLOBAL_MOTION]
    directions = {c.direction for c in horiz}
    assert len(horiz) >= 2, [c.direction for c in horiz]
    assert len(directions) >= 2  # opposite directions -> separate phases
    assert count_direction_reversals(candidates) >= 1


@requires_ffmpeg
def test_uniform_scale_is_scale_change_not_pan(tmp_path: Path) -> None:
    import cv2

    base = _texture()[:, :, 0]
    frames = []
    for i in range(16):
        zoom = 1.0 + i * 0.03
        cw, ch = int(_W / zoom), int(_H / zoom)
        x0, y0 = (_W - cw) // 2, (_H - ch) // 2
        crop = base[y0 : y0 + ch, x0 : x0 + cw]
        resized = cv2.resize(crop, (_W, _H), interpolation=cv2.INTER_LINEAR)
        frames.append(_to_bgr(resized))
    clip = synth_clip(tmp_path / "zoom.mp4", frames)
    candidates = _analyze(clip)
    classes = {c.motion_class for c in candidates}
    # Never a pan; a scale-change candidate is expected.
    assert CameraMotionClass.HORIZONTAL_GLOBAL_MOTION not in classes
    assert CameraMotionClass.SCALE_INCREASE in classes, [c.motion_class.value for c in candidates]


@requires_ffmpeg
def test_static_scene_is_static(tmp_path: Path) -> None:
    base = _texture()
    frames = [base.copy() for _ in range(12)]
    clip = synth_clip(tmp_path / "static.mp4", frames)
    candidates = _analyze(clip)
    movement = [c for c in candidates if c.motion_class in _MOVEMENT]
    assert not movement
