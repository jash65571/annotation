"""Phase 4 slice-2 tests: frame cache, per-frame observation ledger, and
deterministic visual concerns."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from manuscript_reviewer.media import frames as frames_mod
from manuscript_reviewer.media import probe as probe_mod
from manuscript_reviewer.media.clock import AnnotationClock
from manuscript_reviewer.models.frame import FrameLedger, FrameRecord
from manuscript_reviewer.models.review_intelligence import FrameObservation
from manuscript_reviewer.validation.visual_validator import validate_frame_observations
from manuscript_reviewer.visual.concerns import (
    OVEREXPOSURE,
    UNDEREXPOSURE,
    FrameConcernInputs,
    detect_frame_concerns,
)
from manuscript_reviewer.visual.decode import FrameCache
from manuscript_reviewer.visual.observations import build_frame_observations

from .conftest import requires_ffmpeg

# --------------------------------------------------------------------------
# Pure-logic: concerns
# --------------------------------------------------------------------------


def test_overexposure_concern() -> None:
    codes = detect_frame_concerns(
        FrameConcernInputs(
            brightness=0.98,
            sharpness=0.02,
            near_white_fraction=0.8,
            near_black_fraction=0.0,
            motion_magnitude=0.0,
        )
    )
    assert OVEREXPOSURE in codes


def test_underexposure_concern() -> None:
    codes = detect_frame_concerns(
        FrameConcernInputs(
            brightness=0.02,
            sharpness=0.02,
            near_white_fraction=0.0,
            near_black_fraction=0.9,
            motion_magnitude=0.0,
        )
    )
    assert UNDEREXPOSURE in codes


def test_exposure_flash_is_concern_not_text() -> None:
    # A bright flash produces an exposure concern and never any text signal.
    codes = detect_frame_concerns(
        FrameConcernInputs(
            brightness=0.97,
            sharpness=0.01,
            near_white_fraction=0.7,
            near_black_fraction=0.0,
            motion_magnitude=0.0,
        )
    )
    assert OVEREXPOSURE in codes
    assert all("TEXT" not in c for c in codes)


# --------------------------------------------------------------------------
# Pure-logic: observation validator
# --------------------------------------------------------------------------


def _ledger(n: int) -> FrameLedger:
    from fractions import Fraction

    frames = [
        FrameRecord(
            frame_index=i,
            pts=i * 100,
            pts_time_seconds=Fraction(i, 10),
            key_frame=(i == 0),
        )
        for i in range(n)
    ]
    return FrameLedger(stream_index=0, time_base=Fraction(1, 1000), frames=frames)


def _obs(i: int, with_time: bool = True) -> FrameObservation:
    from fractions import Fraction

    return FrameObservation(
        frame_index=i,
        source_pts_time_exact=Fraction(i, 10) if with_time else None,
        annotation_time_exact=Fraction(i, 10) if with_time else None,
        brightness=0.5,
        contrast=0.1,
        sharpness=0.01,
        motion_magnitude=0.0,
        global_camera_motion=0.0,
        foreground_motion=0.0,
    )


def test_validator_flags_count_mismatch() -> None:
    ledger = _ledger(5)
    issues = validate_frame_observations([_obs(0), _obs(1)], ledger)
    assert any(i.rule_id == "P4-OBS-001" for i in issues)


def test_validator_flags_missing_exact_identity() -> None:
    ledger = _ledger(2)
    issues = validate_frame_observations([_obs(0), _obs(1, with_time=False)], ledger)
    assert any(i.rule_id == "P4-OBS-003" for i in issues)


def test_validator_passes_aligned_observations() -> None:
    ledger = _ledger(3)
    obs = [_obs(i) for i in range(3)]
    assert not validate_frame_observations(obs, ledger)


# --------------------------------------------------------------------------
# ffmpeg integration: real frame observations on a synthetic testsrc2 clip
# --------------------------------------------------------------------------


def _build_ledger(path: Path) -> FrameLedger:
    media, _ = probe_mod.probe_media(path)
    stream = media.video_streams[0]
    return frames_mod.enumerate_frames(path, stream.time_base, stream_index=0)


@requires_ffmpeg
def test_frame_observations_align_to_ledger(clip_24fps: Path) -> None:
    ledger = _build_ledger(clip_24fps)
    cache = FrameCache(clip_24fps, ledger)
    clock = AnnotationClock.from_ledger(ledger)
    observations = build_frame_observations(cache, ledger, clock, None)

    assert len(observations) == ledger.frame_count
    # Exact frame identity + monotonic order.
    for i, obs in enumerate(observations):
        assert obs.frame_index == i
        assert obs.source_pts_time_exact is not None
    # The shared gray grid is decoded exactly once.
    assert cache.cache_stats()["gray_decode_count"] == 1
    # testsrc2 is animated -> some frame shows global motion.
    assert any(o.global_camera_motion > 0 for o in observations[1:])


@requires_ffmpeg
def test_frame_cache_color_lru(clip_24fps: Path) -> None:
    ledger = _build_ledger(clip_24fps)
    cache = FrameCache(clip_24fps, ledger, color_cache_size=2)
    a = cache.color_frame(0)
    assert a.ndim == 3 and a.shape[2] == 3
    cache.color_frame(1)
    cache.color_frame(0)  # hit
    cache.color_frame(2)  # evicts least-recent
    stats = cache.cache_stats()
    assert stats["color_hits"] >= 1
    assert stats["color_cached"] <= 2


@requires_ffmpeg
def test_gray_frames_decoded_once(clip_24fps: Path) -> None:
    ledger = _build_ledger(clip_24fps)
    cache = FrameCache(clip_24fps, ledger)
    first = cache.gray_frames()
    second = cache.gray_frames()
    assert first is second
    assert cache.gray_decode_count == 1
    assert first.shape[0] == ledger.frame_count
    # numpy array is real image data.
    assert first.dtype == np.uint8
