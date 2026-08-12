"""Phase 4 playback-speed evidence tests (Y).

Covers the safety rules: fast motion is not accelerated playback, motion blur is
not slow motion, and duplicated frames read as a slow-motion candidate."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from manuscript_reviewer.media import frames as frames_mod
from manuscript_reviewer.media import probe as probe_mod
from manuscript_reviewer.models.review_intelligence import SpeedConclusion
from manuscript_reviewer.models.shot_truth import (
    CandidateStatus,
    ShotProposal,
    ShotTruthResult,
    TransitionStatus,
)
from manuscript_reviewer.speed.evidence import build_playback_speed_evidence
from manuscript_reviewer.visual.decode import FrameCache

from .conftest import requires_ffmpeg, synth_clip

_W, _H = 320, 180


def _texture(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 256, size=(_H, _W), dtype=np.uint8)
    return np.stack([g, g, g], axis=2)


def _one_shot_truth(frame_count: int) -> ShotTruthResult:
    prop = ShotProposal(
        shot_index=1, start_frame_index=0, end_frame_index=frame_count - 1,
        start_exact=None, end_exact=None, last_owned_frame_start_exact=None,
        start_manuscript=None, end_manuscript=None, transition_into_shot="Opening shot",
        transition_status=TransitionStatus.PROPOSED, supporting_boundary_id=None,
        review_status=CandidateStatus.SUPPORTED,
    )
    return ShotTruthResult(
        frame_count=frame_count, adjacent_pair_count=0, raw_candidate_count=0,
        merged_candidate_count=0, supported_count=0, rejected_count=0,
        review_required_count=0, proposed_shot_count=1, overall_status="PASS",
        candidates=[], shots=[prop],
    )


def _evidence(path: Path, n: int) -> SpeedConclusion:
    ledger = _build_ledger(path)
    cache = FrameCache(path, ledger)
    ev = build_playback_speed_evidence(
        cache.gray_frames(), _one_shot_truth(ledger.frame_count), ledger
    )
    return ev[0].conclusion


def _build_ledger(path: Path):  # type: ignore[no-untyped-def]
    media, _ = probe_mod.probe_media(path)
    stream = media.video_streams[0]
    return frames_mod.enumerate_frames(path, stream.time_base, stream_index=0)


@requires_ffmpeg
def test_regular_playback_is_regular_supported(tmp_path: Path) -> None:
    base = _texture(1)[:, :, 0]
    frames = [np.stack([np.roll(base, i * 4, axis=1)] * 3, axis=2) for i in range(16)]
    clip = synth_clip(tmp_path / "regular.mp4", frames)
    assert _evidence(clip, 16) == SpeedConclusion.REGULAR_SUPPORTED


@requires_ffmpeg
def test_duplicated_frames_read_as_slow_motion_candidate(tmp_path: Path) -> None:
    base = _texture(2)[:, :, 0]
    frames = []
    for i in range(10):
        f = np.stack([np.roll(base, i * 6, axis=1)] * 3, axis=2)
        frames.append(f)
        frames.append(f.copy())  # each frame duplicated -> slow-motion look
    clip = synth_clip(tmp_path / "slow.mp4", frames)
    assert _evidence(clip, 20) == SpeedConclusion.SLOW_MOTION_CANDIDATE


def test_regular_needs_positive_evidence_not_absence() -> None:
    from manuscript_reviewer.speed.cadence import ShotCadence
    from manuscript_reviewer.speed.evidence import conclude_speed

    cadence = ShotCadence(pair_count=10, duplicate_ratio=0.0, motion_present=True,
                          first_half_dup=0.0, second_half_dup=0.0, diffs=[0.05] * 10)
    # No positive PTS-cadence evidence -> not REGULAR_SUPPORTED (item 9).
    concl, review = conclude_speed(cadence, pts_regular=False)
    assert concl == SpeedConclusion.REVIEW_REQUIRED and review
    # With positive PTS cadence evidence -> REGULAR_SUPPORTED.
    concl2, review2 = conclude_speed(cadence, pts_regular=True)
    assert concl2 == SpeedConclusion.REGULAR_SUPPORTED and not review2


def test_accelerated_is_reachable_with_evidence() -> None:
    from manuscript_reviewer.speed.cadence import ShotCadence
    from manuscript_reviewer.speed.evidence import conclude_speed

    cadence = ShotCadence(pair_count=10, duplicate_ratio=0.0, motion_present=True,
                          first_half_dup=0.0, second_half_dup=0.0, diffs=[0.05] * 10)
    concl, review = conclude_speed(cadence, pts_regular=True, accelerated_evidence=True)
    assert concl == SpeedConclusion.ACCELERATED_CANDIDATE and review


@requires_ffmpeg
def test_fast_motion_is_not_accelerated(tmp_path: Path) -> None:
    base = _texture(3)[:, :, 0]
    frames = [np.stack([np.roll(base, i * 40, axis=1)] * 3, axis=2) for i in range(16)]
    clip = synth_clip(tmp_path / "fast.mp4", frames)
    # Fast motion must NEVER be concluded as accelerated playback.
    assert _evidence(clip, 16) != SpeedConclusion.ACCELERATED_CANDIDATE
