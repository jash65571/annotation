"""Frame ledger tests: every-frame accounting and exact timing on real media."""

from __future__ import annotations

import itertools
from fractions import Fraction
from pathlib import Path

from manuscript_reviewer.media.frames import enumerate_frames, extract_evidence_frames
from manuscript_reviewer.media.probe import count_decoded_frames, probe_media
from manuscript_reviewer.media.timestamps import cfr_expected_time
from tests.conftest import requires_ffmpeg


@requires_ffmpeg
def test_24fps_every_frame_accounted(clip_24fps: Path) -> None:
    media, _ = probe_media(clip_24fps)
    video = media.video_streams[0]
    ledger = enumerate_frames(clip_24fps, video.time_base)

    assert ledger.frame_count == 48  # 2 s x 24 fps
    assert ledger.frame_count == count_decoded_frames(clip_24fps)
    assert ledger.frame_count == video.declared_frame_count

    # Indexes 0..N-1 sequential; first frame at t=0; strictly increasing pts.
    assert [f.frame_index for f in ledger.frames] == list(range(48))
    assert ledger.frames[0].pts == 0
    assert ledger.frames[0].pts_time_seconds == Fraction(0)
    pts_values = [f.pts for f in ledger.frames]
    assert all(isinstance(p, int) for p in pts_values)
    assert all(b > a for a, b in itertools.pairwise(pts_values))  # type: ignore[operator]

    # Last frame at exactly (N-1)/24 s for this CFR clip.
    assert ledger.frames[-1].pts_time_seconds == Fraction(47, 24)


@requires_ffmpeg
def test_cfr_cross_check_matches_pts(clip_24fps: Path) -> None:
    """frame_index/fps agrees with PTS timing for CFR media (cross-check only)."""
    media, _ = probe_media(clip_24fps)
    video = media.video_streams[0]
    ledger = enumerate_frames(clip_24fps, video.time_base)
    rate = video.nominal_frame_rate
    assert rate is not None
    for frame in ledger.frames:
        assert frame.pts_time_seconds == cfr_expected_time(frame.frame_index, rate)


@requires_ffmpeg
def test_ntsc_2997_exact_timing(clip_2997fps: Path) -> None:
    media, _ = probe_media(clip_2997fps)
    video = media.video_streams[0]
    ledger = enumerate_frames(clip_2997fps, video.time_base)

    assert ledger.frame_count == count_decoded_frames(clip_2997fps)
    rate = Fraction(30000, 1001)
    # Frame N sits exactly on the 1001/30000 grid — no float drift.
    for frame in ledger.frames:
        assert frame.pts_time_seconds == frame.frame_index / rate


@requires_ffmpeg
def test_60fps_with_audio_ledger(clip_60fps_audio: Path) -> None:
    media, _ = probe_media(clip_60fps_audio)
    video = media.video_streams[0]
    ledger = enumerate_frames(clip_60fps_audio, video.time_base)
    assert ledger.frame_count == 120  # 2 s x 60 fps
    assert ledger.frames[-1].pts_time_seconds == Fraction(119, 60)


@requires_ffmpeg
def test_extract_evidence_frames_names(clip_24fps: Path, tmp_path: Path) -> None:
    media, _ = probe_media(clip_24fps)
    ledger = enumerate_frames(clip_24fps, media.video_streams[0].time_base)
    out = extract_evidence_frames(clip_24fps, ledger, tmp_path / "frames")
    assert len(out) == ledger.frame_count
    # Named by index + exact microsecond time, never the 0.1 s display value.
    assert out[0].name == "F000000_0.000000.png"
    assert out[1].name == "F000001_0.041667.png"
    assert all(p.stat().st_size > 0 for p in out)
