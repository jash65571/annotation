"""Media probe tests against generated fixtures."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from manuscript_reviewer.media.probe import probe_media
from tests.conftest import requires_ffmpeg


@requires_ffmpeg
def test_probe_24fps_metadata(clip_24fps: Path) -> None:
    media, raw = probe_media(clip_24fps)
    assert media.file_name == "clip_24fps.mp4"
    assert media.file_size_bytes > 0
    assert "mp4" in media.container_format
    video = media.video_streams[0]
    assert video.codec_name == "h264"
    assert (video.width, video.height) == (320, 240)
    assert video.nominal_frame_rate == Fraction(24)
    assert video.average_frame_rate == Fraction(24)
    assert video.time_base.denominator > 0
    assert media.audio_streams == []
    assert "format" in raw and "streams" in raw


@requires_ffmpeg
def test_probe_ntsc_rate_is_exact_rational(clip_2997fps: Path) -> None:
    media, _ = probe_media(clip_2997fps)
    video = media.video_streams[0]
    assert video.nominal_frame_rate == Fraction(30000, 1001)  # never 29.97 as a float


@requires_ffmpeg
def test_probe_audio_stream(clip_60fps_audio: Path) -> None:
    media, _ = probe_media(clip_60fps_audio)
    assert media.video_streams[0].nominal_frame_rate == Fraction(60)
    assert len(media.audio_streams) == 1
    audio = media.audio_streams[0]
    assert audio.codec_name == "aac"
    assert audio.sample_rate is not None and audio.sample_rate > 0
    assert audio.channels is not None and audio.channels >= 1


@requires_ffmpeg
def test_probe_corrupt_file_raises(corrupt_file: Path) -> None:
    with pytest.raises(Exception):  # ToolExecutionError or ProbeError  # noqa: B017
        probe_media(corrupt_file)


@requires_ffmpeg
def test_probe_declared_duration_present(clip_24fps: Path) -> None:
    media, _ = probe_media(clip_24fps)
    assert media.container_duration_seconds is not None
    assert media.video_streams[0].declared_duration_seconds is not None
