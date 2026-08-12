"""Shared fixtures: tiny synthetic clips generated with ffmpeg (never committed).

If ffmpeg/ffprobe is unavailable, tests that need real media are skipped with a
clear message; pure-logic tests still run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from manuscript_reviewer.media.ffmpeg_tools import FFmpegNotFoundError, find_tool, run_tool

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def ffmpeg_available() -> bool:
    try:
        find_tool("ffmpeg")
        find_tool("ffprobe")
    except FFmpegNotFoundError:
        return False
    return True


requires_ffmpeg = pytest.mark.skipif(
    not ffmpeg_available(), reason="ffmpeg/ffprobe not found on PATH or MANUSCRIPT_FFMPEG_DIR"
)


def _generate_clip(
    path: Path,
    rate: str,
    duration: str = "2",
    with_audio: bool = False,
    size: str = "320x240",
) -> Path:
    if path.exists():
        return path
    ffmpeg = find_tool("ffmpeg")
    args = [
        "-v", "error",
        "-f", "lavfi",
        "-i", f"testsrc2=duration={duration}:size={size}:rate={rate}",
    ]
    if with_audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if with_audio:
        args += ["-c:a", "aac", "-shortest"]
    args += ["-y", str(path)]
    run_tool(ffmpeg, args)
    return path


@pytest.fixture(scope="session")
def clip_24fps() -> Path:
    """2 s, 24 fps, no audio."""
    FIXTURES_DIR.mkdir(exist_ok=True)
    return _generate_clip(FIXTURES_DIR / "clip_24fps.mp4", rate="24")


@pytest.fixture(scope="session")
def clip_2997fps() -> Path:
    """2 s, 30000/1001 fps (NTSC), no audio."""
    FIXTURES_DIR.mkdir(exist_ok=True)
    return _generate_clip(FIXTURES_DIR / "clip_2997fps.mp4", rate="30000/1001")


@pytest.fixture(scope="session")
def clip_60fps_audio() -> Path:
    """2 s, 60 fps, AAC sine-tone audio."""
    FIXTURES_DIR.mkdir(exist_ok=True)
    return _generate_clip(FIXTURES_DIR / "clip_60fps_audio.mp4", rate="60", with_audio=True)


@pytest.fixture
def corrupt_file(tmp_path: Path) -> Path:
    path = tmp_path / "not_a_video.mp4"
    path.write_bytes(b"this is not video data at all" * 100)
    return path
