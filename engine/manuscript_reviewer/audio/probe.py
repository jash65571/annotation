"""Audio frame enumeration: every decoded audio frame with exact PTS + samples.

Same authority model as the video ledger: ffprobe -show_frames over the first
audio stream. Codec priming/skip metadata is captured from the stream when the
container declares it (AAC does not necessarily start at sample zero).
"""

from __future__ import annotations

import json
import logging
from fractions import Fraction
from pathlib import Path
from typing import Any

from ..media.ffmpeg_tools import find_tool, run_tool
from ..media.timestamps import pts_to_seconds
from ..models.audio import AudioFrameRecord

logger = logging.getLogger(__name__)


class AudioProbeError(RuntimeError):
    """Audio frame enumeration was not possible."""


def _int_or_none(value: Any) -> int | None:
    if value is None or str(value).upper() == "N/A":
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def enumerate_audio_frames(
    video_path: Path,
    time_base: Fraction,
    annotation_origin: Fraction,
    stream_index: int = 0,
) -> list[AudioFrameRecord]:
    """Enumerate every decoded audio frame of one audio stream, in order."""
    ffprobe = find_tool("ffprobe")
    result = run_tool(
        ffprobe,
        [
            "-v", "error",
            "-select_streams", f"a:{stream_index}",
            "-show_frames",
            "-show_entries",
            "frame=pts,best_effort_timestamp,duration,nb_samples",
            "-output_format", "json",
            str(video_path),
        ],
        timeout=1800.0,
    )
    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AudioProbeError(f"ffprobe audio -show_frames invalid JSON: {exc}") from exc

    frames = raw.get("frames")
    if not frames:
        raise AudioProbeError(f"No decoded audio frames for {video_path.name}")

    records: list[AudioFrameRecord] = []
    for index, frame in enumerate(frames):
        pts = _int_or_none(frame.get("pts"))
        if pts is None:
            pts = _int_or_none(frame.get("best_effort_timestamp"))
        duration = _int_or_none(frame.get("duration"))
        pts_time = pts_to_seconds(pts, time_base) if pts is not None else None
        records.append(
            AudioFrameRecord(
                audio_frame_index=index,
                pts=pts,
                pts_time_source=pts_time,
                annotation_time=(
                    pts_time - annotation_origin if pts_time is not None else None
                ),
                duration=duration,
                duration_time=(
                    pts_to_seconds(duration, time_base) if duration is not None else None
                ),
                nb_samples=_int_or_none(frame.get("nb_samples")),
            )
        )
    return records


def stream_initial_padding(video_path: Path, stream_index: int = 0) -> int | None:
    """Codec priming samples declared by the container/stream, if any."""
    ffprobe = find_tool("ffprobe")
    result = run_tool(
        ffprobe,
        [
            "-v", "error",
            "-select_streams", f"a:{stream_index}",
            "-show_entries", "stream=initial_padding",
            "-output_format", "json",
            str(video_path),
        ],
    )
    try:
        raw = json.loads(result.stdout)
        streams = raw.get("streams", [])
        if streams:
            return _int_or_none(streams[0].get("initial_padding"))
    except json.JSONDecodeError:
        pass
    return None
