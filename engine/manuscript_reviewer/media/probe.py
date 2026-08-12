"""Media probe: ffprobe JSON → typed MediaInfo, keeping container AND stream views."""

from __future__ import annotations

import json
import logging
from fractions import Fraction
from pathlib import Path
from typing import Any

from ..models.media import AudioStreamInfo, MediaInfo, VideoStreamInfo
from .ffmpeg_tools import find_tool, run_tool
from .timestamps import parse_rational

logger = logging.getLogger(__name__)


class ProbeError(RuntimeError):
    """The file could not be probed as media."""


def _rational_or_none(value: Any) -> Fraction | None:
    """Parse an ffprobe rational/decimal string; undefined ('0/0', 'N/A') → None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    try:
        rational = parse_rational(text)
    except (ValueError, ZeroDivisionError):
        return None
    return rational


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def probe_media(video_path: Path) -> tuple[MediaInfo, dict[str, Any]]:
    """Probe a media file. Returns (typed MediaInfo, raw ffprobe JSON).

    The raw JSON is preserved verbatim in media.json so no probed fact is ever
    lost to model simplification.
    """
    ffprobe = find_tool("ffprobe")
    result = run_tool(
        ffprobe,
        [
            "-v", "error",
            "-show_format",
            "-show_streams",
            "-output_format", "json",
            str(video_path),
        ],
    )
    try:
        raw: dict[str, Any] = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ProbeError(f"ffprobe produced invalid JSON for {video_path}") from exc
    if "format" not in raw:
        raise ProbeError(f"ffprobe returned no format section for {video_path}")

    fmt = raw["format"]
    video_streams: list[VideoStreamInfo] = []
    audio_streams: list[AudioStreamInfo] = []

    for stream in raw.get("streams", []):
        codec_type = stream.get("codec_type")
        if codec_type == "video":
            time_base = _rational_or_none(stream.get("time_base"))
            if time_base is None:
                raise ProbeError(
                    f"Video stream {stream.get('index')} has no usable time_base; "
                    "exact timing is impossible."
                )
            video_streams.append(
                VideoStreamInfo(
                    stream_index=int(stream["index"]),
                    codec_name=str(stream.get("codec_name", "unknown")),
                    profile=stream.get("profile"),
                    pix_fmt=stream.get("pix_fmt"),
                    width=int(stream["width"]),
                    height=int(stream["height"]),
                    sample_aspect_ratio=stream.get("sample_aspect_ratio"),
                    display_aspect_ratio=stream.get("display_aspect_ratio"),
                    nominal_frame_rate=_rational_or_none(stream.get("r_frame_rate")),
                    average_frame_rate=_rational_or_none(stream.get("avg_frame_rate")),
                    time_base=time_base,
                    start_pts=_int_or_none(stream.get("start_pts")),
                    declared_duration_seconds=_rational_or_none(stream.get("duration")),
                    declared_frame_count=_int_or_none(stream.get("nb_frames")),
                )
            )
        elif codec_type == "audio":
            audio_streams.append(
                AudioStreamInfo(
                    stream_index=int(stream["index"]),
                    codec_name=str(stream.get("codec_name", "unknown")),
                    sample_rate=_int_or_none(stream.get("sample_rate")),
                    channels=_int_or_none(stream.get("channels")),
                    channel_layout=stream.get("channel_layout"),
                    time_base=_rational_or_none(stream.get("time_base")),
                    start_pts=_int_or_none(stream.get("start_pts")),
                    declared_duration_seconds=_rational_or_none(stream.get("duration")),
                )
            )

    media = MediaInfo(
        file_name=video_path.name,
        file_size_bytes=video_path.stat().st_size,
        container_format=str(fmt.get("format_name", "unknown")),
        container_duration_seconds=_rational_or_none(fmt.get("duration")),
        container_start_time_seconds=_rational_or_none(fmt.get("start_time")),
        container_bit_rate=_int_or_none(fmt.get("bit_rate")),
        video_streams=video_streams,
        audio_streams=audio_streams,
    )
    return media, raw


def count_decoded_frames(video_path: Path, stream_index: int = 0) -> int | None:
    """Independent cross-check signal: ffprobe -count_frames (full decode).

    Runs a separate full-decode pass and reads ``nb_read_frames``. Returns None
    if ffprobe cannot report it for this container/codec.
    """
    ffprobe = find_tool("ffprobe")
    result = run_tool(
        ffprobe,
        [
            "-v", "error",
            "-select_streams", f"v:{stream_index}",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames",
            "-output_format", "json",
            str(video_path),
        ],
    )
    try:
        raw = json.loads(result.stdout)
        streams = raw.get("streams", [])
        if not streams:
            return None
        return _int_or_none(streams[0].get("nb_read_frames"))
    except json.JSONDecodeError:
        return None
