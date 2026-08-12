"""Audio frame enumeration: every decoded audio frame with exact PTS + samples.

Same authority model as the video ledger: ffprobe -show_frames over the first
audio stream. Codec priming/skip metadata is captured from the stream when the
container declares it (AAC does not necessarily start at sample zero).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from ..media.ffmpeg_tools import find_tool, run_tool
from ..media.timestamps import pts_to_seconds
from ..models.audio import AudioFrameRecord

logger = logging.getLogger(__name__)


class AudioProbeError(RuntimeError):
    """Audio frame enumeration was not possible."""


@dataclass(frozen=True)
class AudioPriming:
    """Raw codec priming/skip evidence for the AAC/MP4 sample-anchor question.

    ``initial_padding`` is a stream-level declaration; ``skip_samples`` /
    ``discard_padding`` come from packet side data (the AAC encoder delay the
    decoder trims). Any non-zero value means decoded PCM sample 0 does NOT
    trivially equal the first encoded packet PTS — sample-perfect anchoring is
    not assumed without this evidence (§16)."""

    codec_name: str | None = None
    stream_start_pts: int | None = None
    initial_padding: int | None = None
    skip_samples: int | None = None
    discard_padding: int | None = None

    @property
    def has_priming(self) -> bool:
        return bool(
            (self.initial_padding or 0)
            or (self.skip_samples or 0)
            or (self.discard_padding or 0)
        )


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


def _skip_from_packets(video_path: Path, stream_index: int) -> tuple[int | None, int | None]:
    """First-packet ``skip_samples`` / ``discard_padding`` from packet side data.

    AAC in MP4/M4A carries encoder delay as a "Skip Samples" side-data block on
    the first packet; ffprobe exposes it under ``side_data_list``. Absence of the
    block (older ffprobe / non-AAC) returns ``(None, None)`` — never a fake 0.
    """
    ffprobe = find_tool("ffprobe")
    result = run_tool(
        ffprobe,
        [
            "-v", "error",
            "-select_streams", f"a:{stream_index}",
            "-read_intervals", "%+#1",
            "-show_packets",
            "-show_entries", "packet=pts:packet_side_data=skip_samples,discard_padding",
            "-output_format", "json",
            str(video_path),
        ],
    )
    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None, None
    for packet in raw.get("packets", []):
        for side in packet.get("side_data_list", []):
            skip = side.get("skip_samples")
            discard = side.get("discard_padding")
            if skip is not None or discard is not None:
                return _int_or_none(skip), _int_or_none(discard)
    return None, None


def probe_audio_priming(video_path: Path, stream_index: int = 0) -> AudioPriming:
    """Collect codec priming/skip evidence (stream + packet side data)."""
    ffprobe = find_tool("ffprobe")
    result = run_tool(
        ffprobe,
        [
            "-v", "error",
            "-select_streams", f"a:{stream_index}",
            "-show_entries", "stream=codec_name,start_pts,initial_padding",
            "-output_format", "json",
            str(video_path),
        ],
    )
    codec_name: str | None = None
    start_pts: int | None = None
    initial_padding: int | None = None
    try:
        streams = json.loads(result.stdout).get("streams", [])
        if streams:
            codec_name = streams[0].get("codec_name")
            start_pts = _int_or_none(streams[0].get("start_pts"))
            initial_padding = _int_or_none(streams[0].get("initial_padding"))
    except json.JSONDecodeError:
        pass
    skip_samples, discard_padding = _skip_from_packets(video_path, stream_index)
    return AudioPriming(
        codec_name=codec_name,
        stream_start_pts=start_pts,
        initial_padding=initial_padding,
        skip_samples=skip_samples,
        discard_padding=discard_padding,
    )
