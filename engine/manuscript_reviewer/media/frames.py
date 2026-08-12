"""Every-frame enumeration and optional evidence-frame extraction.

Primary enumeration method: ``ffprobe -show_frames`` over the first video
stream, JSON output. This walks the real decoder output frame by frame and
reports each frame's integer PTS in the exact stream time base — the media
authority for timing (AI never owns the clock).

The independent cross-check (``ffprobe -count_frames``) lives in
:mod:`probe`; comparing the two is the ledger validator's job.
"""

from __future__ import annotations

import json
import logging
from fractions import Fraction
from pathlib import Path
from typing import Any

from ..models.frame import FrameLedger, FrameRecord
from .ffmpeg_tools import find_tool, run_tool
from .timestamps import pts_to_seconds, seconds_to_decimal

logger = logging.getLogger(__name__)


class FrameEnumerationError(RuntimeError):
    """Reliable frame enumeration was not possible for this file."""


def _int_or_none(value: Any) -> int | None:
    if value is None or str(value).upper() == "N/A":
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def enumerate_frames(
    video_path: Path, time_base: Fraction, stream_index: int = 0
) -> FrameLedger:
    """Enumerate every decoded frame of one video stream, in presentation order.

    Uses best_effort_timestamp when pts is missing (some codecs omit raw pts on
    individual frames); a frame with neither is recorded with ``pts=None`` and
    the validator downgrades the run.
    """
    ffprobe = find_tool("ffprobe")
    result = run_tool(
        ffprobe,
        [
            "-v", "error",
            "-select_streams", f"v:{stream_index}",
            "-show_frames",
            "-show_entries",
            "frame=pts,best_effort_timestamp,pkt_dts,duration,key_frame,pict_type,width,height",
            "-output_format", "json",
            str(video_path),
        ],
        timeout=1800.0,
    )
    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FrameEnumerationError(f"ffprobe -show_frames returned invalid JSON: {exc}") from exc

    raw_frames = raw.get("frames")
    if not raw_frames:
        raise FrameEnumerationError(
            f"ffprobe -show_frames returned no frames for {video_path.name}"
        )

    records: list[FrameRecord] = []
    for index, frame in enumerate(raw_frames):
        pts = _int_or_none(frame.get("pts"))
        if pts is None:
            pts = _int_or_none(frame.get("best_effort_timestamp"))
        duration = _int_or_none(frame.get("duration"))
        records.append(
            FrameRecord(
                frame_index=index,
                pts=pts,
                pts_time_seconds=pts_to_seconds(pts, time_base) if pts is not None else None,
                dts=_int_or_none(frame.get("pkt_dts")),
                duration=duration,
                duration_seconds=(
                    pts_to_seconds(duration, time_base) if duration is not None else None
                ),
                key_frame=bool(_int_or_none(frame.get("key_frame"))),
                pict_type=frame.get("pict_type"),
                width=_int_or_none(frame.get("width")),
                height=_int_or_none(frame.get("height")),
            )
        )

    return FrameLedger(stream_index=stream_index, time_base=time_base, frames=records)


def extract_evidence_frames(
    video_path: Path, ledger: FrameLedger, out_dir: Path, stream_index: int = 0
) -> list[Path]:
    """Extract every ledger frame as a lossless PNG named by index + exact time.

    Naming: ``F000706_11.766667.png`` — enumeration index plus exact pts_time
    at microsecond precision (NEVER the rounded 0.1 s display value).

    Phase 1 keeps this behind ``--extract-frames``; future phases extract
    lazily/on-demand instead.
    """
    ffmpeg = find_tool("ffmpeg")
    out_dir.mkdir(parents=True, exist_ok=True)
    # -fps_mode passthrough: one output image per decoded frame, no dup/drop,
    # so image N corresponds exactly to ledger frame_index N.
    run_tool(
        ffmpeg,
        [
            "-v", "error",
            "-i", str(video_path),
            "-map", f"0:v:{stream_index}",
            "-fps_mode", "passthrough",
            "-start_number", "0",
            str(out_dir / "F%06d.png"),
        ],
        timeout=3600.0,
    )
    extracted = sorted(out_dir.glob("F??????.png"))
    if len(extracted) != ledger.frame_count:
        raise FrameEnumerationError(
            f"Extracted image count ({len(extracted)}) does not match ledger "
            f"frame count ({ledger.frame_count}); refusing to mislabel evidence."
        )
    renamed: list[Path] = []
    for record, path in zip(ledger.frames, extracted, strict=True):
        if record.pts_time_seconds is not None:
            time_label = str(seconds_to_decimal(record.pts_time_seconds))
        else:
            time_label = "unknown"
        target = out_dir / f"F{record.frame_index:06d}_{time_label}.png"
        path.rename(target)
        renamed.append(target)
    return renamed
