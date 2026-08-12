"""Metric decode: every ledger frame as a small grayscale image, in ledger order.

One additional full decode of the stream, scaled to a deterministic metric grid
(160x90, grayscale) with ``-fps_mode passthrough`` so decoded image N is ledger
frame N. The decoded count MUST equal the ledger count or the shot stage fails —
metrics must never silently misalign with frame identity.

The Phase 1 ledger (``ffprobe -show_frames``) remains the timing authority;
this pass only supplies pixels.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ..media.ffmpeg_tools import ToolExecutionError, find_tool

logger = logging.getLogger(__name__)

#: Deterministic metric grid (documented in docs/06). 16:9-ish, small enough to
#: keep a 20 s / 60 fps clip (~1200 frames) at ~17 MB in memory.
METRIC_WIDTH = 160
METRIC_HEIGHT = 90

GrayFrames = npt.NDArray[np.uint8]  # shape (n_frames, METRIC_HEIGHT, METRIC_WIDTH)


class MetricDecodeError(RuntimeError):
    """The metric decode did not produce one image per ledger frame."""


def decode_metric_frames(
    video_path: Path,
    expected_frame_count: int,
    stream_index: int = 0,
) -> GrayFrames:
    """Decode every frame of the stream to the metric grid.

    Returns an array of shape (N, H, W) uint8 where row i is ledger frame i.
    Raises :class:`MetricDecodeError` if the decoded count differs from the
    ledger count.
    """
    ffmpeg = find_tool("ffmpeg")
    command = [
        str(ffmpeg),
        "-v", "error",
        "-i", str(video_path),
        "-map", f"0:v:{stream_index}",
        "-fps_mode", "passthrough",
        "-vf", f"scale={METRIC_WIDTH}:{METRIC_HEIGHT}:flags=area",
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-",
    ]
    logger.debug("metric decode: %s", " ".join(command))
    completed = subprocess.run(
        command,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=1800.0,
        check=False,
    )
    if completed.returncode != 0:
        raise ToolExecutionError(
            command, completed.returncode, completed.stderr.decode("utf-8", "replace")
        )

    frame_bytes = METRIC_WIDTH * METRIC_HEIGHT
    data = completed.stdout
    if len(data) % frame_bytes != 0:
        raise MetricDecodeError(
            f"Metric decode produced {len(data)} bytes, not a multiple of "
            f"{frame_bytes}-byte frames."
        )
    count = len(data) // frame_bytes
    if count != expected_frame_count:
        raise MetricDecodeError(
            f"Metric decode produced {count} frames but the ledger has "
            f"{expected_frame_count}; refusing to compute misaligned metrics."
        )
    array = np.frombuffer(data, dtype=np.uint8)
    return array.reshape((count, METRIC_HEIGHT, METRIC_WIDTH)).copy()
