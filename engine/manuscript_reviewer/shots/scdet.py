"""FFmpeg ``scdet`` adapter: an independent scene-change signal, evidence only.

Runs one extra decode with ``-vf scdet,metadata=print`` and parses the
per-frame ``lavfi.scd.score`` / ``lavfi.scd.mafd`` values. The filter's own
threshold/decision output is deliberately ignored — scores are attached to our
pair records (score of frame N describes the N-1 → N pair) and treated as one
more candidate source, never as shot truth. Frame identity comes from decode
order under ``-fps_mode passthrough``, matching the Phase 1 ledger exactly.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from ..media.ffmpeg_tools import FFmpegNotFoundError, ToolExecutionError, find_tool, run_tool

logger = logging.getLogger(__name__)

_FRAME_RE = re.compile(r"^frame:(\d+)\b")
_KV_RE = re.compile(r"^lavfi\.scd\.(score|mafd)=([0-9.eE+-]+)")


def scdet_scores(video_path: Path, stream_index: int = 0) -> dict[int, tuple[float, float]]:
    """Return {decoded_frame_index: (score, mafd)} from one scdet pass.

    Returns an empty dict (with a warning) if the filter is unavailable so the
    shot stage degrades gracefully to internal metrics only.
    """
    try:
        ffmpeg = find_tool("ffmpeg")
        result = run_tool(
            ffmpeg,
            [
                "-v", "error",
                "-i", str(video_path),
                "-map", f"0:v:{stream_index}",
                "-fps_mode", "passthrough",
                "-vf", "scdet=s=0,metadata=mode=print:file=-",
                "-f", "null",
                "-",
            ],
            timeout=1800.0,
        )
    except (ToolExecutionError, FFmpegNotFoundError, OSError) as exc:
        # Only expected tool/environment failures degrade gracefully; a
        # programming error must NOT silently become "scdet unavailable".
        logger.warning("scdet pass failed (%s); continuing without scdet evidence", exc)
        return {}

    scores: dict[int, tuple[float, float]] = {}
    current_frame: int | None = None
    current_score: float | None = None
    current_mafd: float | None = None
    for line in result.stdout.splitlines():
        frame_match = _FRAME_RE.match(line)
        if frame_match:
            if current_frame is not None and current_score is not None:
                scores[current_frame] = (current_score, current_mafd or 0.0)
            current_frame = int(frame_match.group(1))
            current_score = None
            current_mafd = None
            continue
        kv = _KV_RE.match(line)
        if kv and current_frame is not None:
            if kv.group(1) == "score":
                current_score = float(kv.group(2))
            else:
                current_mafd = float(kv.group(2))
    if current_frame is not None and current_score is not None:
        scores[current_frame] = (current_score, current_mafd or 0.0)
    return scores
