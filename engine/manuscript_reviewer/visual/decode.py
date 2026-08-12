"""One shared, bounded frame-access layer for all Phase 4 visual consumers.

The mandate (prompt §FRAME CACHE): tracking, OCR, camera, action boundaries and
crop extraction must reuse decoded frames — no five independent full decodes.

This cache decodes the deterministic gray metric grid exactly once (reusing the
Phase 2 count==ledger guarantee so image N is always ledger frame N) and serves
it to every consumer. Full-resolution colour frames are decoded on demand and
kept in a small bounded LRU so evidence crops never trigger a whole re-decode.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ..media.ffmpeg_tools import find_tool, run_tool_binary
from ..models.frame import FrameLedger
from ..shots.decode import GrayFrames, decode_metric_frames

logger = logging.getLogger(__name__)

ColorFrame = npt.NDArray[np.uint8]  # (H, W, 3) rgb24


class FrameCache:
    """Bounded frame access keyed by ``frame_index``.

    ``gray_frames`` is decoded once and reused. ``color_frame`` decodes a single
    full-resolution frame on demand (bounded LRU). Every access is by exact
    ledger index — never by approximate time.
    """

    def __init__(
        self,
        video_path: Path,
        ledger: FrameLedger,
        stream_index: int = 0,
        color_cache_size: int = 32,
    ) -> None:
        self._video_path = video_path
        self._ledger = ledger
        self._stream_index = stream_index
        self._gray: GrayFrames | None = None
        self._color: OrderedDict[int, ColorFrame] = OrderedDict()
        self._color_cache_size = color_cache_size
        self.gray_decode_count = 0
        self.color_hits = 0
        self.color_misses = 0

    @property
    def frame_count(self) -> int:
        return self._ledger.frame_count

    def gray_frames(self) -> GrayFrames:
        """The full (N, H, W) uint8 gray metric grid, decoded once."""
        if self._gray is None:
            self._gray = decode_metric_frames(
                self._video_path, self._ledger.frame_count, self._stream_index
            )
            self.gray_decode_count += 1
        return self._gray

    def gray_frame(self, frame_index: int) -> npt.NDArray[np.uint8]:
        """One gray metric frame (H, W). Raises IndexError if out of range."""
        frames = self.gray_frames()
        if frame_index < 0 or frame_index >= frames.shape[0]:
            raise IndexError(f"frame_index {frame_index} out of range 0..{frames.shape[0] - 1}")
        row: npt.NDArray[np.uint8] = frames[frame_index]
        return row

    def color_frame(self, frame_index: int) -> ColorFrame:
        """One full-resolution RGB frame, decoded on demand (bounded LRU).

        Uses an exact-index select so the decoded pixels correspond to ledger
        frame ``frame_index`` (not an approximate seek).
        """
        if frame_index < 0 or frame_index >= self._ledger.frame_count:
            raise IndexError(
                f"frame_index {frame_index} out of range 0..{self._ledger.frame_count - 1}"
            )
        cached = self._color.get(frame_index)
        if cached is not None:
            self._color.move_to_end(frame_index)
            self.color_hits += 1
            return cached
        self.color_misses += 1
        frame = self._decode_color(frame_index)
        self._color[frame_index] = frame
        self._color.move_to_end(frame_index)
        while len(self._color) > self._color_cache_size:
            self._color.popitem(last=False)
        return frame

    def _decode_color(self, frame_index: int) -> ColorFrame:
        ffmpeg = find_tool("ffmpeg")
        result = run_tool_binary(
            ffmpeg,
            [
                "-v", "error",
                "-i", str(self._video_path),
                "-map", f"0:v:{self._stream_index}",
                "-vf", f"select=eq(n\\,{frame_index})",
                "-vframes", "1",
                "-fps_mode", "passthrough",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-",
            ],
            timeout=120.0,
        )
        width = self._ledger.frames[frame_index].width
        height = self._ledger.frames[frame_index].height
        data = result.stdout
        if width is None or height is None or len(data) != width * height * 3:
            raise ValueError(
                f"Colour decode for frame {frame_index} produced {len(data)} bytes; "
                f"expected {width}x{height}x3."
            )
        array = np.frombuffer(data, dtype=np.uint8)
        return array.reshape((height, width, 3)).copy()

    def cache_stats(self) -> dict[str, int]:
        return {
            "gray_decode_count": self.gray_decode_count,
            "color_hits": self.color_hits,
            "color_misses": self.color_misses,
            "color_cached": len(self._color),
        }
