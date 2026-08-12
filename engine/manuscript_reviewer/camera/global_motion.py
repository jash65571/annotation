"""Deterministic global-motion estimation between two frames.

Translation is estimated with windowed FFT phase correlation
(``cv2.phaseCorrelate``); scale is estimated with a log-polar phase correlation.
Both are RNG-free and deterministic. The ``response`` (0..1) is the correlation
peak strength — low response means the global model does not explain the change
(large foreground occlusion / independent subject motion), so the pair is left
UNRESOLVED rather than called a camera move.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

# Phase correlation carries no RNG, but seed OpenCV globally for reproducibility.
cv2.setRNGSeed(0)


@dataclass(frozen=True)
class PairMotion:
    left_frame: int
    right_frame: int
    dx: float  # image content shift (pixels on the metric grid), x
    dy: float  # image content shift, y
    response: float  # translation peak strength (0..1)
    scale: float  # >1 = content magnified (zoom/push in)
    scale_response: float


def _hanning(shape: tuple[int, int]) -> npt.NDArray[np.float32]:
    # cv2 expects (width, height).
    win = cv2.createHanningWindow((shape[1], shape[0]), cv2.CV_32F)
    return win.astype(np.float32)


def estimate_translation(
    prev: npt.NDArray[np.float32],
    cur: npt.NDArray[np.float32],
    window: npt.NDArray[np.float32],
) -> tuple[float, float, float]:
    (dx, dy), response = cv2.phaseCorrelate(prev * window, cur * window)
    return float(dx), float(dy), float(response)


def estimate_scale(
    prev: npt.NDArray[np.float32],
    cur: npt.NDArray[np.float32],
) -> tuple[float, float]:
    """Estimate uniform scale via log-polar phase correlation.

    Returns ``(scale, response)`` where scale > 1 means the content grew.
    """
    h, w = prev.shape
    center = (w / 2.0, h / 2.0)
    max_radius = float(min(center))
    if max_radius <= 1:
        return 1.0, 0.0
    flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LOG
    lp_prev = cv2.warpPolar(prev, (w, h), center, max_radius, flags).astype(np.float32)
    lp_cur = cv2.warpPolar(cur, (w, h), center, max_radius, flags).astype(np.float32)
    window = _hanning(lp_prev.shape)
    (dxr, _dyr), response = cv2.phaseCorrelate(lp_prev * window, lp_cur * window)
    # Log-polar: a uniform scale s shifts the radial (column) axis by
    # log(s) * w / log(max_radius); invert to recover s.
    scale = float(math.exp(dxr * math.log(max_radius) / w))
    return scale, float(response)


def estimate_pair_motion(
    prev_gray: npt.NDArray[np.uint8],
    cur_gray: npt.NDArray[np.uint8],
    left_frame: int,
    right_frame: int,
    window: npt.NDArray[np.float32] | None = None,
) -> PairMotion:
    prev = prev_gray.astype(np.float32)
    cur = cur_gray.astype(np.float32)
    win = window if window is not None else _hanning(prev.shape)
    dx, dy, response = estimate_translation(prev, cur, win)
    scale, scale_response = estimate_scale(prev, cur)
    return PairMotion(
        left_frame=left_frame,
        right_frame=right_frame,
        dx=dx,
        dy=dy,
        response=response,
        scale=scale,
        scale_response=scale_response,
    )


def hanning_for(shape: tuple[int, int]) -> npt.NDArray[np.float32]:
    return _hanning(shape)
