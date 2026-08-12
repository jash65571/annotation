"""Deterministic text-region detection (before OCR).

Uses OpenCV MSER to find high-contrast, text-like connected regions so OCR (and
the eventual reviewer) is not told that every rectangle is text. This is a
candidate detector only — a region is not proof of text.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from ..models.review_intelligence import DetectedTextRegion

#: Geometry gates on candidate boxes (fractions of frame dimensions / raw).
_MIN_AREA = 30
_MAX_ASPECT = 25.0
_MIN_ASPECT = 0.05


def detect_text_regions(
    gray: npt.NDArray[np.uint8], region_id_prefix: str = "TR"
) -> list[DetectedTextRegion]:
    """Return candidate text regions in one gray frame (deterministic)."""
    mser = cv2.MSER_create()  # type: ignore[attr-defined]
    try:
        regions, _boxes = mser.detectRegions(gray)
    except cv2.error:
        return []
    seen: set[tuple[int, int, int, int]] = set()
    out: list[DetectedTextRegion] = []
    counter = 0
    for points in regions:
        x, y, w, h = cv2.boundingRect(points.reshape(-1, 1, 2))
        if w * h < _MIN_AREA:
            continue
        aspect = w / h if h else 0.0
        if aspect < _MIN_ASPECT or aspect > _MAX_ASPECT:
            continue
        key = (int(x), int(y), int(w), int(h))
        if key in seen:
            continue
        seen.add(key)
        patch = gray[y : y + h, x : x + w]
        contrast = float(patch.std()) / 255.0 if patch.size else 0.0
        counter += 1
        out.append(
            DetectedTextRegion(
                region_id=f"{region_id_prefix}-{counter:04d}",
                x=int(x),
                y=int(y),
                width=int(w),
                height=int(h),
                contrast=round(contrast, 4),
            )
        )
    return out
