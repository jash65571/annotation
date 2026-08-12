"""Classify one frame pair's global motion into a safe low-level class.

Camera-vs-subject defense: a low correlation response (the global model does not
explain the change) is UNRESOLVED, never a camera move; a near-zero dominant
shift is STATIC even when a foreground subject is moving, because phase
correlation returns the dominant (background) shift. 2D motion never yields a
semantic label (pan/tilt/dolly) here — only STATIC / HORIZONTAL / VERTICAL /
DIAGONAL / SCALE_* / UNRESOLVED.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..models.review_intelligence import CameraMotionClass
from .global_motion import PairMotion

#: Metric grid is 160x90; thresholds are in grid pixels.
_RESPONSE_MIN = 0.10
_STATIC_MAX_PX = 0.75
_AXIS_DOMINANCE = 2.0
_SCALE_LOG_MIN = 0.012  # ~1.2% per-frame scale change
_SCALE_RESPONSE_MIN = 0.08


@dataclass(frozen=True)
class PairClassification:
    motion_class: CameraMotionClass
    direction: str | None
    strength: float
    response: float


def classify_pair(pm: PairMotion) -> PairClassification:
    magnitude = math.hypot(pm.dx, pm.dy)

    # Translation dominates when the phase-correlation response supports it.
    if magnitude >= _STATIC_MAX_PX and pm.response >= _RESPONSE_MIN:
        return _classify_translation(pm, magnitude)

    # Otherwise consider a uniform scale change (zoom / push / crop).
    log_scale = math.log(pm.scale) if pm.scale > 0 else 0.0
    if abs(log_scale) >= _SCALE_LOG_MIN and pm.scale_response >= _SCALE_RESPONSE_MIN:
        cls = (
            CameraMotionClass.SCALE_INCREASE
            if log_scale > 0
            else CameraMotionClass.SCALE_DECREASE
        )
        return PairClassification(cls, None, round(abs(log_scale), 5), pm.scale_response)

    # Low response with a large frame change => the global model fails: UNRESOLVED.
    if magnitude >= _STATIC_MAX_PX and pm.response < _RESPONSE_MIN:
        return PairClassification(CameraMotionClass.UNRESOLVED, None, magnitude, pm.response)

    return PairClassification(CameraMotionClass.STATIC, None, magnitude, pm.response)


def _classify_translation(pm: PairMotion, magnitude: float) -> PairClassification:
    ax, ay = abs(pm.dx), abs(pm.dy)
    # Camera direction is opposite to the content shift; report screen direction.
    if ax > _AXIS_DOMINANCE * ay:
        direction = "screen-right" if pm.dx < 0 else "screen-left"
        cls = CameraMotionClass.HORIZONTAL_GLOBAL_MOTION
    elif ay > _AXIS_DOMINANCE * ax:
        direction = "up" if pm.dy < 0 else "down"
        cls = CameraMotionClass.VERTICAL_GLOBAL_MOTION
    else:
        horiz = "screen-right" if pm.dx < 0 else "screen-left"
        vert = "up" if pm.dy < 0 else "down"
        direction = f"{horiz}+{vert}"
        cls = CameraMotionClass.DIAGONAL_GLOBAL_MOTION
    return PairClassification(cls, direction, round(magnitude, 4), pm.response)
