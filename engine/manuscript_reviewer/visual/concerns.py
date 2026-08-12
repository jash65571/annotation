"""Deterministic visual-concern candidates.

These are structured evidence candidates (never concern prose). Thresholds are
fixed and documented; a candidate only flags a lead for review. Text-related
concerns (TEXT_TOO_SMALL) and identity concerns (IDENTITY_UNRESOLVED) are raised
by the OCR / tracking slices, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

# Concern codes (subset produced deterministically from single-frame stats).
MOTION_BLUR = "MOTION_BLUR"
LOW_SHARPNESS = "LOW_SHARPNESS"
OVEREXPOSURE = "OVEREXPOSURE"
UNDEREXPOSURE = "UNDEREXPOSURE"
FOCUS_UNCERTAIN = "FOCUS_UNCERTAIN"

# Thresholds on 0..1 normalized stats (gray metric grid).
_LOW_SHARPNESS = 0.004
_MOTION_BLUR = 0.0015
_OVEREXPOSURE_BRIGHTNESS = 0.92
_OVEREXPOSURE_NEAR_WHITE = 0.5
_UNDEREXPOSURE_BRIGHTNESS = 0.06
_UNDEREXPOSURE_NEAR_BLACK = 0.6


@dataclass(frozen=True)
class FrameConcernInputs:
    brightness: float  # mean luma / 255
    sharpness: float  # normalized variance-of-Laplacian
    near_white_fraction: float
    near_black_fraction: float
    motion_magnitude: float  # mean abs frame diff / 255


def detect_frame_concerns(inputs: FrameConcernInputs) -> list[str]:
    """Return deterministic concern codes for one frame."""
    concerns: list[str] = []
    if inputs.sharpness < _LOW_SHARPNESS:
        concerns.append(LOW_SHARPNESS)
        # High motion + low sharpness is characteristic of motion blur.
        if inputs.motion_magnitude > _MOTION_BLUR:
            concerns.append(MOTION_BLUR)
        else:
            concerns.append(FOCUS_UNCERTAIN)
    if (
        inputs.brightness > _OVEREXPOSURE_BRIGHTNESS
        and inputs.near_white_fraction > _OVEREXPOSURE_NEAR_WHITE
    ):
        concerns.append(OVEREXPOSURE)
    if (
        inputs.brightness < _UNDEREXPOSURE_BRIGHTNESS
        and inputs.near_black_fraction > _UNDEREXPOSURE_NEAR_BLACK
    ):
        concerns.append(UNDEREXPOSURE)
    return concerns
