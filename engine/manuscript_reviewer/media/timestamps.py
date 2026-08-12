"""Central exact-timing module.

This is the ONLY place in the codebase where media time is converted between
representations. All internal time is exact:

- A frame's presentation time is ``pts * time_base`` where ``pts`` is an
  integer from the stream and ``time_base`` is an exact rational
  (e.g. 1/15360). We represent the result as :class:`fractions.Fraction`.
- Float arithmetic is never used to accumulate or derive authoritative time.
- The Manuscript II *display* value (0.1-second precision) is a lossy
  presentation-layer projection produced by :func:`to_manuscript_display`.
  It must never flow back into internal timing.

Frame-index/nominal-FPS arithmetic is only ever a cross-check for CFR media
(:func:`cfr_expected_time`); it is never the source of truth, and is
meaningless for VFR media.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

#: Manuscript II final display precision (seconds).
MANUSCRIPT_DISPLAY_STEP = Fraction(1, 10)


def parse_rational(text: str) -> Fraction:
    """Parse an ffprobe rational string like ``"30000/1001"`` or ``"1/15360"``.

    ffprobe uses ``"0/0"`` for undefined rates; that raises ``ZeroDivisionError``
    here on purpose — callers must treat undefined rates explicitly.
    """
    text = text.strip()
    if "/" in text:
        num_s, den_s = text.split("/", 1)
        return Fraction(int(num_s), int(den_s))
    return Fraction(text)


def pts_to_seconds(pts: int, time_base: Fraction) -> Fraction:
    """Exact presentation time in seconds for an integer PTS in a given time base."""
    return pts * time_base


def seconds_to_decimal(seconds: Fraction, places: int = 6) -> Decimal:
    """Render an exact time as a fixed-point Decimal (for filenames / JSON display).

    This is display formatting, not authoritative time; ``places=6``
    (microseconds) matches ffprobe's own ``pts_time`` rendering.
    """
    quantum = Decimal(1).scaleb(-places)
    return (Decimal(seconds.numerator) / Decimal(seconds.denominator)).quantize(
        quantum, rounding=ROUND_HALF_UP
    )


def to_manuscript_display(seconds: Fraction) -> Decimal:
    """Project an exact event time onto the Manuscript II 0.1 s display grid.

    Rounds half-up to one decimal place (e.g. 11.766667s → 11.8). This value
    is for the final caption presentation layer ONLY — internal records keep
    the exact Fraction/PTS.
    """
    return (Decimal(seconds.numerator) / Decimal(seconds.denominator)).quantize(
        Decimal("0.1"), rounding=ROUND_HALF_UP
    )


def format_manuscript_display(seconds: Fraction) -> str:
    """Format an exact time as the Manuscript display string, e.g. ``"11.8s"``."""
    return f"{to_manuscript_display(seconds)}s"


def cfr_expected_time(frame_index: int, frame_rate: Fraction) -> Fraction:
    """Expected presentation time of frame N under a constant frame rate.

    CROSS-CHECK ONLY. For CFR media this should match the real PTS-derived
    time; a disagreement suggests VFR content, timestamp gaps, or a wrong
    nominal rate. Never use this as the authoritative timestamp.
    """
    return frame_index / frame_rate
