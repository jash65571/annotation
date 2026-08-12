"""Canonical annotation clock: the ONE place source time maps to annotation time.

The Manuscript annotation timeline starts at 0.0 at the first presented video
frame, even when the container/stream PTS starts non-zero:

    annotation_time = source_media_time - annotation_timeline_origin

Both clocks are preserved everywhere — raw source PTS values are never
overwritten; annotation-relative values are what the Manuscript display layer
receives. No other module may subtract the origin itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ..models.frame import FrameLedger


@dataclass(frozen=True)
class AnnotationClock:
    """Maps between raw source media time and annotation-relative time."""

    #: Source PTS time of the first presented video frame (exact rational).
    origin: Fraction

    @classmethod
    def from_ledger(cls, ledger: FrameLedger) -> AnnotationClock:
        """Origin = first enumerated video frame's source PTS time.

        Falls back to 0 when the first frame has no usable PTS (the run is
        already PARTIAL in that case).
        """
        if ledger.frames and ledger.frames[0].pts_time_seconds is not None:
            return cls(origin=ledger.frames[0].pts_time_seconds)
        return cls(origin=Fraction(0))

    def to_annotation(self, source_time: Fraction) -> Fraction:
        return source_time - self.origin

    def to_source(self, annotation_time: Fraction) -> Fraction:
        return annotation_time + self.origin
