"""Frame ledger records — one per encoded/display frame, exact timing preserved."""

from __future__ import annotations

from .common import ExactFraction, StrictModel


class FrameRecord(StrictModel):
    """One enumerated frame.

    ``pts`` plus ``time_base`` (stored once, on the ledger/stream) reconstructs
    the exact presentation time; ``pts_time_seconds`` is the same value as an
    exact rational for convenience, never a float.
    """

    frame_index: int
    #: Integer presentation timestamp in stream time_base units. None only when
    #: the codec/container genuinely provides no PTS (run is then PARTIAL/FAILED).
    pts: int | None
    #: Exact pts * time_base, in seconds.
    pts_time_seconds: ExactFraction | None
    #: Decode timestamp, when meaningfully different from PTS.
    dts: int | None = None
    #: Frame duration in time_base units, if reported.
    duration: int | None = None
    #: Exact duration in seconds, if reported.
    duration_seconds: ExactFraction | None = None
    key_frame: bool
    pict_type: str | None = None
    width: int | None = None
    height: int | None = None


class FrameLedger(StrictModel):
    """Every enumerated frame of one video stream, in presentation order."""

    stream_index: int
    time_base: ExactFraction
    frames: list[FrameRecord]

    @property
    def frame_count(self) -> int:
        return len(self.frames)
