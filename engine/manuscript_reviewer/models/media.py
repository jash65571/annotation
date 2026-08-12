"""Media metadata models: what ffprobe tells us about the file and its streams.

Container-level and stream-level values are kept separately so discrepancies
can be surfaced instead of hidden (RULE: expose discrepancies, never pick the
convenient number).
"""

from __future__ import annotations

from .common import ExactFraction, StrictModel


class VideoStreamInfo(StrictModel):
    stream_index: int
    codec_name: str
    profile: str | None = None
    pix_fmt: str | None = None
    width: int
    height: int
    sample_aspect_ratio: str | None = None
    display_aspect_ratio: str | None = None
    #: r_frame_rate — the nominal/base rate ffprobe guesses from the stream.
    nominal_frame_rate: ExactFraction | None = None
    #: avg_frame_rate — total frames / total duration; differs from nominal for VFR.
    average_frame_rate: ExactFraction | None = None
    #: Exact stream time base (e.g. 1/15360). Required for exact timing.
    time_base: ExactFraction
    #: Stream start_pts in time_base units, if declared.
    start_pts: int | None = None
    #: Stream duration in seconds as declared by the container/stream (may be absent).
    declared_duration_seconds: ExactFraction | None = None
    #: nb_frames as declared in metadata — a claim, not a measurement.
    declared_frame_count: int | None = None


class AudioStreamInfo(StrictModel):
    stream_index: int
    codec_name: str
    sample_rate: int | None = None
    channels: int | None = None
    channel_layout: str | None = None
    time_base: ExactFraction | None = None
    start_pts: int | None = None
    declared_duration_seconds: ExactFraction | None = None


class MediaInfo(StrictModel):
    """Probed facts about one media file. Declared values are hypotheses;
    enumerated/measured values live in the frame ledger and QC report."""

    file_name: str
    file_size_bytes: int
    container_format: str
    container_duration_seconds: ExactFraction | None = None
    container_start_time_seconds: ExactFraction | None = None
    container_bit_rate: int | None = None
    video_streams: list[VideoStreamInfo]
    audio_streams: list[AudioStreamInfo]
    #: Raw ffprobe JSON is preserved verbatim alongside (media.json contains both).
