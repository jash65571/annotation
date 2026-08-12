"""AudioTimeline construction + PCM sample-count cross-check.

Canonical anchor: annotation time of evidence sample N =
``annotation_audio_offset + N / evidence_sample_rate`` — exact Fraction math,
never accumulated floats.
"""

from __future__ import annotations

import logging
from fractions import Fraction

from ..media.clock import AnnotationClock
from ..models.audio import AudioFrameRecord, AudioTimeline
from ..models.media import AudioStreamInfo
from ..models.validation import Severity, ValidatorIssue
from .decode import DecodedWav

logger = logging.getLogger(__name__)

#: Codec priming/padding tolerance for sample-count cross-checks (seconds).
#: AAC priming is typically ~1024-2112 samples (< 50 ms at 44.1/48 kHz).
SAMPLE_COUNT_TOLERANCE_SECONDS = Fraction(1, 20)


def build_audio_timeline(
    stream: AudioStreamInfo,
    audio_frames: list[AudioFrameRecord],
    source_wav: DecodedWav,
    clock: AnnotationClock,
    initial_padding: int | None,
    asr_wav: DecodedWav | None,
) -> AudioTimeline:
    first_pts = audio_frames[0].pts if audio_frames else None
    first_time = audio_frames[0].pts_time_source if audio_frames else None
    offset = (
        clock.to_annotation(first_time) if first_time is not None else Fraction(0)
    )
    time_base = stream.time_base if stream.time_base is not None else Fraction(1, 1)
    return AudioTimeline(
        source_stream_index=stream.stream_index,
        source_time_base=time_base,
        source_start_pts=stream.start_pts,
        source_start_time=first_time,
        first_decoded_audio_pts=first_pts,
        annotation_timeline_origin=clock.origin,
        annotation_audio_offset=offset,
        initial_padding_samples=initial_padding,
        source_sample_rate=stream.sample_rate or source_wav.sample_rate,
        source_channels=stream.channels or source_wav.channels,
        evidence_sample_rate=source_wav.sample_rate,
        evidence_channels=source_wav.channels,
        evidence_sample_count=source_wav.sample_count,
        evidence_duration_seconds=Fraction(source_wav.sample_count, source_wav.sample_rate),
        asr_sample_rate=asr_wav.sample_rate if asr_wav else None,
        asr_sample_count=asr_wav.sample_count if asr_wav else None,
        asr_duration_seconds=(
            Fraction(asr_wav.sample_count, asr_wav.sample_rate) if asr_wav else None
        ),
    )


def sample_to_annotation(timeline: AudioTimeline, sample_index: int) -> Fraction:
    """Exact annotation time at which evidence sample N plays."""
    return timeline.annotation_audio_offset + Fraction(
        sample_index, timeline.evidence_sample_rate
    )


def annotation_to_sample(timeline: AudioTimeline, annotation_time: Fraction) -> int:
    """Nearest evidence sample index for an annotation time (floor)."""
    relative = annotation_time - timeline.annotation_audio_offset
    value = relative * timeline.evidence_sample_rate
    return max(0, min(timeline.evidence_sample_count, int(value)))


def cross_check_sample_count(
    timeline: AudioTimeline,
    audio_frames: list[AudioFrameRecord],
    stream: AudioStreamInfo,
) -> list[ValidatorIssue]:
    """Compare decoded PCM sample count against independent signals.

    Signals: sum of enumerated frame nb_samples; declared stream duration x
    sample rate. Disagreements beyond the priming tolerance are surfaced --
    never silently resolved.
    """
    issues: list[ValidatorIssue] = []
    decoded = timeline.evidence_sample_count
    rate = timeline.evidence_sample_rate
    tolerance = int(SAMPLE_COUNT_TOLERANCE_SECONDS * rate)

    frame_sum = sum(f.nb_samples or 0 for f in audio_frames)
    if frame_sum:
        delta = abs(frame_sum - decoded)
        if delta > tolerance:
            issues.append(
                ValidatorIssue(
                    rule_id="P3-AUDIO-007",
                    severity=Severity.WARN,
                    location="audio timeline",
                    message=(
                        f"Decoded PCM sample count ({decoded}) differs from the sum of "
                        f"enumerated audio-frame nb_samples ({frame_sum}) by {delta} "
                        f"samples (> priming tolerance {tolerance})."
                    ),
                )
            )

    if stream.declared_duration_seconds is not None:
        expected = int(stream.declared_duration_seconds * rate)
        delta = abs(expected - decoded)
        if delta > tolerance:
            issues.append(
                ValidatorIssue(
                    rule_id="P3-AUDIO-008",
                    severity=Severity.WARN,
                    location="audio timeline",
                    message=(
                        f"Decoded PCM sample count ({decoded}) differs from declared "
                        f"stream duration x rate ({expected}) by {delta} samples."
                    ),
                )
            )
    return issues
