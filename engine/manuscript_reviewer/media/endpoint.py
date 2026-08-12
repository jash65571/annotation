"""Canonical annotation-endpoint computation.

An incorrect annotation endpoint is a permanent failure class (rules v1.1.0,
timing_integrity). The annotation interval's end is the media presentation
endpoint — NEVER the final frame's start PTS. This is the single helper every
consumer must use.

Evidence considered, strongest first:

1. final frame PTS + final frame duration (direct media evidence, absolute
   source time)
2. video stream declared duration (a DURATION)
3. container declared duration (a DURATION)
4. source segment endpoint encoded in the filename
   (``<id>_<start>_<end>.mp4`` → expected length = end - start, a DURATION)

Duration signals are NOT absolute raw-PTS endpoints: with a non-zero
annotation timeline origin they are converted as
``source_endpoint_candidate = origin + duration`` before comparison, and the
annotation endpoint is ``source_endpoint - origin``. An absolute raw PTS is
never compared against an unshifted duration.

Material conflicts between available signals are surfaced (``conflict=True``)
so the caller can mark the run REVIEW_REQUIRED/PARTIAL instead of guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction

from ..models.frame import FrameLedger
from ..models.media import MediaInfo
from .clock import AnnotationClock

#: Signals disagreeing by more than this are a material conflict.
MATERIAL_CONFLICT_TOLERANCE = Fraction(1, 10)

_SEGMENT_NAME_RE = re.compile(
    r"^[0-9a-f]{6,}_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)$", re.IGNORECASE
)


@dataclass
class EndpointResult:
    """Canonical endpoint plus the evidence that produced it.

    ``endpoint`` is in the SOURCE clock (absolute raw media time);
    ``annotation_endpoint`` is the same instant on the annotation clock.
    """

    endpoint: Fraction | None
    annotation_endpoint: Fraction | None
    method: str
    signals: dict[str, Fraction] = field(default_factory=dict)
    conflict: bool = False
    notes: list[str] = field(default_factory=list)


def _filename_expected_length(stem: str) -> Fraction | None:
    match = _SEGMENT_NAME_RE.match(stem)
    if not match:
        return None
    start = Fraction(match.group(1))
    end = Fraction(match.group(2))
    if end <= start:
        return None
    return end - start


def compute_annotation_endpoint(
    media: MediaInfo,
    ledger: FrameLedger,
    source_stem: str,
    clock: AnnotationClock | None = None,
) -> EndpointResult:
    """Compute the canonical annotation endpoint from media evidence.

    All signals are normalized to the SOURCE clock before comparison:
    duration-type signals become ``origin + duration``.
    """
    if clock is None:
        clock = AnnotationClock.from_ledger(ledger)
    origin = clock.origin
    signals: dict[str, Fraction] = {}
    notes: list[str] = []

    last = ledger.frames[-1] if ledger.frames else None
    if last is not None and last.pts_time_seconds is not None:
        if last.duration_seconds is not None:
            signals["final_frame_presentation_end"] = (
                last.pts_time_seconds + last.duration_seconds
            )
        else:
            notes.append("Final frame reports no duration; presentation end unavailable.")

    # Duration signals → source-clock endpoints (origin + duration).
    if media.video_streams:
        video = media.video_streams[0]
        if video.declared_duration_seconds is not None:
            signals["video_stream_duration"] = origin + video.declared_duration_seconds
    if media.container_duration_seconds is not None:
        signals["container_duration"] = origin + media.container_duration_seconds

    filename_length = _filename_expected_length(source_stem)
    if filename_length is not None:
        signals["filename_segment_length"] = origin + filename_length

    if not signals:
        if last is not None and last.pts_time_seconds is not None:
            notes.append(
                "No duration evidence at all; degraded to final frame START PTS. "
                "Endpoint is NOT verified."
            )
            return EndpointResult(
                endpoint=last.pts_time_seconds,
                annotation_endpoint=clock.to_annotation(last.pts_time_seconds),
                method="degraded_final_frame_start",
                signals=signals,
                conflict=True,
                notes=notes,
            )
        return EndpointResult(
            endpoint=None,
            annotation_endpoint=None,
            method="unavailable",
            signals=signals,
            conflict=True,
            notes=notes,
        )

    # Preference order: direct media evidence beats declared metadata beats
    # filename claims.
    for method in (
        "final_frame_presentation_end",
        "video_stream_duration",
        "container_duration",
        "filename_segment_length",
    ):
        if method in signals:
            endpoint = signals[method]
            break

    conflict = False
    for name, value in signals.items():
        if abs(value - endpoint) > MATERIAL_CONFLICT_TOLERANCE:
            conflict = True
            notes.append(
                f"Endpoint signal {name}={float(value):.6f}s disagrees with chosen "
                f"{method}={float(endpoint):.6f}s by more than "
                f"{float(MATERIAL_CONFLICT_TOLERANCE)}s."
            )

    # Sanity: the endpoint can never precede the final frame's start.
    if (
        last is not None
        and last.pts_time_seconds is not None
        and endpoint < last.pts_time_seconds
    ):
        conflict = True
        notes.append(
            f"Chosen endpoint {float(endpoint):.6f}s precedes the final frame start "
            f"{float(last.pts_time_seconds):.6f}s."
        )
        endpoint = last.pts_time_seconds

    return EndpointResult(
        endpoint=endpoint,
        annotation_endpoint=clock.to_annotation(endpoint),
        method=method,
        signals=signals,
        conflict=conflict,
        notes=notes,
    )
