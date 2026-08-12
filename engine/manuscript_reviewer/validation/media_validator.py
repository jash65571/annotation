"""Media-level validation: stream presence and metadata-consistency warnings.

Warning thresholds (documented in docs/05-phase-1-verification.md):

- Container vs video-stream duration: WARN if they differ by more than one
  nominal frame duration (or 0.1 s when the rate is unknown). Containers
  legitimately include header/start offsets, so small deltas are normal.
- Nominal (r_frame_rate) vs average (avg_frame_rate): WARN on any inequality —
  this is the primary VFR signal.
- Audio vs video stream duration: WARN if they differ by more than 0.5 s.
  Audio priming/padding commonly adds tens of milliseconds.
"""

from __future__ import annotations

from fractions import Fraction

from ..models.media import MediaInfo
from ..models.validation import Severity, ValidatorIssue

#: Fallback duration-delta threshold when no frame rate is known.
DEFAULT_DURATION_TOLERANCE = Fraction(1, 10)
AUDIO_VIDEO_DURATION_TOLERANCE = Fraction(1, 2)


def validate_media(media: MediaInfo) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []

    if not media.video_streams:
        issues.append(
            ValidatorIssue(
                rule_id="P1-MEDIA-001",
                severity=Severity.FAIL,
                location=media.file_name,
                message="No video stream exists in this file.",
                suggested_fix="Supply a file containing at least one video stream.",
            )
        )
        return issues

    video = media.video_streams[0]

    if len(media.video_streams) > 1:
        issues.append(
            ValidatorIssue(
                rule_id="P1-MEDIA-002",
                severity=Severity.WARN,
                location=media.file_name,
                message=(
                    f"File contains {len(media.video_streams)} video streams; "
                    f"only stream index {video.stream_index} is audited."
                ),
            )
        )

    # Container vs stream duration.
    if (
        media.container_duration_seconds is not None
        and video.declared_duration_seconds is not None
    ):
        tolerance = DEFAULT_DURATION_TOLERANCE
        if video.nominal_frame_rate:
            tolerance = 1 / video.nominal_frame_rate
        delta = abs(media.container_duration_seconds - video.declared_duration_seconds)
        if delta > tolerance:
            issues.append(
                ValidatorIssue(
                    rule_id="P1-MEDIA-003",
                    severity=Severity.WARN,
                    location=media.file_name,
                    message=(
                        f"Container duration {float(media.container_duration_seconds):.6f}s "
                        f"differs from video stream duration "
                        f"{float(video.declared_duration_seconds):.6f}s "
                        f"by {float(delta):.6f}s (> {float(tolerance):.6f}s)."
                    ),
                )
            )

    # Nominal vs average frame rate (VFR signal).
    if (
        video.nominal_frame_rate is not None
        and video.average_frame_rate is not None
        and video.nominal_frame_rate != video.average_frame_rate
    ):
        issues.append(
            ValidatorIssue(
                rule_id="P1-MEDIA-004",
                severity=Severity.WARN,
                location=f"{media.file_name} v:{video.stream_index}",
                message=(
                    f"Nominal frame rate {video.nominal_frame_rate} differs from average "
                    f"frame rate {video.average_frame_rate}. Content may be VFR; "
                    "frame_index/fps arithmetic is NOT valid for this file."
                ),
            )
        )

    # Audio vs video duration.
    for audio in media.audio_streams:
        if (
            audio.declared_duration_seconds is not None
            and video.declared_duration_seconds is not None
        ):
            delta = abs(audio.declared_duration_seconds - video.declared_duration_seconds)
            if delta > AUDIO_VIDEO_DURATION_TOLERANCE:
                issues.append(
                    ValidatorIssue(
                        rule_id="P1-MEDIA-005",
                        severity=Severity.WARN,
                        location=f"{media.file_name} a:{audio.stream_index}",
                        message=(
                            f"Audio stream duration {float(audio.declared_duration_seconds):.6f}s "
                            f"differs from video duration "
                            f"{float(video.declared_duration_seconds):.6f}s "
                            f"by {float(delta):.6f}s (> 0.5s)."
                        ),
                    )
                )

    return issues
