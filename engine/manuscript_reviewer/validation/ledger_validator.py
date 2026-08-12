"""Frame-ledger integrity validation and frame-count cross-check.

Hard rule: if reliable enumeration cannot be verified, the run is FAILED or
PARTIAL — never a silently incomplete ledger.
"""

from __future__ import annotations

from ..models.frame import FrameLedger
from ..models.media import MediaInfo
from ..models.validation import (
    FrameCountSignal,
    RunStatus,
    Severity,
    ValidatorIssue,
)


def validate_ledger(ledger: FrameLedger) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []

    if ledger.frame_count == 0:
        issues.append(
            ValidatorIssue(
                rule_id="P1-LEDGER-001",
                severity=Severity.FAIL,
                location="frames",
                message="Frame ledger contains zero frames.",
            )
        )
        return issues

    # Sequential, duplicate-free indexes starting at 0.
    expected = list(range(ledger.frame_count))
    actual = [frame.frame_index for frame in ledger.frames]
    if actual != expected:
        duplicates = len(actual) != len(set(actual))
        issues.append(
            ValidatorIssue(
                rule_id="P1-LEDGER-002" if not duplicates else "P1-LEDGER-003",
                severity=Severity.FAIL,
                location="frames",
                message=(
                    "Frame indexes contain duplicates."
                    if duplicates
                    else "Frame indexes are not sequential from 0."
                ),
            )
        )

    # PTS presence and monotonicity.
    missing_pts = [f.frame_index for f in ledger.frames if f.pts is None]
    if missing_pts:
        issues.append(
            ValidatorIssue(
                rule_id="P1-LEDGER-004",
                severity=Severity.WARN,
                location=f"frames {missing_pts[:10]}{'...' if len(missing_pts) > 10 else ''}",
                message=(
                    f"{len(missing_pts)} frame(s) have no usable PTS; exact timing for "
                    "those frames is unavailable. Run is at best PARTIAL."
                ),
            )
        )

    previous_pts: int | None = None
    for frame in ledger.frames:
        if frame.pts is None:
            continue
        if previous_pts is not None and frame.pts <= previous_pts:
            issues.append(
                ValidatorIssue(
                    rule_id="P1-LEDGER-005",
                    severity=Severity.FAIL,
                    location=f"frame {frame.frame_index}",
                    message=(
                        f"Presentation timestamps are not strictly increasing: "
                        f"pts {frame.pts} follows {previous_pts}."
                    ),
                )
            )
            break
        previous_pts = frame.pts

    return issues


def cross_check_frame_count(
    ledger: FrameLedger,
    media: MediaInfo,
    decoded_count: int | None,
) -> tuple[list[FrameCountSignal], list[ValidatorIssue]]:
    """Compare independent frame-count signals; disagreements are always visible.

    Signals:
    1. ``show_frames`` enumeration (authoritative — it produced the ledger).
    2. ``count_frames`` full decode (independent ffprobe pass) — mismatch FAILS.
    3. Declared ``nb_frames`` metadata (a claim, not a measurement) — mismatch WARNS.
    """
    issues: list[ValidatorIssue] = []
    declared = (
        media.video_streams[0].declared_frame_count if media.video_streams else None
    )
    signals = [
        FrameCountSignal(
            method="ffprobe -show_frames enumeration",
            count=ledger.frame_count,
            authoritative=True,
        ),
        FrameCountSignal(
            method="ffprobe -count_frames decode",
            count=decoded_count,
            authoritative=False,
            notes=None if decoded_count is not None else "not reported for this container",
        ),
        FrameCountSignal(
            method="container-declared nb_frames",
            count=declared,
            authoritative=False,
            notes="metadata claim, not a measurement",
        ),
    ]

    if decoded_count is None:
        issues.append(
            ValidatorIssue(
                rule_id="P1-COUNT-001",
                severity=Severity.WARN,
                location="frame count cross-check",
                message=(
                    "Independent -count_frames verification unavailable for this "
                    "container; frame accounting rests on a single signal."
                ),
            )
        )
    elif decoded_count != ledger.frame_count:
        issues.append(
            ValidatorIssue(
                rule_id="P1-COUNT-002",
                severity=Severity.FAIL,
                location="frame count cross-check",
                message=(
                    f"Enumerated frame count ({ledger.frame_count}) does not match the "
                    f"independent decode count ({decoded_count})."
                ),
            )
        )

    if declared is not None and declared != ledger.frame_count:
        issues.append(
            ValidatorIssue(
                rule_id="P1-COUNT-003",
                severity=Severity.WARN,
                location="frame count cross-check",
                message=(
                    f"Container declares nb_frames={declared} but enumeration found "
                    f"{ledger.frame_count}. Enumerated count is authoritative."
                ),
            )
        )

    return signals, issues


def compute_run_status(issues: list[ValidatorIssue], ledger: FrameLedger | None) -> RunStatus:
    """FAILED on any FAIL; PARTIAL when enumeration exists but exact timing is
    incomplete (missing PTS); PASS otherwise."""
    if any(issue.severity == Severity.FAIL for issue in issues):
        return RunStatus.FAILED
    if ledger is None:
        return RunStatus.FAILED
    if any(frame.pts is None for frame in ledger.frames):
        return RunStatus.PARTIAL
    return RunStatus.PASS
