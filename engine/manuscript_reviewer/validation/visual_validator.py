"""Phase 4 visual validators (P4-OBS-*).

Enforce that the visual observation ledger stays exactly aligned with the Phase 1
frame ledger and that every observation carries exact frame identity — the
invariant behind "exact frame identity controls all timing".
"""

from __future__ import annotations

from ..models.frame import FrameLedger
from ..models.review_intelligence import FrameObservation
from ..models.validation import Severity, ValidatorIssue


def validate_frame_observations(
    observations: list[FrameObservation], ledger: FrameLedger
) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    if len(observations) != ledger.frame_count:
        issues.append(
            ValidatorIssue(
                rule_id="P4-OBS-001",
                severity=Severity.FAIL,
                location="frame_observations",
                message=(
                    f"Observation count {len(observations)} != ledger frame count "
                    f"{ledger.frame_count}; refusing to mislabel visual evidence."
                ),
            )
        )
        return issues
    for i, obs in enumerate(observations):
        if obs.frame_index != i:
            issues.append(
                ValidatorIssue(
                    rule_id="P4-OBS-002",
                    severity=Severity.FAIL,
                    location=f"observation {i}",
                    message=f"Observation out of order: frame_index {obs.frame_index} at row {i}.",
                )
            )
        # Exact frame identity: a frame with a PTS must retain its exact time.
        record = ledger.frames[i]
        if record.pts_time_seconds is not None and obs.source_pts_time_exact is None:
            issues.append(
                ValidatorIssue(
                    rule_id="P4-OBS-003",
                    severity=Severity.FAIL,
                    location=f"observation {i}",
                    message="Observation lost its exact source frame time.",
                )
            )
    return issues
