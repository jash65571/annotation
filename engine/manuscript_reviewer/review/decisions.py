"""Human review-decision persistence.

Human decisions must survive reruns. A decision is applied only when it is bound
to the current video SHA-256 and rules version; a decision from another video (or
after the media/rules changed) is detected as stale and never applied.

Machine code never fabricates a human decision: ``decided_by`` comes from the
supplied decision file. Any decision missing an explicit human ``decided_by`` is
rejected (validator P4-REVIEW-001 territory).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from ..models.review_intelligence import DecisionApplication, HumanReviewDecision

logger = logging.getLogger(__name__)


class DecisionLoadError(RuntimeError):
    """The review-decisions file could not be read or parsed."""


def load_decisions(path: Path) -> list[HumanReviewDecision]:
    """Load human decisions from a JSON file.

    Expected shape: ``{"decisions": [ {decision_id, subject_id, decision_type,
    value, decided_by, ...}, ... ]}``. A decision with a machine/empty
    ``decided_by`` is rejected — machine code cannot supply human decisions.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DecisionLoadError(f"Cannot read decisions file {path}: {exc}") from exc
    entries = raw.get("decisions", []) if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise DecisionLoadError("Decisions file must contain a 'decisions' list.")
    decisions: list[HumanReviewDecision] = []
    for entry in entries:
        try:
            decision = HumanReviewDecision.model_validate(entry)
        except ValidationError as exc:
            raise DecisionLoadError(f"Invalid decision entry: {exc}") from exc
        decided_by = (decision.decided_by or "").strip().lower()
        if decided_by in ("", "machine", "ai", "system"):
            raise DecisionLoadError(
                f"Decision {decision.decision_id} has non-human decided_by="
                f"{decision.decided_by!r}; machine decisions are not accepted."
            )
        decisions.append(decision)
    return decisions


def apply_decisions(
    decisions: list[HumanReviewDecision],
    video_sha256: str,
    rules_version: str,
) -> list[DecisionApplication]:
    """Return per-decision application status, rejecting stale/mismatched ones."""
    applications: list[DecisionApplication] = []
    for decision in decisions:
        if decision.bound_video_sha256 is not None and decision.bound_video_sha256 != video_sha256:
            applications.append(
                DecisionApplication(
                    decision_id=decision.decision_id,
                    applied=False,
                    stale=True,
                    stale_reason="bound to a different video SHA-256",
                )
            )
            logger.warning(
                "Skipping decision %s: bound to a different video", decision.decision_id
            )
            continue
        if (
            decision.bound_rules_version is not None
            and decision.bound_rules_version != rules_version
        ):
            applications.append(
                DecisionApplication(
                    decision_id=decision.decision_id,
                    applied=False,
                    stale=True,
                    stale_reason=(
                        f"bound to rules {decision.bound_rules_version}, current {rules_version}"
                    ),
                )
            )
            continue
        applications.append(
            DecisionApplication(decision_id=decision.decision_id, applied=True)
        )
    return applications
