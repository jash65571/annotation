"""Human review-decision persistence and application (F).

Human decisions survive reruns and are applied as an evidence-override layer:
``apply_decisions_to_claims`` mutates claim evidence, then the normal
comparison→proposals→queue→qc recompute reflects them (a decision visibly changes
the next run). A decision is applied only when bound to the current video SHA-256
and rules version; a decision from another video/rules is STALE and never applied.

Machine code never fabricates a human decision: ``decided_by`` and
``decided_at_utc`` come from the supplied file and are never stamped fresh.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from ..models.caption import SeedClaim
from ..models.evidence import EvidenceReference, EvidenceType
from ..models.review_intelligence import (
    ClaimReviewStatus,
    DecisionApplication,
    DecisionOutcome,
    DecisionType,
    EvidenceStatus,
    HumanReviewDecision,
)

logger = logging.getLogger(__name__)

_SPEED_VALUES = {"slow_motion", "regular", "accelerated"}
_EVIDENCE_VALUES = {s.value for s in EvidenceStatus}


class DecisionLoadError(RuntimeError):
    """The review-decisions file could not be read, parsed, or was unbound."""


def load_decisions(path: Path) -> list[HumanReviewDecision]:
    """Load human decisions from JSON. Rejects machine-authored or unbound
    decisions (a valid decision MUST carry both binding keys)."""
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
        if not decision.bound_video_sha256 or not decision.bound_rules_version:
            raise DecisionLoadError(
                f"Decision {decision.decision_id} is unbound (missing "
                "bound_video_sha256/bound_rules_version); unbound decisions are rejected."
            )
        decisions.append(decision)
    return decisions


def _validate_value(decision: HumanReviewDecision) -> bool:
    value = decision.value.strip()
    if not value:
        return False
    if decision.decision_type == DecisionType.PLAYBACK_SPEED:
        return value in _SPEED_VALUES
    if decision.decision_type == DecisionType.CLAIM_EVIDENCE:
        return value in _EVIDENCE_VALUES
    return True  # free-text kinds (OCR text, notes, ...) just need to be non-empty


def _resolved_status(decision: HumanReviewDecision) -> EvidenceStatus:
    if decision.decision_type == DecisionType.CLAIM_EVIDENCE:
        return EvidenceStatus(decision.value.strip())
    # A human confirming OCR text / identity / a semantic label supports the claim.
    return EvidenceStatus.SUPPORTED


def apply_decisions_to_claims(
    decisions: list[HumanReviewDecision],
    claims: list[SeedClaim],
    video_sha256: str,
    rules_version: str,
) -> list[DecisionApplication]:
    """Apply valid, bound decisions to claim evidence; return per-decision status."""
    by_claim = {c.claim_id: c for c in claims}
    # CONFLICT: two decisions on the same subject with different values.
    subject_values: dict[str, set[str]] = {}
    for d in decisions:
        subject_values.setdefault(d.subject_id, set()).add(d.value.strip())
    conflicted = {s for s, vals in subject_values.items() if len(vals) > 1}

    applications: list[DecisionApplication] = []
    for decision in decisions:
        outcome, reason = _apply_one(
            decision, by_claim, video_sha256, rules_version, conflicted
        )
        applications.append(
            DecisionApplication(
                decision_id=decision.decision_id,
                outcome=outcome,
                applied=outcome == DecisionOutcome.APPLIED,
                stale=outcome == DecisionOutcome.STALE,
                reason=reason,
            )
        )
    return applications


def _apply_one(
    decision: HumanReviewDecision,
    by_claim: dict[str, SeedClaim],
    video_sha256: str,
    rules_version: str,
    conflicted: set[str],
) -> tuple[DecisionOutcome, str | None]:
    if decision.subject_id in conflicted:
        return DecisionOutcome.CONFLICT, "multiple decisions with different values"
    if decision.bound_video_sha256 != video_sha256:
        logger.warning("Skipping decision %s: bound to a different video", decision.decision_id)
        return DecisionOutcome.STALE, "bound to a different video SHA-256"
    if decision.bound_rules_version != rules_version:
        return DecisionOutcome.STALE, (
            f"bound to rules {decision.bound_rules_version}, current {rules_version}"
        )
    claim = by_claim.get(decision.subject_id)
    if claim is None:
        return DecisionOutcome.INVALID_SUBJECT, f"no claim {decision.subject_id}"
    if not _validate_value(decision):
        return DecisionOutcome.INVALID_VALUE, (
            f"value {decision.value!r} not admissible for {decision.decision_type.value}"
        )
    claim.evidence_status = _resolved_status(decision)
    claim.review_status = ClaimReviewStatus.HUMAN_RESOLVED
    claim.evidence.append(
        EvidenceReference(
            evidence_id=f"EV-HUMAN-{decision.decision_id}",
            evidence_type=EvidenceType.HUMAN_VERIFICATION,
            source=decision.decided_by,
            notes=(
                f"{decision.decision_type.value}={decision.value} "
                f"(decided_at={decision.decided_at_utc})"
            ),
        )
    )
    return DecisionOutcome.APPLIED, None
