"""Human review-decision persistence and application (F, item 17).

Human decisions survive reruns and are applied as an evidence-override layer:
``apply_decisions`` routes each decision to the ONE typed registry its kind is
allowed to touch and performs a fixed, whitelisted mutation on the resolved
target. The normal comparison->proposals->queue->qc recompute then reflects them
(a decision visibly changes the next run). A decision is applied only when bound
to the current video SHA-256 and rules version; a decision from another
video/rules is STALE and never applied.

Two safety properties hold structurally, not by convention:

* A ``HumanReviewDecision`` is a StrictModel carrying only
  ``(decision_type, subject_id, value)`` — JSON cannot express "set field X to Y",
  so no decision can mutate an arbitrary model field.
* Each ``DecisionType`` maps to exactly one registry and one applier. A subject
  that is not present in that kind's registry is ``INVALID_SUBJECT`` — an
  OCR decision can never reach a camera candidate, etc.

Machine code never fabricates a human decision: ``decided_by`` and
``decided_at_utc`` come from the supplied file and are never stamped fresh.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from ..models.caption import SeedClaim
from ..models.evidence import EvidenceReference, EvidenceType
from ..models.review_intelligence import (
    ActionCandidate,
    CameraMotionCandidate,
    CameraMotionClass,
    ClaimReviewStatus,
    DecisionApplication,
    DecisionOutcome,
    DecisionType,
    EntityTrack,
    EvidenceStatus,
    HumanReviewDecision,
    PlaybackSpeedEvidence,
    ReviewProposal,
    ReviewProposalOutcome,
    SpeedConclusion,
)

logger = logging.getLogger(__name__)

_SPEED_VALUES = {"slow_motion": SpeedConclusion.SLOW_MOTION_CANDIDATE,
                 "regular": SpeedConclusion.REGULAR_SUPPORTED,
                 "accelerated": SpeedConclusion.ACCELERATED_CANDIDATE}
_EVIDENCE_VALUES = {s.value for s in EvidenceStatus}
_CAMERA_VALUES = {c.value for c in CameraMotionClass}
_PROPOSAL_VALUES = {o.value for o in ReviewProposalOutcome}


class DecisionLoadError(RuntimeError):
    """The review-decisions file could not be read, parsed, or was unbound."""


@dataclass
class DecisionTargets:
    """Typed registries a decision kind is allowed to touch. Each is keyed by the
    target's own id; a decision whose subject is absent from its kind's registry
    is INVALID_SUBJECT (a decision can never mutate a target of the wrong type)."""

    claims: dict[str, SeedClaim] = field(default_factory=dict)
    entity_tracks: dict[str, EntityTrack] = field(default_factory=dict)
    camera_candidates: dict[str, CameraMotionCandidate] = field(default_factory=dict)
    action_candidates: dict[str, ActionCandidate] = field(default_factory=dict)
    speed_evidence: dict[str, PlaybackSpeedEvidence] = field(default_factory=dict)
    proposals: dict[str, ReviewProposal] = field(default_factory=dict)


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


# --------------------------------------------------------------------------
# Typed dispatch: (registry selector, value validator, whitelisted applier)
# --------------------------------------------------------------------------


def _human_ref(decision: HumanReviewDecision) -> EvidenceReference:
    """A graded HUMAN_VERIFICATION evidence ref (is_factual by type)."""
    return EvidenceReference(
        evidence_id=f"EV-HUMAN-{decision.decision_id}",
        evidence_type=EvidenceType.HUMAN_VERIFICATION,
        source=decision.decided_by,
        notes=(
            f"{decision.decision_type.value}={decision.value} "
            f"(decided_at={decision.decided_at_utc})"
        ),
    )


def _apply_claim(target: object, decision: HumanReviewDecision) -> None:
    assert isinstance(target, SeedClaim)
    if decision.decision_type == DecisionType.CLAIM_EVIDENCE:
        target.evidence_status = EvidenceStatus(decision.value.strip())
    else:  # OCR_TEXT / OCR_TIMING: a human confirming the text/timing supports it.
        target.evidence_status = EvidenceStatus.SUPPORTED
    target.review_status = ClaimReviewStatus.HUMAN_RESOLVED
    target.evidence.append(_human_ref(decision))


def _apply_identity(target: object, decision: HumanReviewDecision) -> None:
    assert isinstance(target, EntityTrack)
    # A human confirming identity clears the reacquired flag (the item-8 HIGH
    # machine item for this track disappears on recompute).
    target.reacquired = False
    target.notes.append(f"identity confirmed: {decision.value} ({decision.decided_by})")


def _apply_camera(target: object, decision: HumanReviewDecision) -> None:
    assert isinstance(target, CameraMotionCandidate)
    target.motion_class = CameraMotionClass(decision.value.strip())
    target.review_required = False
    target.notes.append(f"human classification: {decision.value} ({decision.decided_by})")


def _apply_action_semantics(target: object, decision: HumanReviewDecision) -> None:
    assert isinstance(target, ActionCandidate)
    target.semantic_label = decision.value.strip()
    target.review_required = False


def _apply_action_boundary(target: object, decision: HumanReviewDecision) -> None:
    assert isinstance(target, ActionCandidate)
    lo, hi = (int(p) for p in decision.value.split("-", 1))
    target.start_frame, target.end_frame = lo, hi
    target.review_required = False


def _apply_speed(target: object, decision: HumanReviewDecision) -> None:
    assert isinstance(target, PlaybackSpeedEvidence)
    target.conclusion = _SPEED_VALUES[decision.value.strip()]
    target.review_required = False


def _apply_proposal(target: object, decision: HumanReviewDecision) -> None:
    assert isinstance(target, ReviewProposal)
    target.outcome = ReviewProposalOutcome(decision.value.strip())


def _valid_boundary(value: str) -> bool:
    parts = value.split("-", 1)
    return len(parts) == 2 and all(p.strip().isdigit() for p in parts)


_Applier = Callable[[object, HumanReviewDecision], None]
_Validator = Callable[[str], bool]
_Selector = Callable[[DecisionTargets], Mapping[str, object]]

#: DecisionType -> (registry selector, value validator, whitelisted applier).
_DISPATCH: dict[DecisionType, tuple[_Selector, _Validator, _Applier]] = {
    DecisionType.CLAIM_EVIDENCE: (
        lambda t: t.claims, lambda v: v.strip() in _EVIDENCE_VALUES, _apply_claim),
    DecisionType.OCR_TEXT: (
        lambda t: t.claims, lambda v: bool(v.strip()), _apply_claim),
    DecisionType.OCR_TIMING: (
        lambda t: t.claims, lambda v: bool(v.strip()), _apply_claim),
    DecisionType.IDENTITY_MAPPING: (
        lambda t: t.entity_tracks, lambda v: bool(v.strip()), _apply_identity),
    DecisionType.CAMERA_CLASSIFICATION: (
        lambda t: t.camera_candidates, lambda v: v.strip() in _CAMERA_VALUES, _apply_camera),
    DecisionType.ACTION_SEMANTICS: (
        lambda t: t.action_candidates, lambda v: bool(v.strip()), _apply_action_semantics),
    DecisionType.ACTION_BOUNDARY: (
        lambda t: t.action_candidates, _valid_boundary, _apply_action_boundary),
    DecisionType.PLAYBACK_SPEED: (
        lambda t: t.speed_evidence, lambda v: v.strip() in _SPEED_VALUES, _apply_speed),
    DecisionType.REVIEW_PROPOSAL_OUTCOME: (
        lambda t: t.proposals, lambda v: v.strip() in _PROPOSAL_VALUES, _apply_proposal),
}


def apply_decisions(
    decisions: list[HumanReviewDecision],
    targets: DecisionTargets,
    video_sha256: str,
    rules_version: str,
) -> list[DecisionApplication]:
    """Route each decision to the ONE registry its kind is allowed to touch and
    apply a fixed whitelisted mutation. Returns a per-decision application record."""
    # CONFLICT: two decisions on the same subject with different values.
    subject_values: dict[str, set[str]] = {}
    for d in decisions:
        subject_values.setdefault(d.subject_id, set()).add(d.value.strip())
    conflicted = {s for s, vals in subject_values.items() if len(vals) > 1}

    applications: list[DecisionApplication] = []
    for decision in decisions:
        outcome, reason = _apply_one(
            decision, targets, video_sha256, rules_version, conflicted
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


def apply_decisions_to_claims(
    decisions: list[HumanReviewDecision],
    claims: list[SeedClaim],
    video_sha256: str,
    rules_version: str,
) -> list[DecisionApplication]:
    """Claims-only convenience wrapper (claim-evidence and OCR decisions). Non-claim
    decision kinds resolve to INVALID_SUBJECT because their registries are empty."""
    targets = DecisionTargets(claims={c.claim_id: c for c in claims})
    return apply_decisions(decisions, targets, video_sha256, rules_version)


def _apply_one(
    decision: HumanReviewDecision,
    targets: DecisionTargets,
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
    selector, validator, applier = _DISPATCH[decision.decision_type]
    registry = selector(targets)
    target = registry.get(decision.subject_id)
    if target is None:
        return DecisionOutcome.INVALID_SUBJECT, (
            f"no {decision.decision_type.value} target {decision.subject_id}"
        )
    if not validator(decision.value):
        return DecisionOutcome.INVALID_VALUE, (
            f"value {decision.value!r} not admissible for {decision.decision_type.value}"
        )
    applier(target, decision)
    return DecisionOutcome.APPLIED, None
