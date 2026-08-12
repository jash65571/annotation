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
from fractions import Fraction
from pathlib import Path

from pydantic import ValidationError

from ..models.audio import SourceVerificationStatus, SpeechRegion
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
    SourceTextVerificationStatus,
    SpeedConclusion,
    TextTrack,
)
from ..models.shot_truth import ShotProposal, TransitionStatus
from ..rules.loader import load_rules

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
    #: Phase 5.1: real evidence records a reviewer can resolve directly.
    speech_regions: dict[str, SpeechRegion] = field(default_factory=dict)
    text_tracks: dict[str, TextTrack] = field(default_factory=dict)
    #: Keyed "TRANSITION-<shot_index>".
    transitions: dict[str, ShotProposal] = field(default_factory=dict)
    #: Context (not a registry): ledger frame index -> exact ANNOTATION time.
    #: Required so an ACTION_BOUNDARY decision recomputes exact timing — the
    #: frame ledger owns timing; frame indices alone are never accepted.
    frame_to_time: Callable[[int], Fraction | None] | None = None
    #: Context: verified shot index -> inclusive (start_frame, end_frame).
    #: Boundary/timing decisions must stay inside their shot (§5.2-4).
    shot_frame_ranges: dict[int, tuple[int, int]] = field(default_factory=dict)


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


def _apply_claim(target: object, decision: HumanReviewDecision, _t: DecisionTargets) -> None:
    assert isinstance(target, SeedClaim)
    if decision.decision_type == DecisionType.CLAIM_EVIDENCE:
        target.evidence_status = EvidenceStatus(decision.value.strip())
    else:  # OCR_TEXT / OCR_TIMING: a human confirming the text/timing supports it.
        target.evidence_status = EvidenceStatus.SUPPORTED
    target.review_status = ClaimReviewStatus.HUMAN_RESOLVED
    target.evidence.append(_human_ref(decision))


def _apply_identity(target: object, decision: HumanReviewDecision, _t: DecisionTargets) -> None:
    assert isinstance(target, EntityTrack)
    # A human identity resolution explicitly resolves BOTH ambiguity conditions
    # it reviewed, with provenance — never a blind flag flip (Phase 5.1 item 9).
    previous = (
        f"reacquired={target.reacquired}, identity_ambiguous={target.identity_ambiguous}"
    )
    target.reacquired = False
    target.identity_ambiguous = False
    target.evidence_refs.append(_human_ref(decision))
    target.notes.append(
        f"identity resolved by human: {decision.value} ({decision.decided_by}); "
        f"previous state: {previous}"
    )


def _apply_camera(target: object, decision: HumanReviewDecision, _t: DecisionTargets) -> None:
    assert isinstance(target, CameraMotionCandidate)
    target.motion_class = CameraMotionClass(decision.value.strip())
    target.review_required = False
    target.notes.append(f"human classification: {decision.value} ({decision.decided_by})")


def _apply_action_semantics(
    target: object, decision: HumanReviewDecision, _t: DecisionTargets
) -> None:
    assert isinstance(target, ActionCandidate)
    target.semantic_label = decision.value.strip()
    target.review_required = False


def _apply_action_boundary(
    target: object, decision: HumanReviewDecision, targets: DecisionTargets
) -> None:
    """The frame ledger owns timing: new boundary frames MUST resolve to exact
    ledger times or the decision is INVALID_VALUE — stale exact timing is never
    accepted (Phase 5.1 item 10)."""
    assert isinstance(target, ActionCandidate)
    lo, hi = (int(p) for p in decision.value.split("-", 1))
    if lo > hi:
        raise ValueError(f"boundary start {lo} > end {hi}")
    if targets.frame_to_time is None:
        raise ValueError(
            "no frame-ledger time resolver available; an ACTION_BOUNDARY "
            "decision cannot be applied without exact frame timing"
        )
    start_exact = targets.frame_to_time(lo)
    end_exact = targets.frame_to_time(hi)
    if start_exact is None or end_exact is None:
        raise ValueError(f"frames {lo}-{hi} do not exist in the frame ledger")
    if target.shot_number is not None:
        shot_range = targets.shot_frame_ranges.get(target.shot_number)
        if shot_range is not None and (lo < shot_range[0] or hi > shot_range[1] + 1):
            raise ValueError(
                f"frames {lo}-{hi} fall outside shot {target.shot_number} "
                f"(frames {shot_range[0]}-{shot_range[1]})"
            )
    target.start_frame, target.end_frame = lo, hi
    target.start_exact, target.end_exact = start_exact, end_exact
    target.review_required = False
    target.evidence_refs.append(_human_ref(decision))


def _apply_speed(target: object, decision: HumanReviewDecision, _t: DecisionTargets) -> None:
    assert isinstance(target, PlaybackSpeedEvidence)
    target.conclusion = _SPEED_VALUES[decision.value.strip()]
    target.review_required = False


def _apply_proposal(target: object, decision: HumanReviewDecision, _t: DecisionTargets) -> None:
    assert isinstance(target, ReviewProposal)
    target.outcome = ReviewProposalOutcome(decision.value.strip())


def _apply_speech_verification(
    target: object, decision: HumanReviewDecision, _t: DecisionTargets
) -> None:
    """The human listened to the source audio (§5.1-6). The original ASR text
    candidate is preserved untouched — only the human-listen axis moves."""
    assert isinstance(target, SpeechRegion)
    value = decision.value.strip().lower()
    target.source_verification_status = (
        SourceVerificationStatus.HUMAN_VERIFIED
        if value == "verified"
        else SourceVerificationStatus.REJECTED
    )


def _apply_speech_correction(
    target: object, decision: HumanReviewDecision, _t: DecisionTargets
) -> None:
    assert isinstance(target, SpeechRegion)
    target.source_verification_status = SourceVerificationStatus.HUMAN_CORRECTED
    target.corrected_text = decision.value
    # text_candidate (the original ASR transcript) is deliberately untouched.


def _apply_text_verification(
    target: object, decision: HumanReviewDecision, _t: DecisionTargets
) -> None:
    assert isinstance(target, TextTrack)
    value = decision.value.strip().lower()
    if value == "verified":
        target.verification_status = SourceTextVerificationStatus.HUMAN_VERIFIED
        target.caption_text_eligible = True
        target.review_required = False
    else:
        target.verification_status = SourceTextVerificationStatus.REJECTED
        target.caption_text_eligible = False
        target.review_required = False
    target.evidence_refs.append(_human_ref(decision))


def _apply_text_correction(
    target: object, decision: HumanReviewDecision, _t: DecisionTargets
) -> None:
    assert isinstance(target, TextTrack)
    target.verification_status = SourceTextVerificationStatus.HUMAN_CORRECTED
    target.corrected_text = decision.value
    target.caption_text_eligible = True
    target.review_required = False
    target.evidence_refs.append(_human_ref(decision))
    # Raw OCR observations and machine consensus are never overwritten.


def _apply_text_timing(
    target: object, decision: HumanReviewDecision, targets: DecisionTargets
) -> None:
    assert isinstance(target, TextTrack)
    lo, hi = (int(p) for p in decision.value.split("-", 1))
    if lo > hi:
        raise ValueError(f"timing start {lo} > end {hi}")
    if targets.frame_to_time is not None and (
        targets.frame_to_time(lo) is None or targets.frame_to_time(hi) is None
    ):
        raise ValueError(f"frames {lo}-{hi} do not exist in the frame ledger")
    if targets.shot_frame_ranges:
        # The corrected range must lie entirely inside ONE verified shot — a
        # text overlay never spans an unrelated shot (§5.2-4).
        owning = next(
            (
                (idx, rng)
                for idx, rng in targets.shot_frame_ranges.items()
                if rng[0] <= lo <= rng[1]
            ),
            None,
        )
        if owning is None:
            raise ValueError(f"frame {lo} is not inside any verified shot")
        if hi > owning[1][1] + 1:
            raise ValueError(
                f"text range {lo}-{hi} spans beyond shot {owning[0]} "
                f"(frames {owning[1][0]}-{owning[1][1]})"
            )
    target.first_stable_frame = lo
    target.last_stable_frame = hi
    target.evidence_refs.append(_human_ref(decision))


def _apply_transition(
    target: object, decision: HumanReviewDecision, _t: DecisionTargets
) -> None:
    """Human transition resolution (§5.1-8). Value comes from the rule-file
    menu; Shot 1 must remain Opening shot and no later shot may become it."""
    assert isinstance(target, ShotProposal)
    value = decision.value.strip()
    opening = str(load_rules().get("shots.shot_one_transition", "Opening shot"))
    if target.shot_index == 1 and value != opening:
        raise ValueError(f"Shot 1 must remain {opening!r}")
    if target.shot_index > 1 and value == opening:
        raise ValueError(f"{opening!r} may appear nowhere except Shot 1")
    target.transition_into_shot = value
    target.transition_status = TransitionStatus.PROPOSED
    target.evidence_refs.append(f"EV-HUMAN-{decision.decision_id}")


def _valid_boundary(value: str) -> bool:
    parts = value.split("-", 1)
    return len(parts) == 2 and all(p.strip().isdigit() for p in parts)


def _valid_transition(value: str) -> bool:
    allowed = {str(t) for t in load_rules().get("shots.allowed_transition_types", [])}
    return value.strip() in allowed


_VERIFY_VALUES = frozenset({"verified", "rejected"})

_Applier = Callable[[object, HumanReviewDecision, DecisionTargets], None]
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
    DecisionType.SPEECH_VERIFICATION: (
        lambda t: t.speech_regions, lambda v: v.strip().lower() in _VERIFY_VALUES,
        _apply_speech_verification),
    DecisionType.SPEECH_CORRECTION: (
        lambda t: t.speech_regions, lambda v: bool(v.strip()), _apply_speech_correction),
    DecisionType.TEXT_VERIFICATION: (
        lambda t: t.text_tracks, lambda v: v.strip().lower() in _VERIFY_VALUES,
        _apply_text_verification),
    DecisionType.TEXT_CORRECTION: (
        lambda t: t.text_tracks, lambda v: bool(v.strip()), _apply_text_correction),
    DecisionType.TEXT_TIMING: (
        lambda t: t.text_tracks, _valid_boundary, _apply_text_timing),
    DecisionType.TRANSITION_CLASSIFICATION: (
        lambda t: t.transitions, _valid_transition, _apply_transition),
}


def apply_decisions(
    decisions: list[HumanReviewDecision],
    targets: DecisionTargets,
    video_sha256: str,
    rules_version: str,
) -> list[DecisionApplication]:
    """Route each decision to the ONE registry its kind is allowed to touch and
    apply a fixed whitelisted mutation. Returns a per-decision application record."""
    # CONFLICT: two decisions of the SAME KIND on the same subject with
    # different values (an ACTION_SEMANTICS + ACTION_BOUNDARY pair on one
    # candidate is complementary, not conflicting).
    subject_values: dict[tuple[DecisionType, str], set[str]] = {}
    for d in decisions:
        subject_values.setdefault((d.decision_type, d.subject_id), set()).add(
            d.value.strip()
        )
    conflicted = {key for key, vals in subject_values.items() if len(vals) > 1}

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
    conflicted: set[tuple[DecisionType, str]],
) -> tuple[DecisionOutcome, str | None]:
    if (decision.decision_type, decision.subject_id) in conflicted:
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
    try:
        applier(target, decision, targets)
    except ValueError as exc:
        return DecisionOutcome.INVALID_VALUE, str(exc)
    return DecisionOutcome.APPLIED, None
