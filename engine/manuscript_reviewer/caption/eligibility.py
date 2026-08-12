"""Central caption-eligibility policy (§5-§7). ONE resolver controls the
matrix; builders never decide eligibility themselves.

CANDIDATE != FINAL FACT, enforced in code:

* An enum value alone never proves human verification (§4). A
  ``SpeedConclusion.REGULAR_SUPPORTED`` that is not backed by an APPLIED bound
  ``HumanReviewDecision`` (or human-verification evidence / a human-added fact)
  stays REVIEW_REQUIRED.
* Machine ASR text, machine OCR text, machine language detection, diarization,
  generic ActionCandidates, identity-ambiguous tracks and protected-trait
  inferences are never eligible on their own.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction

from ..models.audio import SourceVerificationStatus, SpeechRegion
from ..models.caption_brain import (
    CaptionEligibility,
    CaptionFactType,
    EligibilityBasis,
    HumanCaptionFact,
)
from ..models.evidence import EvidenceReference, EvidenceType
from ..models.review_intelligence import (
    ActionCandidate,
    CameraMotionCandidate,
    CameraMotionClass,
    ClaimReviewStatus,
    DecisionApplication,
    DecisionType,
    EvidenceStatus,
    HumanReviewDecision,
    PlaybackSpeedEvidence,
    SeedClaimType,
    SourceTextVerificationStatus,
    SpeedConclusion,
    TextTrack,
)
from ..models.shot_truth import CandidateStatus, ShotProposal, TransitionStatus

#: One assessment: (eligibility, basis-or-None, human-readable reason).
Assessment = tuple[CaptionEligibility, EligibilityBasis | None, str]


@dataclass
class EligibilityContext:
    """Provenance the resolver inspects. Only APPLIED decisions count."""

    #: (decision_type, subject_id) -> decision, for APPLIED applications only.
    applied_decisions: dict[tuple[DecisionType, str], HumanReviewDecision] = field(
        default_factory=dict
    )
    #: Track ids whose identity is ambiguous / reacquired without human
    #: resolution (facts referencing them are ineligible).
    ambiguous_track_ids: frozenset[str] = frozenset()

    @classmethod
    def build(
        cls,
        decisions: list[HumanReviewDecision],
        applications: list[DecisionApplication],
        ambiguous_track_ids: frozenset[str] = frozenset(),
    ) -> EligibilityContext:
        applied_ids = {a.decision_id for a in applications if a.applied}
        index = {
            (d.decision_type, d.subject_id): d
            for d in decisions
            if d.decision_id in applied_ids
        }
        return cls(applied_decisions=index, ambiguous_track_ids=ambiguous_track_ids)

    def applied(self, decision_type: DecisionType, subject_id: str) -> HumanReviewDecision | None:
        return self.applied_decisions.get((decision_type, subject_id))


def has_human_verification(refs: list[EvidenceReference]) -> bool:
    """An explicit HUMAN_VERIFICATION evidence reference (§4 basis 3)."""
    return any(r.evidence_type == EvidenceType.HUMAN_VERIFICATION for r in refs)


# ---------------------------------------------------------------------------
# Per-domain assessments (the eligibility matrix, §7)
# ---------------------------------------------------------------------------


def assess_media_identity() -> Assessment:
    return (
        CaptionEligibility.ELIGIBLE,
        EligibilityBasis.DETERMINISTIC_EVIDENCE,
        "Phase 1 deterministic media verification",
    )


def assess_shot_boundary(shot: ShotProposal) -> Assessment:
    if shot.start_exact is None or shot.end_exact is None:
        return (
            CaptionEligibility.REVIEW_REQUIRED,
            None,
            "shot interval lacks exact annotation times",
        )
    if shot.review_status == CandidateStatus.SUPPORTED:
        return (
            CaptionEligibility.ELIGIBLE,
            EligibilityBasis.DETERMINISTIC_EVIDENCE,
            "supported ShotProposal exact timeline",
        )
    return (
        CaptionEligibility.REVIEW_REQUIRED,
        None,
        f"shot review_status={shot.review_status.value}",
    )


#: Transitions whose semantics require verified source relationship (§23).
_AUDIO_SEMANTIC_TRANSITIONS = frozenset({"L-cut", "J-cut"})


def assess_transition(shot: ShotProposal, ctx: EligibilityContext) -> Assessment:
    """Never default an unresolved transition; L/J never from audio crossing
    alone (§22/§23). Human resolution comes through the typed
    TRANSITION_CLASSIFICATION decision targeting the actual ShotProposal."""
    label = shot.transition_into_shot
    human = ctx.applied(
        DecisionType.TRANSITION_CLASSIFICATION, f"TRANSITION-{shot.shot_index}"
    )
    if human is not None:
        return (
            CaptionEligibility.ELIGIBLE,
            EligibilityBasis.APPLIED_HUMAN_DECISION,
            f"human TRANSITION_CLASSIFICATION decision {human.decision_id}",
        )
    if label in _AUDIO_SEMANTIC_TRANSITIONS:
        return (
            CaptionEligibility.REVIEW_REQUIRED,
            None,
            f"{label} requires human-verified source relationship "
            "(audio crossing a cut is not proof)",
        )
    if shot.transition_status == TransitionStatus.PROPOSED and label:
        return (
            CaptionEligibility.ELIGIBLE,
            EligibilityBasis.DETERMINISTIC_EVIDENCE,
            "transition supported by the Phase 2 verifier",
        )
    return (
        CaptionEligibility.REVIEW_REQUIRED,
        None,
        f"transition unresolved (status={shot.transition_status.value}); "
        "never defaulted to Hard cut",
    )


def assess_speech(region: SpeechRegion) -> Assessment:
    """A speech line renders only from HUMAN_VERIFIED / HUMAN_CORRECTED source
    audio (§27). UNVERIFIED ASR is never final dialogue; REJECTED is never used."""
    status = region.source_verification_status
    if status == SourceVerificationStatus.REJECTED:
        return (CaptionEligibility.REJECTED, None, "human listened and rejected the ASR claim")
    if status == SourceVerificationStatus.UNVERIFIED:
        return (
            CaptionEligibility.INELIGIBLE,
            None,
            "ASR text is unverified evidence; a human source-audio listen is mandatory",
        )
    if not region.caption_text_eligible:
        return (
            CaptionEligibility.INELIGIBLE,
            None,
            "HUMAN_CORRECTED region without corrected_text — no quotable text",
        )
    return (
        CaptionEligibility.ELIGIBLE,
        EligibilityBasis.HUMAN_VERIFICATION_EVIDENCE,
        f"source verification {status.value}",
    )


def assess_speaker_attribution(speaker_id: str | None) -> Assessment:
    """Every final speech line needs a verified speaker C ID (§28)."""
    if speaker_id:
        return (
            CaptionEligibility.ELIGIBLE,
            EligibilityBasis.HUMAN_ADDED_FACT,
            "speaker attributed by human review input",
        )
    return (
        CaptionEligibility.REVIEW_REQUIRED,
        None,
        "speaker identity unresolved; dialogue line blocked from final output",
    )


def assess_on_screen_text(track: TextTrack) -> Assessment:
    """Machine OCR is never final text truth (§35)."""
    if track.verification_status in (
        SourceTextVerificationStatus.HUMAN_VERIFIED,
        SourceTextVerificationStatus.HUMAN_CORRECTED,
    ):
        return (
            CaptionEligibility.ELIGIBLE,
            EligibilityBasis.HUMAN_VERIFICATION_EVIDENCE,
            f"on-screen text {track.verification_status.value}",
        )
    if track.verification_status == SourceTextVerificationStatus.REJECTED:
        return (CaptionEligibility.REJECTED, None, "human rejected the OCR text")
    return (
        CaptionEligibility.INELIGIBLE,
        None,
        "machine OCR text is never caption-eligible without source verification",
    )


def assess_visual_action(candidate: ActionCandidate, ctx: EligibilityContext) -> Assessment:
    """A generic ActionCandidate is not final semantics (§37). Eligibility
    requires an APPLIED ACTION_SEMANTICS decision (or human-verification
    evidence) — a populated ``semantic_label`` alone is NOT trusted (§4)."""
    for track_id in [*candidate.subject_track_ids, *candidate.object_track_ids]:
        if track_id in ctx.ambiguous_track_ids:
            return (
                CaptionEligibility.INELIGIBLE,
                None,
                f"track {track_id} identity is ambiguous; needs human identity resolution",
            )
    decision = ctx.applied(DecisionType.ACTION_SEMANTICS, candidate.candidate_id)
    if decision is not None and candidate.semantic_label:
        return (
            CaptionEligibility.ELIGIBLE,
            EligibilityBasis.APPLIED_HUMAN_DECISION,
            f"human ACTION_SEMANTICS decision {decision.decision_id}",
        )
    if candidate.semantic_label and has_human_verification(candidate.evidence_refs):
        return (
            CaptionEligibility.ELIGIBLE,
            EligibilityBasis.HUMAN_VERIFICATION_EVIDENCE,
            "semantic label backed by explicit human-verification evidence",
        )
    if candidate.semantic_label:
        return (
            CaptionEligibility.REVIEW_REQUIRED,
            None,
            "semantic label present but no applied human decision backs it "
            "(a mutated field alone is not verification)",
        )
    return (
        CaptionEligibility.INELIGIBLE,
        None,
        f"generic {candidate.action_class.value} candidate has no verified semantics",
    )


def assess_playback_speed(
    evidence: PlaybackSpeedEvidence, ctx: EligibilityContext
) -> Assessment:
    """REGULAR/SLOW/ACCELERATED *candidates* are never final (§26). Even a
    REGULAR_SUPPORTED enum is trusted only with provenance (§4)."""
    decision = ctx.applied(DecisionType.PLAYBACK_SPEED, f"SPEED-{evidence.shot_number}")
    if decision is not None:
        return (
            CaptionEligibility.ELIGIBLE,
            EligibilityBasis.APPLIED_HUMAN_DECISION,
            f"human PLAYBACK_SPEED decision {decision.decision_id}",
        )
    if has_human_verification(evidence.evidence_refs):
        return (
            CaptionEligibility.ELIGIBLE,
            EligibilityBasis.HUMAN_VERIFICATION_EVIDENCE,
            "speed backed by explicit human-verification evidence",
        )
    if evidence.conclusion == SpeedConclusion.REGULAR_SUPPORTED:
        return (
            CaptionEligibility.REVIEW_REQUIRED,
            None,
            "REGULAR_SUPPORTED enum without an applied human decision — "
            "provenance is required, the enum alone proves nothing",
        )
    return (
        CaptionEligibility.REVIEW_REQUIRED,
        None,
        f"{evidence.conclusion.value} is a candidate, never a final speed",
    )


def assess_camera_movement(
    candidate: CameraMotionCandidate, ctx: EligibilityContext
) -> Assessment:
    """Low-level camera motion needs verification before final wording (§25).
    2D global motion never proves dolly/track/zoom semantics."""
    decision = ctx.applied(DecisionType.CAMERA_CLASSIFICATION, candidate.candidate_id)
    if decision is not None:
        return (
            CaptionEligibility.ELIGIBLE,
            EligibilityBasis.APPLIED_HUMAN_DECISION,
            f"human CAMERA_CLASSIFICATION decision {decision.decision_id}",
        )
    if candidate.motion_class == CameraMotionClass.UNRESOLVED:
        return (CaptionEligibility.INELIGIBLE, None, "unresolved camera motion class")
    return (
        CaptionEligibility.REVIEW_REQUIRED,
        None,
        f"{candidate.motion_class.value} is machine evidence; safe wording "
        "requires human verification",
    )


#: Seed claim types that must never become final facts from visuals (§15).
_PROTECTED = frozenset({SeedClaimType.PROTECTED_TRAIT})


def assess_seed_claim(
    claim_type: SeedClaimType,
    evidence_status: EvidenceStatus | None,
    review_status: ClaimReviewStatus | None,
    evidence_refs: list[EvidenceReference],
) -> Assessment:
    """Supported seed wording may be reused; unsupported clauses never (§14)."""
    if claim_type in _PROTECTED:
        if review_status == ClaimReviewStatus.HUMAN_RESOLVED and has_human_verification(
            evidence_refs
        ):
            return (
                CaptionEligibility.ELIGIBLE,
                EligibilityBasis.HUMAN_VERIFICATION_EVIDENCE,
                "protected trait explicitly human-verified",
            )
        return (
            CaptionEligibility.INELIGIBLE,
            None,
            "protected traits are never inferred from appearance",
        )
    if evidence_status == EvidenceStatus.CONTRADICTED:
        return (CaptionEligibility.REJECTED, None, "claim contradicted by media evidence")
    if evidence_status == EvidenceStatus.SUPPORTED:
        if review_status == ClaimReviewStatus.HUMAN_RESOLVED and has_human_verification(
            evidence_refs
        ):
            return (
                CaptionEligibility.ELIGIBLE,
                EligibilityBasis.APPLIED_HUMAN_DECISION,
                "claim human-resolved as supported",
            )
        if any(r.is_factual for r in evidence_refs):
            return (
                CaptionEligibility.ELIGIBLE,
                EligibilityBasis.DETERMINISTIC_EVIDENCE,
                "claim supported by deterministic factual evidence",
            )
        return (
            CaptionEligibility.REVIEW_REQUIRED,
            None,
            "SUPPORTED status without factual evidence provenance (§4)",
        )
    if evidence_status == EvidenceStatus.PARTIALLY_SUPPORTED:
        return (
            CaptionEligibility.REVIEW_REQUIRED,
            None,
            "partially supported — only the supported atomic pieces may be reused",
        )
    return (
        CaptionEligibility.REVIEW_REQUIRED,
        None,
        f"claim unresolved (status={evidence_status.value if evidence_status else 'none'})",
    )


#: Fact types whose events are timed and must carry an exact window.
_TIMED_HUMAN_FACTS = frozenset(
    {
        CaptionFactType.VISUAL_ACTION,
        CaptionFactType.SPEECH,
        CaptionFactType.SOUND,
        CaptionFactType.ON_SCREEN_TEXT,
        CaptionFactType.CAMERA_MOVEMENT,
        CaptionFactType.SPEED_CHANGE,
    }
)

#: Fact types that must carry prose/verbatim text.
_TEXT_HUMAN_FACTS = frozenset(
    {
        CaptionFactType.VISUAL_ACTION,
        CaptionFactType.SOUND,
        CaptionFactType.ON_SCREEN_TEXT,
        CaptionFactType.SCENE,
        CaptionFactType.STYLE,
        CaptionFactType.OVERVIEW_AUDIO,
        CaptionFactType.CHARACTER,
        CaptionFactType.OBJECT,
        CaptionFactType.VISUAL_CONCERN,
        CaptionFactType.AUDIO_CONCERN,
        CaptionFactType.CAMERA_FRAMING,
    }
)

#: Conservative protected/unsupported-trait detector for human fact text (§13).
_PROTECTED_TRAIT_PATTERN = re.compile(
    r"\b(american|british|australian|indian|chinese|japanese|asian|african|"
    r"caucasian|hispanic|latino|latina|white (?:man|woman)|black (?:man|woman)|"
    r"\d{1,3}[- ]year[- ]old|years old|accent|ethnicity|nationality|race)\b",
    re.IGNORECASE,
)

_SPEED_LABEL_VALUES = frozenset({"slow_motion", "regular", "accelerated"})


@dataclass(frozen=True)
class ShotBounds:
    """Verified Shot Truth bounds one shot exposes to human-fact validation."""

    start_exact: Fraction | None
    end_exact: Fraction | None
    start_frame: int
    end_frame: int


def assess_human_fact(
    fact: HumanCaptionFact,
    shot_bounds: Mapping[int, ShotBounds] | None = None,
    frame_to_time: Callable[[int], Fraction | None] | None = None,
) -> Assessment:
    """A bound, non-stale HumanCaptionFact is a human basis — but never an
    unconditional factual bypass (§5.1-11/12/13). Every media-factual human
    fact must carry traceable evidence and pass type-specific validation,
    including full Shot Truth containment for timed facts (§5.2-3)."""
    bounds = shot_bounds or {}
    if not fact.evidence_refs and not fact.bound_evidence_ids:
        return (
            CaptionEligibility.REVIEW_REQUIRED,
            None,
            "human fact carries no evidence reference; every factual caption "
            "assertion must be traceable to media evidence",
        )
    if fact.shot_number is not None and bounds and fact.shot_number not in bounds:
        return (
            CaptionEligibility.REVIEW_REQUIRED,
            None,
            f"human fact targets shot {fact.shot_number}, which is not a "
            "verified shot",
        )
    if fact.fact_type in _TIMED_HUMAN_FACTS:
        if (fact.start_exact is None or fact.end_exact is None) and (
            fact.start_frame is None or fact.end_frame is None
        ):
            return (
                CaptionEligibility.REVIEW_REQUIRED,
                None,
                "timed human fact lacks an exact verified time/frame range",
            )
        if (
            fact.start_exact is not None
            and fact.end_exact is not None
            and fact.start_exact > fact.end_exact
        ):
            return (
                CaptionEligibility.REVIEW_REQUIRED,
                None,
                "timed human fact has start > end",
            )
        if (
            fact.start_frame is not None
            and fact.end_frame is not None
            and fact.start_frame > fact.end_frame
        ):
            return (
                CaptionEligibility.REVIEW_REQUIRED,
                None,
                "timed human fact has start frame > end frame",
            )
        if (
            frame_to_time is not None
            and fact.start_frame is not None
            and fact.end_frame is not None
            and (
                frame_to_time(fact.start_frame) is None
                or frame_to_time(fact.end_frame) is None
            )
        ):
            return (
                CaptionEligibility.REVIEW_REQUIRED,
                None,
                "timed human fact references frames that do not exist in the "
                "frame ledger",
            )
        shot = bounds.get(fact.shot_number) if fact.shot_number is not None else None
        if shot is not None:
            if (
                fact.start_exact is not None
                and fact.end_exact is not None
                and shot.start_exact is not None
                and shot.end_exact is not None
                and (
                    fact.start_exact < shot.start_exact
                    or fact.end_exact > shot.end_exact
                )
            ):
                return (
                    CaptionEligibility.REVIEW_REQUIRED,
                    None,
                    f"timed human fact range falls outside verified shot "
                    f"{fact.shot_number}",
                )
            if (
                fact.start_frame is not None
                and fact.end_frame is not None
                and (
                    fact.start_frame < shot.start_frame
                    or fact.end_frame > shot.end_frame + 1
                )
            ):
                return (
                    CaptionEligibility.REVIEW_REQUIRED,
                    None,
                    f"timed human fact frame range falls outside verified shot "
                    f"{fact.shot_number}",
                )
    if fact.fact_type in _TEXT_HUMAN_FACTS and not (fact.text_value or "").strip():
        return (
            CaptionEligibility.REVIEW_REQUIRED,
            None,
            f"{fact.fact_type.value} human fact requires text",
        )
    if fact.fact_type == CaptionFactType.PLAYBACK_SPEED and (
        (fact.text_value or "").strip() not in _SPEED_LABEL_VALUES
    ):
        return (
            CaptionEligibility.REVIEW_REQUIRED,
            None,
            f"playback speed {fact.text_value!r} is not an allowed value",
        )
    if fact.fact_type == CaptionFactType.SPEECH and not (
        fact.character_ids or fact.semantic_value.get("speaker_id")
    ):
        return (
            CaptionEligibility.REVIEW_REQUIRED,
            None,
            "human speech fact lacks speaker attribution",
        )
    if fact.fact_type == CaptionFactType.CHARACTER and fact.text_value:
        hit = _PROTECTED_TRAIT_PATTERN.search(fact.text_value)
        if hit is not None and not (
            fact.semantic_value.get("protected_trait_source")
            and has_human_verification(fact.evidence_refs)
        ):
            return (
                CaptionEligibility.REVIEW_REQUIRED,
                None,
                f"possible protected/unsupported trait {hit.group(0)!r}: visual "
                "appearance alone is insufficient; name an allowed explicit "
                "source (protected_trait_source) with human verification",
            )
    return (
        CaptionEligibility.ELIGIBLE,
        EligibilityBasis.HUMAN_ADDED_FACT,
        "human-added bound caption fact with traceable evidence",
    )


def assess_speech_enrichment(fact: HumanCaptionFact) -> Assessment:
    """An enrichment/split HumanCaptionFact (semantic ``region_id`` /
    ``splits_region_id``) is NOT a standalone fact, but it still passes the
    same evidence gate before it may alter any rendered semantic — speaker,
    tone, off-screen state, language level/name, split behavior (§5.2-1).
    Timing comes from the machine region, so the timed-range rule does not
    apply; the evidence rule always does."""
    if not fact.evidence_refs and not fact.bound_evidence_ids:
        return (
            CaptionEligibility.REVIEW_REQUIRED,
            None,
            "speech enrichment carries no evidence reference; an evidence-free "
            "enrichment never alters final caption text",
        )
    return (
        CaptionEligibility.ELIGIBLE,
        EligibilityBasis.HUMAN_ADDED_FACT,
        "evidence-backed speech enrichment",
    )


def assess_sound_semantics(ctx: EligibilityContext, subject_id: str) -> Assessment:
    """TRANSIENT_CANDIDATE never becomes gunshot/clap/door slam (§44)."""
    decision = ctx.applied(DecisionType.CLAIM_EVIDENCE, subject_id)
    if decision is not None:
        return (
            CaptionEligibility.ELIGIBLE,
            EligibilityBasis.APPLIED_HUMAN_DECISION,
            f"human sound-semantics decision {decision.decision_id}",
        )
    return (
        CaptionEligibility.INELIGIBLE,
        None,
        "generic waveform classes never produce semantic sound labels",
    )
