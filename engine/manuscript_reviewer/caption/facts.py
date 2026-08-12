"""Caption fact graph (§8): typed facts BEFORE prose.

Builds :class:`CaptionFact` records from Phase 1-4 evidence, applied human
decisions and human-added facts. Eligibility is resolved exclusively through
:mod:`.eligibility`; this module never invents facts, never moves timing, and
never upgrades a machine candidate.

Timing authority by fact type (§49): shots use Shot Truth exact intervals;
speech/sound use source-audio verified timing; OCR uses source-verified
TextTrack frame boundaries; actions/camera use their verified frame boundaries.
Seed timestamps are never timing authority. All display values are canonical
0.1 s projections of exact annotation times (ROUND_HALF_UP) — no local
rounding logic exists here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from ..media.timestamps import format_manuscript_display
from ..models.audio import AudioQCResult, SourceVerificationStatus
from ..models.caption import SeedClaim
from ..models.caption_brain import (
    CaptionEligibility,
    CaptionFact,
    CaptionFactType,
    EligibilityBasis,
    FactMateriality,
    FactSourceKind,
    HumanCaptionFact,
    LanguageRenderLevel,
)
from ..models.review_intelligence import (
    ActionCandidate,
    CameraMotionCandidate,
    ContinuityLink,
    DecisionType,
    EntityTrack,
    FinalStateCheck,
    PlaybackSpeedEvidence,
    SeedClaimType,
    TextTrack,
)
from ..models.shot_truth import ShotProposal, ShotTruthResult
from . import eligibility as elig
from .eligibility import EligibilityContext

_SPEED_LABELS = {"slow_motion", "regular", "accelerated"}


@dataclass
class FactGraph:
    facts: list[CaptionFact] = field(default_factory=list)

    def by_id(self) -> dict[str, CaptionFact]:
        return {f.fact_id: f for f in self.facts}

    def of_type(self, fact_type: CaptionFactType) -> list[CaptionFact]:
        return [f for f in self.facts if f.fact_type == fact_type]

    def eligible(self) -> list[CaptionFact]:
        return [f for f in self.facts if f.eligibility == CaptionEligibility.ELIGIBLE]


@dataclass
class FactBuildInputs:
    video_id: str | None = None
    video_sha256: str | None = None
    rules_version: str | None = None
    shot_truth: ShotTruthResult | None = None
    audio_truth: AudioQCResult | None = None
    seed_claims: list[SeedClaim] = field(default_factory=list)
    speed_evidence: list[PlaybackSpeedEvidence] = field(default_factory=list)
    camera_candidates: list[CameraMotionCandidate] = field(default_factory=list)
    text_tracks: list[TextTrack] = field(default_factory=list)
    action_candidates: list[ActionCandidate] = field(default_factory=list)
    final_state_checks: list[FinalStateCheck] = field(default_factory=list)
    entity_tracks: list[EntityTrack] = field(default_factory=list)
    continuity_links: list[ContinuityLink] = field(default_factory=list)
    human_facts: list[HumanCaptionFact] = field(default_factory=list)
    ctx: EligibilityContext = field(default_factory=EligibilityContext)


class _Counter:
    def __init__(self) -> None:
        self._n = 0

    def next(self) -> str:
        self._n += 1
        return f"CF-{self._n:04d}"


def ambiguous_track_ids(
    tracks: list[EntityTrack], links: list[ContinuityLink]
) -> frozenset[str]:
    """Identity-ambiguous / unresolved-reacquired tracks and unresolved
    SAME_ENTITY_CANDIDATE links: never eligible identity sources (§3)."""
    ids = {t.track_id for t in tracks if t.identity_ambiguous or t.reacquired}
    for link in links:
        if link.relationship == "SAME_ENTITY_CANDIDATE" and link.review_required:
            ids.add(link.from_track_id)
            ids.add(link.to_track_id)
    return frozenset(ids)


def _disp(value: Fraction | None) -> str | None:
    return format_manuscript_display(value) if value is not None else None


def build_fact_graph(inputs: FactBuildInputs) -> FactGraph:
    graph = FactGraph()
    counter = _Counter()
    _build_media_fact(graph, counter, inputs)
    _build_shot_facts(graph, counter, inputs)
    _build_speech_facts(graph, counter, inputs)
    _build_text_facts(graph, counter, inputs)
    _build_action_facts(graph, counter, inputs)
    _build_camera_facts(graph, counter, inputs)
    _build_speed_facts(graph, counter, inputs)
    _build_final_state_facts(graph, counter, inputs)
    _build_seed_claim_facts(graph, counter, inputs)
    _build_human_facts(graph, counter, inputs)
    return graph


# ---------------------------------------------------------------------------
# Media / shots
# ---------------------------------------------------------------------------


def _build_media_fact(graph: FactGraph, counter: _Counter, inputs: FactBuildInputs) -> None:
    if inputs.video_id is None:
        return
    status, basis, reason = elig.assess_media_identity()
    graph.facts.append(
        CaptionFact(
            fact_id=counter.next(),
            fact_type=CaptionFactType.MEDIA,
            text_value=inputs.video_id,
            semantic_value={"video_sha256": inputs.video_sha256 or ""},
            eligibility=status,
            eligibility_basis=basis,
            eligibility_reason=reason,
            source_kind=FactSourceKind.MEDIA_TRUTH,
            required_for_caption=True,
            materiality=FactMateriality.REQUIRED,
        )
    )


def _build_shot_facts(graph: FactGraph, counter: _Counter, inputs: FactBuildInputs) -> None:
    if inputs.shot_truth is None:
        return
    for shot in inputs.shot_truth.shots:
        status, basis, reason = elig.assess_shot_boundary(shot)
        graph.facts.append(
            CaptionFact(
                fact_id=counter.next(),
                fact_type=CaptionFactType.SHOT_BOUNDARY,
                shot_number=shot.shot_index,
                start_frame=shot.start_frame_index,
                end_frame=shot.end_frame_index,
                start_exact=shot.start_exact,
                end_exact=shot.end_exact,
                display_start=_disp(shot.start_exact),
                display_end=_disp(shot.end_exact),
                eligibility=status,
                eligibility_basis=basis,
                eligibility_reason=reason,
                source_kind=FactSourceKind.SHOT_TRUTH,
                source_id=f"SHOT-{shot.shot_index}",
                required_for_caption=True,
                materiality=FactMateriality.REQUIRED,
            )
        )
        _build_transition_fact(graph, counter, inputs, shot)


def _build_transition_fact(
    graph: FactGraph, counter: _Counter, inputs: FactBuildInputs, shot: ShotProposal
) -> None:
    status, basis, reason = elig.assess_transition(shot, inputs.ctx)
    decision = inputs.ctx.applied(
        DecisionType.CLAIM_EVIDENCE, f"TRANSITION-{shot.shot_index}"
    )
    graph.facts.append(
        CaptionFact(
            fact_id=counter.next(),
            fact_type=CaptionFactType.TRANSITION,
            shot_number=shot.shot_index,
            start_exact=shot.start_exact,
            display_start=_disp(shot.start_exact),
            text_value=shot.transition_into_shot,
            semantic_value={"transition_status": shot.transition_status.value},
            eligibility=status,
            eligibility_basis=basis,
            eligibility_reason=reason,
            source_kind=(
                FactSourceKind.HUMAN_DECISION
                if basis == EligibilityBasis.APPLIED_HUMAN_DECISION
                else FactSourceKind.SHOT_TRUTH
            ),
            source_id=f"TRANSITION-{shot.shot_index}",
            human_decision_ids=[decision.decision_id] if decision is not None else [],
            required_for_caption=True,
            materiality=FactMateriality.REQUIRED,
        )
    )


# ---------------------------------------------------------------------------
# Speech (§27-§34)
# ---------------------------------------------------------------------------


def _speech_enrichment(
    inputs: FactBuildInputs, region_id: str
) -> HumanCaptionFact | None:
    """A human SPEECH fact that ENRICHES a machine region (speaker/tone/level)
    references it via ``semantic_value["region_id"]``."""
    for hf in inputs.human_facts:
        if (
            hf.fact_type == CaptionFactType.SPEECH
            and hf.semantic_value.get("region_id") == region_id
        ):
            return hf
    return None


def _region_split_facts(inputs: FactBuildInputs, region_id: str) -> list[HumanCaptionFact]:
    """Human SPEECH facts that SPLIT a machine region at a shot boundary
    (§34) reference it via ``semantic_value["splits_region_id"]``."""
    return [
        hf
        for hf in inputs.human_facts
        if hf.fact_type == CaptionFactType.SPEECH
        and hf.semantic_value.get("splits_region_id") == region_id
    ]


def _shot_for_time(
    shot_truth: ShotTruthResult | None, start: Fraction | None, end: Fraction | None
) -> tuple[int | None, bool]:
    """(containing shot, crosses_boundary). Containment is inclusive of the
    shot start and exclusive of the end except for the final endpoint."""
    if shot_truth is None or start is None or end is None:
        return None, False
    containing: int | None = None
    for shot in shot_truth.shots:
        if shot.start_exact is None or shot.end_exact is None:
            continue
        if shot.start_exact <= start < shot.end_exact or (
            start == shot.end_exact == end
        ):
            containing = shot.shot_index
            crosses = end > shot.end_exact
            return containing, crosses
    return None, False


def _build_speech_facts(graph: FactGraph, counter: _Counter, inputs: FactBuildInputs) -> None:
    if inputs.audio_truth is None:
        return
    for region in inputs.audio_truth.speech_regions:
        status, basis, reason = elig.assess_speech(region)
        enrichment = _speech_enrichment(inputs, region.region_id)
        splits = _region_split_facts(inputs, region.region_id)
        if splits:
            # The human-supplied per-shot fragments are built later from
            # human_facts; the machine region itself is superseded.
            continue
        speaker_id: str | None = None
        semantic: dict[str, str] = {}
        character_ids: list[str] = []
        if enrichment is not None:
            speaker_id = enrichment.character_ids[0] if enrichment.character_ids else None
            character_ids = list(enrichment.character_ids)
            semantic.update(
                {k: v for k, v in enrichment.semantic_value.items() if k != "region_id"}
            )
        shot_number, crosses = _shot_for_time(
            inputs.shot_truth, region.start_exact, region.end_exact
        )
        if crosses and status == CaptionEligibility.ELIGIBLE:
            status, basis = CaptionEligibility.REVIEW_REQUIRED, None
            reason = (
                "speech crosses a visual cut; supply per-shot fragments via "
                "human facts (splits_region_id) — never rendered across shots"
            )
        if status == CaptionEligibility.ELIGIBLE:
            sp_status, _, sp_reason = elig.assess_speaker_attribution(speaker_id)
            if sp_status != CaptionEligibility.ELIGIBLE:
                status, basis, reason = sp_status, None, sp_reason
        semantic.setdefault("language_level", LanguageRenderLevel.VERBATIM.value)
        if speaker_id:
            semantic["speaker_id"] = speaker_id
        if (
            region.source_verification_status == SourceVerificationStatus.HUMAN_CORRECTED
            and region.corrected_text is not None
        ):
            semantic["text_source"] = "human_corrected"
        graph.facts.append(
            CaptionFact(
                fact_id=counter.next(),
                fact_type=CaptionFactType.SPEECH,
                shot_number=shot_number,
                character_ids=character_ids,
                start_exact=region.start_exact,
                end_exact=region.end_exact,
                display_start=_disp(region.start_exact),
                display_end=_disp(region.end_exact),
                text_value=region.caption_text,
                semantic_value=semantic,
                eligibility=status,
                eligibility_basis=basis,
                eligibility_reason=reason,
                source_kind=FactSourceKind.AUDIO_TRUTH,
                source_id=region.region_id,
                human_fact_id=enrichment.fact_id if enrichment is not None else None,
                required_for_caption=status == CaptionEligibility.ELIGIBLE,
                materiality=FactMateriality.REQUIRED,
            )
        )


# ---------------------------------------------------------------------------
# On-screen text (§35/§36)
# ---------------------------------------------------------------------------


def _build_text_facts(graph: FactGraph, counter: _Counter, inputs: FactBuildInputs) -> None:
    for track in inputs.text_tracks:
        status, basis, reason = elig.assess_on_screen_text(track)
        text = track.consensus.consensus_text if track.consensus is not None else None
        first = track.first_stable_frame
        last = track.last_stable_frame
        graph.facts.append(
            CaptionFact(
                fact_id=counter.next(),
                fact_type=CaptionFactType.ON_SCREEN_TEXT,
                start_frame=first if first is not None else track.first_candidate_frame,
                end_frame=last if last is not None else track.disappearance_frame,
                text_value=text,
                semantic_value={"verification": track.verification_status.value},
                eligibility=status,
                eligibility_basis=basis,
                eligibility_reason=reason,
                source_kind=FactSourceKind.VISUAL_EVIDENCE,
                source_id=track.track_id,
                required_for_caption=status == CaptionEligibility.ELIGIBLE,
                materiality=FactMateriality.REQUIRED,
            )
        )


# ---------------------------------------------------------------------------
# Visual actions (§37/§38)
# ---------------------------------------------------------------------------


def _build_action_facts(graph: FactGraph, counter: _Counter, inputs: FactBuildInputs) -> None:
    for candidate in inputs.action_candidates:
        status, basis, reason = elig.assess_visual_action(candidate, inputs.ctx)
        decision = inputs.ctx.applied(DecisionType.ACTION_SEMANTICS, candidate.candidate_id)
        graph.facts.append(
            CaptionFact(
                fact_id=counter.next(),
                fact_type=CaptionFactType.VISUAL_ACTION,
                shot_number=candidate.shot_number,
                start_frame=candidate.start_frame,
                end_frame=candidate.end_frame,
                start_exact=candidate.start_exact,
                end_exact=candidate.end_exact,
                display_start=_disp(candidate.start_exact),
                display_end=_disp(candidate.end_exact),
                text_value=candidate.semantic_label,
                semantic_value={
                    "action_class": candidate.action_class.value,
                    **({"pre_state": candidate.pre_state} if candidate.pre_state else {}),
                    **({"post_state": candidate.post_state} if candidate.post_state else {}),
                },
                eligibility=status,
                eligibility_basis=basis,
                eligibility_reason=reason,
                source_kind=(
                    FactSourceKind.HUMAN_DECISION
                    if basis == EligibilityBasis.APPLIED_HUMAN_DECISION
                    else FactSourceKind.VISUAL_EVIDENCE
                ),
                source_id=candidate.candidate_id,
                human_decision_ids=[decision.decision_id] if decision is not None else [],
                required_for_caption=status == CaptionEligibility.ELIGIBLE,
                materiality=FactMateriality.REQUIRED,
            )
        )


# ---------------------------------------------------------------------------
# Camera (§24/§25)
# ---------------------------------------------------------------------------


def _build_camera_facts(graph: FactGraph, counter: _Counter, inputs: FactBuildInputs) -> None:
    for candidate in inputs.camera_candidates:
        status, basis, reason = elig.assess_camera_movement(candidate, inputs.ctx)
        decision = inputs.ctx.applied(
            DecisionType.CAMERA_CLASSIFICATION, candidate.candidate_id
        )
        graph.facts.append(
            CaptionFact(
                fact_id=counter.next(),
                fact_type=CaptionFactType.CAMERA_MOVEMENT,
                shot_number=candidate.shot_number,
                start_frame=candidate.start_frame,
                end_frame=candidate.end_frame,
                start_exact=candidate.start_exact,
                end_exact=candidate.end_exact,
                display_start=_disp(candidate.start_exact),
                display_end=_disp(candidate.end_exact),
                semantic_value={
                    "motion_class": candidate.motion_class.value,
                    **({"direction": candidate.direction} if candidate.direction else {}),
                },
                eligibility=status,
                eligibility_basis=basis,
                eligibility_reason=reason,
                source_kind=(
                    FactSourceKind.HUMAN_DECISION
                    if basis == EligibilityBasis.APPLIED_HUMAN_DECISION
                    else FactSourceKind.VISUAL_EVIDENCE
                ),
                source_id=candidate.candidate_id,
                human_decision_ids=[decision.decision_id] if decision is not None else [],
                required_for_caption=status == CaptionEligibility.ELIGIBLE,
                materiality=FactMateriality.MATERIAL,
            )
        )


# ---------------------------------------------------------------------------
# Playback speed (§26)
# ---------------------------------------------------------------------------


def _build_speed_facts(graph: FactGraph, counter: _Counter, inputs: FactBuildInputs) -> None:
    for evidence in inputs.speed_evidence:
        status, basis, reason = elig.assess_playback_speed(evidence, inputs.ctx)
        decision = inputs.ctx.applied(
            DecisionType.PLAYBACK_SPEED, f"SPEED-{evidence.shot_number}"
        )
        value: str | None = None
        if decision is not None and decision.value.strip() in _SPEED_LABELS:
            value = decision.value.strip()
        graph.facts.append(
            CaptionFact(
                fact_id=counter.next(),
                fact_type=CaptionFactType.PLAYBACK_SPEED,
                shot_number=evidence.shot_number,
                text_value=value,
                semantic_value={"machine_conclusion": evidence.conclusion.value},
                eligibility=status,
                eligibility_basis=basis,
                eligibility_reason=reason,
                source_kind=(
                    FactSourceKind.HUMAN_DECISION
                    if basis == EligibilityBasis.APPLIED_HUMAN_DECISION
                    else FactSourceKind.VISUAL_EVIDENCE
                ),
                source_id=f"SPEED-{evidence.shot_number}",
                human_decision_ids=[decision.decision_id] if decision is not None else [],
                required_for_caption=True,
                materiality=FactMateriality.REQUIRED,
            )
        )
        # Machine mid-shot speed-change candidates stay review-required (§26).
        for frame in evidence.speed_change_frames:
            graph.facts.append(
                CaptionFact(
                    fact_id=counter.next(),
                    fact_type=CaptionFactType.SPEED_CHANGE,
                    shot_number=evidence.shot_number,
                    start_frame=frame,
                    eligibility=CaptionEligibility.REVIEW_REQUIRED,
                    eligibility_reason=(
                        "mid-shot speed-change candidate; editorial retiming "
                        "must be verified before a Speed Changes entry"
                    ),
                    source_kind=FactSourceKind.VISUAL_EVIDENCE,
                    source_id=f"SPEEDCHG-{evidence.shot_number}-{frame}",
                    materiality=FactMateriality.MATERIAL,
                )
            )


# ---------------------------------------------------------------------------
# Final object state (§47)
# ---------------------------------------------------------------------------


def _build_final_state_facts(
    graph: FactGraph, counter: _Counter, inputs: FactBuildInputs
) -> None:
    for check in inputs.final_state_checks:
        status = (
            CaptionEligibility.REVIEW_REQUIRED
            if not check.resolved
            else CaptionEligibility.ELIGIBLE
        )
        basis = EligibilityBasis.DETERMINISTIC_EVIDENCE if check.resolved else None
        graph.facts.append(
            CaptionFact(
                fact_id=counter.next(),
                fact_type=CaptionFactType.FINAL_OBJECT_STATE,
                shot_number=check.shot_number,
                object_ids=[check.entity_id],
                end_frame=check.final_visible_frame,
                semantic_value={
                    "final_state": check.final_state.value,
                    "still_visible_at_shot_end": str(check.still_visible_at_shot_end).lower(),
                },
                eligibility=status,
                eligibility_basis=basis,
                eligibility_reason=(
                    "resolved final-state check"
                    if check.resolved
                    else check.review_reason or "final state unresolved"
                ),
                source_kind=FactSourceKind.VISUAL_EVIDENCE,
                source_id=f"FINAL-{check.shot_number}-{check.entity_id}",
                materiality=FactMateriality.MATERIAL,
            )
        )


# ---------------------------------------------------------------------------
# Seed claims (§13/§14) — supported wording may be reused, atomically
# ---------------------------------------------------------------------------

_CLAIM_FACT_TYPES: dict[SeedClaimType, CaptionFactType] = {
    SeedClaimType.CHARACTER_EXISTS: CaptionFactType.CHARACTER,
    SeedClaimType.CHARACTER_TRAIT: CaptionFactType.CHARACTER,
    SeedClaimType.CHARACTER_VISIBILITY: CaptionFactType.CHARACTER,
    SeedClaimType.CHARACTER_POSITION: CaptionFactType.SCENE,
    SeedClaimType.OBJECT_EXISTS: CaptionFactType.OBJECT,
    SeedClaimType.OBJECT_TRAIT: CaptionFactType.OBJECT,
    SeedClaimType.SCENE_STATE: CaptionFactType.SCENE,
    SeedClaimType.STYLE_STATE: CaptionFactType.STYLE,
    SeedClaimType.CAMERA_FRAMING: CaptionFactType.CAMERA_FRAMING,
    SeedClaimType.ACTION: CaptionFactType.VISUAL_ACTION,
    SeedClaimType.SOUND: CaptionFactType.SOUND,
    SeedClaimType.VISUAL_CONCERN: CaptionFactType.VISUAL_CONCERN,
    SeedClaimType.AUDIO_CONCERN: CaptionFactType.AUDIO_CONCERN,
    SeedClaimType.PROTECTED_TRAIT: CaptionFactType.CHARACTER,
}

#: Claim types whose FINAL truth lives in a dedicated engine, so the seed copy
#: is never an independent fact source (it would duplicate or contradict).
_ENGINE_OWNED = frozenset(
    {
        SeedClaimType.MEDIA_ID,
        SeedClaimType.SHOT_COUNT,
        SeedClaimType.SHOT_BOUNDARY,
        SeedClaimType.TRANSITION,
        SeedClaimType.PLAYBACK_SPEED,
        SeedClaimType.SPEECH,
        SeedClaimType.ON_SCREEN_TEXT,
        SeedClaimType.CAMERA_MOVEMENT,
        SeedClaimType.OBJECT_IDENTITY,
        SeedClaimType.OBJECT_OWNERSHIP,
        SeedClaimType.OBJECT_CONTACT,
        SeedClaimType.UNCLASSIFIED,
    }
)


def _seed_fact_type(claim: SeedClaim) -> CaptionFactType | None:
    if claim.claim_type is None or claim.claim_type in _ENGINE_OWNED:
        return None
    mapped = _CLAIM_FACT_TYPES.get(claim.claim_type)
    if mapped is None:
        return None
    if (
        claim.claim_type == SeedClaimType.SOUND
        and claim.shot_number is None
        and "AUDIO" in claim.source_field.upper()
    ):
        return CaptionFactType.OVERVIEW_AUDIO
    return mapped


def _build_seed_claim_facts(
    graph: FactGraph, counter: _Counter, inputs: FactBuildInputs
) -> None:
    for claim in inputs.seed_claims:
        fact_type = _seed_fact_type(claim)
        if fact_type is None:
            continue
        status, basis, reason = elig.assess_seed_claim(
            claim.claim_type if claim.claim_type is not None else SeedClaimType.UNCLASSIFIED,
            claim.evidence_status,
            claim.review_status,
            claim.evidence,
        )
        # Seed ACTION wording additionally needs verified action semantics —
        # a supported *description* is not a verified *event boundary* (§37).
        if (
            fact_type == CaptionFactType.VISUAL_ACTION
            and status == CaptionEligibility.ELIGIBLE
            and basis == EligibilityBasis.DETERMINISTIC_EVIDENCE
        ):
            status = CaptionEligibility.REVIEW_REQUIRED
            basis = None
            reason = (
                "seed action wording needs human ACTION_SEMANTICS verification "
                "and an exact verified boundary"
            )
        seed_start = claim.seed_time_range.start_seconds if claim.seed_time_range else None
        seed_end = claim.seed_time_range.end_seconds if claim.seed_time_range else None
        graph.facts.append(
            CaptionFact(
                fact_id=counter.next(),
                fact_type=fact_type,
                shot_number=claim.shot_number,
                character_ids=claim.subject_ids,
                object_ids=claim.object_ids,
                # Seed timestamps are NEVER timing authority (§49): kept only
                # as semantic context, not as start/end_exact.
                semantic_value={
                    "claim_type": claim.claim_type.value if claim.claim_type else "",
                    **({"seed_start": str(seed_start)} if seed_start is not None else {}),
                    **({"seed_end": str(seed_end)} if seed_end is not None else {}),
                },
                text_value=claim.text,
                eligibility=status,
                eligibility_basis=basis,
                eligibility_reason=reason,
                source_kind=FactSourceKind.SEED_SUPPORTED,
                source_id=claim.claim_id,
                required_for_caption=(
                    status == CaptionEligibility.ELIGIBLE
                    and fact_type
                    in (
                        CaptionFactType.CHARACTER,
                        CaptionFactType.OBJECT,
                        CaptionFactType.SCENE,
                    )
                ),
                materiality=(
                    FactMateriality.REQUIRED
                    if fact_type in (CaptionFactType.CHARACTER, CaptionFactType.OBJECT)
                    else FactMateriality.MATERIAL
                ),
            )
        )


# ---------------------------------------------------------------------------
# Human-added facts (§9)
# ---------------------------------------------------------------------------


def _build_human_facts(graph: FactGraph, counter: _Counter, inputs: FactBuildInputs) -> None:
    for hf in inputs.human_facts:
        if hf.semantic_value.get("region_id"):
            continue  # enrichment records merge into their machine region fact
        status, basis, reason = elig.assess_human_fact()
        semantic = dict(hf.semantic_value)
        semantic.pop("splits_region_id", None)
        if hf.fact_type == CaptionFactType.SPEECH:
            semantic.setdefault("language_level", LanguageRenderLevel.VERBATIM.value)
            if hf.character_ids:
                semantic.setdefault("speaker_id", hf.character_ids[0])
        graph.facts.append(
            CaptionFact(
                fact_id=counter.next(),
                fact_type=hf.fact_type,
                shot_number=hf.shot_number,
                character_ids=hf.character_ids,
                object_ids=hf.object_ids,
                start_frame=hf.start_frame,
                end_frame=hf.end_frame,
                start_exact=hf.start_exact,
                end_exact=hf.end_exact,
                display_start=_disp(hf.start_exact),
                display_end=_disp(hf.end_exact),
                text_value=hf.text_value,
                semantic_value=semantic,
                evidence_refs=hf.evidence_refs,
                eligibility=status,
                eligibility_basis=basis,
                eligibility_reason=reason,
                source_kind=FactSourceKind.HUMAN_FACT,
                source_id=hf.fact_id,
                human_fact_id=hf.fact_id,
                required_for_caption=True,
                materiality=FactMateriality.REQUIRED,
            )
        )
