"""Phase 5 — Caption Brain data model.

Turns reviewed Phase 1-4 evidence into a complete Manuscript II caption while
preserving the evidence-first safety model. The central rule is structural:

    CANDIDATE != FINAL FACT

Every final caption fact carries an explicit :class:`CaptionEligibility` plus an
:class:`EligibilityBasis` naming WHY it may become final caption text. A machine
candidate (REGULAR_CANDIDATE, OBJECT_PICKUP_CANDIDATE, machine OCR/ASR text,
ambiguous identity, unresolved camera semantics...) can never silently become a
final fact — eligibility is resolved by ONE central policy
(:mod:`..caption.eligibility`), never scattered across builders.

Two deliberately distinct vocabularies:

* ``EvidenceStatus`` (Phase 4) answers "what does the evidence say?"
* :class:`CaptionEligibility` answers "may this become final caption text?"

Layering: imports only :mod:`.common`, :mod:`.evidence` and enum-only names
from the phase models. No caption prose is stored here except as explicit,
eligibility-gated fact values.
"""

from __future__ import annotations

from enum import StrEnum

from .common import ExactFraction, StrictModel
from .evidence import EvidenceReference

# ===========================================================================
# Eligibility
# ===========================================================================


class CaptionEligibility(StrEnum):
    """May this fact become final caption text? (NOT EvidenceStatus.)"""

    ELIGIBLE = "ELIGIBLE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    INELIGIBLE = "INELIGIBLE"
    REJECTED = "REJECTED"


class EligibilityBasis(StrEnum):
    """The explicit basis a final fact must name. A mutated machine enum alone
    is never a basis — provenance is inspected (§4)."""

    DETERMINISTIC_EVIDENCE = "DETERMINISTIC_EVIDENCE"
    APPLIED_HUMAN_DECISION = "APPLIED_HUMAN_DECISION"
    HUMAN_VERIFICATION_EVIDENCE = "HUMAN_VERIFICATION_EVIDENCE"
    HUMAN_ADDED_FACT = "HUMAN_ADDED_FACT"
    CONTROLLING_SOURCE_RULE = "CONTROLLING_SOURCE_RULE"


class FactSourceKind(StrEnum):
    """Where a caption fact came from."""

    MEDIA_TRUTH = "MEDIA_TRUTH"
    SHOT_TRUTH = "SHOT_TRUTH"
    AUDIO_TRUTH = "AUDIO_TRUTH"
    VISUAL_EVIDENCE = "VISUAL_EVIDENCE"
    SEED_SUPPORTED = "SEED_SUPPORTED"
    HUMAN_DECISION = "HUMAN_DECISION"
    HUMAN_FACT = "HUMAN_FACT"
    TASK_FEEDBACK = "TASK_FEEDBACK"


class FactMateriality(StrEnum):
    """Coverage weight: REQUIRED facts must appear (or carry a valid omission
    reason); MINOR facts may be omitted with a reason."""

    REQUIRED = "REQUIRED"
    MATERIAL = "MATERIAL"
    MINOR = "MINOR"


class CaptionFactType(StrEnum):
    MEDIA = "MEDIA"
    CHARACTER = "CHARACTER"
    OBJECT = "OBJECT"
    SCENE = "SCENE"
    STYLE = "STYLE"
    OVERVIEW_AUDIO = "OVERVIEW_AUDIO"
    VISUAL_CONCERN = "VISUAL_CONCERN"
    AUDIO_CONCERN = "AUDIO_CONCERN"
    SHOT_BOUNDARY = "SHOT_BOUNDARY"
    TRANSITION = "TRANSITION"
    CAMERA_FRAMING = "CAMERA_FRAMING"
    CAMERA_MOVEMENT = "CAMERA_MOVEMENT"
    VISUAL_ACTION = "VISUAL_ACTION"
    SPEECH = "SPEECH"
    SOUND = "SOUND"
    ON_SCREEN_TEXT = "ON_SCREEN_TEXT"
    PLAYBACK_SPEED = "PLAYBACK_SPEED"
    SPEED_CHANGE = "SPEED_CHANGE"
    FINAL_OBJECT_STATE = "FINAL_OBJECT_STATE"


class LanguageRenderLevel(StrEnum):
    """The Manuscript II language ladder (§30). Uncertainty is represented
    structurally BEFORE rendering — never as hedges inside caption prose."""

    VERBATIM = "VERBATIM"
    NAMED_LANGUAGE = "NAMED_LANGUAGE"
    LANGUAGE_FAMILY = "LANGUAGE_FAMILY"
    FOREIGN_LANGUAGE = "FOREIGN_LANGUAGE"
    INDISCERNIBLE = "INDISCERNIBLE"


class CaptionFact(StrictModel):
    """One typed, eligibility-gated caption fact. Prose starts from facts,
    never the other way around (§8)."""

    fact_id: str
    fact_type: CaptionFactType
    shot_number: int | None = None
    character_ids: list[str] = []
    object_ids: list[str] = []
    start_frame: int | None = None
    end_frame: int | None = None
    #: Exact annotation-clock interval (authoritative timing).
    start_exact: ExactFraction | None = None
    end_exact: ExactFraction | None = None
    #: Canonical 0.1 s display projections ("4.3s"), derived via AnnotationClock
    #: + to_manuscript_display only. Display never flows back into timing.
    display_start: str | None = None
    display_end: str | None = None
    #: Prose / verbatim value (speech text, action sentence, description...).
    text_value: str | None = None
    #: Structured values (transition type, speed value, speaker id, tone,
    #: language level...). Strings only, admissible per fact type.
    semantic_value: dict[str, str] = {}
    evidence_refs: list[EvidenceReference] = []
    eligibility: CaptionEligibility
    eligibility_basis: EligibilityBasis | None = None
    eligibility_reason: str
    source_kind: FactSourceKind
    #: The originating record's own id (region_id, candidate_id, claim_id...).
    source_id: str | None = None
    human_decision_ids: list[str] = []
    human_fact_id: str | None = None
    required_for_caption: bool = False
    materiality: FactMateriality = FactMateriality.MATERIAL
    notes: list[str] = []


class HumanCaptionFact(StrictModel):
    """A reviewer-added fact that was completely missing from the seed (§9).
    Machine code must NEVER create one; they are loaded only from a
    human-supplied file and are bound to the exact video + rules version."""

    fact_id: str
    fact_type: CaptionFactType
    text_value: str | None = None
    semantic_value: dict[str, str] = {}
    shot_number: int | None = None
    character_ids: list[str] = []
    object_ids: list[str] = []
    start_frame: int | None = None
    end_frame: int | None = None
    start_exact: ExactFraction | None = None
    end_exact: ExactFraction | None = None
    evidence_refs: list[EvidenceReference] = []
    decided_by: str
    decided_at_utc: str | None = None
    bound_video_sha256: str
    bound_rules_version: str
    bound_seed_sha256: str | None = None
    bound_evidence_ids: list[str] = []


# ===========================================================================
# Caption plan
# ===========================================================================


class CharacterPlanEntry(StrictModel):
    character_id: str
    description_fact_ids: list[str] = []
    off_screen_only: bool = False


class ObjectPlanEntry(StrictModel):
    object_id: str
    description_fact_ids: list[str] = []
    final_state_fact_ids: list[str] = []


class OverviewPlan(StrictModel):
    characters: list[CharacterPlanEntry] = []
    objects: list[ObjectPlanEntry] = []
    scene_fact_ids: list[str] = []
    style_fact_ids: list[str] = []
    overview_audio_fact_ids: list[str] = []
    visual_concern_fact_ids: list[str] = []
    audio_concern_fact_ids: list[str] = []


class ShotPlan(StrictModel):
    shot_number: int
    start_exact: ExactFraction
    end_exact: ExactFraction
    display_start: str
    display_end: str
    transition_fact_id: str | None = None
    #: True only when the transition is caption-eligible; an unresolved
    #: transition is NEVER defaulted (no automatic "Hard cut").
    transition_resolved: bool = False
    camera_framing_fact_ids: list[str] = []
    camera_movement_fact_ids: list[str] = []
    scene_fact_ids: list[str] = []
    #: The ordered Action & Audio event union: verified visual actions, speech,
    #: sound, on-screen text. Camera movement is structurally excluded.
    event_fact_ids: list[str] = []
    playback_speed_fact_id: str | None = None
    playback_speed_resolved: bool = False
    speed_change_fact_ids: list[str] = []


class SeedChangeEntry(StrictModel):
    """Final reviewer disposition for one seeded section/subject (§13/§58)."""

    section: str  # overview | cast | objects | scene | style | audio | shot-N | ...
    subject_id: str | None = None
    disposition: str  # KEEP | FIX_ENRICH | REDO_REBUILD
    decided_by_human: bool = False
    reasons: list[str] = []


class CaptionReadiness(StrEnum):
    BLOCKED = "BLOCKED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    READY_FOR_FINAL_REVIEW = "READY_FOR_FINAL_REVIEW"
    READY_TO_ENTER = "READY_TO_ENTER"


class CaptionPlan(StrictModel):
    """Complete structured plan built BEFORE any caption text (§12). The
    renderer consumes this plan and never rediscovers facts."""

    video_id: str | None = None
    video_sha256: str | None = None
    rules_version: str | None = None
    overview_plan: OverviewPlan
    shot_plans: list[ShotPlan] = []
    eligible_fact_ids: list[str] = []
    omitted_fact_ids: list[str] = []
    blocked_fact_ids: list[str] = []
    review_required_fact_ids: list[str] = []
    seed_change_summary: list[SeedChangeEntry] = []
    #: Provisional readiness from planning alone (validators may lower it).
    readiness: CaptionReadiness = CaptionReadiness.REVIEW_REQUIRED
    blockers: list[str] = []


# ===========================================================================
# Rendered caption + assertion map + coverage
# ===========================================================================


class CaptionSection(StrEnum):
    VIDEO_ID = "VIDEO_ID"
    CAST = "CAST"
    OBJECTS = "OBJECTS"
    SCENE = "SCENE"
    STYLE = "STYLE"
    OVERVIEW_AUDIO = "OVERVIEW_AUDIO"
    VISUAL_CONCERNS = "VISUAL_CONCERNS"
    AUDIO_CONCERNS = "AUDIO_CONCERNS"
    SHOT_HEADER = "SHOT_HEADER"
    CUT = "CUT"
    CAMERA = "CAMERA"
    CAMERA_MOVEMENTS = "CAMERA_MOVEMENTS"
    SHOT_SCENE = "SHOT_SCENE"
    ACTION_AUDIO = "ACTION_AUDIO"
    PLAYBACK_SPEED = "PLAYBACK_SPEED"
    SPEED_CHANGES = "SPEED_CHANGES"


class RenderedCaptionLine(StrictModel):
    """One rendered caption line. Every factual assertion maps back to
    CaptionFact ids — a line with no fact source is a hallucination (§60)."""

    line_id: str
    section: CaptionSection
    shot_number: int | None = None
    text: str
    display_start: str | None = None
    display_end: str | None = None
    start_exact: ExactFraction | None = None
    end_exact: ExactFraction | None = None
    fact_ids: list[str] = []
    #: True for pure structural scaffolding ("None." literals, headers) that
    #: asserts no media fact.
    structural: bool = False


class RenderedCaption(StrictModel):
    video_id: str | None = None
    lines: list[RenderedCaptionLine] = []
    #: The full caption exactly as written to disk; hashed for signoff binding.
    markdown: str = ""
    caption_sha256: str = ""


class AssertionStatus(StrEnum):
    MAPPED = "MAPPED"
    UNMAPPED = "UNMAPPED"


class CaptionAssertionRecord(StrictModel):
    """Reverse hallucination check: one factual assertion in one final line and
    the eligible facts that prove it (§60)."""

    assertion_id: str
    line_id: str
    assertion_text: str
    fact_ids: list[str] = []
    status: AssertionStatus


class OmissionReason(StrEnum):
    REDUNDANT_STABLE_FACT = "REDUNDANT_STABLE_FACT"
    NON_MATERIAL_SIGNAL = "NON_MATERIAL_SIGNAL"
    DUPLICATE_EVIDENCE_OF_SAME_FACT = "DUPLICATE_EVIDENCE_OF_SAME_FACT"


class CaptionCoverageItem(StrictModel):
    """One row of the coverage/omission ledger (§59). "The writer forgot it"
    is never a valid omission."""

    fact_id: str
    material: bool
    required: bool
    represented: bool
    caption_line_id: str | None = None
    omission_reason: OmissionReason | None = None


# ===========================================================================
# Final review signoff + status
# ===========================================================================


class FinalReviewSignoff(StrictModel):
    """Bound manual final-review signoff (§54). Machine code must NEVER
    fabricate this record; it is loaded only from a human-supplied file."""

    video_sha256: str
    rules_version: str
    caption_sha256: str
    reviewer: str
    reviewed_at_utc: str | None = None
    golden_example_comparison_complete: bool = False
    platform_semantic_pass_complete: bool = False
    final_adversarial_read_complete: bool = False
    no_known_omissions_confirmed: bool = False
    no_known_hallucinations_confirmed: bool = False
    comments: str | None = None


class GoldenGateStatus(StrEnum):
    PASS = "PASS"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    FAIL = "FAIL"


class GoldenGateCategoryResult(StrictModel):
    category: str
    status: GoldenGateStatus
    detail: str | None = None
    rule_ids: list[str] = []


class GoldenGateResult(StrictModel):
    rules_file: str
    rules_version: str
    overall: GoldenGateStatus
    categories: list[GoldenGateCategoryResult] = []


class AdversarialFinding(StrictModel):
    """One final-adversarial-QC finding: a sentence whose evidence chain was
    challenged (§87)."""

    finding_id: str
    line_id: str | None = None
    category: str
    detail: str
    blocking: bool = False


class CaptionBrainResult(StrictModel):
    """caption/final_status.json payload: the full Phase 5 outcome."""

    caption_brain_version: str
    video_id: str | None = None
    fact_count: int = 0
    eligible_count: int = 0
    review_required_count: int = 0
    ineligible_count: int = 0
    rejected_count: int = 0
    shot_count: int = 0
    character_count: int = 0
    object_count: int = 0
    speech_verified_count: int = 0
    speech_blocked_count: int = 0
    text_verified_count: int = 0
    text_blocked_count: int = 0
    action_verified_count: int = 0
    action_blocked_count: int = 0
    m2_fail_count: int = 0
    m2_review_count: int = 0
    platform_semantic_status: str = "NOT_RUN"
    golden_gate_status: str = "NOT_RUN"
    coverage_missing_required: int = 0
    unmapped_assertions: int = 0
    unresolved_feedback_high: int = 0
    signoff_present: bool = False
    signoff_stale: bool = False
    readiness: CaptionReadiness = CaptionReadiness.REVIEW_REQUIRED
    blockers: list[str] = []
    stage_timings_seconds: dict[str, float] = {}
