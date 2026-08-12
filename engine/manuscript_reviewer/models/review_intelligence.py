"""Phase 4 — Visual Review Intelligence data model.

This module defines the structured, evidence-first types produced by Phase 4:
the parsed seed, atomic seed claims, machine visual/audio-derived observations,
and reviewer *proposals* (never final decisions, never final caption prose).

Layering: this module imports only from :mod:`.common` and :mod:`.evidence`.
It deliberately does NOT import from :mod:`.caption` so that ``caption.py`` can
import the Phase-4 enums it needs (to extend ``SeedClaim``) without a cycle.

Three vocabularies are kept strictly separate and must never be collapsed:

* :class:`EvidenceStatus`   — what the media proves about a claim.
* :class:`ReviewProposalOutcome` — the machine's *proposal* for a section.
* the human ``ReviewDecision`` (in :mod:`.caption`) — an actual human decision.

Machine systems may propose. Evidence determines timing. Human verification
determines final uncertain semantics. AI/CV never owns identity, the clock,
text truth, semantic action truth, or the final review decision.
"""

from __future__ import annotations

from enum import StrEnum

from .common import ExactFraction, StrictModel
from .evidence import EvidenceReference

# ===========================================================================
# Enums
# ===========================================================================


class SeedSectionKind(StrEnum):
    """Recognized top-level seed sections (unrecognized text is UNKNOWN, never
    discarded)."""

    PREAMBLE = "PREAMBLE"
    VIDEO_ID = "VIDEO_ID"
    OVERVIEW = "OVERVIEW"
    CAST = "CAST"
    OBJECTS = "OBJECTS"
    SCENE = "SCENE"
    STYLE = "STYLE"
    AUDIO = "AUDIO"
    VISUAL_CONCERNS = "VISUAL_CONCERNS"
    AUDIO_CONCERNS = "AUDIO_CONCERNS"
    SHOT = "SHOT"
    UNKNOWN = "UNKNOWN"


class SeedFieldKind(StrEnum):
    """Recognized per-shot / per-entry field labels."""

    SHOT_START = "SHOT_START"
    SHOT_END = "SHOT_END"
    TRANSITION = "TRANSITION"
    CAMERA = "CAMERA"
    CAMERA_MOVEMENTS = "CAMERA_MOVEMENTS"
    SCENE = "SCENE"
    ACTION_AUDIO = "ACTION_AUDIO"
    PLAYBACK_SPEED = "PLAYBACK_SPEED"
    SPEED_CHANGES = "SPEED_CHANGES"
    FREEFORM = "FREEFORM"


class SeedParseSeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class SeedClaimType(StrEnum):
    """Atomic claim categories. A paragraph is never one giant claim."""

    MEDIA_ID = "MEDIA_ID"
    SHOT_COUNT = "SHOT_COUNT"
    SHOT_BOUNDARY = "SHOT_BOUNDARY"
    TRANSITION = "TRANSITION"
    CHARACTER_EXISTS = "CHARACTER_EXISTS"
    CHARACTER_TRAIT = "CHARACTER_TRAIT"
    CHARACTER_VISIBILITY = "CHARACTER_VISIBILITY"
    CHARACTER_POSITION = "CHARACTER_POSITION"
    OBJECT_EXISTS = "OBJECT_EXISTS"
    OBJECT_TRAIT = "OBJECT_TRAIT"
    OBJECT_IDENTITY = "OBJECT_IDENTITY"
    OBJECT_OWNERSHIP = "OBJECT_OWNERSHIP"
    OBJECT_CONTACT = "OBJECT_CONTACT"
    SCENE_STATE = "SCENE_STATE"
    STYLE_STATE = "STYLE_STATE"
    CAMERA_FRAMING = "CAMERA_FRAMING"
    CAMERA_MOVEMENT = "CAMERA_MOVEMENT"
    ACTION = "ACTION"
    SPEECH = "SPEECH"
    SOUND = "SOUND"
    ON_SCREEN_TEXT = "ON_SCREEN_TEXT"
    PLAYBACK_SPEED = "PLAYBACK_SPEED"
    VISUAL_CONCERN = "VISUAL_CONCERN"
    AUDIO_CONCERN = "AUDIO_CONCERN"
    #: Protected/unsupported traits (nationality, ethnicity, race, exact age,
    #: gender, accent) — captured so they can be flagged unsupported, never
    #: visually inferred.
    PROTECTED_TRAIT = "PROTECTED_TRAIT"
    UNCLASSIFIED = "UNCLASSIFIED"


class EvidenceStatus(StrEnum):
    """What the media proves about a claim. Distinct from any review outcome."""

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNRESOLVED = "UNRESOLVED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ClaimImportance(StrEnum):
    """Structural importance of a claim; drives rebuild-vs-patch proposals."""

    FOUNDATIONAL = "FOUNDATIONAL"
    LOCAL = "LOCAL"


class ClaimReviewStatus(StrEnum):
    MACHINE_ONLY = "MACHINE_ONLY"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    HUMAN_RESOLVED = "HUMAN_RESOLVED"


class ReviewProposalOutcome(StrEnum):
    """A machine *proposal* for a section — never a human decision. Adds
    HUMAN_DECISION_REQUIRED, which the human ``ReviewOutcome`` deliberately
    lacks."""

    KEEP = "KEEP"
    FIX_ENRICH = "FIX_ENRICH"
    REDO_REBUILD = "REDO_REBUILD"
    HUMAN_DECISION_REQUIRED = "HUMAN_DECISION_REQUIRED"


class ProposalReasonCode(StrEnum):
    """Structured reasoning for a proposal (reasoning is never hidden in prose)."""

    # Foundation / structure
    FOUNDATION_SUPPORTED = "FOUNDATION_SUPPORTED"
    SHOT_COUNT_CONTRADICTION = "SHOT_COUNT_CONTRADICTION"
    SHOT_BOUNDARY_FOUNDATION_INVALID = "SHOT_BOUNDARY_FOUNDATION_INVALID"
    MEDIA_IDENTITY_MISMATCH = "MEDIA_IDENTITY_MISMATCH"
    TIMELINE_FOUNDATION_INVALID = "TIMELINE_FOUNDATION_INVALID"
    # Identity
    CHARACTER_IDENTITY_COLLISION = "CHARACTER_IDENTITY_COLLISION"
    CHARACTER_TRACK_SPLIT = "CHARACTER_TRACK_SPLIT"
    OBJECT_IDENTITY_COLLISION = "OBJECT_IDENTITY_COLLISION"
    OBJECT_DUPLICATE_ID = "OBJECT_DUPLICATE_ID"
    OBJECT_OWNERSHIP_CONTRADICTION = "OBJECT_OWNERSHIP_CONTRADICTION"
    ENTITY_NOT_VISIBLE_AT_CLAIM_TIME = "ENTITY_NOT_VISIBLE_AT_CLAIM_TIME"
    # Local
    LOCAL_FACT_CONTRADICTED = "LOCAL_FACT_CONTRADICTED"
    LOCAL_FACT_WEAK = "LOCAL_FACT_WEAK"
    ENRICHMENT_NEEDED = "ENRICHMENT_NEEDED"
    MISSING_REQUIRED_EVIDENCE = "MISSING_REQUIRED_EVIDENCE"
    # Uncertainty
    UNRESOLVED_HIGH_RISK = "UNRESOLVED_HIGH_RISK"
    EVIDENCE_CANNOT_DECIDE = "EVIDENCE_CANNOT_DECIDE"
    TASK_FEEDBACK_UNRESOLVED = "TASK_FEEDBACK_UNRESOLVED"
    # Positive
    NO_MATERIAL_CONTRADICTION = "NO_MATERIAL_CONTRADICTION"


class ReviewPriority(StrEnum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"


class ReviewerAction(StrEnum):
    VERIFY = "VERIFY"
    CHOOSE_IDENTITY = "CHOOSE_IDENTITY"
    CORRECT_TEXT = "CORRECT_TEXT"
    CORRECT_BOUNDARY = "CORRECT_BOUNDARY"
    REJECT_CLAIM = "REJECT_CLAIM"
    CONFIRM_CLAIM = "CONFIRM_CLAIM"
    REBUILD_SECTION = "REBUILD_SECTION"


class TriageStrategy(StrEnum):
    PATCH = "PATCH"
    REBUILD = "REBUILD"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class FoundationStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNRESOLVED = "UNRESOLVED"
    NOT_EVALUATED = "NOT_EVALUATED"


class FeedbackPriority(StrEnum):
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"


class InterpretationStatus(StrEnum):
    MAPPED = "MAPPED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    UNINTERPRETED = "UNINTERPRETED"


class SourceTextVerificationStatus(StrEnum):
    """Mirrors Phase 3 speech safety for on-screen text (OCR)."""

    UNVERIFIED = "UNVERIFIED"
    HUMAN_VERIFIED = "HUMAN_VERIFIED"
    HUMAN_CORRECTED = "HUMAN_CORRECTED"
    REJECTED = "REJECTED"


# ===========================================================================
# Seed document (immutable snapshot -> structured, recoverable parse)
# ===========================================================================


class SeedSnapshot(StrictModel):
    """Immutable record of the exact seed bytes copied into the run."""

    original_path: str
    stored_relative_path: str  # seed/seed_original.txt
    sha256: str
    byte_count: int
    line_count: int


class SeedParseIssue(StrictModel):
    """A recoverable parse problem. Malformed content is preserved, never
    discarded, so a reviewer can still see and fix it."""

    issue_id: str
    source_line: int | None
    raw_text: str
    message: str
    severity: SeedParseSeverity = SeedParseSeverity.WARNING


class SeedEntry(StrictModel):
    """One parsed seed line/field. The raw source is always retained."""

    entry_id: str
    section_index: int
    source_line: int
    raw_line: str
    field: SeedFieldKind = SeedFieldKind.FREEFORM
    field_label: str | None = None  # the literal label text as written
    shot_number: int | None = None
    value_text: str | None = None
    timestamp_text: str | None = None
    #: Parsed exact range, only when the timestamp text parsed cleanly.
    parsed_start_exact: ExactFraction | None = None
    parsed_end_exact: ExactFraction | None = None
    referenced_character_ids: list[str] = []
    referenced_object_ids: list[str] = []
    quoted_strings: list[str] = []


class SeedSection(StrictModel):
    section_index: int
    kind: SeedSectionKind
    heading_text: str | None = None
    source_line: int
    shot_number: int | None = None  # for SHOT sections
    entries: list[SeedEntry] = []


class SeedDocument(StrictModel):
    """The structured, recoverable parse. The raw snapshot remains authoritative."""

    snapshot: SeedSnapshot | None = None
    video_id: str | None = None
    raw_line_count: int
    sections: list[SeedSection] = []
    issues: list[SeedParseIssue] = []

    @property
    def shot_sections(self) -> list[SeedSection]:
        return [s for s in self.sections if s.kind == SeedSectionKind.SHOT]

    @property
    def seed_shot_count(self) -> int | None:
        shots = {s.shot_number for s in self.shot_sections if s.shot_number is not None}
        return len(shots) if shots else None


# ===========================================================================
# Task-specific feedback
# ===========================================================================


class FeedbackSnapshot(StrictModel):
    original_path: str
    stored_relative_path: str  # feedback/feedback_original.txt
    sha256: str
    byte_count: int
    line_count: int


class FeedbackDirective(StrictModel):
    """One evaluator/task-feedback item. The raw text is always preserved;
    only exact known patterns map to a machine interpretation."""

    directive_id: str
    raw_text: str
    source_line: int | None
    priority: FeedbackPriority = FeedbackPriority.NORMAL
    #: A known check code (e.g. REQUIRE_VOCAL_LYRIC_REVIEW) only for exact
    #: recognized patterns; otherwise None.
    machine_interpretation: str | None = None
    interpretation_status: InterpretationStatus = InterpretationStatus.UNINTERPRETED
    review_required: bool = True


class FeedbackDocument(StrictModel):
    snapshot: FeedbackSnapshot | None = None
    directives: list[FeedbackDirective] = []


# ===========================================================================
# Claim <-> evidence matrix
# ===========================================================================


class ClaimEvidenceRow(StrictModel):
    """One row of the claim <-> evidence matrix. Never SUPPORTED with zero
    evidence (validator P4-CLAIM-001)."""

    claim_id: str
    claim_type: SeedClaimType
    seed_text: str
    seed_shot: int | None = None
    seed_source_line: int | None = None
    seed_start_exact: ExactFraction | None = None
    seed_end_exact: ExactFraction | None = None
    evidence_status: EvidenceStatus
    importance: ClaimImportance
    foundational: bool
    supporting_evidence_refs: list[EvidenceReference] = []
    contradicting_evidence_refs: list[EvidenceReference] = []
    unresolved_reasons: list[str] = []
    review_status: ClaimReviewStatus = ClaimReviewStatus.MACHINE_ONLY
    review_proposal: ReviewProposalOutcome | None = None
    notes: list[str] = []


# ===========================================================================
# Structural foundation checks (zero new CV: uses shot/audio/frame truth)
# ===========================================================================


class FoundationCheck(StrictModel):
    """A high-value structural verification of the seed against deterministic
    Phase 1-3 truth."""

    check_id: str
    subject: str  # e.g. "shot_structure", "media_identity", "C1"
    status: FoundationStatus
    seed_value: str | None = None
    verified_value: str | None = None
    reason_codes: list[ProposalReasonCode] = []
    evidence_refs: list[EvidenceReference] = []
    detail: str | None = None


# ===========================================================================
# Review proposals / queue / decisions / triage
# ===========================================================================


class ReviewProposal(StrictModel):
    """A machine PROPOSAL about a section/claim. NOT a human ReviewDecision.

    Built at several levels via ``level`` (claim/field/character/object/shot/
    Overview/seed)."""

    proposal_id: str
    level: str  # claim | field | character | object | shot | overview | seed
    subject_id: str
    outcome: ReviewProposalOutcome
    reason_codes: list[ProposalReasonCode] = []
    foundational: bool = False
    claim_ids: list[str] = []
    evidence_refs: list[EvidenceReference] = []
    detail: str | None = None
    #: Always machine — a proposal can never be a human decision (P4-REVIEW-001).
    proposed_by: str = "machine"


class ReviewQueueItem(StrictModel):
    """One item in the visual review queue. Says what/why/where + recommended
    reviewer action, with supporting AND contradicting evidence."""

    item_id: str
    priority: ReviewPriority
    title: str
    reason: str
    shot_number: int | None = None
    start_frame: int | None = None
    end_frame: int | None = None
    start_exact: ExactFraction | None = None
    end_exact: ExactFraction | None = None
    supporting_evidence_refs: list[EvidenceReference] = []
    contradicting_evidence_refs: list[EvidenceReference] = []
    recommended_action: ReviewerAction
    related_claim_ids: list[str] = []
    evidence_bundle_dir: str | None = None


class SeedTriage(StrictModel):
    """The early patch-vs-rebuild recommendation ("minute-8 triage"). An early
    recommendation only — it never prevents later correction."""

    triage_runtime_seconds: float
    foundation_status: FoundationStatus
    foundational_conflicts: list[str] = []
    suggested_strategy: TriageStrategy
    reason_codes: list[ProposalReasonCode] = []


# ===========================================================================
# Visual frame observations & entity continuity (populated by later slices)
# ===========================================================================


class DetectedTextRegion(StrictModel):
    region_id: str
    x: int
    y: int
    width: int
    height: int
    contrast: float
    stable: bool = False


class FrameObservation(StrictModel):
    """One machine observation record per source video frame. Not caption prose."""

    frame_index: int
    source_pts: int | None = None
    source_pts_time_exact: ExactFraction | None = None
    annotation_time_exact: ExactFraction | None = None
    shot_number: int | None = None
    brightness: float
    contrast: float
    sharpness: float
    motion_magnitude: float
    global_camera_motion: float
    #: Raw inter-frame difference motion (NOT a global-compensated residual).
    #: Renamed from the misleading "foreground_motion"; the action engine
    #: computes its own residual rather than trusting this as foreground-only.
    raw_interframe_motion: float
    text_region_count: int = 0
    active_track_ids: list[str] = []
    occluded_track_ids: list[str] = []
    entry_candidate_ids: list[str] = []
    exit_candidate_ids: list[str] = []
    contact_state_ids: list[str] = []
    visual_concern_codes: list[str] = []
    notes: list[str] = []


class CropArtifact(StrictModel):
    """An exact crop from one source frame. Never carries a burned-in semantic
    label."""

    crop_id: str
    frame_index: int
    source_pts_time_exact: ExactFraction | None = None
    annotation_time_exact: ExactFraction | None = None
    x: int
    y: int
    width: int
    height: int
    crop_type: str  # face | hand | object | logo | overlay_text | watermark | ...
    subject_id: str | None = None
    reason: str | None = None
    artifact_path: str | None = None


class TrackStatus(StrEnum):
    TRACKED = "TRACKED"
    OCCLUDED = "OCCLUDED"
    LOST = "LOST"
    REACQUIRED = "REACQUIRED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class TrackObservation(StrictModel):
    frame_index: int
    x: int
    y: int
    width: int
    height: int
    status: TrackStatus = TrackStatus.TRACKED
    appearance_similarity: float | None = None
    notes: list[str] = []


class EntityTrack(StrictModel):
    """An explainable local track. A similarity score is NOT identity truth."""

    track_id: str
    entity_type: str  # CHARACTER | OBJECT | UNKNOWN
    anchor_frame_index: int | None = None
    first_frame_index: int
    last_frame_index: int
    shot_numbers: list[int] = []
    observations: list[TrackObservation] = []
    status: TrackStatus = TrackStatus.REVIEW_REQUIRED
    reacquired: bool = False
    evidence_refs: list[EvidenceReference] = []
    notes: list[str] = []


class IdentityEvidence(StrictModel):
    kind: str  # bbox_continuity | trajectory | appearance | clothing | held_object | ...
    detail: str
    supports: bool  # True = supports identity, False = contradicts
    evidence_refs: list[EvidenceReference] = []


class CharacterHypothesis(StrictModel):
    """A proposed character. C IDs are proposed in verified first-appearance
    order — never assigned merely because a detector says "person"."""

    hypothesis_id: str  # provisional "H-C1"; not a final C-label
    proposed_label: str | None = None  # "C1" candidate, review_required
    track_ids: list[str] = []
    first_visible_frame: int | None = None
    last_visible_frame: int | None = None
    shot_appearances: list[int] = []
    identity_evidence: list[IdentityEvidence] = []
    review_required: bool = True
    is_crowd_aggregate: bool = False
    notes: list[str] = []


class ObjectStateKind(StrEnum):
    UNASSIGNED = "UNASSIGNED"
    HELD_BY_CHARACTER = "HELD_BY_CHARACTER"
    IN_CONTACT_WITH_CHARACTER = "IN_CONTACT_WITH_CHARACTER"
    ON_SURFACE = "ON_SURFACE"
    MOVING_INDEPENDENTLY = "MOVING_INDEPENDENTLY"
    OCCLUDED = "OCCLUDED"
    OUT_OF_FRAME = "OUT_OF_FRAME"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class ObjectHypothesis(StrictModel):
    hypothesis_id: str  # provisional "H-O1"
    proposed_label: str | None = None
    track_ids: list[str] = []
    first_visible_frame: int | None = None
    last_visible_frame: int | None = None
    shot_appearances: list[int] = []
    identity_evidence: list[IdentityEvidence] = []
    review_required: bool = True
    notes: list[str] = []


class ContactEventKind(StrEnum):
    CONTACT_BEGINS = "CONTACT_BEGINS"
    CONTACT_ENDS = "CONTACT_ENDS"
    HELD_STATE_BEGINS = "HELD_STATE_BEGINS"
    HELD_STATE_ENDS = "HELD_STATE_ENDS"
    OBJECT_RELEASE = "OBJECT_RELEASE"
    OBJECT_PICKUP_CANDIDATE = "OBJECT_PICKUP_CANDIDATE"
    CONTACT_STATE_CHANGE = "CONTACT_STATE_CHANGE"


class ContactEvent(StrictModel):
    event_id: str
    kind: ContactEventKind
    shot_number: int | None = None
    frame_index: int
    object_track_id: str | None = None
    character_track_id: str | None = None
    pre_state: ObjectStateKind | None = None
    post_state: ObjectStateKind | None = None
    review_required: bool = True
    evidence_refs: list[EvidenceReference] = []


class ContinuityLink(StrictModel):
    link_id: str
    from_track_id: str
    to_track_id: str
    relationship: str  # SAME_ENTITY_CANDIDATE | REACQUISITION | DISTINCT_ENTITY
    review_required: bool = True
    evidence_refs: list[EvidenceReference] = []
    notes: list[str] = []


class FinalStateCheck(StrictModel):
    """Mandatory: for every important object in every shot, inspect the final
    frame interval. Never imply release/removal when evidence stops only because
    the shot ends."""

    shot_number: int
    entity_id: str
    final_visible_frame: int | None = None
    final_state: ObjectStateKind
    resolved: bool = False
    evidence_refs: list[EvidenceReference] = []
    review_reason: str | None = None


# ===========================================================================
# OCR / on-screen text
# ===========================================================================


class OCRStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    LANGUAGE_UNAVAILABLE = "LANGUAGE_UNAVAILABLE"
    #: OCR ran but a meaningful portion of frames failed to execute (M).
    DEGRADED = "DEGRADED"


class OCRFrameStatus(StrEnum):
    """Per-frame OCR execution accounting (M) — a failure is never silently
    converted into 'no text'."""

    OCR_PRESENT = "OCR_PRESENT"  # OCR ran on this frame (may or may not find text)
    OCR_NOT_READ = "OCR_NOT_READ"  # frame not submitted to OCR
    OCR_ENGINE_FAILED = "OCR_ENGINE_FAILED"  # OCR raised on this frame


class TextDisappearanceStatus(StrEnum):
    """Whether/why a text region stopped being read (K) — disappearance is never
    invented from 'last observation + 1'."""

    DISAPPEARANCE_SUPPORTED = "DISAPPEARANCE_SUPPORTED"
    PERSISTS_TO_END = "PERSISTS_TO_END"
    REGION_PRESENT_TEXT_UNREADABLE = "REGION_PRESENT_TEXT_UNREADABLE"
    UNRESOLVED = "UNRESOLVED"


class TextDetectionProvenance(StrEnum):
    REGION_DETECTOR = "REGION_DETECTOR"
    WHOLE_FRAME = "WHOLE_FRAME"


class OCREngineInfo(StrictModel):
    engine: str  # "tesseract" | "mock"
    status: OCRStatus
    version: str | None = None
    language_config: str | None = None
    failure_reason: str | None = None


class OCRObservation(StrictModel):
    """One OCR read from one exact frame. A single read never becomes final
    on-screen text evidence."""

    observation_id: str
    frame_index: int
    source_pts_time_exact: ExactFraction | None = None
    annotation_time_exact: ExactFraction | None = None
    x: int
    y: int
    width: int
    height: int
    raw_text: str
    confidence: float | None = None
    provenance: TextDetectionProvenance = TextDetectionProvenance.WHOLE_FRAME


class TextConsensus(StrictModel):
    consensus_text: str
    support_frames: int
    confidence: float | None = None


class TextTrack(StrictModel):
    """A temporally tracked on-screen text region. Machine OCR text is never
    caption-eligible until source verification."""

    track_id: str
    first_candidate_frame: int
    first_stable_frame: int | None = None
    fully_legible_frame: int | None = None
    last_stable_frame: int | None = None
    disappearance_frame: int | None = None
    disappearance_status: TextDisappearanceStatus = TextDisappearanceStatus.UNRESOLVED
    #: True only when the text is still present in the last inspected frame (K).
    text_persists_to_shot_end: bool = False
    #: Real stability accounting (L).
    consecutive_support_frames: int = 0
    total_support_frames: int = 0
    missed_frames: int = 0
    bbox_path: list[list[int]] = []  # per-frame [frame_index, x, y, w, h]
    observations: list[OCRObservation] = []
    consensus: TextConsensus | None = None
    verification_status: SourceTextVerificationStatus = SourceTextVerificationStatus.UNVERIFIED
    review_required: bool = True
    #: Future property: OCR text is not caption text until source verification.
    caption_text_eligible: bool = False
    evidence_refs: list[EvidenceReference] = []


class WatermarkCandidate(StrictModel):
    """Separate from ordinary overlay text. Persistent game UI is NOT a
    watermark merely because it is fixed."""

    candidate_id: str
    x: int
    y: int
    width: int
    height: int
    first_frame: int
    last_frame: int
    persistence_ratio: float
    screen_position: str  # e.g. "top-left"
    review_required: bool = True


# ===========================================================================
# Camera motion
# ===========================================================================


class CameraMotionClass(StrEnum):
    STATIC = "STATIC"
    HORIZONTAL_GLOBAL_MOTION = "HORIZONTAL_GLOBAL_MOTION"
    VERTICAL_GLOBAL_MOTION = "VERTICAL_GLOBAL_MOTION"
    DIAGONAL_GLOBAL_MOTION = "DIAGONAL_GLOBAL_MOTION"
    SCALE_INCREASE = "SCALE_INCREASE"
    SCALE_DECREASE = "SCALE_DECREASE"
    ROTATION = "ROTATION"
    HANDHELD_DRIFT = "HANDHELD_DRIFT"
    HANDHELD_SHAKE = "HANDHELD_SHAKE"
    UNRESOLVED = "UNRESOLVED"


class CameraMotionCandidate(StrictModel):
    """A per-shot camera-motion phase. Global background/image motion only —
    never inferred from foreground subject movement. 2D global motion never
    claims dolly/track."""

    candidate_id: str
    shot_number: int | None
    start_frame: int
    #: The last frame that actually shows the motion (the right frame of the last
    #: supporting pair). Distinct from the interval end boundary below (P).
    last_supporting_frame: int
    #: Interval end boundary = presentation start of the frame AFTER the last
    #: supporting frame (or the media endpoint), mirroring shot interval ends.
    end_frame: int
    start_exact: ExactFraction | None = None
    end_exact: ExactFraction | None = None
    motion_class: CameraMotionClass
    direction: str | None = None  # screen-left | screen-right | up | down | ...
    strength: float
    inlier_ratio: float
    supporting_pair_ids: list[str] = []
    review_required: bool = True
    evidence_refs: list[EvidenceReference] = []
    notes: list[str] = []


# ===========================================================================
# Actions
# ===========================================================================


class ActionStateClass(StrEnum):
    ENTITY_APPEARS = "ENTITY_APPEARS"
    ENTITY_EXITS = "ENTITY_EXITS"
    MOTION_BEGINS = "MOTION_BEGINS"
    MOTION_ENDS = "MOTION_ENDS"
    DIRECTION_CHANGES = "DIRECTION_CHANGES"
    CONTACT_BEGINS = "CONTACT_BEGINS"
    CONTACT_ENDS = "CONTACT_ENDS"
    OBJECT_SEPARATES = "OBJECT_SEPARATES"
    OBJECT_APPROACHES = "OBJECT_APPROACHES"
    OBJECT_MOVES_WITH_CHARACTER = "OBJECT_MOVES_WITH_CHARACTER"
    OCCLUSION_BEGINS = "OCCLUSION_BEGINS"
    OCCLUSION_ENDS = "OCCLUSION_ENDS"
    POSE_CHANGE_CANDIDATE = "POSE_CHANGE_CANDIDATE"
    FINAL_STATE = "FINAL_STATE"


class ActionCandidate(StrictModel):
    """An atomic, evidence-bounded action candidate. Semantic labels
    (raises hand / picks up / throws) are NEVER forced from generic motion; they
    stay REVIEW_REQUIRED until specific evidence exists."""

    candidate_id: str
    shot_number: int | None
    subject_track_ids: list[str] = []
    object_track_ids: list[str] = []
    action_class: ActionStateClass
    #: Only set when specific evidence supports it; otherwise None + review.
    semantic_label: str | None = None
    start_frame: int
    end_frame: int
    start_exact: ExactFraction | None = None
    end_exact: ExactFraction | None = None
    pre_state: str | None = None
    post_state: str | None = None
    supporting_observation_frames: list[int] = []
    review_required: bool = True
    evidence_refs: list[EvidenceReference] = []
    notes: list[str] = []


# ===========================================================================
# Playback speed
# ===========================================================================


class SpeedConclusion(StrEnum):
    REGULAR_SUPPORTED = "REGULAR_SUPPORTED"
    SLOW_MOTION_CANDIDATE = "SLOW_MOTION_CANDIDATE"
    ACCELERATED_CANDIDATE = "ACCELERATED_CANDIDATE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class PlaybackSpeedEvidence(StrictModel):
    """Per-shot playback-speed evidence. Fast camera motion or motion blur alone
    can never prove accelerated/slow playback."""

    shot_number: int
    duplicate_frame_ratio: float | None = None
    motion_cadence_regularity: float | None = None
    frame_spacing_regular: bool | None = None
    optical_flow_cadence: float | None = None
    audio_cadence_available: bool = False
    sustained_retiming: bool = False
    conclusion: SpeedConclusion
    review_required: bool = True
    #: Candidate exact mid-shot speed-change boundaries (only with sustained
    #: evidence).
    speed_change_frames: list[int] = []
    evidence_refs: list[EvidenceReference] = []
    notes: list[str] = []


# ===========================================================================
# Human decisions (bound to media/rules/subject so they survive reruns safely)
# ===========================================================================


class DecisionType(StrEnum):
    """Typed human-decision kinds. A decision's ``value`` must be admissible for
    its kind (else INVALID_VALUE)."""

    CLAIM_EVIDENCE = "CLAIM_EVIDENCE"
    OCR_TEXT = "OCR_TEXT"
    OCR_TIMING = "OCR_TIMING"
    IDENTITY_MAPPING = "IDENTITY_MAPPING"
    CAMERA_CLASSIFICATION = "CAMERA_CLASSIFICATION"
    ACTION_SEMANTICS = "ACTION_SEMANTICS"
    ACTION_BOUNDARY = "ACTION_BOUNDARY"
    PLAYBACK_SPEED = "PLAYBACK_SPEED"
    REVIEW_PROPOSAL_OUTCOME = "REVIEW_PROPOSAL_OUTCOME"


class DecisionOutcome(StrEnum):
    APPLIED = "APPLIED"
    STALE = "STALE"
    INVALID_SUBJECT = "INVALID_SUBJECT"
    INVALID_VALUE = "INVALID_VALUE"
    CONFLICT = "CONFLICT"


class HumanReviewDecision(StrictModel):
    """A recorded human decision. ``decided_by`` is only ever set by an actual
    human-supplied decision file — machine code never fabricates one."""

    decision_id: str
    subject_id: str
    decision_type: DecisionType
    value: str
    reviewer_note: str | None = None
    evidence_refs: list[EvidenceReference] = []
    #: From the human file; never stamped fresh at apply time (deterministic).
    decided_at_utc: str | None = None
    decided_by: str
    #: Binding keys so a decision from another video/rules is never applied.
    bound_video_sha256: str | None = None
    bound_rules_version: str | None = None
    #: Optional binding to a specific evidence identity/hash.
    bound_evidence_id: str | None = None


class DecisionApplication(StrictModel):
    decision_id: str
    outcome: DecisionOutcome
    applied: bool
    stale: bool = False
    reason: str | None = None


# ===========================================================================
# Top-level Phase 4 result
# ===========================================================================


class VisualIntelligenceResult(StrictModel):
    """visual_qc.json payload: the full Phase 4 outcome for one run.

    ``overall_status`` is PASS | REVIEW_REQUIRED | FAILED and can NEVER be PASS
    while a CRITICAL review item remains (validator P4-QC-001). Phase 4 never
    prints CAPTION VERIFIED / EXPORT READY.
    """

    visual_intelligence_version: str
    seed_present: bool = False
    seed_parsed: bool = False
    seed_claim_count: int = 0
    frame_observation_count: int = 0
    foundation_status: FoundationStatus = FoundationStatus.NOT_EVALUATED
    seed_shot_count: int | None = None
    verified_shot_count: int | None = None
    character_hypothesis_count: int = 0
    object_hypothesis_count: int = 0
    ocr_status: OCRStatus = OCRStatus.UNAVAILABLE
    ocr_track_count: int = 0
    source_verified_ocr_count: int = 0
    camera_phase_count: int = 0
    camera_direction_reversals: int = 0
    action_candidate_count: int = 0
    semantic_review_required_count: int = 0
    proposal_counts: dict[str, int] = {}
    critical_review_item_count: int = 0
    review_item_count: int = 0
    overall_status: str = "REVIEW_REQUIRED"

    triage: SeedTriage | None = None
    foundation_checks: list[FoundationCheck] = []
    proposals: list[ReviewProposal] = []
