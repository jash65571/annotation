"""Future caption/entity schema (Phase 2+ populates these).

Defined now so the evidence-first data model exists before any caption text is
ever generated. Every event/claim carries `evidence` references; display
timestamps are always projections of exact PTS-backed times.

Transition types and field vocabulary come from the versioned rule file
(`rules/manuscript_v1.yaml`), not from code constants, wherever the rule can
change; enums here mirror the stable Manuscript II structure.
"""

from __future__ import annotations

from enum import StrEnum

from .common import ExactFraction, StrictModel
from .evidence import EvidenceReference


class ReviewOutcome(StrEnum):
    """The three valid reviewer outcomes for any seeded section (Handbook §1)."""

    KEEP = "KEEP"
    FIX_ENRICH = "FIX_ENRICH"
    REDO_REBUILD = "REDO_REBUILD"


class PlaybackSpeed(StrEnum):
    SLOW_MOTION = "slow_motion"
    REGULAR = "regular"
    ACCELERATED = "accelerated"


class ExactTimeRange(StrictModel):
    """Exact event window. PTS anchors are authoritative; display strings are derived."""

    start_pts: int | None = None
    end_pts: int | None = None
    start_seconds: ExactFraction
    end_seconds: ExactFraction
    #: Enumeration frame anchors into the run's frame ledger.
    start_frame: int | None = None
    end_frame: int | None = None


class Transition(StrictModel):
    """How a shot begins (cut type). Allowed types live in the rule file."""

    transition_type: str
    at: ExactTimeRange | None = None
    evidence: list[EvidenceReference] = []


class Shot(StrictModel):
    shot_number: int
    time_range: ExactTimeRange
    transition: Transition
    camera_description: str | None = None
    scene_description: str | None = None
    playback_speed: PlaybackSpeed | None = None
    evidence: list[EvidenceReference] = []


class Character(StrictModel):
    character_id: str  # "C1", "C2", ...
    description: str
    off_screen_only: bool = False
    first_appearance: ExactTimeRange | None = None
    evidence: list[EvidenceReference] = []


class ObjectEntity(StrictModel):
    object_id: str  # "O1", "O2", ...
    description: str
    first_appearance: ExactTimeRange | None = None
    evidence: list[EvidenceReference] = []


class CameraEvent(StrictModel):
    """A timed camera movement phase (lives in Camera Movements, never Action & Audio)."""

    shot_number: int
    time_range: ExactTimeRange
    description: str
    reveals_or_hides: str | None = None
    evidence: list[EvidenceReference] = []


class ActionAudioEvent(StrictModel):
    """One Action & Audio line: one sentence, one event, exact window."""

    shot_number: int
    time_range: ExactTimeRange
    text: str
    character_ids: list[str] = []
    object_ids: list[str] = []
    evidence: list[EvidenceReference] = []


class SpeechEvent(StrictModel):
    shot_number: int
    time_range: ExactTimeRange
    speaker_id: str | None = None
    verbatim_text: str
    tone: str | None = None
    off_screen: bool = False
    contains_inaudible: bool = False
    evidence: list[EvidenceReference] = []


class SoundEvent(StrictModel):
    shot_number: int
    time_range: ExactTimeRange
    description: str
    diegetic: bool | None = None
    evidence: list[EvidenceReference] = []


class OnScreenTextEvent(StrictModel):
    shot_number: int
    time_range: ExactTimeRange
    text: str
    location_on_frame: str | None = None
    evidence: list[EvidenceReference] = []


class SeedClaim(StrictModel):
    """One claim extracted from the seeded caption — a hypothesis to verify."""

    claim_id: str
    source_field: str
    text: str
    outcome: ReviewOutcome | None = None
    evidence: list[EvidenceReference] = []


class ConfidenceAssessment(StrictModel):
    """Uncertainty lowers specificity; it never licenses guessing."""

    subject_id: str
    confidence: float  # 0.0-1.0, from deterministic signals or reviewer judgment
    basis: str
    requires_human_verification: bool = True


class ReviewDecision(StrictModel):
    """A recorded human decision about a section, claim, or suggestion."""

    decision_id: str
    subject_id: str
    outcome: ReviewOutcome
    rationale: str | None = None
    decided_by: str = "human"
    evidence: list[EvidenceReference] = []


class ReviewedCaption(StrictModel):
    """The full final caption structure (Phase 2+)."""

    characters: list[Character] = []
    objects: list[ObjectEntity] = []
    scene: str | None = None
    style: str | None = None
    overview_audio: str | None = None
    visual_concerns: str | None = None
    audio_concerns: str | None = None
    shots: list[Shot] = []
    camera_events: list[CameraEvent] = []
    action_audio_events: list[ActionAudioEvent] = []
    speech_events: list[SpeechEvent] = []
    sound_events: list[SoundEvent] = []
    on_screen_text_events: list[OnScreenTextEvent] = []
    review_decisions: list[ReviewDecision] = []
