"""Evidence model: every future caption claim must be traceable to evidence.

AI confidence alone is never factual evidence — `MODEL_OBSERVATION` evidence
can propose, but a claim is only *verified* when backed by deterministic media
evidence (frames/audio) and/or `HUMAN_VERIFICATION`.
"""

from __future__ import annotations

from enum import StrEnum

from .common import ExactFraction, StrictModel


class EvidenceType(StrEnum):
    FRAME = "FRAME"
    FRAME_RANGE = "FRAME_RANGE"
    FRAME_STRIP = "FRAME_STRIP"
    AUDIO_RANGE = "AUDIO_RANGE"
    WAVEFORM_RANGE = "WAVEFORM_RANGE"
    SPECTROGRAM_RANGE = "SPECTROGRAM_RANGE"
    ASR_SEGMENT = "ASR_SEGMENT"
    OCR_RESULT = "OCR_RESULT"
    MODEL_OBSERVATION = "MODEL_OBSERVATION"
    HUMAN_VERIFICATION = "HUMAN_VERIFICATION"
    # --- Phase 4 (Visual Review Intelligence) machine-candidate evidence ---
    # These are machine observations/candidates: they can *propose* but never
    # make a claim factual on their own, so none are in FACTUAL_EVIDENCE_TYPES.
    CROP = "CROP"
    FRAME_OBSERVATION = "FRAME_OBSERVATION"
    TRACK = "TRACK"
    TRACK_RANGE = "TRACK_RANGE"
    OCR_OBSERVATION = "OCR_OBSERVATION"
    OCR_TRACK = "OCR_TRACK"
    CAMERA_METRIC = "CAMERA_METRIC"
    ACTION_STATE_CHANGE = "ACTION_STATE_CHANGE"
    CONTACT_STATE = "CONTACT_STATE"
    FINAL_STATE = "FINAL_STATE"
    SEED_LINE = "SEED_LINE"
    STRUCTURAL_CHECK = "STRUCTURAL_CHECK"


#: Evidence types that can, on their own, make a claim factual.
FACTUAL_EVIDENCE_TYPES = frozenset(
    {
        EvidenceType.FRAME,
        EvidenceType.FRAME_RANGE,
        EvidenceType.FRAME_STRIP,
        EvidenceType.AUDIO_RANGE,
        EvidenceType.WAVEFORM_RANGE,
        EvidenceType.SPECTROGRAM_RANGE,
        EvidenceType.HUMAN_VERIFICATION,
    }
)


class EvidenceReference(StrictModel):
    """A pointer from a claim to the material that supports it."""

    evidence_id: str
    evidence_type: EvidenceType
    #: Frame anchors (enumeration indexes into the run's frame ledger).
    start_frame: int | None = None
    end_frame: int | None = None
    #: Exact PTS anchors in stream time_base units.
    start_pts: int | None = None
    end_pts: int | None = None
    #: Exact time anchors in seconds (derived from PTS, never from a model).
    start_time_seconds: ExactFraction | None = None
    end_time_seconds: ExactFraction | None = None
    #: Artifact paths relative to the run folder (frame images, audio crops...).
    artifact_paths: list[str] = []
    #: Which tool/model/human produced this evidence.
    source: str | None = None
    notes: str | None = None

    @property
    def is_factual(self) -> bool:
        """True if this evidence type can support a factual claim by itself."""
        return self.evidence_type in FACTUAL_EVIDENCE_TYPES
