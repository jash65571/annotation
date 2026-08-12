"""Audio Truth Engine models (Phase 3).

Actual source audio is factual truth. ASR/alignment/language/diarization
outputs are EVIDENCE ONLY — machine probability and human verification are
different concepts and are stored separately (`EvidenceState` vs float
probabilities). No model here ever carries final caption prose.
"""

from __future__ import annotations

from enum import StrEnum

from .common import ExactFraction, StrictModel


class EvidenceState(StrEnum):
    """Verification state of a piece of audio evidence — NOT a probability."""

    MACHINE_CANDIDATE = "MACHINE_CANDIDATE"
    ASR_EVIDENCE = "ASR_EVIDENCE"
    ALIGNED_EVIDENCE = "ALIGNED_EVIDENCE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    HUMAN_VERIFIED = "HUMAN_VERIFIED"
    REJECTED = "REJECTED"


class AudioStatus(StrEnum):
    ANALYZED = "ANALYZED"
    NO_AUDIO_STREAM = "NO_AUDIO_STREAM"
    FAILED = "FAILED"


class ASRStatus(StrEnum):
    PASS = "PASS"
    FAILED = "FAILED"
    UNAVAILABLE = "ASR_UNAVAILABLE"
    DISABLED = "DISABLED"


class AlignmentStatus(StrEnum):
    ALIGNED = "ALIGNED"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    TEXT_MISMATCH = "ALIGNMENT_TEXT_MISMATCH"


class LanguageReviewStatus(StrEnum):
    SUPPORTED_BY_ASR = "SUPPORTED_BY_ASR"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    UNKNOWN = "UNKNOWN"
    HUMAN_VERIFIED = "HUMAN_VERIFIED"


class AudioReviewReason(StrEnum):
    LOW_ASR_CONFIDENCE = "LOW_ASR_CONFIDENCE"
    MISSING_ALIGNMENT = "MISSING_ALIGNMENT"
    AUDIO_WITHOUT_ASR_COVERAGE = "AUDIO_WITHOUT_ASR_COVERAGE"
    POSSIBLE_OVERLAP = "POSSIBLE_OVERLAP"
    SPEECH_AT_CLIP_START = "SPEECH_AT_CLIP_START"
    SPEECH_AT_CLIP_END = "SPEECH_AT_CLIP_END"
    UNKNOWN_LANGUAGE = "UNKNOWN_LANGUAGE"
    LOW_LANGUAGE_CONFIDENCE = "LOW_LANGUAGE_CONFIDENCE"
    VOCAL_REVIEW_REQUIRED = "VOCAL_REVIEW_REQUIRED"
    SHOT_BOUNDARY_AUDIO_CONTINUITY = "SHOT_BOUNDARY_AUDIO_CONTINUITY"
    TRANSIENT_SEMANTICS_UNKNOWN = "TRANSIENT_SEMANTICS_UNKNOWN"
    ASR_DISAGREEMENT = "ASR_DISAGREEMENT"
    ASR_UNAVAILABLE = "ASR_UNAVAILABLE"


class AudioRegionKind(StrEnum):
    """Deterministic signal classes — deliberately NOT semantic labels."""

    SILENCE_CANDIDATE = "SILENCE_CANDIDATE"
    ACTIVE_AUDIO = "ACTIVE_AUDIO"
    TRANSIENT_CANDIDATE = "TRANSIENT_CANDIDATE"
    SUSTAINED_TONAL_AUDIO = "SUSTAINED_TONAL_AUDIO"
    BROADBAND_NOISE = "BROADBAND_NOISE"
    LOW_LEVEL_BACKGROUND_AUDIO = "LOW_LEVEL_BACKGROUND_AUDIO"


class BoundaryAudioStatus(StrEnum):
    CONTINUOUS = "CONTINUOUS"
    DISCONTINUOUS = "DISCONTINUOUS"
    UNCERTAIN = "UNCERTAIN"
    NO_AUDIO = "NO_AUDIO"


class AudioVerificationStatus(StrEnum):
    """Phase 3 resolution of Phase 2's audio_verification_required flag."""

    CHECKED_NO_CROSSING = "CHECKED_NO_CROSSING"
    CROSSING_REVIEW_REQUIRED = "CROSSING_REVIEW_REQUIRED"
    UNAVAILABLE = "UNAVAILABLE"


class AudioFrameRecord(StrictModel):
    """One decoded audio frame from ffprobe enumeration."""

    audio_frame_index: int
    pts: int | None
    pts_time_source: ExactFraction | None
    annotation_time: ExactFraction | None
    duration: int | None = None
    duration_time: ExactFraction | None = None
    nb_samples: int | None = None


class AudioTimeline(StrictModel):
    """Canonical mapping: decoded PCM sample index ↔ source PTS ↔ annotation time.

    annotation time of evidence sample N =
        annotation_audio_offset + N / evidence_sample_rate  (exact Fractions)
    """

    source_stream_index: int
    source_time_base: ExactFraction
    source_start_pts: int | None
    source_start_time: ExactFraction | None
    first_decoded_audio_pts: int | None
    #: Source PTS time of the first presented video frame.
    annotation_timeline_origin: ExactFraction
    #: Annotation time at which decoded evidence sample 0 plays.
    annotation_audio_offset: ExactFraction
    #: Codec priming/skip metadata when the container declares it.
    initial_padding_samples: int | None = None
    source_sample_rate: int
    source_channels: int
    evidence_sample_rate: int
    evidence_channels: int
    evidence_sample_count: int
    evidence_duration_seconds: ExactFraction
    asr_sample_rate: int | None = None
    asr_sample_count: int | None = None
    asr_duration_seconds: ExactFraction | None = None


class AudioEnergyBin(StrictModel):
    """One exact non-overlapping energy bin (nominally 10 ms)."""

    bin_index: int
    start_sample: int
    end_sample: int  # exclusive
    start_annotation_time: ExactFraction
    end_annotation_time: ExactFraction
    rms: float
    peak: float
    dbfs: float
    zero_crossing_rate: float
    spectral_centroid_hz: float
    spectral_flatness: float


class AudioRegion(StrictModel):
    """A deterministic signal-class region (never a semantic claim)."""

    region_id: str
    kind: AudioRegionKind
    start_sample: int
    end_sample: int
    start_annotation_time: ExactFraction
    end_annotation_time: ExactFraction
    mean_dbfs: float
    state: EvidenceState = EvidenceState.MACHINE_CANDIDATE
    notes: list[str] = []


class AudioTransientCandidate(StrictModel):
    candidate_id: str
    start_sample: int
    peak_sample: int
    end_sample: int
    start_annotation_time: ExactFraction
    peak_annotation_time: ExactFraction
    end_annotation_time: ExactFraction
    peak_dbfs: float
    rise_db: float
    state: EvidenceState = EvidenceState.MACHINE_CANDIDATE


class LanguageEvidence(StrictModel):
    """Machine language detection NEVER becomes a final language claim."""

    language_candidate: str | None
    language_probability: float | None
    language_source: str  # e.g. "faster_whisper"
    language_review_status: LanguageReviewStatus


class ASRWord(StrictModel):
    word_index: int
    text: str
    #: ASR-reported float times parsed via Decimal(str(...)) — estimates only.
    asr_start_seconds: str
    asr_end_seconds: str
    #: Mapped onto the annotation clock (exact rational), still ESTIMATED timing.
    start_annotation_time: ExactFraction
    end_annotation_time: ExactFraction
    probability: float | None = None
    timing_source: str = "faster_whisper"


class ASRSegment(StrictModel):
    segment_id: int
    text: str
    asr_start_seconds: str
    asr_end_seconds: str
    start_annotation_time: ExactFraction
    end_annotation_time: ExactFraction
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    words: list[ASRWord] = []


class ASRRuntimeInfo(StrictModel):
    engine: str  # "faster_whisper" | "whisperx"
    package_version: str | None = None
    model_name: str | None = None
    model_revision: str | None = None
    device: str | None = None
    compute_type: str | None = None
    runtime_seconds: float | None = None
    model_cache_state: str | None = None  # cached | downloaded | unavailable
    failure_reason: str | None = None


class ASRResult(StrictModel):
    status: ASRStatus
    runtime: ASRRuntimeInfo
    language: LanguageEvidence | None = None
    segments: list[ASRSegment] = []
    vad_enabled: bool = True
    task: str = "transcribe"  # NEVER "translate"


class AlignmentResult(StrictModel):
    status: AlignmentStatus
    runtime: ASRRuntimeInfo
    #: Aligned words; text must match the faster-whisper transcript under the
    #: documented normalization policy or status becomes TEXT_MISMATCH.
    segments: list[ASRSegment] = []
    text_preserved: bool | None = None
    failure_reason: str | None = None


class SpeakerEvidence(StrictModel):
    """Diarization output. Labels are SPEAKER_NN only — NEVER C# identities."""

    speaker_label: str  # SPEAKER_00, SPEAKER_01, ...
    source: str = "diarization"
    state: EvidenceState = EvidenceState.MACHINE_CANDIDATE


class SpeechRegion(StrictModel):
    region_id: str
    start_exact: ExactFraction  # annotation clock
    end_exact: ExactFraction
    start_manuscript: str
    end_manuscript: str
    sources: list[str]
    text_candidate: str | None = None
    language: LanguageEvidence | None = None
    speaker: SpeakerEvidence | None = None
    mean_word_probability: float | None = None
    alignment_status: AlignmentStatus = AlignmentStatus.NOT_ATTEMPTED
    state: EvidenceState = EvidenceState.MACHINE_CANDIDATE
    possible_overlap: bool = False
    overlap_status: EvidenceState | None = None
    review_reasons: list[AudioReviewReason] = []


class BoundaryAudioEvidence(StrictModel):
    """Audio continuity evidence around one SUPPORTED visual boundary."""

    boundary_candidate_id: str
    visual_boundary_exact: ExactFraction  # annotation clock
    window_seconds: float
    energy_before_dbfs: float | None
    energy_after_dbfs: float | None
    silence_around_boundary: bool
    speech_region_crossing: bool
    asr_word_crossing: bool
    audio_continuity_status: BoundaryAudioStatus
    audio_verification_status: AudioVerificationStatus
    notes: list[str] = []


class AudioReviewItem(StrictModel):
    item_id: str
    reasons: list[AudioReviewReason]
    start_exact: ExactFraction  # annotation clock
    end_exact: ExactFraction
    #: Suggested playback window (with context padding), annotation clock.
    playback_start: ExactFraction
    playback_end: ExactFraction
    asr_text_candidate: str | None = None
    evidence_refs: list[str] = []
    review_clip: str | None = None
    notes: list[str] = []


class AudioConcernCandidate(StrictModel):
    """Structured concern lead for the later caption layer — never final prose."""

    concern_code: str  # e.g. OVERLAPPING_VOICES, CLIPPED_OPENING_WORD, ASR_UNAVAILABLE
    start_exact: ExactFraction | None = None
    end_exact: ExactFraction | None = None
    detail: str | None = None


class AudioQCResult(StrictModel):
    """audio_qc.json payload."""

    audio_status: AudioStatus
    asr_status: ASRStatus
    alignment_status: AlignmentStatus
    overall_status: str  # PASS | REVIEW_REQUIRED | FAILED | NO_AUDIO_STREAM
    timeline: AudioTimeline | None = None
    energy_bin_count: int = 0
    region_count: int = 0
    transient_count: int = 0
    speech_region_count: int = 0
    review_item_count: int = 0
    boundaries_checked: int = 0
    language: LanguageEvidence | None = None
    regions: list[AudioRegion] = []
    transients: list[AudioTransientCandidate] = []
    speech_regions: list[SpeechRegion] = []
    boundary_evidence: list[BoundaryAudioEvidence] = []
    review_items: list[AudioReviewItem] = []
    concerns: list[AudioConcernCandidate] = []
