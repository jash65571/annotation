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
    """Machine-processing state ONLY — never a probability, never a human claim.

    ``ALIGNED_EVIDENCE`` means only that WhisperX produced valid forced-alignment
    timing for machine ASR text; it NEVER means a human confirmed the words.
    Human disposition (verified/corrected/rejected) is a SEPARATE axis and lives
    exclusively in :class:`SourceVerificationStatus` — it must never be written
    back into this machine-confidence field.
    """

    MACHINE_CANDIDATE = "MACHINE_CANDIDATE"
    ASR_EVIDENCE = "ASR_EVIDENCE"
    ALIGNED_EVIDENCE = "ALIGNED_EVIDENCE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class SourceVerificationStatus(StrEnum):
    """Did a human independently listen to the SOURCE audio for this speech act?

    Orthogonal to :class:`EvidenceState` and to any machine probability. ASR
    text, ASR word timing, WhisperX alignment, language detection and speaker
    labels are ALL unverified evidence (Master Frame Audit Protocol): a clean,
    high-confidence, fully-aligned machine result is still ``UNVERIFIED`` until a
    human listens. Only a human may move this off ``UNVERIFIED``.
    """

    UNVERIFIED = "UNVERIFIED"
    HUMAN_VERIFIED = "HUMAN_VERIFIED"
    HUMAN_CORRECTED = "HUMAN_CORRECTED"
    REJECTED = "REJECTED"


#: Human dispositions that RESOLVE a speech region — a human has listened, so it
#: no longer demands a mandatory source-audio-verification listen. ``REJECTED``
#: (the human listened and rejected the ASR claim) is resolved just like
#: ``HUMAN_VERIFIED`` / ``HUMAN_CORRECTED``. Only ``UNVERIFIED`` stays unresolved.
RESOLVED_SOURCE_VERIFICATION: frozenset[SourceVerificationStatus] = frozenset(
    {
        SourceVerificationStatus.HUMAN_VERIFIED,
        SourceVerificationStatus.HUMAN_CORRECTED,
        SourceVerificationStatus.REJECTED,
    }
)


class WordTimingStatus(StrEnum):
    """Timing provenance of one reconciled best-transcript word."""

    #: WhisperX forced-alignment timing was matched to this source word.
    WHISPERX_ALIGNED = "WHISPERX_ALIGNED"
    #: WhisperX had no usable timing for this source word; faster-whisper kept.
    FASTER_WHISPER_FALLBACK = "FASTER_WHISPER_FALLBACK"
    #: Reconciliation was ambiguous; faster-whisper timing kept, review required.
    ALIGNMENT_UNRESOLVED = "ALIGNMENT_UNRESOLVED"


class ReviewPriority(StrEnum):
    """Structured priority for the future review UI — never hides evidence."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"


class SampleAnchorStatus(StrEnum):
    """Confidence that decoded PCM sample 0 maps to the first encoded packet."""

    #: No codec priming/skip evidence found; sample 0 anchors the first frame PTS.
    ANCHORED = "ANCHORED"
    #: Codec priming/skip metadata exists but exact anchoring is unproven.
    ANCHOR_REVIEW_REQUIRED = "AUDIO_SAMPLE_ANCHOR_REVIEW_REQUIRED"


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
    #: Text preserved AND every source word represented AND ordered AND in-bounds.
    ALIGNED = "ALIGNED"
    #: Text preserved but some source words fell back to faster-whisper timing.
    PARTIAL = "ALIGNMENT_PARTIAL"
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
    #: Every machine speech act must be listened to at the source, always.
    MANDATORY_SOURCE_AUDIO_VERIFICATION = "MANDATORY_SOURCE_AUDIO_VERIFICATION"
    LOW_ASR_CONFIDENCE = "LOW_ASR_CONFIDENCE"
    MISSING_ALIGNMENT = "MISSING_ALIGNMENT"
    PARTIAL_ALIGNMENT = "PARTIAL_ALIGNMENT"
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
    #: Stream-level codec priming metadata when the container declares it.
    initial_padding_samples: int | None = None
    #: Packet/frame side-data priming: samples the decoder skips at stream start
    #: (AAC encoder delay). ``None`` = ffprobe exposed no skip_samples side data.
    codec_skip_samples: int | None = None
    #: Packet/frame side-data trailing padding the decoder discards, if declared.
    codec_discard_padding: int | None = None
    #: Container-declared start time in the audio stream's own time base, if any.
    codec_delay: int | None = None
    #: Whether decoded PCM sample 0 provably anchors the first encoded packet.
    sample_anchor_status: SampleAnchorStatus = SampleAnchorStatus.ANCHORED
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
    """One engine-emitted word. Timing is OPTIONAL: WhisperX can emit an aligned
    word with no start/end, and a missing time is missing evidence — NEVER 0.
    Faster-whisper words always carry timing; the model supports absence so no
    stage ever fabricates a timestamp or silently drops a word (§8)."""

    word_index: int
    #: Stable identity of the underlying faster-whisper source word. Survives
    #: alignment so WhisperX timing can be mapped back by identity, not position.
    segment_id: int = 0
    source_word_index: int = 0
    text: str
    #: ASR-reported float times parsed via Decimal(str(...)) — estimates only.
    asr_start_seconds: str | None = None
    asr_end_seconds: str | None = None
    #: Mapped onto the annotation clock (exact rational), still ESTIMATED timing.
    start_annotation_time: ExactFraction | None = None
    end_annotation_time: ExactFraction | None = None
    probability: float | None = None
    timing_source: str = "faster_whisper"


class BestWord(StrictModel):
    """One reconciled best-transcript word: exactly one per faster-whisper source
    word. Wording is ALWAYS the faster-whisper source word (never WhisperX
    re-wording); timing is WhisperX when validly matched, faster-whisper
    otherwise. Timing is non-null because the faster-whisper fallback always
    provides it — WhisperX only ever improves timing, never removes a word."""

    segment_id: int
    source_word_index: int
    text: str
    start_annotation_time: ExactFraction
    end_annotation_time: ExactFraction
    asr_start_seconds: str
    asr_end_seconds: str
    probability: float | None = None
    timing_source: str  # "whisperx" | "faster_whisper"
    timing_status: WordTimingStatus


class ASRSegment(StrictModel):
    segment_id: int
    text: str
    #: Optional so a WhisperX segment with no start/end is NEVER anchored to 0 s;
    #: reconciliation fills it from the preserved faster-whisper segment (§8).
    asr_start_seconds: str | None = None
    asr_end_seconds: str | None = None
    start_annotation_time: ExactFraction | None = None
    end_annotation_time: ExactFraction | None = None
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
    #: Fraction of faster-whisper source words that received WhisperX timing.
    alignment_coverage: float | None = None
    #: True when some source words fell back to faster-whisper timing.
    partial_alignment: bool = False
    source_word_count: int | None = None
    aligned_word_count: int | None = None
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
    #: Human-supplied corrected wording (only when source_verification_status is
    #: HUMAN_CORRECTED). ASR text is NEVER copied here automatically.
    corrected_text: str | None = None
    language: LanguageEvidence | None = None
    speaker: SpeakerEvidence | None = None
    mean_word_probability: float | None = None
    alignment_status: AlignmentStatus = AlignmentStatus.NOT_ATTEMPTED
    alignment_coverage: float | None = None
    state: EvidenceState = EvidenceState.MACHINE_CANDIDATE
    #: Independent human-listen axis. ASR/alignment/probability NEVER change it.
    source_verification_status: SourceVerificationStatus = (
        SourceVerificationStatus.UNVERIFIED
    )
    possible_overlap: bool = False
    overlap_status: EvidenceState | None = None
    review_reasons: list[AudioReviewReason] = []

    @property
    def caption_text_eligible(self) -> bool:
        """Phase-4 safety gate. Machine ASR text is EVIDENCE ONLY and must never
        become quoted final caption dialogue: eligibility requires a human. A
        HUMAN_CORRECTED region is eligible only via its ``corrected_text``."""
        if self.source_verification_status == SourceVerificationStatus.HUMAN_VERIFIED:
            return True
        return (
            self.source_verification_status
            == SourceVerificationStatus.HUMAN_CORRECTED
            and self.corrected_text is not None
        )

    @property
    def caption_text(self) -> str | None:
        """The only text a later builder may quote — or None when ineligible."""
        if self.source_verification_status == SourceVerificationStatus.HUMAN_VERIFIED:
            return self.text_candidate
        if self.source_verification_status == SourceVerificationStatus.HUMAN_CORRECTED:
            return self.corrected_text
        return None

    @property
    def caption_language_eligible(self) -> bool:
        """A machine language guess never becomes a final caption language claim
        (§23). Requires human/source verification of the language itself."""
        return (
            self.language is not None
            and self.language.language_review_status
            == LanguageReviewStatus.HUMAN_VERIFIED
        )


class BoundaryAudioEvidence(StrictModel):
    """Audio continuity evidence around one SUPPORTED visual boundary.

    Each fact is computed and stored SEPARATELY. Energy presence on both sides
    proves only ``audio_present_before`` + ``audio_present_after`` — it does NOT
    prove one sound source crosses the visual cut. A speech region spanning the
    boundary is a WEAKER fact than an actual word spanning it, and neither is
    cloned from the other (§12)."""

    boundary_candidate_id: str
    visual_boundary_exact: ExactFraction  # annotation clock
    window_seconds: float

    #: Independent presence facts.
    audio_present_before: bool
    audio_present_after: bool
    silence_spans_boundary: bool

    #: Distinct crossing facts — computed independently, never cloned.
    speech_region_spans_boundary: bool
    asr_segment_spans_boundary: bool
    asr_word_spans_boundary: bool

    #: Energy evidence.
    energy_before_dbfs: float | None
    energy_after_dbfs: float | None
    energy_delta_db: float | None

    #: Spectral evidence (distinguishes an equal-energy source SWITCH from true
    #: continuity — a 440→880 Hz or tone→noise change keeps energy but moves the
    #: spectrum, so it is NOT continuity).
    spectral_centroid_before: float | None = None
    spectral_centroid_after: float | None = None
    spectral_flatness_before: float | None = None
    spectral_flatness_after: float | None = None
    spectral_change_score: float | None = None

    continuity_evidence_codes: list[str] = []
    audio_continuity_status: BoundaryAudioStatus
    audio_verification_status: AudioVerificationStatus
    notes: list[str] = []


class AudioReviewItem(StrictModel):
    item_id: str
    reasons: list[AudioReviewReason]
    priority: ReviewPriority = ReviewPriority.NORMAL
    start_exact: ExactFraction  # annotation clock
    end_exact: ExactFraction
    #: Suggested playback window (with context padding), annotation clock. Always
    #: clamped to available source-audio bounds — never seeks outside the media.
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
    best_word_count: int = 0
    review_item_count: int = 0
    boundaries_checked: int = 0
    language: LanguageEvidence | None = None
    regions: list[AudioRegion] = []
    transients: list[AudioTransientCandidate] = []
    speech_regions: list[SpeechRegion] = []
    boundary_evidence: list[BoundaryAudioEvidence] = []
    review_items: list[AudioReviewItem] = []
    concerns: list[AudioConcernCandidate] = []
