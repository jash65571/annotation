"""Shot Truth Engine models (Phase 2).

Core principle: a boundary is evidence BETWEEN TWO FRAMES, never a lone
timestamp. ``left_frame_index`` is the final frame before the visual boundary;
``right_frame_index`` is the first frame after it; the boundary's exact time is
anchored to the incoming (right) frame's PTS. A candidate is NOT a shot
boundary; a detector is NOT factual truth.
"""

from __future__ import annotations

from enum import StrEnum

from .common import ExactFraction, StrictModel


class CandidateStatus(StrEnum):
    CANDIDATE = "CANDIDATE"
    SUPPORTED = "SUPPORTED"
    REJECTED = "REJECTED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class ReasonCode(StrEnum):
    """Structured verifier reasoning codes (no free-form reasoning where a code fits)."""

    HIGH_DISCONTINUITY = "HIGH_DISCONTINUITY"
    LOCAL_OUTLIER = "LOCAL_OUTLIER"
    COHERENT_CAMERA_MOTION = "COHERENT_CAMERA_MOTION"
    FLASH_FRAME = "FLASH_FRAME"
    RETURN_TO_PREVIOUS_STATE = "RETURN_TO_PREVIOUS_STATE"
    MULTIFRAME_FADE_PATTERN = "MULTIFRAME_FADE_PATTERN"
    MULTIFRAME_BLEND_PATTERN = "MULTIFRAME_BLEND_PATTERN"
    LARGE_OCCLUSION = "LARGE_OCCLUSION"
    SCOPE_OR_ZOOM_CONTINUITY = "SCOPE_OR_ZOOM_CONTINUITY"
    EFFECT_CONTINUITY = "EFFECT_CONTINUITY"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    MULTI_DETECTOR_SUPPORT = "MULTI_DETECTOR_SUPPORT"
    POSSIBLE_JUMP_CUT = "POSSIBLE_JUMP_CUT"
    SIDES_LOCALLY_STABLE = "SIDES_LOCALLY_STABLE"
    SINGLE_PAIR_SPIKE = "SINGLE_PAIR_SPIKE"


class TransitionStatus(StrEnum):
    """Internal transition resolution state. UNRESOLVED / REVIEW_REQUIRED must
    never silently become a final Manuscript transition type."""

    PROPOSED = "PROPOSED"
    UNRESOLVED = "UNRESOLVED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    AUDIO_VERIFICATION_REQUIRED = "AUDIO_VERIFICATION_REQUIRED"


class FrameStats(StrictModel):
    """Deterministic per-frame statistics from the downsampled metric decode."""

    frame_index: int
    luma_mean: float
    luma_std: float
    edge_density: float
    sharpness: float
    near_black: bool
    near_white: bool


class PairMetrics(StrictModel):
    """One record per adjacent decoded frame pair (F[i] → F[i+1])."""

    left_frame_index: int
    right_frame_index: int
    left_pts: int | None
    right_pts: int | None
    left_pts_time_seconds: ExactFraction | None
    right_pts_time_seconds: ExactFraction | None
    #: Mean absolute luma difference (0-255 scale) on the metric grid.
    mean_abs_diff: float
    #: Bhattacharyya distance between 64-bin gray histograms (0=identical, 1=disjoint).
    hist_distance: float
    #: Hamming distance between 64-bit DCT perceptual hashes (0-64).
    phash_hamming: int
    #: Absolute change in edge density between frames.
    edge_change: float
    #: Change in mean luma (signed, right - left).
    luma_delta: float
    #: Mean optical-flow magnitude (pixels on the flow grid).
    flow_mean_mag: float
    #: |mean flow vector| / mean |flow|; 1.0 = perfectly coherent global motion.
    flow_coherence: float
    #: ffmpeg scdet scene score for the right frame, if the scdet pass ran.
    scdet_score: float | None = None
    #: ffmpeg scdet mafd for the right frame, if available.
    scdet_mafd: float | None = None


class LocalBaseline(StrictModel):
    """Robust local context for one pair: median/MAD of nearby pair metrics."""

    window_pairs: int
    diff_median: float
    diff_mad: float
    diff_z: float
    hist_median: float
    hist_mad: float
    hist_z: float
    neighbor_motion: float


class FadeEvidence(StrictModel):
    """Multi-frame intensity-trend evidence for a fade."""

    direction: str  # "out" (to color) or "in" (from color)
    target_color: str  # "black" | "white"
    transition_start_frame: int
    transition_end_frame: int
    start_pts: int | None
    end_pts: int | None
    status: CandidateStatus


class BlendEvidence(StrictModel):
    """Sustained multi-frame blend (possible cross dissolve)."""

    start_frame: int
    end_frame: int
    start_pts: int | None
    end_pts: int | None
    sustained_pairs: int
    status: CandidateStatus


class TransitionEvidence(StrictModel):
    """A transition proposal attached to a boundary. ``manuscript_type`` may only
    hold values from the rule-file menu; unresolved states live in ``status``."""

    manuscript_type: str | None = None
    status: TransitionStatus
    #: L-cut/J-cut cannot be ruled out without the Phase 3 audio engine.
    audio_verification_required: bool = False
    fade: FadeEvidence | None = None
    blend: BlendEvidence | None = None
    notes: list[str] = []


class BoundaryCandidate(StrictModel):
    """A candidate visual boundary between two exact ledger frames."""

    candidate_id: str
    left_frame_index: int
    right_frame_index: int
    left_pts: int | None
    right_pts: int | None
    #: Exact boundary time = incoming/right frame PTS time.
    boundary_time_exact: ExactFraction | None
    #: Manuscript 0.1 s display projection (string, e.g. "11.8s"). Display only.
    boundary_time_manuscript: str | None
    candidate_sources: list[str]
    metric_snapshot: PairMetrics
    local_baseline: LocalBaseline
    candidate_score: float
    status: CandidateStatus
    reason_codes: list[ReasonCode] = []
    transition: TransitionEvidence | None = None
    evidence_refs: list[str] = []
    notes: list[str] = []


class ShotProposal(StrictModel):
    """A provisional shot spanning two boundaries (or media start/end).

    Two distinct concepts are kept separate and must not be conflated:

    A. **Inclusive frame ownership** — ``start_frame_index``/``end_frame_index``
       are the first and last enumerated frames belonging to this shot.
    B. **Continuous annotation interval** — ``start_exact``/``end_exact`` are
       timeline boundary times: ``start_exact`` is the incoming boundary time
       (PTS of the first owned frame), a non-final ``end_exact`` is the NEXT
       boundary's exact time (PTS of the next shot's first frame), and the
       final shot's ``end_exact`` is the canonical media annotation endpoint.
       Adjacent shots therefore satisfy prev.end_exact == next.start_exact.

    The final owned frame's own start time is preserved separately as
    ``last_owned_frame_start_exact`` — it is NEVER the interval end.
    """

    shot_index: int  # 1-based, matching Manuscript "Shot N"
    #: Frame range: first and last enumerated frame indexes inside this shot.
    start_frame_index: int
    end_frame_index: int
    #: Annotation interval start = incoming boundary exact time.
    start_exact: ExactFraction | None
    #: Annotation interval end = next boundary exact time, or the canonical
    #: media annotation endpoint for the final shot.
    end_exact: ExactFraction | None
    #: Start PTS time of the final owned frame (frame ownership evidence).
    last_owned_frame_start_exact: ExactFraction | None
    start_manuscript: str | None
    end_manuscript: str | None
    #: Transition INTO this shot ("Opening shot" for shot 1 only).
    transition_into_shot: str | None
    transition_status: TransitionStatus
    supporting_boundary_id: str | None
    evidence_refs: list[str] = []
    review_status: CandidateStatus


class ShotTruthResult(StrictModel):
    """shot_qc.json payload: the full Phase 2 outcome for one run."""

    frame_count: int
    adjacent_pair_count: int
    raw_candidate_count: int
    merged_candidate_count: int
    supported_count: int
    rejected_count: int
    review_required_count: int
    proposed_shot_count: int
    overall_status: str  # PASS | REVIEW_REQUIRED | FAILED
    #: Canonical media annotation endpoint used as the final shot's end_exact.
    annotation_endpoint_exact: ExactFraction | None = None
    annotation_endpoint_method: str | None = None
    annotation_endpoint_conflict: bool = False
    candidates: list[BoundaryCandidate]
    fades: list[FadeEvidence] = []
    blends: list[BlendEvidence] = []
    shots: list[ShotProposal] = []
