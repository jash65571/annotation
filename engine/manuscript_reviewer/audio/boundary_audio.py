"""Shot-boundary audio continuity: evidence for every SUPPORTED visual boundary.

Resolves Phase 2's ``audio_verification_required`` flag at the EVIDENCE level
only — the final transition selector is never changed automatically, and L-cut
vs J-cut is never decided from waveform continuity alone (that needs semantic
source relation, P3-BOUNDARY-002).
"""

from __future__ import annotations

import logging
from fractions import Fraction

from ..media.clock import AnnotationClock
from ..models.audio import (
    AudioEnergyBin,
    AudioRegion,
    AudioRegionKind,
    AudioTimeline,
    AudioVerificationStatus,
    BoundaryAudioEvidence,
    BoundaryAudioStatus,
    SpeechRegion,
)
from ..models.shot_truth import BoundaryCandidate, CandidateStatus

logger = logging.getLogger(__name__)

WINDOW_SECONDS = Fraction(1, 2)
SILENT_DBFS = -55.0


def _window_energy(
    bins: list[AudioEnergyBin], start: Fraction, end: Fraction
) -> float | None:
    selected = [
        b.dbfs
        for b in bins
        if b.end_annotation_time > start and b.start_annotation_time < end
    ]
    if not selected:
        return None
    return round(sum(selected) / len(selected), 3)


def analyze_boundary_audio(
    candidates: list[BoundaryCandidate],
    clock: AnnotationClock,
    timeline: AudioTimeline | None,
    bins: list[AudioEnergyBin],
    regions: list[AudioRegion],
    speech_regions: list[SpeechRegion],
) -> list[BoundaryAudioEvidence]:
    evidence: list[BoundaryAudioEvidence] = []
    for candidate in candidates:
        if candidate.status != CandidateStatus.SUPPORTED:
            continue
        if candidate.boundary_time_exact is None:
            continue
        boundary = clock.to_annotation(candidate.boundary_time_exact)

        if timeline is None:
            evidence.append(
                BoundaryAudioEvidence(
                    boundary_candidate_id=candidate.candidate_id,
                    visual_boundary_exact=boundary,
                    window_seconds=float(WINDOW_SECONDS),
                    energy_before_dbfs=None,
                    energy_after_dbfs=None,
                    silence_around_boundary=False,
                    speech_region_crossing=False,
                    asr_word_crossing=False,
                    audio_continuity_status=BoundaryAudioStatus.NO_AUDIO,
                    audio_verification_status=AudioVerificationStatus.UNAVAILABLE,
                )
            )
            continue

        before = _window_energy(bins, boundary - WINDOW_SECONDS, boundary)
        after = _window_energy(bins, boundary, boundary + WINDOW_SECONDS)
        silent = bool(
            any(
                r.kind == AudioRegionKind.SILENCE_CANDIDATE
                and r.start_annotation_time <= boundary <= r.end_annotation_time
                for r in regions
            )
        )
        speech_crossing = any(
            s.start_exact < boundary < s.end_exact for s in speech_regions
        )
        notes: list[str] = []

        if before is None or after is None:
            status = BoundaryAudioStatus.NO_AUDIO
            verification = AudioVerificationStatus.UNAVAILABLE
        elif silent and before <= SILENT_DBFS and after <= SILENT_DBFS:
            status = BoundaryAudioStatus.DISCONTINUOUS
            verification = AudioVerificationStatus.CHECKED_NO_CROSSING
            notes.append("Silence spans the visual boundary.")
        elif speech_crossing:
            status = BoundaryAudioStatus.CONTINUOUS
            verification = AudioVerificationStatus.CROSSING_REVIEW_REQUIRED
            notes.append(
                "AUDIO_CROSSES_VISUAL_BOUNDARY: a speech region spans the visual "
                "edit. L-cut vs J-cut requires semantic source relation — not "
                "decidable from continuity alone."
            )
        elif abs(before - after) >= 20.0:
            status = BoundaryAudioStatus.DISCONTINUOUS
            verification = AudioVerificationStatus.CHECKED_NO_CROSSING
            notes.append(
                f"Energy steps {before:.1f} -> {after:.1f} dBFS across the boundary."
            )
        elif before > SILENT_DBFS and after > SILENT_DBFS:
            status = BoundaryAudioStatus.CONTINUOUS
            verification = AudioVerificationStatus.CROSSING_REVIEW_REQUIRED
            notes.append(
                "AUDIO_CROSSES_VISUAL_BOUNDARY: sustained audio continues across "
                "the visual edit (music/ambience/speech source unresolved)."
            )
        else:
            status = BoundaryAudioStatus.UNCERTAIN
            verification = AudioVerificationStatus.CROSSING_REVIEW_REQUIRED
            notes.append("Low-level audio around the boundary; continuity uncertain.")

        evidence.append(
            BoundaryAudioEvidence(
                boundary_candidate_id=candidate.candidate_id,
                visual_boundary_exact=boundary,
                window_seconds=float(WINDOW_SECONDS),
                energy_before_dbfs=before,
                energy_after_dbfs=after,
                silence_around_boundary=silent,
                speech_region_crossing=speech_crossing,
                asr_word_crossing=speech_crossing,
                audio_continuity_status=status,
                audio_verification_status=verification,
                notes=notes,
            )
        )
    return evidence
