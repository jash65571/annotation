"""Sequential OCR: link per-frame observations into text tracks with exact
first/stable/legible/last/disappearance timing, plus the one-frame defense and
watermark candidates.

Machine OCR text is never caption-eligible without human source verification —
every track defaults to ``UNVERIFIED`` and ``caption_text_eligible=False``.
"""

from __future__ import annotations

from ..models.review_intelligence import (
    OCRObservation,
    SourceTextVerificationStatus,
    TextTrack,
    WatermarkCandidate,
)
from .consensus import temporal_consensus

#: Min consecutive frames a text region must persist to be a "stable" track.
_STABLE_MIN_FRAMES = 2
#: IoU above which two boxes on adjacent frames are the same region.
_IOU_LINK = 0.3
#: Max frame gap that can be bridged within one track (brief occlusion).
_MAX_GAP = 2


def _iou(a: OCRObservation, b: OCRObservation) -> float:
    ax2, ay2 = a.x + a.width, a.y + a.height
    bx2, by2 = b.x + b.width, b.y + b.height
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def link_observations(observations: list[OCRObservation]) -> list[list[OCRObservation]]:
    """Greedily link observations across frames into per-region sequences."""
    ordered = sorted(observations, key=lambda o: o.frame_index)
    tracks: list[list[OCRObservation]] = []
    for obs in ordered:
        best: list[OCRObservation] | None = None
        best_iou = _IOU_LINK
        for track in tracks:
            last = track[-1]
            if 0 < obs.frame_index - last.frame_index <= _MAX_GAP:
                score = _iou(last, obs)
                if score >= best_iou:
                    best_iou = score
                    best = track
        if best is None:
            tracks.append([obs])
        else:
            best.append(obs)
    return tracks


def build_text_tracks(
    observations: list[OCRObservation], track_id_prefix: str = "TT"
) -> list[TextTrack]:
    """Build TextTracks with exact phase frames and the one-frame defense."""
    tracks: list[TextTrack] = []
    for i, seq in enumerate(link_observations(observations), start=1):
        seq_sorted = sorted(seq, key=lambda o: o.frame_index)
        frames = [o.frame_index for o in seq_sorted]
        consensus = temporal_consensus(seq_sorted)
        distinct_frames = len(set(frames))

        # One-frame defense: a text seen in a single isolated frame is never a
        # stable track — it is REVIEW_REQUIRED, not established evidence.
        stable = distinct_frames >= _STABLE_MIN_FRAMES
        first_stable = frames[0] if stable else None
        last_stable = frames[-1] if stable else None
        # "Fully legible" = first frame reaching the consensus text at good conf.
        fully_legible = None
        if consensus is not None:
            for obs in seq_sorted:
                if obs.raw_text.strip() == consensus.consensus_text and (
                    obs.confidence is None or obs.confidence >= 60.0
                ):
                    fully_legible = obs.frame_index
                    break

        tracks.append(
            TextTrack(
                track_id=f"{track_id_prefix}-{i:03d}",
                first_candidate_frame=frames[0],
                first_stable_frame=first_stable,
                fully_legible_frame=fully_legible,
                last_stable_frame=last_stable,
                disappearance_frame=frames[-1] + 1,
                bbox_path=[[o.frame_index, o.x, o.y, o.width, o.height] for o in seq_sorted],
                observations=seq_sorted,
                consensus=consensus,
                verification_status=SourceTextVerificationStatus.UNVERIFIED,
                review_required=True,  # machine OCR always needs human verification
                caption_text_eligible=False,
                evidence_refs=[],
            )
        )
    return tracks


def detect_watermark_candidates(
    tracks: list[TextTrack], total_frames: int, min_persistence: float = 0.8
) -> list[WatermarkCandidate]:
    """A text region present for most of the clip in a fixed position is a
    *candidate* watermark — never auto-classified (persistent game UI is not a
    watermark merely because it is fixed)."""
    candidates: list[WatermarkCandidate] = []
    for i, track in enumerate(tracks, start=1):
        if not track.observations:
            continue
        first = track.first_candidate_frame
        last = track.last_stable_frame or track.first_candidate_frame
        distinct = len({o.frame_index for o in track.observations})
        persistence = distinct / total_frames if total_frames else 0.0
        if persistence < min_persistence:
            continue
        obs0 = track.observations[0]
        candidates.append(
            WatermarkCandidate(
                candidate_id=f"WM-{i:03d}",
                x=obs0.x,
                y=obs0.y,
                width=obs0.width,
                height=obs0.height,
                first_frame=first,
                last_frame=last,
                persistence_ratio=round(persistence, 4),
                screen_position="unknown",
                review_required=True,
            )
        )
    return candidates
