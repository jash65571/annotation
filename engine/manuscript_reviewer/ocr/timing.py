"""Sequential OCR: link per-frame observations into text tracks with exact
first/stable/legible/last timing, honest disappearance semantics, real stability
accounting, and the one-frame defense.

Linking (J) uses spatial overlap AND text similarity, so the same HUD box whose
text changes (``ROUND WON`` -> ``NEXT ROUND``) becomes two tracks, not one.
Disappearance (K) is never invented from ``last + 1``. Stability (L) is a real
consecutive run, not two observations across a gap. Unicode is preserved and
words are never fuzzily corrected. Machine OCR text is never caption-eligible
without human source verification.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable
from difflib import SequenceMatcher

from ..models.review_intelligence import (
    OCRObservation,
    SourceTextVerificationStatus,
    TextDisappearanceStatus,
    TextTrack,
    WatermarkCandidate,
)
from .consensus import temporal_consensus

#: Min consecutive support frames for a track to be "stable".
_STABLE_MIN_FRAMES = 2
#: IoU above which two boxes on nearby frames may be the same region.
_IOU_LINK = 0.3
#: Text-similarity ratio above which two reads are the "same" text.
_TEXT_SIM = 0.6
#: Max frame gap that can be bridged within one track (brief occlusion).
_MAX_GAP = 2

_NORMALIZE = re.compile(r"\s+")


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


def _norm(text: str) -> str:
    return _NORMALIZE.sub(" ", text.strip()).casefold()


def text_similarity(a: str, b: str) -> float:
    na, nb = _norm(a), _norm(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def link_observations(observations: list[OCRObservation]) -> list[list[OCRObservation]]:
    """Greedily link observations across frames by spatial overlap AND text
    similarity — a same-box text change starts a NEW track."""
    ordered = sorted(observations, key=lambda o: o.frame_index)
    tracks: list[list[OCRObservation]] = []
    for obs in ordered:
        best: list[OCRObservation] | None = None
        best_score = 0.0
        for track in tracks:
            last = track[-1]
            if not (0 < obs.frame_index - last.frame_index <= _MAX_GAP):
                continue
            iou = _iou(last, obs)
            if iou < _IOU_LINK:
                continue
            sim = text_similarity(last.raw_text, obs.raw_text)
            if sim < _TEXT_SIM:
                continue  # same box, different text -> different track
            score = iou * sim
            if score > best_score:
                best_score = score
                best = track
        if best is None:
            tracks.append([obs])
        else:
            best.append(obs)
    return tracks


def _longest_run(frames: list[int]) -> tuple[int, int | None]:
    """Return (longest_consecutive_run_length, first_frame_of_first_qualifying_run)."""
    if not frames:
        return 0, None
    best_len = 1
    run_len = 1
    run_start = frames[0]
    first_stable: int | None = None
    for prev, cur in itertools.pairwise(frames):
        if cur == prev + 1:
            run_len += 1
        else:
            run_len = 1
            run_start = cur
        if run_len >= _STABLE_MIN_FRAMES and first_stable is None:
            first_stable = run_start
        best_len = max(best_len, run_len)
    return best_len, first_stable


def build_text_tracks(
    observations: list[OCRObservation],
    last_inspected_frame: int | None = None,
    track_id_prefix: str = "TT",
    shot_bounds_of: Callable[[int], tuple[int, int] | None] | None = None,
) -> list[TextTrack]:
    """Build TextTracks with exact phases, honest disappearance, and the
    one-frame defense. ``persists_to_shot_end`` is SHOT-aware (item 15): a track
    persists only when its stable evidence reaches its containing shot's final
    frame — clip end is never equated with shot end."""
    tracks: list[TextTrack] = []
    for i, seq in enumerate(link_observations(observations), start=1):
        seq_sorted = sorted(seq, key=lambda o: o.frame_index)
        frames = [o.frame_index for o in seq_sorted]
        distinct = sorted(set(frames))
        consensus = temporal_consensus(seq_sorted)
        run_len, first_stable = _longest_run(distinct)
        span = distinct[-1] - distinct[0] + 1
        stable = run_len >= _STABLE_MIN_FRAMES

        fully_legible = None
        if consensus is not None:
            for obs in seq_sorted:
                if obs.raw_text.strip() == consensus.consensus_text and (
                    obs.confidence is None or obs.confidence >= 60.0
                ):
                    fully_legible = obs.frame_index
                    break

        # Disappearance (K): never invented. Persistence is SHOT-aware (item 15):
        # a track persists only when its stable evidence reaches its shot's final
        # frame — otherwise UNRESOLVED (no region-absence evidence).
        last_frame = distinct[-1]
        disappearance_frame = None
        shot_bounds = shot_bounds_of(last_frame) if shot_bounds_of is not None else None
        if stable and shot_bounds is not None and last_frame >= shot_bounds[1]:
            disappearance_status = TextDisappearanceStatus.PERSISTS_TO_END
            persists = True
        elif (
            shot_bounds_of is None
            and last_inspected_frame is not None
            and last_frame >= last_inspected_frame
        ):
            # No shot info: fall back to clip-end presence (best available).
            disappearance_status = TextDisappearanceStatus.PERSISTS_TO_END
            persists = True
        else:
            disappearance_status = TextDisappearanceStatus.UNRESOLVED
            persists = False

        tracks.append(
            TextTrack(
                track_id=f"{track_id_prefix}-{i:03d}",
                first_candidate_frame=distinct[0],
                first_stable_frame=first_stable if stable else None,
                fully_legible_frame=fully_legible,
                last_stable_frame=last_frame if stable else None,
                disappearance_frame=disappearance_frame,
                disappearance_status=disappearance_status,
                text_persists_to_shot_end=persists,
                consecutive_support_frames=run_len,
                total_support_frames=len(distinct),
                missed_frames=max(0, span - len(distinct)),
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
