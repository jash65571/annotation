"""Anchor-seeded forward/backward local tracking on the shared gray grid.

Deterministic (template matching + appearance histogram, no RNG, no detector).
A match SCORE is never identity truth; a reacquisition after a loss is always
marked REVIEW_REQUIRED, never silently accepted. Tracking reuses the already
decoded metric grid — no extra decode.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from ..media.clock import AnnotationClock
from ..models.frame import FrameLedger
from ..models.review_intelligence import EntityTrack, TrackObservation, TrackStatus, VisualAnchor
from ..shots.decode import METRIC_HEIGHT, METRIC_WIDTH, GrayFrames

#: Normalized template-match score for a confident match.
_TRACK_THRESHOLD = 0.55
#: Consecutive misses tolerated as OCCLUSION before declaring LOST.
_OCCLUSION_MAX = 3
#: Give up searching after this many consecutive LOST frames.
_LOST_GIVEUP = 8
_MIN_TEMPLATE = 2


def _to_grid_box(
    anchor: VisualAnchor, full_w: int, full_h: int
) -> tuple[int, int, int, int]:
    sx = METRIC_WIDTH / full_w
    sy = METRIC_HEIGHT / full_h
    gx = round(anchor.x * sx)
    gy = round(anchor.y * sy)
    gw = max(_MIN_TEMPLATE, round(anchor.width * sx))
    gh = max(_MIN_TEMPLATE, round(anchor.height * sy))
    gx = min(gx, METRIC_WIDTH - gw)
    gy = min(gy, METRIC_HEIGHT - gh)
    return max(0, gx), max(0, gy), gw, gh


def _to_full_box(
    gx: int, gy: int, gw: int, gh: int, full_w: int, full_h: int
) -> tuple[int, int, int, int]:
    sx = full_w / METRIC_WIDTH
    sy = full_h / METRIC_HEIGHT
    return round(gx * sx), round(gy * sy), round(gw * sx), round(gh * sy)


def _hist(patch: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    hist = cv2.calcHist([patch], [0], None, [32], [0, 256])
    cv2.normalize(hist, hist)
    return hist.astype(np.float32)


def _match(
    frame: npt.NDArray[np.uint8], template: npt.NDArray[np.uint8]
) -> tuple[int, int, float]:
    th, tw = template.shape
    if frame.shape[0] < th or frame.shape[1] < tw:
        return 0, 0, 0.0
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _min_v, max_v, _min_l, max_l = cv2.minMaxLoc(result)
    return int(max_l[0]), int(max_l[1]), float(max_v)


def track_anchor(
    gray: GrayFrames,
    ledger: FrameLedger,
    clock: AnnotationClock,
    anchor: VisualAnchor,
    full_w: int,
    full_h: int,
    shot_bounds: tuple[int, int] | None = None,
) -> EntityTrack:
    """Track an anchor forward/backward, bounded to its verified shot (item 3).

    Local tracking never crosses an editorial cut; ``shot_bounds`` is the
    anchor's shot frame range. OCCLUDED frames carry the LAST KNOWN position with
    explicit uncertainty — never the original anchor coordinates.
    """
    frame_count = gray.shape[0]
    lo = 0 if shot_bounds is None else max(0, shot_bounds[0])
    hi = frame_count - 1 if shot_bounds is None else min(frame_count - 1, shot_bounds[1])
    gx, gy, gw, gh = _to_grid_box(anchor, full_w, full_h)
    template = gray[anchor.frame_index][gy : gy + gh, gx : gx + gw].copy()
    template_hist = _hist(template)

    observations: dict[int, TrackObservation] = {}
    observations[anchor.frame_index] = _observation(
        anchor.frame_index, gx, gy, gw, gh, 1.0, TrackStatus.TRACKED, full_w, full_h
    )
    reacquired = False
    saw_loss = False

    for direction in (1, -1):
        miss = 0
        lost = False
        last_gx, last_gy = gx, gy  # last KNOWN position for occlusion frames
        f = anchor.frame_index + direction
        while lo <= f <= hi:
            mx, my, score = _match(gray[f], template)
            if score >= _TRACK_THRESHOLD:
                hist_corr = float(
                    cv2.compareHist(
                        template_hist,
                        _hist(gray[f][my : my + gh, mx : mx + gw]),
                        cv2.HISTCMP_CORREL,
                    )
                )
                status = TrackStatus.TRACKED
                if lost:
                    status = TrackStatus.REACQUIRED
                    reacquired = True
                    lost = False
                observations[f] = _observation(
                    f, mx, my, gw, gh, hist_corr, status, full_w, full_h
                )
                last_gx, last_gy = mx, my
                miss = 0
            else:
                miss += 1
                if miss <= _OCCLUSION_MAX:
                    # Uncertain: last-known position, no trustworthy box, no score.
                    observations[f] = _observation(
                        f, last_gx, last_gy, gw, gh, None, TrackStatus.OCCLUDED,
                        full_w, full_h,
                        note="occluded: last-known position, uncertain (not a verified box)",
                    )
                else:
                    saw_loss = True
                    lost = True
                    if miss - _OCCLUSION_MAX > _LOST_GIVEUP:
                        break
            f += direction

    ordered = [observations[k] for k in sorted(observations)]
    frames = [o.frame_index for o in ordered]
    # Similarity never proves identity: a reacquired or lossy track is REVIEW.
    status = (
        TrackStatus.REVIEW_REQUIRED
        if (reacquired or saw_loss)
        else TrackStatus.TRACKED
    )
    return EntityTrack(
        track_id=f"TRK-{anchor.anchor_id}",
        entity_type=anchor.entity_type,
        anchor_frame_index=anchor.frame_index,
        first_frame_index=frames[0],
        last_frame_index=frames[-1],
        observations=ordered,
        status=status,
        reacquired=reacquired,
        notes=[f"seeded by anchor {anchor.anchor_id} ({anchor.temporary_label or 'unlabeled'})"],
    )


def _observation(
    frame_index: int,
    gx: int,
    gy: int,
    gw: int,
    gh: int,
    similarity: float | None,
    status: TrackStatus,
    full_w: int,
    full_h: int,
    note: str | None = None,
) -> TrackObservation:
    fx, fy, fw, fh = _to_full_box(gx, gy, gw, gh, full_w, full_h)
    return TrackObservation(
        frame_index=frame_index,
        x=fx,
        y=fy,
        width=fw,
        height=fh,
        status=status,
        appearance_similarity=round(similarity, 4) if similarity is not None else None,
        notes=[note] if note else [],
    )
