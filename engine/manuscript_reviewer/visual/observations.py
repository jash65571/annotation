"""Per-frame machine observation ledger (visual/frame_observations.*).

One observation record per source video frame, carrying exact frame identity
(frame_index, source PTS, annotation time). Metrics are deterministic numpy/
OpenCV reductions on the shared gray metric grid; the global-motion estimate
uses FFT phase correlation (``cv2.phaseCorrelate``), which is deterministic and
RNG-free. This is a machine observation layer, never caption prose.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from ..media.clock import AnnotationClock
from ..models.frame import FrameLedger
from ..models.review_intelligence import FrameObservation
from ..models.shot_truth import ShotTruthResult
from .concerns import FrameConcernInputs, detect_frame_concerns
from .decode import FrameCache

_NEAR_BLACK = 16
_NEAR_WHITE = 239


def _shot_for_frame(shot_result: ShotTruthResult | None, frame_count: int) -> list[int | None]:
    mapping: list[int | None] = [None] * frame_count
    if shot_result is None:
        return mapping
    for shot in shot_result.shots:
        for i in range(shot.start_frame_index, min(shot.end_frame_index + 1, frame_count)):
            mapping[i] = shot.shot_index
    return mapping


def _sharpness(gray: npt.NDArray[np.uint8]) -> float:
    normalized = gray.astype(np.float64) / 255.0
    lap = cv2.Laplacian(normalized, cv2.CV_64F)
    return float(lap.var())


def build_frame_observations(
    cache: FrameCache,
    ledger: FrameLedger,
    clock: AnnotationClock,
    shot_result: ShotTruthResult | None,
) -> list[FrameObservation]:
    """Build one FrameObservation per ledger frame, in ledger order."""
    gray = cache.gray_frames()
    frame_count = gray.shape[0]
    grid_diag = float(np.hypot(gray.shape[1], gray.shape[2]))
    shot_map = _shot_for_frame(shot_result, frame_count)

    observations: list[FrameObservation] = []
    prev_f: npt.NDArray[np.float32] | None = None
    for i in range(frame_count):
        frame = gray[i]
        frame_f = frame.astype(np.float32)
        brightness = float(frame.mean()) / 255.0
        contrast = float(frame.std()) / 255.0
        sharpness = _sharpness(frame)
        near_white_fraction = float((frame >= _NEAR_WHITE).mean())
        near_black_fraction = float((frame <= _NEAR_BLACK).mean())

        if prev_f is None:
            motion_magnitude = 0.0
            global_motion = 0.0
        else:
            motion_magnitude = float(np.abs(frame_f - prev_f).mean()) / 255.0
            (dx, dy), _response = cv2.phaseCorrelate(prev_f, frame_f)
            global_motion = float(np.hypot(dx, dy)) / grid_diag
        # Raw inter-frame motion only — NOT a global-compensated residual.
        raw_interframe_motion = max(0.0, motion_magnitude)

        concerns = detect_frame_concerns(
            FrameConcernInputs(
                brightness=brightness,
                sharpness=sharpness,
                near_white_fraction=near_white_fraction,
                near_black_fraction=near_black_fraction,
                motion_magnitude=motion_magnitude,
            )
        )

        record = ledger.frames[i]
        source_time = record.pts_time_seconds
        annotation_time = clock.to_annotation(source_time) if source_time is not None else None
        observations.append(
            FrameObservation(
                frame_index=i,
                source_pts=record.pts,
                source_pts_time_exact=source_time,
                annotation_time_exact=annotation_time,
                shot_number=shot_map[i],
                brightness=round(brightness, 6),
                contrast=round(contrast, 6),
                sharpness=round(sharpness, 8),
                motion_magnitude=round(motion_magnitude, 6),
                global_camera_motion=round(global_motion, 6),
                raw_interframe_motion=round(raw_interframe_motion, 6),
                visual_concern_codes=concerns,
            )
        )
        prev_f = frame_f
    return observations
