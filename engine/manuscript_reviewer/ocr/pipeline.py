"""OCR driver: per-frame recognition → observations → text tracks → watermarks.

Decoupled from the frame cache via an image-provider callable so the whole
pipeline is testable with the deterministic mock adapter. Every observation
carries exact frame identity; the adapter never supplies a timestamp.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

from ..media.clock import AnnotationClock
from ..models.frame import FrameLedger
from ..models.review_intelligence import (
    OCREngineInfo,
    OCRObservation,
    OCRStatus,
    TextTrack,
    WatermarkCandidate,
)
from .adapter import OCRWord
from .timing import build_text_tracks, detect_watermark_candidates


class _RecognizingAdapter(Protocol):
    def engine_info(self) -> OCREngineInfo: ...
    def recognize(self, image: npt.NDArray[np.uint8], language: str = ...) -> list[OCRWord]: ...


@dataclass
class OCRResult:
    engine_info: OCREngineInfo
    observations: list[OCRObservation] = field(default_factory=list)
    text_tracks: list[TextTrack] = field(default_factory=list)
    watermark_candidates: list[WatermarkCandidate] = field(default_factory=list)


def run_ocr(
    frame_indices: list[int],
    get_image: Callable[[int], npt.NDArray[np.uint8]],
    ledger: FrameLedger,
    clock: AnnotationClock,
    adapter: _RecognizingAdapter,
    language: str = "eng",
    total_frames: int | None = None,
) -> OCRResult:
    """Recognize text across the given frames and build tracks + watermarks."""
    info = adapter.engine_info()
    if info.status == OCRStatus.UNAVAILABLE:
        return OCRResult(engine_info=info)

    observations: list[OCRObservation] = []
    obs_counter = 0
    for frame_index in frame_indices:
        image = get_image(frame_index)
        try:
            words = adapter.recognize(image, language)
        except (RuntimeError, OSError, ValueError):
            # A per-frame adapter failure degrades to no words for that frame;
            # it never aborts the OCR pass or the run.
            words = []
        record = ledger.frames[frame_index] if frame_index < ledger.frame_count else None
        source_time = record.pts_time_seconds if record is not None else None
        annotation_time = clock.to_annotation(source_time) if source_time is not None else None
        for word in words:
            obs_counter += 1
            observations.append(
                OCRObservation(
                    observation_id=f"OCR-{obs_counter:05d}",
                    frame_index=frame_index,
                    source_pts_time_exact=source_time,
                    annotation_time_exact=annotation_time,
                    x=word.x,
                    y=word.y,
                    width=word.width,
                    height=word.height,
                    raw_text=word.text,
                    confidence=word.confidence,
                )
            )

    tracks = build_text_tracks(observations)
    watermarks = detect_watermark_candidates(
        tracks, total_frames if total_frames is not None else ledger.frame_count
    )
    return OCRResult(
        engine_info=info,
        observations=observations,
        text_tracks=tracks,
        watermark_candidates=watermarks,
    )
