"""OCR driver: per-frame recognition → observations → text tracks → watermarks.

Consumes a stream of ``(frame_index, image)`` from one shared decode (H). A
per-frame OCR failure is accounted for (M), never silently turned into "no text";
if a meaningful fraction of frames fail, OCR status becomes DEGRADED. An optional
region detector (I) focuses OCR on candidate text regions with a whole-frame
fallback. Every observation carries exact frame identity; the adapter never
supplies a timestamp.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np
import numpy.typing as npt

from ..media.clock import AnnotationClock
from ..models.frame import FrameLedger
from ..models.review_intelligence import (
    OCREngineInfo,
    OCRFrameStatus,
    OCRObservation,
    OCRStatus,
    TextDetectionProvenance,
    TextTrack,
    WatermarkCandidate,
)
from ..models.shot_truth import ShotTruthResult
from .adapter import OCRWord
from .timing import build_text_tracks, detect_watermark_candidates

#: Fraction of frames failing OCR above which the run is DEGRADED.
_DEGRADED_FRACTION = 0.2

#: A region detector maps a gray image to candidate (x, y, w, h) boxes.
RegionDetector = Callable[[npt.NDArray[np.uint8]], list[tuple[int, int, int, int]]]


class _RecognizingAdapter(Protocol):
    def engine_info(self) -> OCREngineInfo: ...
    def recognize(self, image: npt.NDArray[np.uint8], language: str = ...) -> list[OCRWord]: ...


@dataclass
class OCRResult:
    engine_info: OCREngineInfo
    observations: list[OCRObservation] = field(default_factory=list)
    text_tracks: list[TextTrack] = field(default_factory=list)
    watermark_candidates: list[WatermarkCandidate] = field(default_factory=list)
    frame_status: dict[int, str] = field(default_factory=dict)
    read_frame_count: int = 0
    failed_frame_count: int = 0


def _shot_bounds_resolver(
    shot_truth: ShotTruthResult | None,
) -> Callable[[int], tuple[int, int] | None] | None:
    if shot_truth is None or not shot_truth.shots:
        return None
    ranges = [(s.start_frame_index, s.end_frame_index) for s in shot_truth.shots]

    def resolve(frame_index: int) -> tuple[int, int] | None:
        for lo, hi in ranges:
            if lo <= frame_index <= hi:
                return lo, hi
        return None

    return resolve


def _to_gray(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    if image.ndim == 3:
        return np.asarray(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), dtype=np.uint8)
    return image


def run_ocr(
    frames: Iterable[tuple[int, npt.NDArray[np.uint8]]],
    ledger: FrameLedger,
    clock: AnnotationClock,
    adapter: _RecognizingAdapter,
    language: str = "eng",
    total_frames: int | None = None,
    region_detector: RegionDetector | None = None,
    shot_truth: ShotTruthResult | None = None,
) -> OCRResult:
    info = adapter.engine_info()
    if info.status == OCRStatus.UNAVAILABLE:
        return OCRResult(engine_info=info)

    observations: list[OCRObservation] = []
    frame_status: dict[int, str] = {}
    obs_counter = 0
    read_count = 0
    failed_count = 0
    last_inspected: int | None = None

    for frame_index, image in frames:
        last_inspected = frame_index if last_inspected is None else max(last_inspected, frame_index)
        record = ledger.frames[frame_index] if frame_index < ledger.frame_count else None
        source_time = record.pts_time_seconds if record is not None else None
        annotation_time = clock.to_annotation(source_time) if source_time is not None else None
        try:
            words, provenance, had_region = _recognize_frame(
                adapter, image, language, region_detector
            )
            read_count += 1
            # Item 14: a successful call with zero words is not "OCR present".
            if words:
                frame_status[frame_index] = OCRFrameStatus.OCR_READ_TEXT.value
            elif had_region:
                frame_status[frame_index] = OCRFrameStatus.REGION_PRESENT_TEXT_UNREADABLE.value
            else:
                frame_status[frame_index] = OCRFrameStatus.OCR_NO_TEXT_READ.value
        except (RuntimeError, OSError, ValueError):
            # M: record the failure explicitly; never pretend the frame had no text.
            frame_status[frame_index] = OCRFrameStatus.OCR_ENGINE_FAILED.value
            failed_count += 1
            continue
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
                    provenance=provenance,
                )
            )

    total = read_count + failed_count
    if total and failed_count / total >= _DEGRADED_FRACTION:
        info = info.model_copy(update={"status": OCRStatus.DEGRADED})

    shot_bounds_of = _shot_bounds_resolver(shot_truth)
    tracks = build_text_tracks(
        observations, last_inspected_frame=last_inspected, shot_bounds_of=shot_bounds_of
    )
    watermarks = detect_watermark_candidates(
        tracks, total_frames if total_frames is not None else ledger.frame_count
    )
    return OCRResult(
        engine_info=info,
        observations=observations,
        text_tracks=tracks,
        watermark_candidates=watermarks,
        frame_status=frame_status,
        read_frame_count=read_count,
        failed_frame_count=failed_count,
    )


def _recognize_frame(
    adapter: _RecognizingAdapter,
    image: npt.NDArray[np.uint8],
    language: str,
    region_detector: RegionDetector | None,
) -> tuple[list[OCRWord], TextDetectionProvenance, bool]:
    """OCR one frame. With a region detector, OCR each candidate region crop and
    map boxes back to full-frame coordinates; if regions exist but yield NO words,
    fall back to a safe whole-frame pass (item 16). Returns
    ``(words, provenance, had_text_region)``."""
    if region_detector is not None:
        boxes = region_detector(_to_gray(image))
        if boxes:
            words: list[OCRWord] = []
            for x, y, w, h in boxes:
                crop = image[y : y + h, x : x + w]
                if crop.size == 0:
                    continue
                for word in adapter.recognize(crop, language):
                    words.append(
                        OCRWord(
                            text=word.text, x=x + word.x, y=y + word.y,
                            width=word.width, height=word.height, confidence=word.confidence,
                        )
                    )
            if words:
                return words, TextDetectionProvenance.REGION_DETECTOR, True
            # Item 16: regions found but unreadable -> safe whole-frame fallback.
            fallback = adapter.recognize(image, language)
            return fallback, TextDetectionProvenance.WHOLE_FRAME_FALLBACK, True
    return adapter.recognize(image, language), TextDetectionProvenance.WHOLE_FRAME, False
