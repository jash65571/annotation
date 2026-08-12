"""OCR adapter interface + a deterministic mock adapter for tests.

An adapter receives an image (numpy RGB/gray) and returns machine-candidate words
with exact bounding boxes. It never returns a timestamp — timing comes from the
frame identity of the image it was given. Adapters are optional; the core OCR
pipeline is tested against the mock so normal CI never depends on system OCR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ..models.review_intelligence import OCREngineInfo, OCRStatus


@dataclass(frozen=True)
class OCRWord:
    """One recognized word with an exact bounding box (image coordinates)."""

    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float | None = None


@runtime_checkable
class OCRAdapter(Protocol):
    """Provider-neutral OCR contract. Never supplies timestamps."""

    def engine_info(self) -> OCREngineInfo: ...

    def recognize(self, image: npt.NDArray[np.uint8], language: str = "eng") -> list[OCRWord]: ...


@dataclass
class MockOCRAdapter:
    """Deterministic adapter for tests: returns a scripted result per call.

    ``scripted`` is a list of per-call word lists; calls beyond the script return
    the last entry (so a stable overlay can be simulated across many frames).
    """

    scripted: list[list[OCRWord]] = field(default_factory=list)
    language_available: bool = True
    _call: int = 0

    def engine_info(self) -> OCREngineInfo:
        status = OCRStatus.AVAILABLE if self.language_available else OCRStatus.LANGUAGE_UNAVAILABLE
        return OCREngineInfo(
            engine="mock",
            status=status,
            version="mock-1.0",
            language_config="eng",
        )

    def recognize(self, image: npt.NDArray[np.uint8], language: str = "eng") -> list[OCRWord]:
        if not self.scripted:
            return []
        index = min(self._call, len(self.scripted) - 1)
        self._call += 1
        return list(self.scripted[index])
