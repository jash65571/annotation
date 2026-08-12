"""Optional local Tesseract OCR adapter.

Execution is routed through the same safe subprocess wrapper as ffmpeg (no shell
invocation, no scattered direct process spawning). Tesseract is NOT mandatory: if
its binary is absent, ``engine_info`` reports UNAVAILABLE and ``recognize`` raises
:class:`OCRUnavailableError`. No cloud OCR is ever used. Unicode text is
preserved (never normalized to ASCII).
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from ..media.ffmpeg_tools import ToolExecutionError, run_tool
from ..models.review_intelligence import OCREngineInfo, OCRStatus
from .adapter import OCRWord

logger = logging.getLogger(__name__)

TESSERACT_DIR_ENV = "MANUSCRIPT_TESSERACT_DIR"


class OCRUnavailableError(RuntimeError):
    """Tesseract could not be located or executed."""


def _find_tesseract() -> Path | None:
    override = os.environ.get(TESSERACT_DIR_ENV)
    if override:
        for name in ("tesseract.exe", "tesseract"):
            candidate = Path(override) / name
            if candidate.is_file():
                return candidate
    found = shutil.which("tesseract")
    return Path(found) if found else None


class TesseractAdapter:
    """Runs the Tesseract CLI in TSV mode over a single image."""

    def __init__(self) -> None:
        self._binary = _find_tesseract()
        self._version: str | None = None

    def _resolve_version(self) -> str | None:
        if self._binary is None:
            return None
        if self._version is None:
            try:
                result = run_tool(self._binary, ["--version"], timeout=30.0)
                self._version = (result.stdout or result.stderr).splitlines()[0].strip()
            except (ToolExecutionError, OSError, IndexError):
                self._version = "unknown"
        return self._version

    def engine_info(self, language: str = "eng") -> OCREngineInfo:
        if self._binary is None:
            return OCREngineInfo(
                engine="tesseract",
                status=OCRStatus.UNAVAILABLE,
                failure_reason=(
                    "tesseract binary not found on PATH or via "
                    f"{TESSERACT_DIR_ENV}"
                ),
            )
        return OCREngineInfo(
            engine="tesseract",
            status=OCRStatus.AVAILABLE,
            version=self._resolve_version(),
            language_config=language,
        )

    def recognize(self, image: npt.NDArray[np.uint8], language: str = "eng") -> list[OCRWord]:
        if self._binary is None:
            raise OCRUnavailableError("tesseract binary not available")
        with tempfile.TemporaryDirectory(prefix="mr_ocr_") as tmp:
            img_path = Path(tmp) / "frame.png"
            if not cv2.imwrite(str(img_path), image):
                raise OCRUnavailableError(f"could not write OCR input image to {img_path}")
            try:
                result = run_tool(
                    self._binary,
                    [str(img_path), "stdout", "-l", language, "tsv"],
                    timeout=60.0,
                )
            except ToolExecutionError as exc:
                raise OCRUnavailableError(f"tesseract failed: {exc}") from exc
        return _parse_tsv(result.stdout)


def _parse_tsv(tsv: str) -> list[OCRWord]:
    words: list[OCRWord] = []
    lines = tsv.splitlines()
    if not lines:
        return words
    header = lines[0].split("\t")
    try:
        idx = {name: header.index(name) for name in
               ("level", "left", "top", "width", "height", "conf", "text")}
    except ValueError:
        return words
    for line in lines[1:]:
        cols = line.split("\t")
        if len(cols) <= idx["text"]:
            continue
        if cols[idx["level"]] != "5":  # word level
            continue
        text = cols[idx["text"]]
        if not text.strip():
            continue
        try:
            conf = float(cols[idx["conf"]])
        except ValueError:
            conf = None
        words.append(
            OCRWord(
                text=text,
                x=int(cols[idx["left"]]),
                y=int(cols[idx["top"]]),
                width=int(cols[idx["width"]]),
                height=int(cols[idx["height"]]),
                confidence=conf,
            )
        )
    return words
