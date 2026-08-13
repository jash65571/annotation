"""Locate ffmpeg / ffprobe without depending on the Manuscript engine.

Order: MANUSCRIPT_FFMPEG_DIR, AUTOSCRIBE_FFMPEG_DIR, then PATH, then C:/ffbin.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def _candidate_dirs() -> list[Path]:
    dirs: list[Path] = []
    for var in ("AUTOSCRIBE_FFMPEG_DIR", "MANUSCRIPT_FFMPEG_DIR"):
        val = os.environ.get(var)
        if val:
            dirs.append(Path(val))
    dirs.append(Path("C:/ffbin"))
    return dirs


def find_tool(name: str) -> str:
    """Return an absolute path to ffmpeg/ffprobe, raising if unavailable."""
    exe = name + (".exe" if os.name == "nt" else "")
    for d in _candidate_dirs():
        cand = d / exe
        if cand.is_file():
            return str(cand)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"{name} not found. Set AUTOSCRIBE_FFMPEG_DIR to its bin directory "
        f"(ffmpeg/ffprobe), add it to PATH, or place it in C:/ffbin."
    )
