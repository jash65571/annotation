"""Robust shot/cut detection: sensitive candidate detection + VLM verification.

Getting the SHOT COUNT right is critical. Two failure modes:
  - Over-counting: fast motion / strobing lights fool a naive detector into
    reporting cuts that aren't there (ContentDetector reports 6 on a clip that is
    really 1 continuous handheld take).
  - Under-counting: a robust detector might smooth over a real quick cut.

Strategy: collect candidate boundaries from BOTH a motion-robust detector
(AdaptiveDetector) and a sensitive one (ContentDetector) so nothing is missed,
then have the vision model look at the frames straddling each candidate and
confirm whether it is a genuine shot change (and, if so, its cut type). Only
verified boundaries become shots.
"""

from __future__ import annotations

import json
from pathlib import Path

from .frames import GridFrame
from .vision import OpenAIVisionBackend, image_content, text_content

CUT_TYPES = (
    "Opening shot", "Hard cut", "Cross dissolve", "Fade in", "Fade out",
    "Match cut", "Jump cut", "Smash cut", "Wipe", "Iris",
    "L-cut", "J-cut", "Whip pan", "Swish pan",
)


def _adaptive(video: Path) -> list[float]:
    from scenedetect import AdaptiveDetector, detect  # type: ignore[import-untyped]

    scenes = detect(str(video), AdaptiveDetector())
    return [round(s.get_seconds(), 2) for i, (s, _e) in enumerate(scenes) if i > 0]


def _content(video: Path, threshold: float = 27.0) -> list[float]:
    from scenedetect import ContentDetector, detect

    scenes = detect(str(video), ContentDetector(threshold=threshold))
    return [round(s.get_seconds(), 2) for i, (s, _e) in enumerate(scenes) if i > 0]


def candidate_boundaries(video: Path, min_gap: float = 0.4) -> list[float]:
    """Union of robust + sensitive detectors, de-duplicated within ``min_gap``."""
    raw = sorted(set(_adaptive(video) + _content(video)))
    merged: list[float] = []
    for t in raw:
        if not merged or t - merged[-1] > min_gap:
            merged.append(t)
    return merged


def _near(
    grid: list[GridFrame], t: float, *, before: bool, n: int = 3, span: float = 0.4
) -> list[GridFrame]:
    if before:
        fs = [f for f in grid if t - span <= f.time_seconds < t]
        return fs[-n:]
    fs = [f for f in grid if t <= f.time_seconds <= t + span]
    return fs[:n]


_VERIFY = (
    "These frames straddle a POSSIBLE shot boundary in a video. The first images are the "
    "last frames before it; the later images are the first frames after it.\n"
    "Decide: is this a GENUINE shot change (a cut or transition to a DIFFERENT shot — the "
    "scene, framing, or subjects change discontinuously), or is it the SAME continuous "
    "shot where only the camera or subjects moved (a pan, fast motion, or a lighting/"
    "strobe change is NOT a cut)?\n"
    "If it is a genuine change, also give the cut type from: Hard cut, Cross dissolve, "
    "Fade in, Fade out, Match cut, Jump cut, Smash cut, Wipe, Iris, Whip pan, Swish pan "
    "(use 'Hard cut' if unsure).\n"
    'Return JSON {"is_cut": true|false, "cut": "<type>"}.'
)


def _verify(
    backend: OpenAIVisionBackend, before: list[GridFrame], after: list[GridFrame]
) -> tuple[bool, str]:
    if not before or not after:
        return False, "Hard cut"
    content: list[dict[str, object]] = [text_content(_VERIFY), text_content("Before:")]
    content += [image_content(f.path) for f in before]
    content.append(text_content("After:"))
    content += [image_content(f.path) for f in after]
    try:
        data = json.loads(backend.complete(content, json_mode=True, max_tokens=60))
    except Exception:
        return False, "Hard cut"
    is_cut = bool(data.get("is_cut"))
    cut = str(data.get("cut", "Hard cut")).strip()
    return is_cut, (cut if cut in CUT_TYPES else "Hard cut")


def resolve_shots(
    backend: OpenAIVisionBackend, video: Path, grid: list[GridFrame], duration: float
) -> list[tuple[float, float, str]]:
    """Return verified shots as (start, end, cut_type). Always >= 1 shot."""
    confirmed: list[tuple[float, str]] = []
    for t in candidate_boundaries(video):
        if t <= 0.3 or t >= duration - 0.3:
            continue
        is_cut, ctype = _verify(backend, _near(grid, t, before=True), _near(grid, t, before=False))
        if is_cut:
            confirmed.append((round(t, 2), ctype))
    marks = [0.0, *[t for t, _ in confirmed], round(duration, 2)]
    shots: list[tuple[float, float, str]] = []
    for i in range(len(marks) - 1):
        cut = "Opening shot" if i == 0 else confirmed[i - 1][1]
        shots.append((marks[i], marks[i + 1], cut))
    return shots
