"""Render a structured Annotation into the canonical Manuscript-II caption text.

Field names follow the live tool's master template, not AutoScribe's own
invention: ``Cast:``, ``Camera Movements:``, ``Playback Speed:`` and
``Speed Changes:`` (emitted only when present). Shot headers use seconds, the
same unit as every action line and the same format the engine's own seed parser
accepts — the old clock-style header made one caption speak two time languages.

Punctuation and empty-section handling are decided HERE, deterministically,
rather than trusted to whatever the model happened to produce.
"""

from __future__ import annotations

from pathlib import Path

from .structured import Annotation, Shot, Timed

#: en dash, per the caption spec
DASH = "–"  # noqa: RUF001


def _sec(t: float) -> str:
    return f"{t:.1f}s"


def _rng(a: float, b: float) -> str:
    return f"{_sec(a)}{DASH}{_sec(b)}"


def _terminated(text: str) -> str:
    """Every rendered sentence ends in terminal punctuation (§ formatting)."""
    text = " ".join(text.split())
    if not text:
        return text
    return text if text[-1] in '.!?"’”' else text + "."  # noqa: RUF001


def _inline(items: list[Timed]) -> list[str]:
    return [f"({_rng(it.start, it.end)}) {_terminated(it.text)}" for it in items]


def _shot(shot: Shot) -> list[str]:
    out: list[str] = []
    out.append(f"[Shot {shot.index}: {_rng(shot.start, shot.end)}]")
    out.append(f"Cut: {_terminated(shot.cut)}")
    camera = shot.camera or shot.shot_type
    if shot.camera and shot.shot_type and shot.shot_type.lower() not in shot.camera.lower():
        camera = f"{_terminated(shot.shot_type)} {shot.camera}"
    out.append(f"Camera: {_terminated(camera)}")
    # Timed phases live in their own block — merging them into the Camera line
    # was flagged as a field-placement violation (Camera = setup only).
    if shot.camera_movements:
        out.append("Camera Movements:")
        out += _inline(shot.camera_movements)
    out.append(f"Scene: {_terminated(shot.scene)}")
    # An empty Action & Audio heading is a formatting defect: either there are
    # lines, or the section is omitted and the gap is visible as a blocker.
    if shot.actions:
        out.append("Action & Audio:")
        out += _inline(shot.actions)
    out.append(f"Playback Speed: {_terminated(shot.playback_speed)}")
    if shot.speed_changes:
        out.append("Speed Changes:")
        out += _inline(shot.speed_changes)
    out.append("")
    return out


def render(ann: Annotation) -> str:
    g = ann.globals
    out: list[str] = ["[Overview]", "Cast:"]
    for c in g.characters:
        out.append(f"{c.id}: {_terminated(c.description)}")
    out.append("")
    if g.objects:
        out.append("Objects:")
        for o in g.objects:
            out.append(f"{o.id}: {_terminated(o.description)}")
        out.append("")
    out.append(f"Scene: {_terminated(g.scene)}")
    out.append("")
    out.append(f"Style: {_terminated(g.style)}")
    out.append("")
    out.append(f"Audio: {_terminated(g.audio)}")
    out.append("")
    # Capital C: the source-of-truth §26 template is "Visual Concerns:" /
    # "Audio Concerns:". Empty means exactly "None." (§6.6).
    out.append(f"Visual Concerns: {_terminated(g.visual_concerns) or 'None.'}")
    out.append(f"Audio Concerns: {_terminated(g.audio_concerns) or 'None.'}")
    out.append("")
    for shot in ann.shots:
        out += _shot(shot)
    return "\n".join(out).rstrip() + "\n"


def write(markdown: str, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.manuscript.md"
    path.write_text(markdown, encoding="utf-8")
    return path
