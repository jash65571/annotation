"""Render a structured Annotation into the canonical Manuscript-II caption text."""

from __future__ import annotations

from pathlib import Path

from .structured import Annotation, Shot, Timed


def _hms(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    return f"{h:02d}:{m:02d}:{t % 60:04.1f}"


def _rng(a: float, b: float) -> str:
    return f"{a:.1f}s–{b:.1f}s"  # noqa: RUF001  (en dash, per spec)


def _inline(items: list[Timed]) -> list[str]:
    return [f"({_rng(it.start, it.end)}) {it.text}" for it in items]


def _shot(shot: Shot) -> list[str]:
    out: list[str] = []
    out.append(f"[Shot {shot.index}: {_hms(shot.start)}–{_hms(shot.end)}]")  # noqa: RUF001
    out.append(f"Cut: {shot.cut}")
    camera = shot.camera or shot.shot_type
    if shot.camera and shot.shot_type and shot.shot_type.lower() not in shot.camera.lower():
        camera = f"{shot.shot_type}. {shot.camera}"
    moves = " ".join(f"({_rng(m.start, m.end)}) {m.text}" for m in shot.camera_movements)
    out.append(f"Camera: {camera}" + (f" {moves}" if moves else ""))
    out.append(f"Scene: {shot.scene}")
    out.append("Action & Audio:")
    out += _inline(shot.actions)
    out.append(f"Video playback speed: {shot.playback_speed}")
    out.append("")
    return out


def render(ann: Annotation) -> str:
    g = ann.globals
    out: list[str] = ["[Overview]", "Characters:"]
    for c in g.characters:
        out.append(f"{c.id}: {c.description}")
    out.append("")
    if g.objects:
        out.append("Objects:")
        for o in g.objects:
            out.append(f"{o.id}: {o.description}")
        out.append("")
    out.append(f"Scene: {g.scene}")
    out.append("")
    out.append(f"Style: {g.style}")
    out.append("")
    out.append(f"Audio: {g.audio}")
    out.append("")
    out.append(f"Visual concerns: {g.visual_concerns or 'None.'}")
    out.append(f"Audio concerns: {g.audio_concerns or 'None.'}")
    out.append("")
    for shot in ann.shots:
        out += _shot(shot)
    return "\n".join(out).rstrip() + "\n"


def write(markdown: str, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.manuscript.md"
    path.write_text(markdown, encoding="utf-8")
    return path
