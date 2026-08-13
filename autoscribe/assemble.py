"""Merge speech + keyframe visual descriptions into one copy-paste annotation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .asr import SpeechSegment


@dataclass(frozen=True)
class Described:
    time_seconds: float
    text: str


def _fmt(t: float) -> str:
    return f"{t:0.1f}s"


def build_markdown(
    *,
    video_name: str,
    duration: float,
    speech: list[SpeechSegment],
    visuals: list[Described],
    hz: float,
    speech_status: str = "ok",
) -> str:
    lines: list[str] = []
    lines.append(f"# {video_name}")
    lines.append("")
    lines.append(
        f"*Auto-generated (AutoScribe). Duration {duration:0.2f}s · "
        f"visual timeline {hz:g} Hz · best-effort description, not a verified transcript.*"
    )
    lines.append("")

    # --- Spoken words -----------------------------------------------------
    lines.append("## Spoken words")
    if speech:
        for seg in speech:
            lines.append(f"- **{_fmt(seg.start)}-{_fmt(seg.end)}** - {seg.text}")
    elif speech_status != "ok":
        lines.append(f"- Speech transcription not run: {speech_status}.")
    else:
        lines.append("- None detected (no intelligible speech).")
    lines.append("")

    # --- Visual timeline --------------------------------------------------
    lines.append("## Visual timeline")
    prev: str | None = None
    for d in visuals:
        marker = "•" if d.text != prev else "·"
        lines.append(f"- **{_fmt(d.time_seconds)}** {marker} {d.text}")
        prev = d.text
    lines.append("")

    # --- Combined narrative ----------------------------------------------
    lines.append("## Combined")
    events: list[tuple[float, str]] = []
    for seg in speech:
        events.append((seg.start, f'SAID: "{seg.text}"'))
    last: str | None = None
    for d in visuals:
        if d.text != last:
            events.append((d.time_seconds, f"SEEN: {d.text}"))
            last = d.text
    for t, text in sorted(events, key=lambda e: e[0]):
        lines.append(f"- **{_fmt(t)}** — {text}")
    lines.append("")
    return "\n".join(lines)


def write_outputs(markdown: str, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{stem}.annotation.md"
    md_path.write_text(markdown, encoding="utf-8")
    return md_path
