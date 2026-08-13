"""Frame extraction: a dense time grid + scene-change keyframes.

The grid sets the timeline resolution (default 10 Hz = every 0.1 s). Scene-change
detection picks which frames actually need a fresh visual description, so we stay
precise (change-aligned) without running the vision model on near-identical frames.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .ffbin import find_tool


@dataclass(frozen=True)
class GridFrame:
    index: int
    time_seconds: float
    path: Path


def probe_duration(video: Path) -> float:
    """Duration in seconds via ffprobe."""
    ffprobe = find_tool("ffprobe")
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def extract_grid(video: Path, out_dir: Path, hz: float = 10.0, width: int = 768) -> list[GridFrame]:
    """Extract one frame every ``1/hz`` seconds, scaled to ``width`` px.

    Frame ``i`` corresponds to input time ``i / hz`` seconds.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_tool("ffmpeg")
    pattern = out_dir / "g%06d.png"
    subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(video),
         "-vf", f"fps={hz},scale={width}:-2", "-start_number", "0", str(pattern)],
        check=True,
    )
    frames: list[GridFrame] = []
    for p in sorted(out_dir.glob("g??????.png")):
        idx = int(p.stem[1:])
        frames.append(GridFrame(index=idx, time_seconds=idx / hz, path=p))
    return frames


_SHOWINFO_TIME = re.compile(r"pts_time:(\d+\.?\d*)")


def scene_change_times(video: Path, threshold: float = 0.15) -> list[float]:
    """Timestamps (s) where ffmpeg detects a scene change above ``threshold``.

    0.0 is always included (the first frame is always a "change").
    """
    ffmpeg = find_tool("ffmpeg")
    proc = subprocess.run(
        [ffmpeg, "-v", "info", "-i", str(video),
         "-vf", f"select='gt(scene\\,{threshold})',showinfo",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    times = {0.0}
    for m in _SHOWINFO_TIME.finditer(proc.stderr):
        times.add(round(float(m.group(1)), 3))
    return sorted(times)


def select_keyframes(
    grid: list[GridFrame], scene_times: list[float], min_gap_seconds: float = 1.0
) -> list[GridFrame]:
    """Grid frames to actually describe: nearest grid frame to each scene change,
    plus a steady cadence so long static stretches still get sampled."""
    if not grid:
        return []
    chosen: dict[int, GridFrame] = {}
    for t in scene_times:
        nearest = min(grid, key=lambda f: abs(f.time_seconds - t))
        chosen[nearest.index] = nearest
    last_t = -1e9
    for f in grid:
        if f.time_seconds - last_t >= min_gap_seconds:
            chosen[f.index] = f
            last_t = f.time_seconds
    return [chosen[i] for i in sorted(chosen)]
