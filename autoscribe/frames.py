"""Frame extraction anchored to real encoded-frame presentation timestamps.

The old implementation asked FFmpeg for `fps=10` and then computed each frame's
time as ``index / hz``. That is a *resampled constant-rate grid*: on
variable-frame-rate media the numbers are simply wrong, and even on CFR media
they name a time no encoded frame actually has. Every downstream timestamp
inherited that error, which is why the tool could not honestly claim 0.1 s
precision.

Now the source frame list is probed first (``ffprobe -show_frames`` → real
``pts_time`` per encoded frame), the sampler picks *actual* frames nearest each
target time, and each extracted image carries the PTS of the source frame it
came from. Timestamps therefore always name a frame that exists.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .blockers import WARNING, BlockerLog
from .ffbin import find_tool


@dataclass(frozen=True)
class GridFrame:
    index: int
    #: TRUE presentation timestamp of the source frame this image came from.
    time_seconds: float
    path: Path
    #: Index of the frame in the source stream (n), for evidence pointers.
    source_index: int = -1

    def evidence(self) -> str:
        """Pointer a human can check: which encoded frame backs this claim."""
        return f"frame n={self.source_index} pts={self.time_seconds:.3f}s"


def probe_duration(video: Path) -> float:
    """Duration in seconds via ffprobe."""
    ffprobe = find_tool("ffprobe")
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def probe_frame_times(video: Path) -> list[float]:
    """Presentation timestamp of every encoded video frame, in display order.

    This is the ledger every AutoScribe timestamp is anchored to. Frames are
    sorted by PTS because decode order is not display order.
    """
    ffprobe = find_tool("ffprobe")
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time,best_effort_timestamp_time",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    times: list[float] = []
    for fr in json.loads(out).get("frames", []):
        raw = fr.get("pts_time", fr.get("best_effort_timestamp_time"))
        if raw in (None, "N/A"):
            continue
        try:
            times.append(float(raw))
        except (TypeError, ValueError):
            continue
    return sorted(times)


def _targets(duration: float, hz: float) -> list[float]:
    if hz <= 0:
        raise ValueError(f"hz must be positive, got {hz!r}")
    step = 1.0 / hz
    out: list[float] = []
    t = 0.0
    while t <= duration + 1e-9:
        out.append(t)
        t += step
    return out


def pick_source_frames(frame_times: list[float], duration: float, hz: float) -> list[int]:
    """Indices of the real source frames nearest each 1/hz target time.

    Deduplicated: when the source runs slower than ``hz`` the same frame is not
    emitted twice, so the grid never invents intermediate moments.
    """
    if not frame_times:
        return []
    picked: list[int] = []
    seen: set[int] = set()
    cursor = 0
    for target in _targets(duration, hz):
        while (cursor + 1 < len(frame_times)
               and abs(frame_times[cursor + 1] - target) <= abs(frame_times[cursor] - target)):
            cursor += 1
        if cursor not in seen:
            seen.add(cursor)
            picked.append(cursor)
    return picked


def _select_expr(indices: list[int]) -> str:
    return "+".join(f"eq(n\\,{i})" for i in indices)


def extract_grid(
    video: Path,
    out_dir: Path,
    hz: float = 10.0,
    width: int = 768,
    blockers: BlockerLog | None = None,
) -> list[GridFrame]:
    """Extract the source frames nearest a 1/hz cadence, tagged with real PTS.

    Falls back to the legacy resampled grid ONLY if the frame ledger cannot be
    probed, and records a blocker when it does — a run whose timestamps are not
    PTS-anchored must not be mistaken for a precise one.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_tool("ffmpeg")
    pattern = out_dir / "g%06d.png"

    frame_times = probe_frame_times(video)
    if not frame_times:
        if blockers is not None:
            blockers.add(
                "TIMING_NOT_PTS_ANCHORED",
                "Could not probe encoded-frame timestamps; timestamps fall back to a "
                "resampled constant-rate grid and are NOT frame-accurate.",
            )
        return _extract_resampled(ffmpeg, video, pattern, out_dir, hz, width)

    duration = frame_times[-1]
    indices = pick_source_frames(frame_times, duration, hz)
    if not indices:
        return []

    # `select` + passthrough frame timing emits exactly the chosen source
    # frames, in order, with no duplication or dropping — so output image k is
    # source frame indices[k]. `-fps_mode passthrough` is the modern spelling;
    # the old `-vsync 0` was REMOVED in FFmpeg 8 and hard-fails on current
    # builds, so it is only used as a fallback for older ones.
    cmd = [ffmpeg, "-v", "error", "-i", str(video),
           "-vf", f"select='{_select_expr(indices)}',scale={width}:-2",
           "-fps_mode", "passthrough", "-start_number", "0", str(pattern)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 and "fps_mode" in (proc.stderr or ""):
        cmd[-4:-2] = ["-vsync", "0"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr,
        )
    paths = sorted(out_dir.glob("g??????.png"))
    frames: list[GridFrame] = []
    for k, path in enumerate(paths):
        if k >= len(indices):
            break
        src = indices[k]
        frames.append(GridFrame(
            index=k, time_seconds=round(frame_times[src], 3), path=path, source_index=src,
        ))
    if blockers is not None and len(paths) != len(indices):
        blockers.add(
            "FRAME_EXTRACTION_INCOMPLETE",
            f"Requested {len(indices)} source frames but FFmpeg wrote {len(paths)}; "
            "some moments were never shown to the vision model.",
            severity=WARNING,
        )
    return frames


def _extract_resampled(
    ffmpeg: str, video: Path, pattern: Path, out_dir: Path, hz: float, width: int
) -> list[GridFrame]:
    """Legacy constant-rate path. Only used when the PTS ledger is unavailable."""
    subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(video),
         "-vf", f"fps={hz},scale={width}:-2", "-start_number", "0", str(pattern)],
        check=True,
    )
    frames: list[GridFrame] = []
    for p in sorted(out_dir.glob("g??????.png")):
        idx = int(p.stem[1:])
        frames.append(GridFrame(index=idx, time_seconds=idx / hz, path=p, source_index=-1))
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
