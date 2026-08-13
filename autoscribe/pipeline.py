"""Orchestrator: video path in → annotation markdown out.

    from autoscribe.pipeline import run
    md_path = run(Path("clip.mp4"), out_dir=Path("out"), vision="cloud", hz=10)

Emits progress via an optional callback so a UI can show live status.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from . import asr, assemble, frames, vision

Progress = Callable[[str, float], None]


def _noop(stage: str, frac: float) -> None:  # pragma: no cover - default
    pass


def run(
    video: Path,
    out_dir: Path,
    *,
    vision_backend: str = "cloud",
    hz: float = 10.0,
    scene_threshold: float = 0.15,
    asr_model: str = "large-v3-turbo",
    progress: Progress = _noop,
) -> Path:
    video = Path(video)
    work = out_dir / video.stem
    work.mkdir(parents=True, exist_ok=True)

    progress("probe", 0.02)
    duration = frames.probe_duration(video)

    progress("audio", 0.08)
    wav = asr.extract_audio(video, work / "audio.wav")
    progress("transcribe", 0.15)
    try:
        speech = asr.transcribe(wav, model_name=asr_model)
        speech_status = "ok"
    except Exception as exc:  # ASR is optional; never block the visual pass
        speech = []
        speech_status = f"unavailable ({type(exc).__name__}: {exc})"

    progress("frames", 0.45)
    grid = frames.extract_grid(video, work / "frames", hz=hz)
    scenes = frames.scene_change_times(video, threshold=scene_threshold)
    keyframes = frames.select_keyframes(grid, scenes)

    progress("vision", 0.55)
    backend = vision.make_backend(vision_backend)
    described: list[assemble.Described] = []
    total = len(keyframes) or 1
    for i, kf in enumerate(keyframes):
        text = backend.describe(kf.path)
        described.append(assemble.Described(time_seconds=kf.time_seconds, text=text))
        progress("vision", 0.55 + 0.4 * (i + 1) / total)

    progress("assemble", 0.97)
    md = assemble.build_markdown(
        video_name=video.name, duration=duration, speech=speech, visuals=described, hz=hz,
        speech_status=speech_status,
    )
    out = assemble.write_outputs(md, out_dir, video.stem)
    progress("done", 1.0)
    return out
