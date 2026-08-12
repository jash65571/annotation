"""Phase 2 runtime benchmark over synthetic clips.

Usage:  uv run python scripts/benchmark.py

Generates 24/30/60 fps synthetic clips (~15 s, two hard cuts each), runs the
full audit with shot analysis, and reports per-stage wall-clock timings from
the run manifest. Peak-memory tracking is intentionally omitted (would need
psutil; stage timings cover the optimization targets).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "engine"))

from manuscript_reviewer.media.ffmpeg_tools import find_tool, run_tool
from manuscript_reviewer.pipeline import run_audit

STAGES = [
    "probe",
    "enumerate_frames",
    "count_frames_crosscheck",
    "shot_metric_decode",
    "shot_pair_metrics",
    "shot_scdet",
    "shot_baselines",
    "shot_regions",
    "shot_candidates",
    "shot_verification",
    "shot_evidence_render",
    "shot_proposals",
    "shot_validation",
    "shot_artifacts",
    "shot_analysis_total",
]


def make_clip(directory: Path, rate: str) -> Path:
    path = directory / f"bench_{rate.replace('/', '_')}.mp4"
    ffmpeg = find_tool("ffmpeg")
    run_tool(
        ffmpeg,
        [
            "-v", "error",
            "-f", "lavfi", "-i", f"testsrc2=duration=5:rate={rate}:size=640x360",
            "-f", "lavfi", "-i", f"smptebars=duration=5:rate={rate}:size=640x360",
            "-f", "lavfi", "-i", f"rgbtestsrc=duration=5:rate={rate}:size=640x360",
            "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-y", str(path),
        ],
    )
    return path


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mr_bench_") as tmp:
        tmp_path = Path(tmp)
        for rate in ("24", "30", "60"):
            clip = make_clip(tmp_path, rate)
            result = run_audit(
                clip,
                artifacts_root=tmp_path / "runs",
                shot_analysis=True,
                extract_shot_evidence=True,
            )
            manifest = result.manifest
            assert manifest is not None
            frames = result.ledger.frame_count if result.ledger else 0
            print(f"\n=== {clip.name}: {frames} frames, 15 s ===")
            for stage in STAGES:
                value = manifest.stage_timings_seconds.get(stage)
                if value is not None:
                    print(f"  {stage:<28}{value:>8.3f}s")
            print(f"  {'TOTAL AUDIT':<28}{manifest.analysis_duration_seconds:>8.3f}s")
            shot = result.shot_truth
            if shot:
                print(
                    f"  candidates={shot.merged_candidate_count} "
                    f"supported={shot.supported_count} "
                    f"review={shot.review_required_count} "
                    f"status={shot.overall_status}"
                )


if __name__ == "__main__":
    main()
