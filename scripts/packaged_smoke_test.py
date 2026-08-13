"""Packaged engine sidecar smoke test (spec §106).

Drives the staged sidecar exe over its real JSONL protocol:
health handshake → synthetic clip audit (no ASR/OCR) → load run →
finalize → exact frame. Run with:  uv run python scripts/packaged_smoke_test.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SIDECAR = (
    REPO
    / "desktop"
    / "src-tauri"
    / "binaries"
    / "manuscript-engine-worker"
    / "manuscript-engine-worker.exe"
)


def find_ffmpeg() -> Path:
    staged = REPO / "desktop" / "src-tauri" / "ffmpeg" / "bin" / "ffmpeg.exe"
    if staged.exists():
        return staged
    env_dir = os.environ.get("MANUSCRIPT_FFMPEG_DIR")
    if env_dir and (Path(env_dir) / "ffmpeg.exe").exists():
        return Path(env_dir) / "ffmpeg.exe"
    which = shutil.which("ffmpeg")
    if which:
        return Path(which)
    fallback = Path.home() / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
    if fallback.exists():
        return fallback
    raise SystemExit("no ffmpeg found for synthetic clip generation")


def main() -> None:
    if not SIDECAR.exists():
        raise SystemExit("Sidecar not staged; run scripts/build_engine_sidecar.ps1")
    work = Path(tempfile.mkdtemp(prefix="mr_packaged_smoke_"))
    clip = work / "smoke.mp4"
    ffmpeg = find_ffmpeg()
    subprocess.run(
        [
            str(ffmpeg), "-v", "error",
            "-f", "lavfi", "-i", "testsrc2=duration=2:rate=24:size=320x240",
            "-pix_fmt", "yuv420p", "-y", str(clip),
        ],
        check=True,
    )

    env = dict(os.environ)
    ffmpeg_dir = ffmpeg.parent
    env["MANUSCRIPT_FFMPEG_DIR"] = str(ffmpeg_dir)
    env["MANUSCRIPT_ASR_WORKERS_DIR"] = str(work / "asr_workers")

    started = time.perf_counter()
    proc = subprocess.Popen(
        [str(SIDECAR)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=env,
        cwd=str(SIDECAR.parent),
        text=True,
        encoding="utf-8",
    )
    assert proc.stdin is not None and proc.stdout is not None

    def request(rid: str, command: str, payload: dict) -> dict:
        line = json.dumps(
            {
                "request_id": rid,
                "command": command,
                "payload": payload,
                "protocol_version": 1,
            }
        )
        proc.stdin.write(line + "\n")
        proc.stdin.flush()
        while True:
            raw = proc.stdout.readline()
            if not raw:
                raise SystemExit(f"sidecar exited during {command}")
            obj = json.loads(raw)
            if obj.get("event") == "progress":
                stage = obj["payload"].get("stage")
                status = obj["payload"].get("status")
                print(f"  progress: {stage} {status}")
                continue
            if obj.get("request_id") != rid:
                continue
            return obj

    health = request("s1", "health", {})
    assert health["status"] == "ok", health
    payload = health["payload"]
    assert payload["protocol_version"] == 1
    print(
        f"health OK in {time.perf_counter() - started:.2f}s  "
        f"engine={payload['engine_version']} rules={payload['rules_version']} "
        f"ffmpeg={payload['ffmpeg']['available']}"
    )
    assert payload["ffmpeg"]["available"], "sidecar must locate the provided ffmpeg"

    audit_start = time.perf_counter()
    audit = request(
        "s2",
        "start_audit",
        {
            "video_path": str(clip),
            "artifacts_root": str(work / "artifacts"),
            "options": {"asr": False, "ocr": False},
        },
    )
    assert audit["status"] == "ok", audit.get("error")
    run_dir = Path(audit["payload"]["run_dir"])
    print(
        f"audit OK in {time.perf_counter() - audit_start:.2f}s  "
        f"status={audit['payload']['status']} run={run_dir}"
    )
    for artifact in (
        "manifest.json",
        "frames.jsonl",
        "shot_qc.json",
        "audio/audio_qc.json",
        "caption/final_status.json",
    ):
        assert (run_dir / artifact).exists(), f"missing artifact {artifact}"

    summary = request("s3", "get_run_summary", {"run_dir": str(run_dir)})
    assert summary["status"] == "ok"

    fin_start = time.perf_counter()
    finalize = request("s4", "finalize", {"run_dir": str(run_dir)})
    assert finalize["status"] == "ok", finalize.get("error")
    print(
        f"finalize OK in {time.perf_counter() - fin_start:.2f}s  "
        f"readiness={finalize['payload']['result']['readiness']}"
    )

    frame = request("s5", "get_exact_frame", {"run_dir": str(run_dir), "frame_index": 5})
    assert frame["status"] == "ok", frame.get("error")
    assert Path(frame["payload"]["path"]).exists()
    print(f"exact frame OK {frame['payload']['path']}")

    proc.stdin.close()
    proc.wait(timeout=10)
    print("PACKAGED SMOKE TEST PASSED")


if __name__ == "__main__":
    sys.exit(main())
