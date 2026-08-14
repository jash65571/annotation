"""Clean local web UI for AutoScribe. Stdlib only — no extra dependencies.

    uv run python -m autoscribe.webapp            # http://localhost:8765
    AUTOSCRIBE_VISION=cloud ANTHROPIC_API_KEY=...  uv run python -m autoscribe.webapp

Upload a video in the browser; it runs the pipeline and shows the resulting
DRAFT annotation.

PRIVACY: in `structured` mode AutoScribe uploads extracted audio and frames to
OpenAI. That is the opposite of the local-only Manuscript Reviewer engine, and
the UI says so on every run — see ``privacy_notice``.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import pipeline, render, structured
from . import review as review_mod
from .blockers import BlockerLog
from .validate import validate_caption

_STATIC = Path(__file__).parent / "static"
_JOBS: dict[str, dict[str, object]] = {}
_SEEDS: dict[str, dict[str, str]] = {}  # seed_id -> {"seed": ..., "feedback": ...}
_LOCK = threading.Lock()

#: Guardrails. A local tool still needs bounds: an unbounded upload fills the
#: disk, an absurd hz melts the CPU and the API bill, and unbounded concurrency
#: means N simultaneous jobs each spawning cloud calls.
MAX_UPLOAD_BYTES = int(os.environ.get("AUTOSCRIBE_MAX_UPLOAD_MB", "512")) * 1024 * 1024
MIN_HZ, MAX_HZ = 1.0, 60.0
MAX_CONCURRENT_JOBS = int(os.environ.get("AUTOSCRIBE_MAX_JOBS", "2"))
#: Finished jobs keep their temp dir this long so results can still be fetched.
_ACTIVE = threading.Semaphore(MAX_CONCURRENT_JOBS)


def privacy_notice() -> str:
    """What actually leaves this machine, stated for the mode in force."""
    mode = os.environ.get("AUTOSCRIBE_MODE", "structured")
    if mode == "structured":
        return (
            "CLOUD MODE: extracted audio and video frames from this file are "
            "uploaded to OpenAI for transcription and vision analysis. Do not use "
            "this mode for material you may not send to a third party."
        )
    backend = os.environ.get("AUTOSCRIBE_VISION", "openai")
    if backend in ("openai", "cloud"):
        return (
            f"CLOUD MODE: video frames are uploaded to the '{backend}' vision "
            f"service. Audio is transcribed locally."
        )
    return "LOCAL MODE: media is processed on this machine only."


def load_dotenv() -> None:
    """Load KEY=VALUE lines from a .env (cwd, autoscribe/, or repo root) into the
    environment without overwriting existing vars. No external dependency."""
    here = Path(__file__).resolve().parent
    for path in (Path.cwd() / ".env", here / ".env", here.parent / ".env"):
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _set(job: str, **kw: object) -> None:
    with _LOCK:
        _JOBS.setdefault(job, {}).update(kw)


def _run_job(job: str, video: Path, hz: float, seed: dict[str, str] | None = None) -> None:
    def progress(stage: str, frac: float) -> None:
        # Analysis fills 0-90% when a review pass follows.
        _set(job, stage=stage, fraction=frac * 0.9 if seed else frac)

    workspace = video.parent
    try:
        out_dir = workspace / "out"
        mode = os.environ.get("AUTOSCRIBE_MODE", "structured")
        blockers = BlockerLog()
        evidence = ""
        evidence_frames: list[tuple[float, Path]] = []
        language = ""
        if mode == "structured":
            ann = structured.analyze(video, out_dir, hz=hz, progress=progress)
            blockers.extend(ann.blockers)
            evidence = ann.evidence_summary()
            evidence_frames = ann.labelled_frames()
            language = ann.detected_language
            md_path = render.write(render.render(ann), out_dir, video.stem)
        else:
            backend = os.environ.get("AUTOSCRIBE_VISION", "openai")
            md_path = pipeline.run(
                video, out_dir, vision_backend=backend, hz=hz, progress=progress,
            )
        fresh = md_path.read_text(encoding="utf-8")
        # The draft is validated before anyone sees it, review pass or not.
        validate_caption(fresh, blockers, detected_language=language)

        if seed:
            _set(job, stage="reviewing seed", fraction=0.92)
            result = review_mod.review(
                fresh, seed["seed"], seed.get("feedback", ""),
                evidence=evidence, blockers=blockers, frames=evidence_frames,
                detected_language=language,
            )
            final_path = md_path.with_suffix(".reviewed.md")
            final_path.write_text(result["final_caption"], encoding="utf-8")
            _set(job, state="done", markdown=result["final_caption"], review=result,
                 blockers=blockers.as_dicts(), ready=result["ready"],
                 readiness_reason=result["readiness_reason"],
                 stage="done", fraction=1.0)
        else:
            ready, reason = blockers.readiness()
            _set(job, state="done", markdown=fresh, blockers=blockers.as_dicts(),
                 ready=ready, readiness_reason=reason, stage="done", fraction=1.0)
    except Exception as exc:  # surfaced to the UI
        _set(job, state="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        _ACTIVE.release()
        _cleanup_frames(workspace)


def _cleanup_frames(workspace: Path) -> None:
    """Delete extracted frames/audio once a job ends.

    A 15-second clip at 10 Hz is ~150 PNGs; the old code left every one of them,
    plus the uploaded video, in the system temp directory forever.
    """
    for path in workspace.rglob("frames"):
        shutil.rmtree(path, ignore_errors=True)
    for wav in workspace.rglob("audio.wav"):
        wav.unlink(missing_ok=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_a: object) -> None:  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict[str, object]) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send(200, (_STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/privacy":
            self._json(200, {"notice": privacy_notice()})
            return
        if parsed.path == "/api/status":
            job = parse_qs(parsed.query).get("job", [""])[0]
            with _LOCK:
                state = dict(_JOBS.get(job, {"state": "unknown"}))
            self._json(200, state)
            return
        self._send(404, b"not found", "text/plain")

    def _read_seed(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1 << 20:  # a caption is text; 1 MB is already generous
            self._send(413, b"seed too large", "text/plain")
            return
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            seed_text = str(data.get("seed", "")).strip()
        except (json.JSONDecodeError, UnicodeDecodeError):
            seed_text = ""
            data = {}
        if not seed_text:
            self._send(400, b"seed caption is empty", "text/plain")
            return
        seed_id = uuid.uuid4().hex
        with _LOCK:
            _SEEDS[seed_id] = {
                "seed": seed_text,
                "feedback": str(data.get("feedback", "")).strip(),
            }
        self._json(200, {"seed_id": seed_id})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/seed":
            self._read_seed()
            return
        if parsed.path != "/api/run":
            self._send(404, b"not found", "text/plain")
            return

        q = parse_qs(parsed.query)
        try:
            hz = float(q.get("hz", ["10"])[0])
        except ValueError:
            self._send(400, b"hz must be a number", "text/plain")
            return
        if not (MIN_HZ <= hz <= MAX_HZ):
            self._send(
                400, f"hz must be between {MIN_HZ} and {MAX_HZ}".encode(), "text/plain",
            )
            return

        name = q.get("name", ["upload.mp4"])[0]
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            self._send(400, b"empty upload", "text/plain")
            return
        if length > MAX_UPLOAD_BYTES:
            self._send(
                413,
                f"upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB".encode(),
                "text/plain",
            )
            return
        if not _ACTIVE.acquire(blocking=False):
            self._send(
                503,
                f"{MAX_CONCURRENT_JOBS} job(s) already running; retry shortly".encode(),
                "text/plain",
            )
            return

        workspace = Path(tempfile.mkdtemp(prefix="autoscribe_"))
        tmp = workspace / Path(name).name
        written = 0
        try:
            with tmp.open("wb") as fh:
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    fh.write(chunk)
                    written += len(chunk)
                    remaining -= len(chunk)
        except OSError as exc:
            _ACTIVE.release()
            shutil.rmtree(workspace, ignore_errors=True)
            self._send(500, f"upload failed: {exc}".encode(), "text/plain")
            return
        if written == 0:
            _ACTIVE.release()
            shutil.rmtree(workspace, ignore_errors=True)
            self._send(400, b"empty upload", "text/plain")
            return

        seed = None
        seed_id = q.get("seed_id", [""])[0]
        if seed_id:
            with _LOCK:
                seed = _SEEDS.pop(seed_id, None)
            if seed is None:
                _ACTIVE.release()
                shutil.rmtree(workspace, ignore_errors=True)
                self._send(400, b"unknown seed_id", "text/plain")
                return

        job = uuid.uuid4().hex
        _set(job, state="running", stage="queued", fraction=0.0,
             privacy=privacy_notice())
        threading.Thread(target=_run_job, args=(job, tmp, hz, seed), daemon=True).start()
        self._json(200, {"job": job})


def main(host: str = "127.0.0.1", port: int = 8765) -> None:
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set — create a .env (see .env.example).")
    server = ThreadingHTTPServer((host, port), Handler)
    backend = os.environ.get("AUTOSCRIBE_VISION", "cloud")
    print(f"AutoScribe UI -> http://{host}:{port}  (vision={backend})")
    print(privacy_notice())
    server.serve_forever()


if __name__ == "__main__":
    main()
