"""Clean local web UI for AutoScribe. Stdlib only — no extra dependencies.

    uv run python -m autoscribe.webapp            # http://localhost:8765
    AUTOSCRIBE_VISION=cloud ANTHROPIC_API_KEY=...  uv run python -m autoscribe.webapp

Upload a video in the browser; it runs the pipeline and shows the final
copy-paste annotation.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import pipeline, render, structured

_STATIC = Path(__file__).parent / "static"
_JOBS: dict[str, dict[str, object]] = {}
_LOCK = threading.Lock()


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


def _run_job(job: str, video: Path, hz: float) -> None:
    def progress(stage: str, frac: float) -> None:
        _set(job, stage=stage, fraction=frac)

    try:
        out_dir = video.parent / "out"
        mode = os.environ.get("AUTOSCRIBE_MODE", "structured")
        if mode == "structured":
            ann = structured.analyze(video, out_dir, hz=hz, progress=progress)
            md_path = render.write(render.render(ann), out_dir, video.stem)
        else:
            backend = os.environ.get("AUTOSCRIBE_VISION", "openai")
            md_path = pipeline.run(
                video, out_dir, vision_backend=backend, hz=hz, progress=progress,
            )
        _set(job, state="done", markdown=md_path.read_text(encoding="utf-8"),
             stage="done", fraction=1.0)
    except Exception as exc:  # surfaced to the UI
        _set(job, state="error", error=f"{type(exc).__name__}: {exc}")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_a: object) -> None:  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send(200, (_STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/status":
            job = parse_qs(parsed.query).get("job", [""])[0]
            with _LOCK:
                state = dict(_JOBS.get(job, {"state": "unknown"}))
            self._send(200, json.dumps(state).encode(), "application/json")
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            self._send(404, b"not found", "text/plain")
            return
        q = parse_qs(parsed.query)
        hz = float(q.get("hz", ["10"])[0])
        name = q.get("name", ["upload.mp4"])[0]
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            self._send(400, b"empty upload", "text/plain")
            return
        tmp = Path(tempfile.mkdtemp(prefix="autoscribe_")) / Path(name).name
        with tmp.open("wb") as fh:
            remaining = length
            while remaining > 0:
                chunk = self.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                fh.write(chunk)
                remaining -= len(chunk)
        job = uuid.uuid4().hex
        _set(job, state="running", stage="queued", fraction=0.0)
        threading.Thread(target=_run_job, args=(job, tmp, hz), daemon=True).start()
        self._send(200, json.dumps({"job": job}).encode(), "application/json")


def main(host: str = "127.0.0.1", port: int = 8765) -> None:
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set — create a .env (see .env.example).")
    server = ThreadingHTTPServer((host, port), Handler)
    backend = os.environ.get("AUTOSCRIBE_VISION", "cloud")
    print(f"AutoScribe UI -> http://{host}:{port}  (vision={backend})")
    server.serve_forever()


if __name__ == "__main__":
    main()
