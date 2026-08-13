"""The UI bridge worker: JSONL over stdin/stdout, one request per line.

Run with::

    uv run python -m manuscript_reviewer.ui_bridge.worker

The worker executes one command at a time. Long jobs (``start_audit``)
interleave structured progress events before the final response. Cancellation
is owned by the supervising process (Rust) terminating the worker — the
worker never leaves half-written human-input files thanks to atomic writes,
and audit artifacts already written remain auditable.

No arbitrary Python execution, no shell access: only the fixed command table
below is reachable, and every input is validated before it touches the engine.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, TextIO

from pydantic import ValidationError

from .. import __version__
from ..caption_brain import CaptionBrainError, finalize_run
from ..media.ffmpeg_tools import FFmpegNotFoundError, find_tool, tool_version
from ..models.caption_brain import FinalReviewSignoff
from ..pipeline import run_audit
from ..progress import StageStatus
from ..rules.loader import load_rules
from ..validation.final_caption_validator import check_signoff
from . import UI_BRIDGE_PROTOCOL_VERSION
from .protocol import (
    BridgeCommandError,
    BridgeErrorCode,
    BridgeEvent,
    BridgeRequest,
    BridgeResponse,
)
from .serializers import (
    append_audit_history,
    audio_review_clip,
    caption_state,
    evidence_bundle,
    exact_frame,
    final_review_path,
    frame_record,
    human_facts_path,
    read_artifact,
    read_audit_history,
    read_review_inputs,
    read_text_artifact,
    read_ui_state,
    require_run_dir,
    review_decisions_path,
    review_queue,
    run_summary,
    save_final_signoff,
    save_human_facts,
    save_review_decisions,
    save_review_inputs,
    save_ui_state,
    save_visual_anchors,
    shots,
    ui_dir,
    waveform_metadata,
)

LOCK_FILE = "run.lock"


@contextlib.contextmanager
def run_lock(run_dir: Path) -> Iterator[None]:
    """A per-run writer lock so audit and finalize never race on one run.

    A lock left by a dead process (crash) is detected via pid liveness and
    recovered; a live holder raises RUN_LOCKED.
    """
    lock_dir = ui_dir(run_dir)
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / LOCK_FILE
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))
            break
        except FileExistsError:
            holder_pid: int | None = None
            with contextlib.suppress(OSError, ValueError):
                holder_pid = int(lock_path.read_text(encoding="utf-8").strip())
            if holder_pid is not None and holder_pid != os.getpid() and _pid_alive(holder_pid):
                raise BridgeCommandError(
                    BridgeErrorCode.RUN_LOCKED,
                    f"Run is locked by another process (pid {holder_pid})",
                ) from None
            # Stale lock from a crashed process: remove and retry.
            with contextlib.suppress(OSError):
                lock_path.unlink()
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            lock_path.unlink()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class _JsonlProgressReporter:
    """Emits pipeline progress as BridgeEvent lines on the worker's stdout."""

    def __init__(self, worker: BridgeWorker, request_id: str) -> None:
        self._worker = worker
        self._request_id = request_id

    def report(
        self,
        stage: str,
        status: StageStatus,
        detail: str | None = None,
        current: int | None = None,
        total: int | None = None,
    ) -> None:
        self._worker.emit_event(
            self._request_id,
            {
                "stage": stage,
                "status": status.value,
                "detail": detail,
                "current": current,
                "total": total,
            },
        )


class BridgeWorker:
    """Dispatches validated bridge requests to engine APIs and artifacts."""

    def __init__(self, stdin: TextIO, stdout: TextIO) -> None:
        self._stdin = stdin
        self._stdout = stdout
        self._handlers: dict[str, Callable[[str, dict[str, Any]], dict[str, Any]]] = {
            "health": self._cmd_health,
            "engine_info": self._cmd_engine_info,
            "get_rules": self._cmd_get_rules,
            "start_audit": self._cmd_start_audit,
            "load_run": self._cmd_run_summary,
            "get_run_summary": self._cmd_run_summary,
            "get_review_queue": self._cmd_review_queue,
            "get_review_item": self._cmd_review_item,
            "get_shots": self._cmd_shots,
            "get_frame_record": self._cmd_frame_record,
            "get_exact_frame": self._cmd_exact_frame,
            "get_evidence_bundle": self._cmd_evidence_bundle,
            "get_audio_review_clip": self._cmd_audio_review_clip,
            "get_waveform_metadata": self._cmd_waveform_metadata,
            "save_review_decisions": self._cmd_save_review_decisions,
            "save_human_facts": self._cmd_save_human_facts,
            "get_review_inputs": self._cmd_get_review_inputs,
            "save_visual_anchors": self._cmd_save_visual_anchors,
            "get_review_resolution": self._cmd_get_review_resolution,
            "get_media_dimensions": self._cmd_get_media_dimensions,
            "save_review_inputs": self._cmd_save_review_inputs,
            "append_audit_history": self._cmd_append_audit_history,
            "get_audit_history": self._cmd_get_audit_history,
            "save_ui_state": self._cmd_save_ui_state,
            "get_ui_state": self._cmd_get_ui_state,
            "resolve_rerun_request": self._cmd_resolve_rerun_request,
            "export_draft": self._cmd_export_draft,
            "finalize": self._cmd_finalize,
            "get_caption_state": self._cmd_caption_state,
            "create_final_signoff": self._cmd_create_final_signoff,
            "validate_final_signoff": self._cmd_validate_final_signoff,
            "export_caption": self._cmd_export_caption,
        }

    # -- transport ---------------------------------------------------------

    def serve_forever(self) -> None:
        for line in self._stdin:
            # Tolerate a UTF-8 BOM from Windows shells/redirects.
            line = line.lstrip("﻿").strip()
            if not line:
                continue
            self._handle_line(line)

    def _handle_line(self, line: str) -> None:
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            self._write_response(
                BridgeResponse(
                    request_id="unknown",
                    status="error",
                    error=BridgeCommandError(
                        BridgeErrorCode.INVALID_INPUT,
                        "Request is not valid JSON",
                        detail=str(exc),
                    ).to_error(),
                )
            )
            return
        try:
            request = BridgeRequest.model_validate(raw)
        except ValidationError as exc:
            request_id = "unknown"
            if isinstance(raw, dict) and isinstance(raw.get("request_id"), str):
                request_id = raw["request_id"]
            self._write_response(
                BridgeResponse(
                    request_id=request_id,
                    status="error",
                    error=BridgeCommandError(
                        BridgeErrorCode.INVALID_INPUT,
                        "Request does not match the bridge protocol",
                        detail=str(exc),
                    ).to_error(),
                )
            )
            return
        self._write_response(self.handle_request(request))

    def handle_request(self, request: BridgeRequest) -> BridgeResponse:
        if request.protocol_version != UI_BRIDGE_PROTOCOL_VERSION:
            return BridgeResponse(
                request_id=request.request_id,
                status="error",
                error=BridgeCommandError(
                    BridgeErrorCode.PROTOCOL_VERSION_MISMATCH,
                    f"Bridge protocol {request.protocol_version} is not supported "
                    f"(worker speaks {UI_BRIDGE_PROTOCOL_VERSION})",
                ).to_error(),
            )
        handler = self._handlers.get(request.command)
        if handler is None:
            return BridgeResponse(
                request_id=request.request_id,
                status="error",
                error=BridgeCommandError(
                    BridgeErrorCode.INVALID_COMMAND,
                    f"Unknown bridge command: {request.command}",
                ).to_error(),
            )
        try:
            payload = handler(request.request_id, request.payload)
        except BridgeCommandError as exc:
            return BridgeResponse(
                request_id=request.request_id, status="error", error=exc.to_error()
            )
        except Exception as exc:  # a worker must never die mid-protocol
            return BridgeResponse(
                request_id=request.request_id,
                status="error",
                error=BridgeCommandError(
                    BridgeErrorCode.ENGINE_CRASH,
                    "Engine command failed unexpectedly",
                    detail=f"{type(exc).__name__}: {exc}",
                ).to_error(),
            )
        return BridgeResponse(request_id=request.request_id, status="ok", payload=payload)

    def emit_event(self, request_id: str, payload: dict[str, Any]) -> None:
        event = BridgeEvent(request_id=request_id, event="progress", payload=payload)
        self._write_line(event.model_dump(mode="json"))

    def _write_response(self, response: BridgeResponse) -> None:
        self._write_line(response.model_dump(mode="json"))

    def _write_line(self, obj: dict[str, Any]) -> None:
        self._stdout.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")
        self._stdout.flush()

    # -- commands ----------------------------------------------------------

    def _cmd_health(self, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        ffmpeg_info: dict[str, Any] = {"available": False, "path": None, "version": None}
        ffprobe_info: dict[str, Any] = {"available": False, "path": None, "version": None}
        try:
            ffmpeg = find_tool("ffmpeg")
            ffmpeg_info = {
                "available": True,
                "path": str(ffmpeg),
                "version": tool_version(ffmpeg),
            }
            ffprobe = find_tool("ffprobe")
            ffprobe_info = {
                "available": True,
                "path": str(ffprobe),
                "version": tool_version(ffprobe),
            }
        except FFmpegNotFoundError:
            pass
        from ..audio.asr import runtime as asr_runtime

        tesseract = shutil.which("tesseract")
        tesseract_dir = os.environ.get("MANUSCRIPT_TESSERACT_DIR")

        def _env_health(name: str) -> dict[str, Any]:
            """Honest ASR env state: a packaged template alone is never
            reported as bootstrapped/cached (§Phase 6.1-17). The effective
            env dir honors MANUSCRIPT_ASR_WORKERS_DIR."""
            template = asr_runtime.WORKERS_DIR / name / "pyproject.toml"
            override = os.environ.get(asr_runtime.WORKERS_DIR_ENV, "").strip()
            effective = (
                Path(override) / name if override else asr_runtime.WORKERS_DIR / name
            )
            return {
                "worker_template_available": template.exists(),
                "effective_env_dir": str(effective),
                "worker_env_bootstrapped": (effective / ".venv").exists(),
            }

        return {
            "protocol_version": UI_BRIDGE_PROTOCOL_VERSION,
            "engine_version": __version__,
            "rules_version": load_rules().version,
            "ffmpeg": ffmpeg_info,
            "ffprobe": ffprobe_info,
            "asr_worker_envs": {
                "fw_env": _env_health("fw_env"),
                "wx_env": _env_health("wx_env"),
            },
            "ocr": {
                "tesseract_on_path": tesseract is not None,
                "tesseract_dir_env": tesseract_dir,
            },
        }

    def _cmd_engine_info(self, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "engine_version": __version__,
            "rules_version": load_rules().version,
            "protocol_version": UI_BRIDGE_PROTOCOL_VERSION,
        }

    def _cmd_get_rules(self, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """The versioned rule data (transition menu, literals...) so the UI
        never hardcodes a rule list."""
        rules = load_rules()
        return {"rules_version": rules.version, "rules": rules.data}

    def _cmd_start_audit(self, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        from ..audio.asr.runtime import ASRConfig

        video_raw = payload.get("video_path")
        if not isinstance(video_raw, str) or not video_raw:
            raise BridgeCommandError(
                BridgeErrorCode.INVALID_INPUT, "start_audit requires video_path"
            )
        video_path = Path(video_raw)
        if not video_path.is_file():
            raise BridgeCommandError(
                BridgeErrorCode.INVALID_INPUT, f"Video not found: {video_raw}"
            )
        artifacts_root = Path(str(payload.get("artifacts_root", "artifacts")))
        options_raw = payload.get("options", {})
        options: dict[str, Any] = options_raw if isinstance(options_raw, dict) else {}

        seed_path = self._optional_input_file(payload, "seed_path", "seed_text", "seed")
        feedback_path = self._optional_input_file(
            payload, "feedback_path", "feedback_text", "feedback"
        )
        anchors_raw = payload.get("visual_anchors_path")
        visual_anchors_path: Path | None = None
        if isinstance(anchors_raw, str) and anchors_raw:
            visual_anchors_path = Path(anchors_raw)
            if not visual_anchors_path.is_file():
                raise BridgeCommandError(
                    BridgeErrorCode.INVALID_INPUT,
                    f"Visual anchors file not found: {anchors_raw}",
                )
        asr_config = ASRConfig(
            model=str(options.get("asr_model", "large-v3-turbo")),
            device=str(options.get("asr_device", "auto")),
            compute_type=str(options.get("asr_compute_type", "auto")),
            language=(
                str(options["asr_language"]) if options.get("asr_language") else None
            ),
            bootstrap=bool(options.get("asr_bootstrap", True)),
        )
        reporter = _JsonlProgressReporter(self, request_id)
        result = run_audit(
            video_path,
            artifacts_root,
            seed_path=seed_path,
            extract_frames=bool(options.get("extract_frames", False)),
            shot_analysis=bool(options.get("shot_analysis", True)),
            shot_sensitivity=str(options.get("shot_sensitivity", "normal")),
            extract_shot_evidence=bool(options.get("extract_shot_evidence", True)),
            use_scdet=bool(options.get("scdet", True)),
            audio_analysis=bool(options.get("audio_analysis", True)),
            asr_enabled=bool(options.get("asr", True)),
            asr_config=asr_config,
            visual_intelligence=bool(options.get("visual_intelligence", True)),
            feedback_path=feedback_path,
            visual_anchors_path=visual_anchors_path,
            ocr_enabled=bool(options.get("ocr", True)),
            caption_brain=bool(options.get("caption_brain", True)),
            progress=reporter,
        )
        if result.fatal_error is not None:
            raise BridgeCommandError(
                BridgeErrorCode.FFMPEG_UNAVAILABLE
                if "ffmpeg" in result.fatal_error.lower()
                else BridgeErrorCode.ENGINE_CRASH,
                "Audit could not run",
                detail=result.fatal_error,
            )
        return {
            "run_id": result.run_id,
            "run_dir": str(result.run_dir),
            "status": result.status.value,
            "summary": run_summary(result.run_dir),
        }

    @staticmethod
    def _optional_input_file(
        payload: dict[str, Any], path_key: str, text_key: str, label: str
    ) -> Path | None:
        """A task input, either an existing file or pasted text snapshotted to
        an immutable temp file (the engine copies + hashes it into the run)."""
        raw_path = payload.get(path_key)
        if isinstance(raw_path, str) and raw_path:
            path = Path(raw_path)
            if not path.is_file():
                raise BridgeCommandError(
                    BridgeErrorCode.INVALID_INPUT, f"{label} file not found: {raw_path}"
                )
            return path
        raw_text = payload.get(text_key)
        if isinstance(raw_text, str) and raw_text.strip():
            fd, tmp_name = tempfile.mkstemp(prefix=f"mr_{label}_", suffix=".txt")
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(raw_text)
            return Path(tmp_name)
        return None

    def _cmd_run_summary(self, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        run_dir = require_run_dir(str(payload.get("run_dir", "")))
        return run_summary(run_dir)

    def _cmd_review_queue(self, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        run_dir = require_run_dir(str(payload.get("run_dir", "")))
        return review_queue(run_dir)

    def _cmd_review_item(self, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        run_dir = require_run_dir(str(payload.get("run_dir", "")))
        item_id = str(payload.get("item_id", ""))
        queue = review_queue(run_dir)
        for source in ("visual_items", "audio_items"):
            for item in queue[source]:
                if item.get("item_id") == item_id:
                    return {"source": source, "item": item}
        raise BridgeCommandError(
            BridgeErrorCode.INVALID_INPUT, f"Unknown review item: {item_id}"
        )

    def _cmd_shots(self, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        run_dir = require_run_dir(str(payload.get("run_dir", "")))
        return shots(run_dir)

    def _cmd_frame_record(self, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        run_dir = require_run_dir(str(payload.get("run_dir", "")))
        return frame_record(run_dir, self._require_int(payload, "frame_index"))

    def _cmd_exact_frame(self, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        run_dir = require_run_dir(str(payload.get("run_dir", "")))
        return exact_frame(run_dir, self._require_int(payload, "frame_index"))

    def _cmd_evidence_bundle(
        self, request_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        run_dir = require_run_dir(str(payload.get("run_dir", "")))
        return evidence_bundle(run_dir, str(payload.get("bundle_dir", "")))

    def _cmd_audio_review_clip(
        self, request_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        run_dir = require_run_dir(str(payload.get("run_dir", "")))
        return audio_review_clip(run_dir, str(payload.get("item_id", "")))

    def _cmd_waveform_metadata(
        self, request_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        run_dir = require_run_dir(str(payload.get("run_dir", "")))
        return waveform_metadata(run_dir)

    def _cmd_save_review_decisions(
        self, request_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        run_dir = require_run_dir(str(payload.get("run_dir", "")))
        decisions = payload.get("decisions")
        if not isinstance(decisions, list):
            raise BridgeCommandError(
                BridgeErrorCode.INVALID_INPUT, "save_review_decisions requires decisions[]"
            )
        with run_lock(run_dir):
            return save_review_decisions(run_dir, decisions)

    def _cmd_save_human_facts(
        self, request_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        run_dir = require_run_dir(str(payload.get("run_dir", "")))
        facts = payload.get("facts")
        if not isinstance(facts, list):
            raise BridgeCommandError(
                BridgeErrorCode.INVALID_INPUT, "save_human_facts requires facts[]"
            )
        with run_lock(run_dir):
            return save_human_facts(run_dir, facts)

    def _cmd_get_review_inputs(
        self, request_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        run_dir = require_run_dir(str(payload.get("run_dir", "")))
        return read_review_inputs(run_dir)

    def _cmd_save_visual_anchors(
        self, request_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        run_dir = require_run_dir(str(payload.get("run_dir", "")))
        anchors = payload.get("anchors")
        if not isinstance(anchors, list):
            raise BridgeCommandError(
                BridgeErrorCode.INVALID_INPUT, "save_visual_anchors requires anchors[]"
            )
        with run_lock(run_dir):
            return save_visual_anchors(run_dir, anchors)

    def _cmd_get_review_resolution(
        self, request_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        from .resolution import review_resolution

        run_dir = require_run_dir(str(payload.get("run_dir", "")))
        return review_resolution(run_dir)

    def _cmd_get_media_dimensions(
        self, request_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        from .resolution import media_dimensions

        run_dir = require_run_dir(str(payload.get("run_dir", "")))
        return media_dimensions(run_dir)

    def _cmd_save_review_inputs(
        self, request_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        run_dir = require_run_dir(str(payload.get("run_dir", "")))
        decisions = payload.get("decisions")
        facts = payload.get("facts")
        if not isinstance(decisions, list) or not isinstance(facts, list):
            raise BridgeCommandError(
                BridgeErrorCode.INVALID_INPUT,
                "save_review_inputs requires decisions[] and facts[]",
            )
        with run_lock(run_dir):
            return save_review_inputs(run_dir, decisions, facts)

    def _cmd_append_audit_history(
        self, request_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        run_dir = require_run_dir(str(payload.get("run_dir", "")))
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise BridgeCommandError(
                BridgeErrorCode.INVALID_INPUT, "append_audit_history requires entries[]"
            )
        return append_audit_history(run_dir, entries)

    def _cmd_get_audit_history(
        self, request_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        run_dir = require_run_dir(str(payload.get("run_dir", "")))
        return read_audit_history(run_dir)

    def _cmd_save_ui_state(self, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        run_dir = require_run_dir(str(payload.get("run_dir", "")))
        state = payload.get("state")
        if not isinstance(state, dict):
            raise BridgeCommandError(
                BridgeErrorCode.INVALID_INPUT, "save_ui_state requires state{}"
            )
        return save_ui_state(run_dir, state)

    def _cmd_get_ui_state(self, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        run_dir = require_run_dir(str(payload.get("run_dir", "")))
        return read_ui_state(run_dir)

    def _cmd_resolve_rerun_request(
        self, request_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Build a start_audit payload for RE-RUN WITH ANCHORS from the
        verified run's own provenance (manifest + snapshots + saved anchors).
        The frontend supplies only the run_dir; every input path comes from
        the engine's records — never from the renderer."""
        run_dir = require_run_dir(str(payload.get("run_dir", "")))
        manifest = read_artifact(run_dir, "manifest.json")
        source_video = Path(str(manifest.get("source_video_path", "")))
        if not source_video.is_file():
            raise BridgeCommandError(
                BridgeErrorCode.ARTIFACT_NOT_FOUND,
                "Source video is no longer at its recorded path; cannot re-run",
                detail=str(source_video),
            )
        anchors_file = ui_dir(run_dir) / "visual_anchors.json"
        if not anchors_file.is_file():
            raise BridgeCommandError(
                BridgeErrorCode.ARTIFACT_NOT_FOUND,
                "This run has no saved visual anchors to re-run with",
            )
        request: dict[str, Any] = {
            "video_path": str(source_video),
            # New sibling run under the same artifacts root, provenance intact.
            "artifacts_root": str(run_dir.parent.parent),
            "visual_anchors_path": str(anchors_file),
        }
        # Reuse the run's immutable seed/feedback snapshots when present.
        seed_snapshot = next(
            (p for p in run_dir.glob("seed.*") if p.is_file()), None
        )
        if seed_snapshot is not None:
            request["seed_path"] = str(seed_snapshot)
        feedback_snapshot = run_dir / "feedback" / "feedback_original.txt"
        if feedback_snapshot.is_file():
            request["feedback_path"] = str(feedback_snapshot)
        return {"request": request}

    def _cmd_export_draft(self, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """The current draft artifact for 'Save review draft as…'. Clearly a
        draft: served only from draft_review_only.md, never the ready file."""
        run_dir = require_run_dir(str(payload.get("run_dir", "")))
        text = read_text_artifact(run_dir, "caption/draft_review_only.md", required=False)
        if text is None:
            raise BridgeCommandError(
                BridgeErrorCode.ARTIFACT_NOT_FOUND, "Run has no draft caption artifact"
            )
        return {
            "markdown": text,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "path": str(run_dir / "caption" / "draft_review_only.md"),
        }

    def _cmd_finalize(self, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        run_dir = require_run_dir(str(payload.get("run_dir", "")))
        with run_lock(run_dir):
            try:
                output = finalize_run(
                    run_dir,
                    review_decisions_path=review_decisions_path(run_dir),
                    human_facts_path=human_facts_path(run_dir),
                    final_review_path=final_review_path(run_dir),
                )
            except CaptionBrainError as exc:
                message = str(exc)
                code = (
                    BridgeErrorCode.MANIFEST_MISMATCH
                    if "sha" in message.lower() or "manifest" in message.lower()
                    else BridgeErrorCode.ENGINE_CRASH
                )
                raise BridgeCommandError(
                    code, "Re-finalization failed", detail=message
                ) from exc
        return {
            "result": output.result.model_dump(mode="json"),
            "caption_state": caption_state(run_dir),
        }

    def _cmd_caption_state(self, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        run_dir = require_run_dir(str(payload.get("run_dir", "")))
        return caption_state(run_dir)

    def _cmd_create_final_signoff(
        self, request_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        run_dir = require_run_dir(str(payload.get("run_dir", "")))
        signoff_raw = payload.get("signoff")
        if not isinstance(signoff_raw, dict):
            raise BridgeCommandError(
                BridgeErrorCode.INVALID_INPUT, "create_final_signoff requires signoff{}"
            )
        with run_lock(run_dir):
            return save_final_signoff(run_dir, signoff_raw)

    def _cmd_validate_final_signoff(
        self, request_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Check the saved signoff against the CURRENT run identity + caption
        using the engine's own staleness check."""
        run_dir = require_run_dir(str(payload.get("run_dir", "")))
        path = final_review_path(run_dir)
        if path is None:
            return {"present": False, "valid": False, "stale": False, "reasons": []}
        try:
            signoff = FinalReviewSignoff.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise BridgeCommandError(
                BridgeErrorCode.INVALID_DECISION,
                "Saved signoff is unreadable",
                detail=str(exc),
            ) from exc
        manifest = read_artifact(run_dir, "manifest.json")
        caption_manifest = read_artifact(
            run_dir, "caption/caption_manifest.json", required=False
        )
        caption_sha = ""
        if isinstance(caption_manifest, dict):
            caption_sha = str(caption_manifest.get("rendered_caption_sha256", ""))
        check = check_signoff(
            signoff,
            manifest.get("source_video_sha256"),
            load_rules().version,
            caption_sha,
        )
        return {
            "present": True,
            "valid": check.valid,
            "stale": check.stale,
            "reasons": list(check.reasons),
        }

    def _cmd_export_caption(self, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """The exact ready artifact, only at READY_TO_ENTER. Byte-identical
        text: the UI must never reformat it."""
        run_dir = require_run_dir(str(payload.get("run_dir", "")))
        final_status = read_artifact(run_dir, "caption/final_status.json", required=False)
        readiness = (
            final_status.get("readiness") if isinstance(final_status, dict) else None
        )
        if readiness != "READY_TO_ENTER":
            raise BridgeCommandError(
                BridgeErrorCode.NOT_READY,
                f"Caption is {readiness or 'UNKNOWN'}, not READY_TO_ENTER; "
                "export is not allowed",
            )
        text = read_text_artifact(run_dir, "caption/ready_to_enter.md")
        assert text is not None
        return {
            "markdown": text,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "path": str(run_dir / "caption" / "ready_to_enter.md"),
        }

    @staticmethod
    def _require_int(payload: dict[str, Any], key: str) -> int:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise BridgeCommandError(
                BridgeErrorCode.INVALID_INPUT, f"{key} must be an integer"
            )
        return value


def main() -> None:
    # The protocol is UTF-8 JSONL regardless of console codepage — without
    # this, a packaged worker on Windows emits cp1252 bytes (e.g. em dashes
    # in blocker messages) that break the UTF-8 reader on the Rust side.
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    worker = BridgeWorker(sys.stdin, sys.stdout)
    worker.serve_forever()


if __name__ == "__main__":
    main()
