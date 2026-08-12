"""UI bridge protocol + worker tests (Phase 6, spec §100).

Every request id must match its response; malformed input must produce typed
errors, never arbitrary execution; the worker must expose engine truth without
re-implementing it.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from manuscript_reviewer.ui_bridge import UI_BRIDGE_PROTOCOL_VERSION
from manuscript_reviewer.ui_bridge.protocol import (
    BridgeCommandError,
    BridgeErrorCode,
    BridgeRequest,
)
from manuscript_reviewer.ui_bridge.serializers import (
    atomic_write_json,
    require_run_dir,
)
from manuscript_reviewer.ui_bridge.worker import BridgeWorker, run_lock

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def make_worker() -> tuple[BridgeWorker, io.StringIO]:
    out = io.StringIO()
    return BridgeWorker(io.StringIO(), out), out


def request(command: str, payload: dict | None = None, request_id: str = "req-1",
            protocol_version: int = UI_BRIDGE_PROTOCOL_VERSION) -> BridgeRequest:
    return BridgeRequest(
        request_id=request_id,
        command=command,
        payload=payload or {},
        protocol_version=protocol_version,
    )


# ---------------------------------------------------------------------------
# Protocol / transport
# ---------------------------------------------------------------------------


def test_bad_json_line_yields_typed_error() -> None:
    worker, out = make_worker()
    worker._handle_line("{this is not json")
    response = json.loads(out.getvalue())
    assert response["status"] == "error"
    assert response["error"]["code"] == "INVALID_INPUT"


def test_malformed_request_echoes_request_id_when_present() -> None:
    worker, out = make_worker()
    worker._handle_line(json.dumps({"request_id": "abc", "bogus": True}))
    response = json.loads(out.getvalue())
    assert response["request_id"] == "abc"
    assert response["status"] == "error"
    assert response["error"]["code"] == "INVALID_INPUT"


def test_protocol_version_mismatch_is_blocking_and_typed() -> None:
    worker, _ = make_worker()
    response = worker.handle_request(request("health", protocol_version=999))
    assert response.status == "error"
    assert response.error is not None
    assert response.error.code == BridgeErrorCode.PROTOCOL_VERSION_MISMATCH
    assert response.request_id == "req-1"


def test_unknown_command_is_invalid_command() -> None:
    worker, _ = make_worker()
    response = worker.handle_request(request("run_arbitrary_python"))
    assert response.status == "error"
    assert response.error is not None
    assert response.error.code == BridgeErrorCode.INVALID_COMMAND


def test_request_id_matches_on_success() -> None:
    worker, _ = make_worker()
    response = worker.handle_request(request("engine_info", request_id="match-me"))
    assert response.status == "ok"
    assert response.request_id == "match-me"


# ---------------------------------------------------------------------------
# Health / info / rules
# ---------------------------------------------------------------------------


def test_health_reports_versions_and_tools() -> None:
    worker, _ = make_worker()
    response = worker.handle_request(request("health"))
    assert response.status == "ok"
    payload = response.payload
    assert payload is not None
    assert payload["protocol_version"] == UI_BRIDGE_PROTOCOL_VERSION
    assert payload["engine_version"]
    assert payload["rules_version"]
    assert set(payload["ffmpeg"]) == {"available", "path", "version"}


def test_rules_exposed_so_ui_never_hardcodes_menus() -> None:
    worker, _ = make_worker()
    response = worker.handle_request(request("get_rules"))
    assert response.status == "ok"
    assert response.payload is not None
    assert response.payload["rules_version"]
    assert isinstance(response.payload["rules"], dict)


# ---------------------------------------------------------------------------
# Run directory guards
# ---------------------------------------------------------------------------


def test_missing_run_dir_is_run_not_found(tmp_path: Path) -> None:
    with pytest.raises(BridgeCommandError) as excinfo:
        require_run_dir(str(tmp_path / "nope"))
    assert excinfo.value.code == BridgeErrorCode.RUN_NOT_FOUND


def test_get_run_summary_missing_run(tmp_path: Path) -> None:
    worker, _ = make_worker()
    response = worker.handle_request(
        request("get_run_summary", {"run_dir": str(tmp_path / "missing")})
    )
    assert response.status == "error"
    assert response.error is not None
    assert response.error.code == BridgeErrorCode.RUN_NOT_FOUND


# ---------------------------------------------------------------------------
# Human-input persistence guards
# ---------------------------------------------------------------------------


def _fake_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    return run_dir


def _decision(**overrides: object) -> dict:
    base: dict = {
        "decision_id": "D-1",
        "subject_id": "SR-0001",
        "decision_type": "SPEECH_VERIFICATION",
        "value": "verified",
        "decided_by": "reviewer@example",
        "bound_video_sha256": "a" * 64,
        "bound_rules_version": "v1.3.0",
    }
    base.update(overrides)
    return base


def test_save_review_decisions_rejects_machine_author(tmp_path: Path) -> None:
    worker, _ = make_worker()
    run_dir = _fake_run_dir(tmp_path)
    response = worker.handle_request(
        request(
            "save_review_decisions",
            {"run_dir": str(run_dir), "decisions": [_decision(decided_by="machine")]},
        )
    )
    assert response.status == "error"
    assert response.error is not None
    assert response.error.code == BridgeErrorCode.INVALID_DECISION


def test_save_review_decisions_rejects_unbound(tmp_path: Path) -> None:
    worker, _ = make_worker()
    run_dir = _fake_run_dir(tmp_path)
    response = worker.handle_request(
        request(
            "save_review_decisions",
            {"run_dir": str(run_dir), "decisions": [_decision(bound_video_sha256=None)]},
        )
    )
    assert response.status == "error"
    assert response.error is not None
    assert response.error.code == BridgeErrorCode.INVALID_DECISION


def test_save_review_decisions_persists_valid_file(tmp_path: Path) -> None:
    worker, _ = make_worker()
    run_dir = _fake_run_dir(tmp_path)
    response = worker.handle_request(
        request(
            "save_review_decisions",
            {"run_dir": str(run_dir), "decisions": [_decision()]},
        )
    )
    assert response.status == "ok"
    saved = json.loads(
        (run_dir / "ui" / "review_decisions.json").read_text(encoding="utf-8")
    )
    assert saved["decisions"][0]["decision_id"] == "D-1"
    assert saved["decisions"][0]["decided_by"] == "reviewer@example"


def test_save_human_fact_without_evidence_is_rejected(tmp_path: Path) -> None:
    worker, _ = make_worker()
    run_dir = _fake_run_dir(tmp_path)
    fact = {
        "fact_id": "HF-1",
        "fact_type": "SCENE",
        "text_value": "A kitchen",
        "decided_by": "reviewer@example",
        "bound_video_sha256": "a" * 64,
        "bound_rules_version": "v1.3.0",
        "evidence_refs": [],
    }
    response = worker.handle_request(
        request("save_human_facts", {"run_dir": str(run_dir), "facts": [fact]})
    )
    assert response.status == "error"
    assert response.error is not None
    assert response.error.code == BridgeErrorCode.INVALID_DECISION
    assert "evidence" in response.error.message


def test_export_caption_blocked_unless_ready(tmp_path: Path) -> None:
    worker, _ = make_worker()
    run_dir = _fake_run_dir(tmp_path)
    (run_dir / "caption").mkdir()
    (run_dir / "caption" / "final_status.json").write_text(
        json.dumps({"readiness": "REVIEW_REQUIRED"}), encoding="utf-8"
    )
    response = worker.handle_request(request("export_caption", {"run_dir": str(run_dir)}))
    assert response.status == "error"
    assert response.error is not None
    assert response.error.code == BridgeErrorCode.NOT_READY


def test_export_caption_returns_exact_bytes_when_ready(tmp_path: Path) -> None:
    worker, _ = make_worker()
    run_dir = _fake_run_dir(tmp_path)
    (run_dir / "caption").mkdir()
    (run_dir / "caption" / "final_status.json").write_text(
        json.dumps({"readiness": "READY_TO_ENTER"}), encoding="utf-8"
    )
    exact = "b3729c4211b2\n\n[Shot 1: 0.0s-1.0s]\nHard cut.\n"
    (run_dir / "caption" / "ready_to_enter.md").write_text(
        exact, encoding="utf-8", newline=""
    )
    response = worker.handle_request(request("export_caption", {"run_dir": str(run_dir)}))
    assert response.status == "ok"
    assert response.payload is not None
    assert response.payload["markdown"] == exact


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------


def test_run_lock_conflict_is_typed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = _fake_run_dir(tmp_path)
    (run_dir / "ui").mkdir()
    (run_dir / "ui" / "run.lock").write_text("999999", encoding="utf-8")
    monkeypatch.setattr(
        "manuscript_reviewer.ui_bridge.worker._pid_alive", lambda pid: True
    )
    with pytest.raises(BridgeCommandError) as excinfo, run_lock(run_dir):
        pass
    assert excinfo.value.code == BridgeErrorCode.RUN_LOCKED


def test_stale_lock_is_recovered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = _fake_run_dir(tmp_path)
    (run_dir / "ui").mkdir()
    (run_dir / "ui" / "run.lock").write_text("999999", encoding="utf-8")
    monkeypatch.setattr(
        "manuscript_reviewer.ui_bridge.worker._pid_alive", lambda pid: False
    )
    with run_lock(run_dir):
        assert (run_dir / "ui" / "run.lock").exists()
    assert not (run_dir / "ui" / "run.lock").exists()


def test_atomic_write_json_replaces_not_corrupts(tmp_path: Path) -> None:
    target = tmp_path / "ui" / "review_decisions.json"
    atomic_write_json(target, {"decisions": [1]})
    atomic_write_json(target, {"decisions": [1, 2]})
    assert json.loads(target.read_text(encoding="utf-8")) == {"decisions": [1, 2]}
    leftovers = [p for p in target.parent.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


# ---------------------------------------------------------------------------
# Full worker flow on a real (synthetic) clip: audit → summary → frames →
# decisions → finalize. ASR/OCR disabled: deterministic evidence only.
# ---------------------------------------------------------------------------


def test_worker_full_flow(clip_24fps: Path, tmp_path: Path) -> None:
    worker, out = make_worker()
    response = worker.handle_request(
        request(
            "start_audit",
            {
                "video_path": str(clip_24fps),
                "artifacts_root": str(tmp_path / "artifacts"),
                "options": {"asr": False, "ocr": False},
            },
            request_id="audit-1",
        )
    )
    assert response.status == "ok", response.error
    assert response.payload is not None
    run_dir = response.payload["run_dir"]
    assert response.payload["run_id"]

    # Progress events were emitted for the audit request, well-formed JSONL.
    events = [json.loads(line) for line in out.getvalue().splitlines() if line]
    stages = [e["payload"]["stage"] for e in events if e.get("event") == "progress"]
    assert "media_verification" in stages
    assert "frame_ledger" in stages
    assert "done" in stages
    assert all(e["request_id"] == "audit-1" for e in events)

    summary = worker.handle_request(request("get_run_summary", {"run_dir": run_dir}))
    assert summary.status == "ok"
    assert summary.payload is not None
    assert summary.payload["manifest"]["run_id"] == response.payload["run_id"]

    record = worker.handle_request(
        request("get_frame_record", {"run_dir": run_dir, "frame_index": 0})
    )
    assert record.status == "ok"
    assert record.payload is not None
    assert record.payload["frame_index"] == 0

    frame = worker.handle_request(
        request("get_exact_frame", {"run_dir": run_dir, "frame_index": 3})
    )
    assert frame.status == "ok"
    assert frame.payload is not None
    assert Path(frame.payload["path"]).exists()
    # Served again from cache, same identity.
    frame2 = worker.handle_request(
        request("get_exact_frame", {"run_dir": run_dir, "frame_index": 3})
    )
    assert frame2.status == "ok"
    assert frame2.payload is not None
    assert frame2.payload["path"] == frame.payload["path"]

    queue = worker.handle_request(request("get_review_queue", {"run_dir": run_dir}))
    assert queue.status == "ok"

    caption = worker.handle_request(request("get_caption_state", {"run_dir": run_dir}))
    assert caption.status == "ok"
    assert caption.payload is not None
    assert caption.payload["final_status"] is not None

    finalized = worker.handle_request(request("finalize", {"run_dir": run_dir}))
    assert finalized.status == "ok", finalized.error
    assert finalized.payload is not None
    assert finalized.payload["result"]["readiness"] in (
        "BLOCKED",
        "REVIEW_REQUIRED",
        "READY_FOR_FINAL_REVIEW",
    )

    signoff_state = worker.handle_request(
        request("validate_final_signoff", {"run_dir": run_dir})
    )
    assert signoff_state.status == "ok"
    assert signoff_state.payload is not None
    assert signoff_state.payload["present"] is False
