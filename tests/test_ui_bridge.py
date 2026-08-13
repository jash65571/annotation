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
# Full worker flow on a real (synthetic) clip: audit -> summary -> frames ->
# decisions -> finalize. ASR/OCR disabled: deterministic evidence only.
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
    # Signoff binding hash must be exposed (FinalScreen binds signoffs to it).
    assert caption.payload["rendered_caption_sha256"]

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

    # Anchor flow (§Phase 6.1-2): save an anchor, resolve a rerun request from
    # run provenance, run it, and verify the new run consumed the anchors
    # (hashed + snapshotted by the pipeline). The locked engine's tracking
    # slice is reserved, so consumption/provenance is the honest assertion.
    dims = worker.handle_request(request("get_media_dimensions", {"run_dir": run_dir}))
    assert dims.status == "ok"
    assert dims.payload is not None
    assert dims.payload["width"] > 0 and dims.payload["height"] > 0

    anchors_saved = worker.handle_request(
        request(
            "save_visual_anchors",
            {
                "run_dir": run_dir,
                "anchors": [
                    {
                        "frame_index": 3,
                        "x": 10,
                        "y": 12,
                        "width": 40,
                        "height": 30,
                        "entity_type": "object",
                        "label": "test-box",
                    }
                ],
            },
        )
    )
    assert anchors_saved.status == "ok"

    rerun = worker.handle_request(request("resolve_rerun_request", {"run_dir": run_dir}))
    assert rerun.status == "ok", rerun.error
    assert rerun.payload is not None
    rerun_request = rerun.payload["request"]
    assert rerun_request["visual_anchors_path"].endswith("visual_anchors.json")

    rerun_request["options"] = {"asr": False, "ocr": False}
    second_run = worker.handle_request(
        request("start_audit", rerun_request, request_id="audit-2")
    )
    assert second_run.status == "ok", second_run.error
    assert second_run.payload is not None
    new_run_dir = Path(second_run.payload["run_dir"])
    assert str(new_run_dir) != run_dir
    new_manifest = json.loads((new_run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert new_manifest["source_visual_anchors_sha256"], (
        "rerun must record the consumed anchors hash in provenance"
    )
    snapshots = list(new_run_dir.glob("visual_anchors*"))
    assert snapshots, "anchors snapshot must be copied into the new run"


# ---------------------------------------------------------------------------
# Full E2E through the worker to READY_TO_ENTER (spec §98): fixture run with
# a hash-complete manifest -> typed speed decision -> human fact -> finalize ->
# READY_FOR_FINAL_REVIEW -> human signoff -> READY_TO_ENTER -> export equals the
# ready artifact byte-for-byte.
# ---------------------------------------------------------------------------


def _manifested_ready_run(tmp_path: Path) -> Path:
    from fractions import Fraction

    from manuscript_reviewer.artifacts.writer import sha256_file
    from manuscript_reviewer.models.review_intelligence import SeedClaimType

    from .phase5_helpers import (
        RULES_VERSION,
        VIDEO_ID,
        VIDEO_SHA,
        make_shot,
        make_shot_truth,
        supported_claim,
        write_json,
        write_run_dir,
    )

    shots = make_shot_truth([make_shot(1, Fraction(0), Fraction(2), "Opening shot")])
    claims = [
        supported_claim(
            "CLM-C1",
            SeedClaimType.CHARACTER_EXISTS,
            "A person in a dark green jacket.",
            subject_ids=["C1"],
        ),
        supported_claim(
            "CLM-SC",
            SeedClaimType.SCENE_STATE,
            "A flat dirt path beside a canal.",
            source_field="SCENE",
        ),
    ]
    run_dir = write_run_dir(tmp_path, shots, seed_claims=claims)
    write_json(
        run_dir / "visual" / "speed" / "playback_speed_evidence.json",
        {
            "playback_speed_evidence": [
                {"shot_number": 1, "conclusion": "REGULAR_CANDIDATE", "review_required": True}
            ]
        },
    )
    # A REAL manifest with the actual hash of every artifact: STRICT evidence
    # verification passes because nothing was tampered with.
    artifacts = [
        {"path": p.relative_to(run_dir).as_posix(), "sha256": sha256_file(p)}
        for p in sorted(run_dir.rglob("*"))
        if p.is_file()
    ]
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source_video_sha256": VIDEO_SHA,
                "rules_version": RULES_VERSION,
                "source_video_path": f"C:/videos/{VIDEO_ID}.mp4",
                "run_id": "fixture-ready-run",
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_worker_reaches_ready_to_enter_and_exports_exact_bytes(tmp_path: Path) -> None:
    from .phase5_helpers import RULES_VERSION, VIDEO_SHA

    worker, _ = make_worker()
    run_dir = str(_manifested_ready_run(tmp_path))
    reviewer = "reviewer@test"

    saved = worker.handle_request(
        request(
            "save_review_decisions",
            {
                "run_dir": run_dir,
                "decisions": [
                    {
                        "decision_id": "D-SPEED",
                        "subject_id": "SPEED-1",
                        "decision_type": "PLAYBACK_SPEED",
                        "value": "regular",
                        "decided_by": reviewer,
                        "decided_at_utc": "2026-08-12T00:00:00Z",
                        "bound_video_sha256": VIDEO_SHA,
                        "bound_rules_version": RULES_VERSION,
                    }
                ],
            },
        )
    )
    assert saved.status == "ok", saved.error
    facts = worker.handle_request(
        request(
            "save_human_facts",
            {
                "run_dir": run_dir,
                "facts": [
                    {
                        "fact_id": "HF-ACT",
                        "fact_type": "VISUAL_ACTION",
                        "text_value": "C1 stands beside the canal.",
                        "shot_number": 1,
                        "character_ids": ["C1"],
                        "start_exact": "0",
                        "end_exact": "1",
                        "evidence_refs": [
                            {
                                "evidence_id": "EV-HF",
                                "evidence_type": "FRAME_RANGE",
                                "start_frame": 0,
                                "end_frame": 24,
                                "source": reviewer,
                            }
                        ],
                        "decided_by": reviewer,
                        "bound_video_sha256": VIDEO_SHA,
                        "bound_rules_version": RULES_VERSION,
                    }
                ],
            },
        )
    )
    assert facts.status == "ok", facts.error

    first = worker.handle_request(request("finalize", {"run_dir": run_dir}))
    assert first.status == "ok", first.error
    assert first.payload is not None
    assert first.payload["result"]["readiness"] == "READY_FOR_FINAL_REVIEW"
    caption_sha = first.payload["caption_state"]["rendered_caption_sha256"]
    assert caption_sha

    # Export must be refused below READY_TO_ENTER.
    refused = worker.handle_request(request("export_caption", {"run_dir": run_dir}))
    assert refused.status == "error"
    assert refused.error is not None
    assert refused.error.code == BridgeErrorCode.NOT_READY

    signed = worker.handle_request(
        request(
            "create_final_signoff",
            {
                "run_dir": run_dir,
                "signoff": {
                    "video_sha256": VIDEO_SHA,
                    "rules_version": RULES_VERSION,
                    "caption_sha256": caption_sha,
                    "reviewer": reviewer,
                    "reviewed_at_utc": "2026-08-12T01:00:00Z",
                    "golden_example_comparison_complete": True,
                    "platform_semantic_pass_complete": True,
                    "final_adversarial_read_complete": True,
                    "no_known_omissions_confirmed": True,
                    "no_known_hallucinations_confirmed": True,
                },
            },
        )
    )
    assert signed.status == "ok", signed.error

    second = worker.handle_request(request("finalize", {"run_dir": run_dir}))
    assert second.status == "ok", second.error
    assert second.payload is not None
    assert second.payload["result"]["readiness"] == "READY_TO_ENTER"

    exported = worker.handle_request(request("export_caption", {"run_dir": run_dir}))
    assert exported.status == "ok", exported.error
    assert exported.payload is not None
    on_disk = (Path(run_dir) / "caption" / "ready_to_enter.md").read_text(encoding="utf-8")
    assert exported.payload["markdown"] == on_disk

    validated = worker.handle_request(
        request("validate_final_signoff", {"run_dir": run_dir})
    )
    assert validated.status == "ok"
    assert validated.payload is not None
    assert validated.payload["valid"] is True
    assert validated.payload["stale"] is False


# ---------------------------------------------------------------------------
# Phase 6.1 hardening regressions
# ---------------------------------------------------------------------------


def test_evidence_bundle_sibling_prefix_escape_rejected(tmp_path: Path) -> None:
    """C:\\runs\\task1 must never admit C:\\runs\\task10 (string-prefix bug)."""
    from manuscript_reviewer.ui_bridge.serializers import evidence_bundle

    run_dir = tmp_path / "task1"
    sibling = tmp_path / "task10"
    run_dir.mkdir()
    sibling.mkdir()
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (sibling / "secret.png").write_bytes(b"png")
    with pytest.raises(BridgeCommandError) as excinfo:
        evidence_bundle(run_dir, "..\\task10")
    assert excinfo.value.code == BridgeErrorCode.INVALID_INPUT


def test_save_review_inputs_is_transactional(tmp_path: Path) -> None:
    """An invalid fact must leave BOTH previous files untouched."""
    worker, _ = make_worker()
    run_dir = _fake_run_dir(tmp_path)
    first = worker.handle_request(
        request(
            "save_review_inputs",
            {"run_dir": str(run_dir), "decisions": [_decision()], "facts": []},
        )
    )
    assert first.status == "ok"
    assert first.payload is not None
    assert first.payload["decision_count"] == 1
    revision = first.payload["review_input_revision_id"]
    assert revision

    bad_fact = {"fact_id": "HF-BAD", "fact_type": "SCENE", "decided_by": "machine"}
    second = worker.handle_request(
        request(
            "save_review_inputs",
            {
                "run_dir": str(run_dir),
                "decisions": [_decision(decision_id="D-2")],
                "facts": [bad_fact],
            },
        )
    )
    assert second.status == "error"
    # The previous pair is intact: decision D-1 from revision 1, no facts file
    # half-written with the bad fact.
    saved = json.loads(
        (run_dir / "ui" / "review_decisions.json").read_text(encoding="utf-8")
    )
    assert [d["decision_id"] for d in saved["decisions"]] == ["D-1"]
    assert saved["revision"] == revision
    leftovers = list((run_dir / "ui").glob("*.tmp"))
    assert leftovers == []


def test_audit_history_persists_and_survives(tmp_path: Path) -> None:
    worker, _ = make_worker()
    run_dir = _fake_run_dir(tmp_path)
    appended = worker.handle_request(
        request(
            "append_audit_history",
            {
                "run_dir": str(run_dir),
                "entries": [
                    {
                        "at_utc": "2026-08-13T00:00:00Z",
                        "reviewer": "reviewer@test",
                        "operation": "decision_saved",
                        "subject": "SPEED-1",
                    }
                ],
            },
        )
    )
    assert appended.status == "ok"
    # A fresh worker (app restart) still reads the history from disk.
    fresh_worker, _ = make_worker()
    history = fresh_worker.handle_request(
        request("get_audit_history", {"run_dir": str(run_dir)})
    )
    assert history.status == "ok"
    assert history.payload is not None
    assert history.payload["entries"][0]["operation"] == "decision_saved"


def test_ui_state_persists_skipped_items(tmp_path: Path) -> None:
    worker, _ = make_worker()
    run_dir = _fake_run_dir(tmp_path)
    saved = worker.handle_request(
        request(
            "save_ui_state",
            {"run_dir": str(run_dir), "state": {"skipped_item_ids": ["areview_0002"]}},
        )
    )
    assert saved.status == "ok"
    fresh_worker, _ = make_worker()
    loaded = fresh_worker.handle_request(request("get_ui_state", {"run_dir": str(run_dir)}))
    assert loaded.status == "ok"
    assert loaded.payload is not None
    assert loaded.payload["skipped_item_ids"] == ["areview_0002"]


def test_asr_health_reports_env_truth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A packaged worker template alone is never reported as bootstrapped."""
    from manuscript_reviewer.audio.asr import runtime as asr_runtime

    monkeypatch.setenv(asr_runtime.WORKERS_DIR_ENV, str(tmp_path))
    worker, _ = make_worker()
    response = worker.handle_request(request("health"))
    assert response.status == "ok"
    assert response.payload is not None
    fw = response.payload["asr_worker_envs"]["fw_env"]
    assert fw["worker_template_available"] is True  # repo templates exist
    assert fw["worker_env_bootstrapped"] is False  # no .venv in override dir
    assert str(tmp_path) in fw["effective_env_dir"]
    # A materialized .venv flips the bootstrapped flag.
    (tmp_path / "fw_env" / ".venv").mkdir(parents=True)
    again = worker.handle_request(request("health"))
    assert again.payload is not None
    assert again.payload["asr_worker_envs"]["fw_env"]["worker_env_bootstrapped"] is True


def test_review_resolution_typed_targets_and_subject_mismatch(tmp_path: Path) -> None:
    """Engine-owned resolution: a PLAYBACK_SPEED decision on SPEED-1 resolves
    the speed target even though no queue item id matches the subject; an
    unbound/stale decision leaves it OPEN."""
    from .phase5_helpers import RULES_VERSION, VIDEO_SHA

    worker, _ = make_worker()
    run_dir = str(_manifested_ready_run(tmp_path))

    before = worker.handle_request(request("get_review_resolution", {"run_dir": run_dir}))
    assert before.status == "ok", before.error
    assert before.payload is not None
    speed = next(
        t for t in before.payload["speed_targets"] if t["subject_id"] == "SPEED-1"
    )
    assert speed["target_kind"] == "PLAYBACK_SPEED"
    assert speed["allowed_decision_types"] == ["PLAYBACK_SPEED"]
    assert speed["resolution_status"] == "OPEN"
    transition = next(
        t
        for t in before.payload["transition_targets"]
        if t["subject_id"] == "TRANSITION-1"
    )
    assert transition["target_kind"] == "SHOT_TRANSITION"

    # A decision bound to ANOTHER video is STALE and must not resolve.
    stale = worker.handle_request(
        request(
            "save_review_decisions",
            {
                "run_dir": run_dir,
                "decisions": [
                    {
                        "decision_id": "D-STALE",
                        "subject_id": "SPEED-1",
                        "decision_type": "PLAYBACK_SPEED",
                        "value": "regular",
                        "decided_by": "reviewer@test",
                        "bound_video_sha256": "b" * 64,
                        "bound_rules_version": RULES_VERSION,
                    }
                ],
            },
        )
    )
    assert stale.status == "ok"
    still_open = worker.handle_request(
        request("get_review_resolution", {"run_dir": run_dir})
    )
    assert still_open.payload is not None
    speed_after_stale = next(
        t for t in still_open.payload["speed_targets"] if t["subject_id"] == "SPEED-1"
    )
    assert speed_after_stale["resolution_status"] == "OPEN"
    stale_app = next(
        a for a in still_open.payload["applications"] if a["decision_id"] == "D-STALE"
    )
    assert stale_app["outcome"] == "STALE"

    # A properly bound decision resolves it.
    applied = worker.handle_request(
        request(
            "save_review_decisions",
            {
                "run_dir": run_dir,
                "decisions": [
                    {
                        "decision_id": "D-OK",
                        "subject_id": "SPEED-1",
                        "decision_type": "PLAYBACK_SPEED",
                        "value": "regular",
                        "decided_by": "reviewer@test",
                        "bound_video_sha256": VIDEO_SHA,
                        "bound_rules_version": RULES_VERSION,
                    }
                ],
            },
        )
    )
    assert applied.status == "ok"
    resolved = worker.handle_request(
        request("get_review_resolution", {"run_dir": run_dir})
    )
    assert resolved.payload is not None
    speed_resolved = next(
        t for t in resolved.payload["speed_targets"] if t["subject_id"] == "SPEED-1"
    )
    assert speed_resolved["resolution_status"] == "RESOLVED"
    assert speed_resolved["resolved_by_decision_ids"] == ["D-OK"]


def test_media_dimensions_from_media_truth(tmp_path: Path) -> None:
    """Anchor source dimensions come from Phase 1 media.json — portrait too."""
    worker, _ = make_worker()
    run_dir = _fake_run_dir(tmp_path)
    (run_dir / "media.json").write_text(
        json.dumps({"video_streams": [{"width": 1080, "height": 1920}]}),
        encoding="utf-8",
    )
    response = worker.handle_request(
        request("get_media_dimensions", {"run_dir": str(run_dir)})
    )
    assert response.status == "ok"
    assert response.payload == {"width": 1080, "height": 1920}


def test_resolve_rerun_request_requires_saved_anchors(tmp_path: Path) -> None:
    worker, _ = make_worker()
    run_dir = _fake_run_dir(tmp_path)
    response = worker.handle_request(
        request("resolve_rerun_request", {"run_dir": str(run_dir)})
    )
    assert response.status == "error"
    assert response.error is not None
    assert response.error.code == BridgeErrorCode.ARTIFACT_NOT_FOUND
