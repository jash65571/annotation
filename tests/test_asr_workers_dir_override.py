"""MANUSCRIPT_ASR_WORKERS_DIR override (Phase 6 packaging, spec §15).

Packaged installs keep worker envs in writable app-local data; the same
pinned worker templates are materialized there. No factual logic involved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from manuscript_reviewer.audio.asr import runtime


def test_default_resolution_is_in_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(runtime.WORKERS_DIR_ENV, raising=False)
    assert runtime._resolve_env_dir("fw_env") == runtime.WORKERS_DIR / "fw_env"


def test_override_copies_templates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(runtime.WORKERS_DIR_ENV, str(tmp_path))
    resolved = runtime._resolve_env_dir("fw_env")
    assert resolved == tmp_path / "fw_env"
    assert (resolved / "pyproject.toml").exists()
    assert (resolved / "worker.py").exists()
    # Byte-identical templates: the packaged app runs the exact same pinned
    # worker definition as development.
    source = runtime.WORKERS_DIR / "fw_env"
    assert (resolved / "pyproject.toml").read_bytes() == (
        source / "pyproject.toml"
    ).read_bytes()


def test_override_never_overwrites_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(runtime.WORKERS_DIR_ENV, str(tmp_path))
    env_dir = tmp_path / "wx_env"
    env_dir.mkdir()
    (env_dir / "pyproject.toml").write_text("custom-preserved", encoding="utf-8")
    resolved = runtime._resolve_env_dir("wx_env")
    assert (resolved / "pyproject.toml").read_text(encoding="utf-8") == "custom-preserved"
