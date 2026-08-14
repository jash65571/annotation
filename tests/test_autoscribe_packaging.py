"""AutoScribe must be installable, importable, and complete.

The wheel used to package only ``engine/manuscript_reviewer``, so an installed
copy of this project had no AutoScribe at all, and the ``autoscribe`` extra
omitted a dependency its own flat mode imports.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

MODULES = [
    "autoscribe.asr",
    "autoscribe.audio_timeline",
    "autoscribe.blockers",
    "autoscribe.cuts",
    "autoscribe.ffbin",
    "autoscribe.frames",
    "autoscribe.pipeline",
    "autoscribe.render",
    "autoscribe.review",
    "autoscribe.structured",
    "autoscribe.transcribe",
    "autoscribe.validate",
    "autoscribe.vision",
    "autoscribe.webapp",
]


@pytest.fixture(scope="module")
def config() -> dict[str, object]:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


@pytest.mark.parametrize("module", MODULES)
def test_every_module_imports(module: str) -> None:
    importlib.import_module(module)


def test_autoscribe_is_packaged_in_the_wheel(config: dict) -> None:
    packages = config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert "autoscribe" in packages, "AutoScribe would not ship in an install"


def test_autoscribe_is_type_checked_strictly(config: dict) -> None:
    assert "autoscribe" in config["tool"]["mypy"]["packages"]
    assert config["tool"]["mypy"]["strict"] is True


def test_extra_declares_every_dependency_the_code_imports(config: dict) -> None:
    extra = config["project"]["optional-dependencies"]["autoscribe"]
    names = [d.split(">")[0].split("=")[0].split("[")[0].strip() for d in extra]
    # cuts.py imports scenedetect; asr.py (flat mode) imports faster_whisper.
    assert "scenedetect" in names
    assert "faster-whisper" in names


def test_web_ui_asset_exists() -> None:
    from autoscribe import webapp

    index = webapp._STATIC / "index.html"
    assert index.is_file(), "the web UI template is missing from the package"


def test_ui_never_claims_ready_to_deliver() -> None:
    """The UI must not label model output as a final RTD caption."""
    from autoscribe import webapp

    html = (webapp._STATIC / "index.html").read_text(encoding="utf-8")
    assert "Final RTD caption" not in html
    assert "requires human verification" in html
    assert "NOT READY TO DELIVER" in html
