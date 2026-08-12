"""Load human/detector visual anchors for assisted local tracking."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from ..models.review_intelligence import VisualAnchor


class AnchorLoadError(RuntimeError):
    """The visual-anchors file could not be read or parsed."""


def load_anchors(path: Path) -> list[VisualAnchor]:
    """Load anchors from JSON: ``{"anchors": [{anchor_id, frame_index, x, y,
    width, height, entity_type, temporary_label?}, ...]}``."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnchorLoadError(f"Cannot read anchors file {path}: {exc}") from exc
    entries = raw.get("anchors", []) if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise AnchorLoadError("Anchors file must contain an 'anchors' list.")
    anchors: list[VisualAnchor] = []
    for entry in entries:
        try:
            anchors.append(VisualAnchor.model_validate(entry))
        except ValidationError as exc:
            raise AnchorLoadError(f"Invalid anchor entry: {exc}") from exc
    return anchors
