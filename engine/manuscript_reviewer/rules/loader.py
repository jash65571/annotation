"""Typed loader for the versioned Manuscript rule file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

RULES_FILE = Path(__file__).parent / "manuscript_v1.yaml"


class RuleSet:
    """Read-only view over the YAML rule document."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @property
    def version(self) -> str:
        return str(self._data["rules_version"])

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Fetch a rule via dotted path, e.g. ``timing.timestamp_precision_seconds``."""
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


@lru_cache(maxsize=1)
def load_rules(path: Path | None = None) -> RuleSet:
    rules_path = path or RULES_FILE
    with rules_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "rules_version" not in data:
        raise ValueError(f"Rule file {rules_path} is missing rules_version")
    return RuleSet(data)
