"""Human-added caption facts (§9/§10): loading + binding.

Machine code must NEVER create a :class:`HumanCaptionFact`. They are loaded
only from a human-supplied ``--human-facts`` file, must carry a human
``decided_by``, and must be bound to the current video SHA-256 and rules
version. Stale facts from another video/rules are rejected — old review input
is never silently reused.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from ..models.caption_brain import HumanCaptionFact

logger = logging.getLogger(__name__)

_NON_HUMAN = frozenset({"", "machine", "ai", "system"})


class HumanFactLoadError(RuntimeError):
    """The human-facts file could not be read, parsed, or was machine-authored."""


@dataclass
class HumanFactLoadResult:
    accepted: list[HumanCaptionFact] = field(default_factory=list)
    #: (fact_id, reason) for stale/rejected entries — never silently dropped.
    rejected: list[tuple[str, str]] = field(default_factory=list)


def load_human_facts(
    path: Path,
    video_sha256: str,
    rules_version: str,
    seed_sha256: str | None = None,
) -> HumanFactLoadResult:
    """Load and bind human caption facts. A machine-authored entry fails the
    whole file (the file itself is untrustworthy); a stale binding rejects
    only that entry, with the reason recorded."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HumanFactLoadError(f"Cannot read human facts file {path}: {exc}") from exc
    entries = raw.get("facts", []) if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise HumanFactLoadError("Human facts file must contain a 'facts' list.")

    result = HumanFactLoadResult()
    for entry in entries:
        try:
            fact = HumanCaptionFact.model_validate(entry)
        except ValidationError as exc:
            raise HumanFactLoadError(f"Invalid human fact entry: {exc}") from exc
        if (fact.decided_by or "").strip().lower() in _NON_HUMAN:
            raise HumanFactLoadError(
                f"Human fact {fact.fact_id} has non-human decided_by="
                f"{fact.decided_by!r}; machine-created facts are not accepted."
            )
        if fact.bound_video_sha256 != video_sha256:
            result.rejected.append(
                (fact.fact_id, "bound to a different video SHA-256 (stale)")
            )
            logger.warning("Rejecting human fact %s: wrong video binding", fact.fact_id)
            continue
        if fact.bound_rules_version != rules_version:
            result.rejected.append(
                (
                    fact.fact_id,
                    f"bound to rules {fact.bound_rules_version}, current {rules_version}",
                )
            )
            continue
        if (
            fact.bound_seed_sha256 is not None
            and seed_sha256 is not None
            and fact.bound_seed_sha256 != seed_sha256
        ):
            result.rejected.append((fact.fact_id, "bound to a different seed SHA-256"))
            continue
        result.accepted.append(fact)
    return result
