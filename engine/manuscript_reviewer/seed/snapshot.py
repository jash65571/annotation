"""Immutable seed / feedback snapshots.

The original file is copied byte-for-byte and hashed. It is never normalized,
never repaired, never reordered — it remains immutable evidence. Parsing (in
:mod:`.parser`) produces a separate representation from a decoded *copy* of
these bytes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..models.review_intelligence import FeedbackSnapshot, SeedSnapshot


def _sha256_bytes(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def _line_count(data: bytes) -> int:
    if not data:
        return 0
    # Count lines without assuming a trailing newline; do not alter the bytes.
    text = data.decode("utf-8", errors="replace")
    return len(text.splitlines())


def snapshot_seed(seed_path: Path, seed_dir: Path) -> SeedSnapshot:
    """Copy the seed verbatim to ``seed_dir/seed_original.txt`` and hash it.

    Also writes ``seed_dir/seed_sha256.txt``. The stored copy preserves the
    exact source bytes (no normalization).
    """
    seed_dir.mkdir(parents=True, exist_ok=True)
    data = seed_path.read_bytes()
    sha = _sha256_bytes(data)
    original = seed_dir / "seed_original.txt"
    original.write_bytes(data)
    (seed_dir / "seed_sha256.txt").write_text(sha + "\n", encoding="utf-8")
    return SeedSnapshot(
        original_path=str(seed_path.resolve()),
        stored_relative_path="seed/seed_original.txt",
        sha256=sha,
        byte_count=len(data),
        line_count=_line_count(data),
    )


def snapshot_feedback(feedback_path: Path, feedback_dir: Path) -> FeedbackSnapshot:
    """Copy task feedback verbatim to ``feedback_dir/feedback_original.txt``."""
    feedback_dir.mkdir(parents=True, exist_ok=True)
    data = feedback_path.read_bytes()
    sha = _sha256_bytes(data)
    original = feedback_dir / "feedback_original.txt"
    original.write_bytes(data)
    return FeedbackSnapshot(
        original_path=str(feedback_path.resolve()),
        stored_relative_path="feedback/feedback_original.txt",
        sha256=sha,
        byte_count=len(data),
        line_count=_line_count(data),
    )
