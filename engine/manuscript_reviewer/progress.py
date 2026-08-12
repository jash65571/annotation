"""Additive progress reporting for long-running pipeline stages.

The reporter is presentation-only plumbing: no timing or factual algorithm may
ever depend on it. The default is a no-op so CLI/library behavior is unchanged;
the UI bridge supplies a reporter that emits structured progress events.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol


class StageStatus(StrEnum):
    STARTED = "STARTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class ProgressReporter(Protocol):
    """Receives structured stage progress. Implementations must never raise
    into the pipeline and must never influence analysis results."""

    def report(
        self,
        stage: str,
        status: StageStatus,
        detail: str | None = None,
        current: int | None = None,
        total: int | None = None,
    ) -> None: ...


class NoOpProgressReporter:
    """Default reporter: does nothing (CLI behavior unchanged)."""

    def report(
        self,
        stage: str,
        status: StageStatus,
        detail: str | None = None,
        current: int | None = None,
        total: int | None = None,
    ) -> None:
        return None
