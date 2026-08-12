"""Validation and QC models: issues, QC report, run status."""

from __future__ import annotations

from enum import StrEnum

from .common import StrictModel


class Severity(StrEnum):
    FAIL = "FAIL"
    WARN = "WARN"
    INFO = "INFO"


class RunStatus(StrEnum):
    PASS = "PASS"
    #: Deterministic checks passed but unresolved evidence (e.g. possible real
    #: cuts, unverified annotation endpoint) awaits human review. Never
    #: silently collapsed into PASS.
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class ValidatorIssue(StrictModel):
    """One finding. ``rule_id`` uses the M2-* registry (docs/00, §validator)."""

    rule_id: str
    severity: Severity
    location: str
    message: str
    suggested_fix: str | None = None


class FrameCountSignal(StrictModel):
    """One independent frame-count measurement used in the cross-check."""

    method: str
    count: int | None
    authoritative: bool
    notes: str | None = None


class QCReport(StrictModel):
    """qc.json — every check that ran, every signal, every disagreement."""

    status: RunStatus
    issues: list[ValidatorIssue]
    frame_count_signals: list[FrameCountSignal]
    checks_run: list[str]

    @property
    def failures(self) -> list[ValidatorIssue]:
        return [i for i in self.issues if i.severity == Severity.FAIL]

    @property
    def warnings(self) -> list[ValidatorIssue]:
        return [i for i in self.issues if i.severity == Severity.WARN]
