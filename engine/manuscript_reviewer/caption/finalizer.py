"""Readiness computation + signoff loading (§53/§54/§90/§91).

Exact rules:

* BLOCKED — hard validation failure (any M2/platform FAIL), wrong media,
  invalid timeline, missing required structure.
* REVIEW_REQUIRED — required facts unresolved, candidate-only facts needed by
  the output, unverified speech/OCR/speed, unresolved task feedback, golden
  gate not passing.
* READY_FOR_FINAL_REVIEW — every factual/structural machine gate passes, no
  known omission/hallucination, but no valid manual signoff yet.
* READY_TO_ENTER — READY_FOR_FINAL_REVIEW + valid bound FinalReviewSignoff
  whose caption hash matches, with nothing changed after signoff.

Machine code NEVER fabricates a signoff — it is loaded from a human-supplied
file only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from ..models.caption_brain import (
    CaptionPlan,
    CaptionReadiness,
    FinalReviewSignoff,
    GoldenGateResult,
    GoldenGateStatus,
)
from ..models.validation import Severity, ValidatorIssue
from ..validation.final_caption_validator import AdversarialQCResult, SignoffCheck
from ..validation.platform_semantic_validator import PlatformSemanticReport

logger = logging.getLogger(__name__)


class SignoffLoadError(RuntimeError):
    """The final-review file could not be read or parsed."""


def load_signoff(path: Path) -> FinalReviewSignoff:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SignoffLoadError(f"Cannot read final-review file {path}: {exc}") from exc
    try:
        return FinalReviewSignoff.model_validate(raw)
    except ValidationError as exc:
        raise SignoffLoadError(f"Invalid final-review signoff: {exc}") from exc


@dataclass
class ReadinessInputs:
    plan: CaptionPlan
    m2_issues: list[ValidatorIssue]
    platform_report: PlatformSemanticReport
    golden: GoldenGateResult
    adversarial: AdversarialQCResult
    signoff_check: SignoffCheck
    unresolved_high_feedback: list[str] = field(default_factory=list)
    #: Recomputed Phase 4 CRITICAL/HIGH machine review items (§5.1-4). Any
    #: entry keeps the caption from reaching READY_FOR_FINAL_REVIEW.
    visual_review_blockers: list[str] = field(default_factory=list)


@dataclass
class ReadinessResult:
    readiness: CaptionReadiness
    blockers: list[str] = field(default_factory=list)


def compute_readiness(inputs: ReadinessInputs) -> ReadinessResult:
    blockers: list[str] = []

    m2_fails = [i for i in inputs.m2_issues if i.severity == Severity.FAIL]
    if m2_fails:
        blockers.extend(f"{i.rule_id}: {i.message}" for i in m2_fails)
    if inputs.platform_report.status == "FAIL":
        blockers.extend(
            f"{i.rule_id}: {i.message}"
            for i in inputs.platform_report.issues
            if i.severity == Severity.FAIL
        )
    if inputs.plan.readiness == CaptionReadiness.BLOCKED:
        blockers.extend(inputs.plan.blockers)
    if blockers:
        return ReadinessResult(CaptionReadiness.BLOCKED, blockers)

    review: list[str] = []
    review.extend(inputs.plan.blockers)
    review.extend(
        f"{i.rule_id}: {i.message}"
        for i in inputs.m2_issues
        if i.severity == Severity.WARN
    )
    if inputs.platform_report.status == "REVIEW_REQUIRED":
        review.append("platform-semantic pass requires review")
    if inputs.golden.overall != GoldenGateStatus.PASS:
        review.append(f"golden behavior gate: {inputs.golden.overall.value}")
    if inputs.adversarial.blocking_count:
        review.append(
            f"{inputs.adversarial.blocking_count} blocking adversarial finding(s)"
        )
    if inputs.adversarial.missing_required_facts:
        review.append("required eligible facts missing from the caption")
    for directive_id in inputs.unresolved_high_feedback:
        review.append(f"unresolved HIGH feedback directive {directive_id}")
    review.extend(inputs.visual_review_blockers)
    if review:
        return ReadinessResult(CaptionReadiness.REVIEW_REQUIRED, review)

    check = inputs.signoff_check
    if check.present and check.valid:
        return ReadinessResult(CaptionReadiness.READY_TO_ENTER, [])
    reasons: list[str] = []
    if check.present and not check.valid:
        reasons = [f"signoff invalid/stale: {r}" for r in check.reasons] or [
            "signoff invalid"
        ]
    return ReadinessResult(CaptionReadiness.READY_FOR_FINAL_REVIEW, reasons)
