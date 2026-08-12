"""Final adversarial QC (§87/§88) + signoff validity (§54/§91).

After rendering, the finished caption is treated as potentially wrong: every
final sentence must answer "what exact evidence proves this?", and the reverse
question — "what eligible material evidence did the renderer fail to mention?"
— is asked as well.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..caption.coverage import AssertionCheckResult, CoverageResult
from ..models.caption_brain import (
    AdversarialFinding,
    CaptionEligibility,
    CaptionFact,
    CaptionFactType,
    CaptionPlan,
    FinalReviewSignoff,
    RenderedCaption,
)
from ..models.validation import Severity, ValidatorIssue


@dataclass
class AdversarialQCResult:
    findings: list[AdversarialFinding] = field(default_factory=list)
    missing_required_facts: list[str] = field(default_factory=list)

    @property
    def blocking_count(self) -> int:
        return sum(1 for f in self.findings if f.blocking)


def run_adversarial_qc(
    plan: CaptionPlan,
    caption: RenderedCaption,
    facts: list[CaptionFact],
    coverage: CoverageResult,
    assertions: AssertionCheckResult,
    m2_issues: list[ValidatorIssue],
) -> AdversarialQCResult:
    result = AdversarialQCResult()
    n = 0

    def add(category: str, detail: str, line_id: str | None = None, blocking: bool = False) -> None:
        nonlocal n
        n += 1
        result.findings.append(
            AdversarialFinding(
                finding_id=f"QC-{n:04d}",
                line_id=line_id,
                category=category,
                detail=detail,
                blocking=blocking,
            )
        )

    # Hallucination direction: unmapped assertions / ineligible references.
    for record in assertions.unmapped:
        add(
            "UNSUPPORTED_ASSERTION",
            f"No evidence chain for: {record.assertion_text[:120]!r}",
            line_id=record.line_id,
            blocking=True,
        )
    for assertion_id, fid in assertions.ineligible_refs:
        add(
            "INELIGIBLE_EVIDENCE",
            f"Assertion {assertion_id} cites non-eligible fact {fid}",
            blocking=True,
        )

    # Omission direction (§88): the renderer failed to mention eligible facts.
    result.missing_required_facts = list(coverage.missing_required_fact_ids)
    for fid in coverage.missing_required_fact_ids:
        add("OMISSION", f"Eligible material fact {fid} is not represented", blocking=True)

    # Unresolved evidence the caption silently rides over.
    for fact in facts:
        if fact.eligibility == CaptionEligibility.REVIEW_REQUIRED and fact.fact_type in (
            CaptionFactType.SPEECH,
            CaptionFactType.TRANSITION,
            CaptionFactType.PLAYBACK_SPEED,
        ):
            add(
                f"UNRESOLVED_{fact.fact_type.value}",
                f"{fact.fact_id}: {fact.eligibility_reason}",
                blocking=False,
            )
        if fact.eligibility == CaptionEligibility.INELIGIBLE and fact.fact_type in (
            CaptionFactType.SPEECH,
            CaptionFactType.ON_SCREEN_TEXT,
        ):
            add(
                f"BLOCKED_{fact.fact_type.value}",
                f"{fact.fact_id}: {fact.eligibility_reason} — possible missing "
                "dialogue/overlay until verified",
                blocking=False,
            )

    # Every M2 FAIL is an adversarial finding too (wrong identity/object/
    # speaker/timing/fields all surface as M2 rules).
    for issue in m2_issues:
        if issue.severity == Severity.FAIL:
            add("M2_FAIL", f"{issue.rule_id}: {issue.message}", blocking=True)
    _ = plan, caption
    return result


# ---------------------------------------------------------------------------
# Signoff validity (§54/§91)
# ---------------------------------------------------------------------------


@dataclass
class SignoffCheck:
    present: bool = False
    valid: bool = False
    stale: bool = False
    reasons: list[str] = field(default_factory=list)


_NON_HUMAN = frozenset({"", "machine", "ai", "system"})


def check_signoff(
    signoff: FinalReviewSignoff | None,
    video_sha256: str | None,
    rules_version: str | None,
    caption_sha256: str,
) -> SignoffCheck:
    """A signoff is valid only when bound to the exact video, rules version and
    caption hash, with every manual confirmation true. If caption content
    changed after review, the old signoff is stale."""
    check = SignoffCheck()
    if signoff is None:
        return check
    check.present = True
    if (signoff.reviewer or "").strip().lower() in _NON_HUMAN:
        check.reasons.append("signoff reviewer is not a human identity")
        return check
    if video_sha256 is not None and signoff.video_sha256 != video_sha256:
        check.stale = True
        check.reasons.append("signoff bound to a different video SHA-256")
    if rules_version is not None and signoff.rules_version != rules_version:
        check.stale = True
        check.reasons.append(
            f"signoff bound to rules {signoff.rules_version}, current {rules_version}"
        )
    if signoff.caption_sha256 != caption_sha256:
        check.stale = True
        check.reasons.append(
            "signoff caption hash does not match the rendered caption (content "
            "changed after final review)"
        )
    confirmations = {
        "golden_example_comparison_complete": signoff.golden_example_comparison_complete,
        "platform_semantic_pass_complete": signoff.platform_semantic_pass_complete,
        "final_adversarial_read_complete": signoff.final_adversarial_read_complete,
        "no_known_omissions_confirmed": signoff.no_known_omissions_confirmed,
        "no_known_hallucinations_confirmed": signoff.no_known_hallucinations_confirmed,
    }
    for name, value in confirmations.items():
        if not value:
            check.reasons.append(f"signoff confirmation missing: {name}")
    check.valid = not check.stale and not check.reasons
    return check
