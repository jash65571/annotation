"""Coverage / omission ledger (§59) and the reverse hallucination check (§60).

No omissions is as important as no hallucinations:

* Every material ELIGIBLE fact must appear in the final caption OR carry a
  valid explicit omission reason. "The writer forgot it" is never valid.
* Every final caption assertion must map to one or more eligible CaptionFacts.
  An unmapped phrase FAILS the hallucination gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models.caption_brain import (
    AssertionStatus,
    CaptionAssertionRecord,
    CaptionCoverageItem,
    CaptionEligibility,
    CaptionFact,
    FactMateriality,
    OmissionReason,
)


@dataclass
class CoverageResult:
    items: list[CaptionCoverageItem] = field(default_factory=list)
    missing_required_fact_ids: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.missing_required_fact_ids


def build_coverage(
    facts: list[CaptionFact],
    rendered_fact_lines: dict[str, str],
    omissions: dict[str, OmissionReason] | None = None,
) -> CoverageResult:
    """One ledger row per ELIGIBLE fact. ``omissions`` maps fact_id -> the
    explicit reason a planner/renderer deliberately left it out."""
    omissions = omissions or {}
    result = CoverageResult()
    for fact in facts:
        if fact.eligibility != CaptionEligibility.ELIGIBLE:
            continue
        line_id = rendered_fact_lines.get(fact.fact_id)
        represented = line_id is not None
        material = fact.materiality in (FactMateriality.REQUIRED, FactMateriality.MATERIAL)
        required = fact.required_for_caption or fact.materiality == FactMateriality.REQUIRED
        reason = omissions.get(fact.fact_id)
        result.items.append(
            CaptionCoverageItem(
                fact_id=fact.fact_id,
                material=material,
                required=required,
                represented=represented,
                caption_line_id=line_id,
                omission_reason=reason if not represented else None,
            )
        )
        if not represented and material and reason is None:
            result.missing_required_fact_ids.append(fact.fact_id)
        elif not represented and required and reason is not None:
            # A REQUIRED fact can never be omitted, even with a reason.
            result.missing_required_fact_ids.append(fact.fact_id)
    return result


@dataclass
class AssertionCheckResult:
    unmapped: list[CaptionAssertionRecord] = field(default_factory=list)
    ineligible_refs: list[tuple[str, str]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.unmapped and not self.ineligible_refs


def check_assertions(
    assertions: list[CaptionAssertionRecord],
    facts_by_id: dict[str, CaptionFact],
) -> AssertionCheckResult:
    """Reverse direction (§60): every assertion maps to eligible facts."""
    result = AssertionCheckResult()
    for assertion in assertions:
        if assertion.status == AssertionStatus.UNMAPPED or not assertion.fact_ids:
            result.unmapped.append(assertion)
            continue
        for fid in assertion.fact_ids:
            fact = facts_by_id.get(fid)
            if fact is None or fact.eligibility != CaptionEligibility.ELIGIBLE:
                result.ineligible_refs.append((assertion.assertion_id, fid))
    return result
