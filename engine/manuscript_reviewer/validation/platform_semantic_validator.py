"""Platform-semantic validator (§78-§80): a separate defense modeled on known
export blockers. Uses the structured event plan FIRST (each Action & Audio line
carries exactly one fact); text checks are the second defense — never regex
alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..caption.textcheck import (
    find_quote_spans,
    hidden_second_action,
    sentence_count,
    strip_quotes,
    subject_verb_subjects,
)
from ..media.timestamps import format_manuscript_display
from ..models.caption_brain import (
    CaptionFact,
    CaptionFactType,
    CaptionSection,
    RenderedCaption,
)
from ..models.validation import Severity, ValidatorIssue

PLATFORM_VALIDATOR_VERSION = "1.0.0"

_SAYS = re.compile(r"\b(says|shouts|whispers|replies|asks)\b", re.IGNORECASE)


@dataclass
class PlatformSemanticReport:
    issues: list[ValidatorIssue] = field(default_factory=list)

    @property
    def status(self) -> str:
        if any(i.severity == Severity.FAIL for i in self.issues):
            return "FAIL"
        if any(i.severity == Severity.WARN for i in self.issues):
            return "REVIEW_REQUIRED"
        return "PASS"


def validate_platform_semantics(
    caption: RenderedCaption, facts_by_id: dict[str, CaptionFact]
) -> PlatformSemanticReport:
    report = PlatformSemanticReport()
    action_lines = [
        ln for ln in caption.lines if ln.section == CaptionSection.ACTION_AUDIO
    ]

    seen_pairs: dict[tuple[int | None, str | None, str | None], int] = {}
    for line in action_lines:
        fact = facts_by_id.get(line.fact_ids[0]) if line.fact_ids else None
        body = line.text
        if "] " in body:
            body = body.split("] ", 1)[1]  # strip the leading [a-b] stamp

        # 1. Multiple sentences (robust splitter; quotes excluded).
        count = sentence_count(body)
        if count > 1:
            report.issues.append(
                ValidatorIssue(
                    rule_id="M2-PLATFORM-001",
                    severity=Severity.FAIL,
                    location=line.line_id,
                    message=f"Action & Audio entry contains {count} sentences; "
                    "one entry = one sentence.",
                )
            )

        # 2. Multiple independently quoted spans (§43). UI text combines
        # simultaneous lines into ONE quoted string, so >1 span always fails.
        spans = find_quote_spans(body)
        if len(spans) > 1:
            report.issues.append(
                ValidatorIssue(
                    rule_id="M2-PLATFORM-002",
                    severity=Severity.FAIL,
                    location=line.line_id,
                    message=f"{len(spans)} quoted spans in one entry; one speech act "
                    "= one quote, one simultaneous UI object = one combined quote.",
                )
            )

        # 3. Multiple speech acts in one entry.
        if len(_SAYS.findall(strip_quotes(body))) > 1:
            report.issues.append(
                ValidatorIssue(
                    rule_id="M2-PLATFORM-003",
                    severity=Severity.FAIL,
                    location=line.line_id,
                    message="Multiple speech acts in one Action & Audio entry.",
                )
            )

        # 4. Two independent subjects (structural first: one fact per line;
        # then the text defense).
        subjects = subject_verb_subjects(body)
        if len(subjects) >= 2 and hidden_second_action(body):
            report.issues.append(
                ValidatorIssue(
                    rule_id="M2-PLATFORM-004",
                    severity=Severity.FAIL,
                    location=line.line_id,
                    message=f"Two independent subject actions in one entry "
                    f"({', '.join(subjects)}); split into separate overlapping lines.",
                )
            )

        # 5. Structural mixing: a line must carry exactly one event fact, and
        # never a camera fact.
        if len(line.fact_ids) != 1:
            report.issues.append(
                ValidatorIssue(
                    rule_id="M2-PLATFORM-005",
                    severity=Severity.FAIL,
                    location=line.line_id,
                    message="Entry does not map to exactly one event fact "
                    "(mixed independent events).",
                )
            )
        if fact is not None and fact.fact_type == CaptionFactType.CAMERA_MOVEMENT:
            report.issues.append(
                ValidatorIssue(
                    rule_id="M2-PLATFORM-006",
                    severity=Severity.FAIL,
                    location=line.line_id,
                    message="Camera movement rendered as an Action & Audio entry.",
                )
            )

        # 6. Duplicate displayed pair + nudging defense.
        key = (line.shot_number, line.display_start, line.display_end)
        seen_pairs[key] = seen_pairs.get(key, 0) + 1
        if (
            line.start_exact is not None
            and line.display_start is not None
            and line.display_start != format_manuscript_display(line.start_exact)
        ):
            report.issues.append(
                ValidatorIssue(
                    rule_id="M2-PLATFORM-007",
                    severity=Severity.FAIL,
                    location=line.line_id,
                    message="Displayed start is not the canonical projection of the "
                    "exact time (artificial 0.1 s nudging).",
                )
            )

    for (shot_number, d_start, d_end), n in seen_pairs.items():
        if n > 1 and d_start is not None:
            report.issues.append(
                ValidatorIssue(
                    rule_id="M2-PLATFORM-008",
                    severity=Severity.WARN,
                    location=f"shot {shot_number}",
                    message=f"{n} entries display the identical time pair "
                    f"({d_start}-{d_end}); the editor may reject duplicate windows. "
                    "Never fabricate timing to pass validation.",
                )
            )
    return report
