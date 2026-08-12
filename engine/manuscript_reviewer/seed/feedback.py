"""Task-specific feedback parsing.

Task feedback is higher priority than seed content (source hierarchy §1). It is
snapshotted verbatim (see :mod:`.snapshot`); here it is turned into structured
:class:`FeedbackDirective` records. Free-form evaluator feedback is NOT
over-interpreted: only exact, known patterns map to a known check code; anything
uncertain stays ``REVIEW_REQUIRED`` with ``review_required=True``.
"""

from __future__ import annotations

import re

from ..models.review_intelligence import (
    FeedbackDirective,
    FeedbackDocument,
    FeedbackPriority,
    FeedbackSnapshot,
    InterpretationStatus,
)

#: Exact, conservative pattern -> known check code mappings.
_KNOWN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"missing\s+music\s+lyrics", re.IGNORECASE), "REQUIRE_VOCAL_LYRIC_REVIEW"),
    (re.compile(r"music\s+lyrics", re.IGNORECASE), "REQUIRE_VOCAL_LYRIC_REVIEW"),
    (re.compile(r"singing\s+(?:is\s+)?not\s+transcribed", re.IGNORECASE),
     "REQUIRE_VOCAL_LYRIC_REVIEW"),
    (re.compile(r"english\s+singing", re.IGNORECASE), "REQUIRE_VOCAL_LYRIC_REVIEW"),
    (re.compile(r"lyrics\s+(?:are\s+)?missing", re.IGNORECASE), "REQUIRE_VOCAL_LYRIC_REVIEW"),
]

_HIGH_PRIORITY = re.compile(r"\b(critical|must|required|high priority|blocker)\b", re.IGNORECASE)
_BULLET = re.compile(r"^[\s>#*\-•\d.)]+")


def parse_feedback_text(
    text: str, snapshot: FeedbackSnapshot | None = None
) -> FeedbackDocument:
    """Parse feedback text into directives. One directive per non-empty line."""
    directives: list[FeedbackDirective] = []
    counter = 0
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        stripped = _BULLET.sub("", raw_line).strip()
        if not stripped:
            continue
        counter += 1
        code: str | None = None
        for pattern, mapped in _KNOWN_PATTERNS:
            if pattern.search(stripped):
                code = mapped
                break
        if code is not None:
            status = InterpretationStatus.MAPPED
        else:
            status = InterpretationStatus.REVIEW_REQUIRED
        priority = (
            FeedbackPriority.HIGH if _HIGH_PRIORITY.search(stripped) else FeedbackPriority.NORMAL
        )
        directives.append(
            FeedbackDirective(
                directive_id=f"FBK-{counter:04d}",
                raw_text=raw_line,
                source_line=line_no,
                priority=priority,
                machine_interpretation=code,
                interpretation_status=status,
                # Even a mapped directive requires human confirmation of the fix.
                review_required=True,
            )
        )
    return FeedbackDocument(snapshot=snapshot, directives=directives)
