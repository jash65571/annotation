"""Deterministic validation of a rendered caption, run on the TEXT itself.

This is the half of the RTD gate that does not involve a language model. It runs
after the first draft AND again after any reviewer rewrite — the previous
reviewer pass wrote its output straight to disk, so a model that introduced a
broken timestamp, an unbalanced quote, a ghost C-ID or a pronoun produced a
"final" file nobody had checked.

Shared text primitives (quote-span pairing, pronoun detection outside quotes,
sentence splitting) come from the engine's ``caption.textcheck`` so AutoScribe
and Manuscript Reviewer cannot drift apart on what counts as a quote.
"""

from __future__ import annotations

import re

from manuscript_reviewer.caption.textcheck import find_quote_spans, pronoun_hits

from .blockers import WARNING, BlockerLog
from .cuts import CUT_TYPES

#: Canonical section labels of the live tool's master template.
REQUIRED_OVERVIEW_FIELDS = ("Cast:", "Scene:", "Style:", "Audio:")
CANONICAL_SHOT_FIELDS = ("Cut:", "Camera:", "Scene:", "Playback Speed:")

#: Field names AutoScribe used to emit that are NOT the canonical ones.
LEGACY_FIELDS = {
    "Characters:": "Cast:",
    "Camera movements:": "Camera Movements:",
    "Video playback speed:": "Playback Speed:",
}

BLOCKED_PRONOUNS = [
    "he", "she", "they", "him", "her", "them", "his", "hers", "their", "theirs",
]

_SHOT_HEADER = re.compile(r"^\[Shot (\d+):\s*([\d.]+)s?\s*[–\-—]\s*([\d.]+)s?\]")  # noqa: RUF001
_TIMED_LINE = re.compile(r"^\(([\d.]+)s\s*[–\-—]\s*([\d.]+)s\)\s*(.+)$")  # noqa: RUF001
_ID_REF = re.compile(r"\b([CO]\d+)\b")
_ID_DEF = re.compile(r"^([CO]\d+):\s*(.+)$")
_FILLER = re.compile(
    r"^\(?[^)]*\)?\s*no\s+(speech|dialogue|music|audio|sound)\b", re.IGNORECASE
)
_TIMING_FILLER = re.compile(
    r"\b(near the end|around \d+ seconds?|partway through|in the final frames|"
    r"toward the end|at the start of the shot|midway through)\b",
    re.IGNORECASE,
)
_HEDGE = re.compile(
    r"\b(cannot be determined|not verifiable|unclear whether|impossible to tell|"
    r"unable to determine)\b",
    re.IGNORECASE,
)
_PROTECTED_TRAIT = re.compile(
    r"\b(nationality|ethnicity|ethnic|racial)\b|\brace\b(?!\s*(?:car|track|day))",
    re.IGNORECASE,
)


def _quotes_balanced(line: str) -> bool:
    """A straight-quote span that runs to end-of-line means an unclosed quote."""
    for span in find_quote_spans(line):
        if span.end >= len(line) and not line.rstrip().endswith(('"', "”")):
            return False
    return line.count('"') % 2 == 0


def validate_caption(text: str, blockers: BlockerLog | None = None) -> BlockerLog:
    """Check a rendered caption and return the log of everything wrong with it.

    Every finding is BLOCKING unless it is genuinely stylistic — this gate
    exists to stop bad captions being delivered, not to produce advice.
    """
    log = blockers if blockers is not None else BlockerLog()
    lines = text.splitlines()

    for legacy, canonical in LEGACY_FIELDS.items():
        if any(line.strip().startswith(legacy) for line in lines):
            log.add(
                "FIELD_NAME_NOT_CANONICAL",
                f"Caption uses '{legacy}' where the master template requires "
                f"'{canonical}'.",
            )

    for field in REQUIRED_OVERVIEW_FIELDS:
        if not any(line.startswith(field) for line in lines):
            log.add("OVERVIEW_FIELD_MISSING", f"Overview is missing '{field}'.")

    defined_ids = {m.group(1) for line in lines if (m := _ID_DEF.match(line.strip()))}
    referenced_ids: set[str] = set()

    shot_index = 0
    shot_start = shot_end = 0.0
    in_shot = False
    seen_ranges: set[tuple[int, float, float]] = set()
    shot_numbers: list[int] = []
    prev_shot_end: float | None = None

    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue

        header = _SHOT_HEADER.match(line)
        if header:
            shot_index = int(header.group(1))
            shot_start, shot_end = float(header.group(2)), float(header.group(3))
            shot_numbers.append(shot_index)
            in_shot = True
            if shot_end <= shot_start:
                log.add(
                    "SHOT_RANGE_INVALID",
                    f"Shot {shot_index} ends at or before it starts "
                    f"({shot_start:.1f}s-{shot_end:.1f}s).",
                    start=shot_start, end=shot_end,
                )
            if prev_shot_end is not None and abs(shot_start - prev_shot_end) > 0.05:
                log.add(
                    "SHOT_TIMELINE_GAP",
                    f"Shot {shot_index} starts at {shot_start:.1f}s but the previous "
                    f"shot ended at {prev_shot_end:.1f}s — the timeline is not "
                    f"continuous.",
                    start=prev_shot_end, end=shot_start,
                )
            prev_shot_end = shot_end
            continue

        if line.startswith("Cut:"):
            cut = line[4:].strip().rstrip(".")
            if cut and cut not in CUT_TYPES:
                log.add(
                    "CUT_TYPE_UNKNOWN",
                    f"Shot {shot_index} declares cut type '{cut}', which is not a "
                    f"recognised transition.",
                )

        if not _quotes_balanced(line):
            log.add("QUOTES_UNBALANCED", f"Line {lineno} has an unclosed quote: {line[:80]!r}")

        if _FILLER.match(line):
            log.add(
                "FILLER_LINE",
                f"Line {lineno} states the absence of audio ({line[:60]!r}); absence is "
                f"expressed by omission.",
            )
        if _TIMING_FILLER.search(line):
            log.add(
                "TIMING_FILLER",
                f"Line {lineno} uses a vague timing phrase; timestamps carry the timing: "
                f"{line[:80]!r}",
            )
        if _HEDGE.search(line):
            log.add("HEDGE_TEXT", f"Line {lineno} hedges instead of omitting: {line[:80]!r}")
        if _PROTECTED_TRAIT.search(line):
            log.add(
                "PROTECTED_TRAIT",
                f"Line {lineno} asserts a protected trait (race/ethnicity/nationality), "
                f"which is not supportable from footage: {line[:80]!r}",
            )

        referenced_ids.update(_ID_REF.findall(line))

        hits = pronoun_hits(line, BLOCKED_PRONOUNS)
        if hits and not _ID_DEF.match(line):
            log.add(
                "PRONOUN_OUTSIDE_QUOTES",
                f"Line {lineno} uses pronoun(s) {sorted(set(hits))} outside quoted "
                f"speech: {line[:80]!r}",
            )

        timed = _TIMED_LINE.match(line)
        if timed:
            start, end, body = float(timed.group(1)), float(timed.group(2)), timed.group(3)
            if end < start:
                log.add(
                    "TIMESTAMP_REVERSED",
                    f"Line {lineno} ends before it starts ({start:.1f}s-{end:.1f}s).",
                    start=start, end=end,
                )
            if in_shot and (start < shot_start - 0.05 or end > shot_end + 0.05):
                log.add(
                    "TIMESTAMP_OUTSIDE_SHOT",
                    f"Line {lineno} ({start:.1f}s-{end:.1f}s) falls outside Shot "
                    f"{shot_index} ({shot_start:.1f}s-{shot_end:.1f}s).",
                    start=start, end=end,
                )
            key = (shot_index, start, end)
            if key in seen_ranges:
                # Two genuinely simultaneous events are legitimate and must NOT be
                # silently merged — but they must be a deliberate, confirmed
                # decision, so they surface for review rather than passing quietly.
                log.add(
                    "SIMULTANEOUS_EVENTS_UNCONFIRMED",
                    f"Shot {shot_index} has more than one entry at "
                    f"{start:.1f}s-{end:.1f}s. If these are truly simultaneous this is "
                    f"correct and must stay as separate lines — confirm it.",
                    severity=WARNING, start=start, end=end,
                )
            seen_ranges.add(key)
            if body and body.strip()[-1] not in '.!?"’”':  # noqa: RUF001
                log.add(
                    "PUNCTUATION_MISSING",
                    f"Line {lineno} has no terminal punctuation: {body[:60]!r}",
                )

    ghosts = sorted(referenced_ids - defined_ids)
    if ghosts:
        log.add(
            "GHOST_ID",
            f"Caption references {ghosts} which are never defined in the Overview.",
        )
    unused = sorted(defined_ids - referenced_ids)
    if unused:
        log.add(
            "UNREFERENCED_ID",
            f"Overview defines {unused} which no shot ever references.",
            severity=WARNING,
        )

    if shot_numbers and shot_numbers != list(range(1, len(shot_numbers) + 1)):
        log.add(
            "SHOT_NUMBERING",
            f"Shots are numbered {shot_numbers}; they must run 1..N without gaps.",
        )
    if not shot_numbers:
        log.add("NO_SHOTS", "Caption contains no shot sections.")

    return log
