"""Robust Manuscript II seed parser.

Accepts Markdown, plain text, or copied editor output. Recognizes the Overview
sections and per-shot fields documented in the reference corpus, and preserves
for every line: the raw source line, the source line number, the section, the
shot number, timestamp text (and a parsed exact range only when it is valid),
referenced C/O IDs, and quoted strings.

Robustness contract (mandatory):

* The original bytes are never mutated (that is :mod:`.snapshot`'s job).
* A line is never thrown away because parsing failed — malformed content is
  preserved as a FREEFORM entry and, where relevant, a recoverable
  :class:`SeedParseIssue` is recorded.
* Malformed timestamp syntax is not silently repaired: ``timestamp_text`` is
  kept, ``parsed_*`` stays ``None``, and an issue is logged.
"""

from __future__ import annotations

import re
from fractions import Fraction

from ..models.review_intelligence import (
    SeedDocument,
    SeedEntry,
    SeedFieldKind,
    SeedParseIssue,
    SeedParseSeverity,
    SeedSection,
    SeedSectionKind,
    SeedSnapshot,
)

# --- ID / quote extraction ------------------------------------------------

_CHARACTER_RE = re.compile(r"(?<![A-Za-z0-9])@?C(\d+)(?![0-9])")
_OBJECT_RE = re.compile(r"(?<![A-Za-z0-9])@?O(\d+)(?![0-9])")
#: Straight and curly double quotes.
_QUOTE_RE = re.compile(r"[\"“”]([^\"“”]*)[\"“”]")

# --- Timestamps -----------------------------------------------------------

#: A single time token: bare seconds, seconds with 's' suffix, or mm:ss(.s).
_TIME_TOKEN = r"(?:\d+:)?\d+(?:\.\d+)?s?"
#: A range: token <sep> token, where sep is en-dash, em-dash, hyphen, or 'to'.
_RANGE_RE = re.compile(
    rf"(?P<start>{_TIME_TOKEN})\s*(?:[–—-]|to)\s*(?P<end>{_TIME_TOKEN})"  # noqa: RUF001
)
#: Something that *looks* like a timestamp range but is malformed (e.g. an open
#: bracket range or a stray dash between numbers with junk).
_LOOKS_LIKE_TIME = re.compile(r"[\[(]\s*\d[\d:.\ss]*[–—-]")  # noqa: RUF001


def parse_time_token(token: str) -> Fraction | None:
    """Parse one time token to exact seconds, or ``None`` if malformed.

    Accepts ``12``, ``12.3``, ``4.3s``, ``00:03.4``, ``1:08.5``. Never returns a
    float — exact rationals only.
    """
    raw = token.strip().rstrip("s").strip()
    if not raw:
        return None
    try:
        if ":" in raw:
            minutes_str, seconds_str = raw.rsplit(":", 1)
            minutes = int(minutes_str)
            seconds = Fraction(seconds_str)
            if minutes < 0 or seconds < 0:
                return None
            return minutes * 60 + seconds
        value = Fraction(raw)
    except (ValueError, ZeroDivisionError):
        return None
    return value if value >= 0 else None


def find_time_range(text: str) -> tuple[str, Fraction | None, Fraction | None] | None:
    """Find the first timestamp range in ``text``.

    Returns ``(matched_text, start, end)`` where start/end are exact or ``None``
    if the tokens were malformed. Returns ``None`` when no range-like text is
    present.
    """
    match = _RANGE_RE.search(text)
    if match is None:
        return None
    start = parse_time_token(match.group("start"))
    end = parse_time_token(match.group("end"))
    return match.group(0), start, end


_SINGLE_FIELDS = frozenset({SeedFieldKind.SHOT_START, SeedFieldKind.SHOT_END})
_LEADING_BRACKET_TIME = re.compile(rf"^[\[(]\s*(?P<t>{_TIME_TOKEN})\s*[\])]")
_HAS_DIGIT = re.compile(r"\d")


class _TimestampResult:
    __slots__ = ("end", "malformed", "start", "text")

    def __init__(
        self,
        text: str | None,
        start: Fraction | None,
        end: Fraction | None,
        malformed: bool,
    ) -> None:
        self.text = text
        self.start = start
        self.end = end
        self.malformed = malformed


def _extract_timestamp(field: SeedFieldKind, value: str) -> _TimestampResult:
    """Extract a timestamp range, a single-point time, or flag malformed text.

    Ranges take priority. ``Start``/``End`` fields carry a single time. A value
    that starts with a bracketed single time (e.g. ``(00:03.4) ...``) is a point
    event. Bare numbers inside prose are deliberately NOT treated as timestamps
    (false-positive defense).
    """
    found = find_time_range(value)
    if found is not None:
        text, start, end = found
        return _TimestampResult(text, start, end, malformed=start is None or end is None)

    if field in _SINGLE_FIELDS:
        token = value.strip()
        single = parse_time_token(token)
        if single is not None:
            return _TimestampResult(token, single, None, malformed=False)
        if token and _HAS_DIGIT.search(token):
            return _TimestampResult(token, None, None, malformed=True)
        return _TimestampResult(None, None, None, malformed=False)

    bracket = _LEADING_BRACKET_TIME.match(value)
    if bracket is not None:
        single = parse_time_token(bracket.group("t"))
        return _TimestampResult(bracket.group(0), single, None, malformed=single is None)

    if _LOOKS_LIKE_TIME.search(value):
        return _TimestampResult(None, None, None, malformed=True)
    return _TimestampResult(None, None, None, malformed=False)


# --- Headings / fields ----------------------------------------------------

_OVERVIEW_HEADINGS: dict[str, SeedSectionKind] = {
    "overview": SeedSectionKind.OVERVIEW,
    "characters": SeedSectionKind.CAST,
    "character": SeedSectionKind.CAST,
    "cast": SeedSectionKind.CAST,
    "objects": SeedSectionKind.OBJECTS,
    "object": SeedSectionKind.OBJECTS,
    "scene": SeedSectionKind.SCENE,
    "style": SeedSectionKind.STYLE,
    "audio": SeedSectionKind.AUDIO,
    "visual concerns": SeedSectionKind.VISUAL_CONCERNS,
    "visual concern": SeedSectionKind.VISUAL_CONCERNS,
    "audio concerns": SeedSectionKind.AUDIO_CONCERNS,
    "audio concern": SeedSectionKind.AUDIO_CONCERNS,
}

#: Per-shot field labels. Order matters for prefix disambiguation is handled by
#: exact-label matching on the left of the first colon, so order here is only
#: cosmetic.
_SHOT_FIELDS: dict[str, SeedFieldKind] = {
    "start": SeedFieldKind.SHOT_START,
    "shot start": SeedFieldKind.SHOT_START,
    "end": SeedFieldKind.SHOT_END,
    "shot end": SeedFieldKind.SHOT_END,
    "cut": SeedFieldKind.TRANSITION,
    "cut into this shot": SeedFieldKind.TRANSITION,
    "transition": SeedFieldKind.TRANSITION,
    "camera movements": SeedFieldKind.CAMERA_MOVEMENTS,
    "camera movement": SeedFieldKind.CAMERA_MOVEMENTS,
    "camera": SeedFieldKind.CAMERA,
    "scene": SeedFieldKind.SCENE,
    "action & audio": SeedFieldKind.ACTION_AUDIO,
    "action and audio": SeedFieldKind.ACTION_AUDIO,
    "action/audio": SeedFieldKind.ACTION_AUDIO,
    "movements": SeedFieldKind.ACTION_AUDIO,
    "playback speed": SeedFieldKind.PLAYBACK_SPEED,
    "video playback speed": SeedFieldKind.PLAYBACK_SPEED,
    "speed changes": SeedFieldKind.SPEED_CHANGES,
    "speed change": SeedFieldKind.SPEED_CHANGES,
}

_SHOT_HEADER_RE = re.compile(r"^\[?\s*shot\s+(\d+)\b", re.IGNORECASE)
_VIDEO_ID_RE = re.compile(r"^video[\s_]*id\s*[:=]\s*(.+)$", re.IGNORECASE)
#: Leading Markdown / list decoration that can precede a heading or label.
_LEADING_DECOR = re.compile(r"^[\s>#*\-•]+")
_TRAILING_DECOR = re.compile(r"[\s*_:#]+$")


def _strip_decoration(line: str) -> str:
    without_lead = _LEADING_DECOR.sub("", line)
    return without_lead.strip()


def _normalize_label(label: str) -> str:
    cleaned = label.strip().strip("*_ ").strip().lower()
    cleaned = _TRAILING_DECOR.sub("", cleaned).strip()
    # Collapse internal whitespace.
    return re.sub(r"\s+", " ", cleaned)


def _match_overview_heading(stripped: str) -> tuple[SeedSectionKind, str] | None:
    """A heading is a known label alone (optionally with a trailing colon and no
    substantive value)."""
    # Allow "Characters" or "Characters:" but not "Character positions: ...".
    label_part, sep, value_part = stripped.partition(":")
    key = _normalize_label(label_part)
    if key in _OVERVIEW_HEADINGS and (not sep or not value_part.strip()):
        return _OVERVIEW_HEADINGS[key], stripped
    # Also accept a bare heading with no colon.
    if not sep:
        key2 = _normalize_label(stripped)
        if key2 in _OVERVIEW_HEADINGS:
            return _OVERVIEW_HEADINGS[key2], stripped
    return None


def _match_shot_field(stripped: str) -> tuple[SeedFieldKind, str, str] | None:
    label_part, sep, value_part = stripped.partition(":")
    if not sep:
        return None
    key = _normalize_label(label_part)
    kind = _SHOT_FIELDS.get(key)
    if kind is None:
        return None
    return kind, label_part.strip(), value_part.strip()


def _extract_ids(text: str) -> tuple[list[str], list[str]]:
    characters = [f"C{m.group(1)}" for m in _CHARACTER_RE.finditer(text)]
    objects = [f"O{m.group(1)}" for m in _OBJECT_RE.finditer(text)]
    # Deduplicate preserving order.
    return _dedup(characters), _dedup(objects)


def _dedup(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _extract_quotes(text: str) -> list[str]:
    return [m.group(1) for m in _QUOTE_RE.finditer(text)]


def parse_seed_text(text: str, snapshot: SeedSnapshot | None = None) -> SeedDocument:
    """Parse decoded seed text into a recoverable :class:`SeedDocument`."""
    lines = text.splitlines()
    sections: list[SeedSection] = []
    issues: list[SeedParseIssue] = []
    video_id: str | None = None

    # Start with a PREAMBLE section so no early line is ever dropped.
    preamble = SeedSection(section_index=0, kind=SeedSectionKind.PREAMBLE, source_line=1)
    sections.append(preamble)
    current = preamble
    entry_counter = 0
    issue_counter = 0

    for line_no, raw_line in enumerate(lines, start=1):
        stripped = _strip_decoration(raw_line)

        if not stripped:
            continue

        # Video ID (anywhere; first wins).
        vid = _VIDEO_ID_RE.match(stripped)
        if vid is not None and video_id is None:
            video_id = vid.group(1).strip()
            section = SeedSection(
                section_index=len(sections),
                kind=SeedSectionKind.VIDEO_ID,
                heading_text=stripped,
                source_line=line_no,
            )
            sections.append(section)
            current = section
            continue

        # Shot header -> new SHOT section.
        shot_match = _SHOT_HEADER_RE.match(stripped)
        if shot_match is not None:
            shot_number = int(shot_match.group(1))
            section = SeedSection(
                section_index=len(sections),
                kind=SeedSectionKind.SHOT,
                heading_text=stripped,
                source_line=line_no,
                shot_number=shot_number,
            )
            sections.append(section)
            current = section
            # The header itself may carry the shot's start-end range.
            found = find_time_range(stripped)
            if found is not None:
                entry_counter += 1
                matched, start, end = found
                if (start is None or end is None):
                    issue_counter += 1
                    issues.append(
                        SeedParseIssue(
                            issue_id=f"SPI-{issue_counter:04d}",
                            source_line=line_no,
                            raw_text=raw_line,
                            message="Shot header timestamp range could not be fully parsed.",
                            severity=SeedParseSeverity.WARNING,
                        )
                    )
                current.entries.append(
                    SeedEntry(
                        entry_id=f"SE-{entry_counter:04d}",
                        section_index=section.section_index,
                        source_line=line_no,
                        raw_line=raw_line,
                        field=SeedFieldKind.FREEFORM,
                        field_label="Shot header",
                        shot_number=shot_number,
                        value_text=stripped,
                        timestamp_text=matched,
                        parsed_start_exact=start,
                        parsed_end_exact=end,
                    )
                )
            continue

        # Inside a shot: try per-shot field labels first.
        if current.kind == SeedSectionKind.SHOT:
            field_match = _match_shot_field(stripped)
            if field_match is not None:
                kind, label, value = field_match
                entry_counter += 1
                current.entries.append(
                    _build_entry(
                        entry_counter,
                        current,
                        line_no,
                        raw_line,
                        kind,
                        label,
                        value,
                        issues,
                    )
                )
                continue
        else:
            # Not in a shot: an Overview heading opens a new section.
            heading = _match_overview_heading(stripped)
            if heading is not None:
                kind_h, heading_text = heading
                section = SeedSection(
                    section_index=len(sections),
                    kind=kind_h,
                    heading_text=heading_text,
                    source_line=line_no,
                )
                sections.append(section)
                current = section
                continue

        # Otherwise: a freeform entry belonging to the current section.
        entry_counter += 1
        current.entries.append(
            _build_entry(
                entry_counter,
                current,
                line_no,
                raw_line,
                SeedFieldKind.FREEFORM,
                None,
                stripped,
                issues,
            )
        )

    # Drop the PREAMBLE if it never collected anything.
    sections = [s for s in sections if not (s.kind == SeedSectionKind.PREAMBLE and not s.entries)]
    # Reindex sections so section_index stays contiguous and stable.
    reindexed = _reindex(sections)

    return SeedDocument(
        snapshot=snapshot,
        video_id=video_id,
        raw_line_count=len(lines),
        sections=reindexed,
        issues=issues,
    )


def _build_entry(
    counter: int,
    section: SeedSection,
    line_no: int,
    raw_line: str,
    field: SeedFieldKind,
    label: str | None,
    value: str,
    issues: list[SeedParseIssue],
) -> SeedEntry:
    characters, objects = _extract_ids(value)
    quotes = _extract_quotes(value)
    ts = _extract_timestamp(field, value)
    if ts.malformed:
        issues.append(
            SeedParseIssue(
                issue_id=f"SPI-{len(issues) + 1:04d}",
                source_line=line_no,
                raw_text=raw_line,
                message="Timestamp-like text could not be parsed to an exact value; "
                "left unrepaired for review.",
                severity=SeedParseSeverity.WARNING,
            )
        )
    return SeedEntry(
        entry_id=f"SE-{counter:04d}",
        section_index=section.section_index,
        source_line=line_no,
        raw_line=raw_line,
        field=field,
        field_label=label,
        shot_number=section.shot_number,
        value_text=value,
        timestamp_text=ts.text,
        parsed_start_exact=ts.start,
        parsed_end_exact=ts.end,
        referenced_character_ids=characters,
        referenced_object_ids=objects,
        quoted_strings=quotes,
    )


def _reindex(sections: list[SeedSection]) -> list[SeedSection]:
    result: list[SeedSection] = []
    for new_index, section in enumerate(sections):
        entries = [e.model_copy(update={"section_index": new_index}) for e in section.entries]
        result.append(section.model_copy(update={"section_index": new_index, "entries": entries}))
    return result
