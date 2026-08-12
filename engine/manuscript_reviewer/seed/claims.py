"""Atomic seed-claim extraction.

Turns a parsed :class:`SeedDocument` into a list of *atomic* claims (the
extended :class:`~manuscript_reviewer.models.caption.SeedClaim`). A paragraph is
never one giant claim: each defined entity, each shot boundary, each transition,
each timestamped action/overlay/speed field becomes its own independently
reviewable claim.

Claim *importance* (FOUNDATIONAL vs LOCAL) is set here; ``evidence_status`` is
left ``None`` and is filled in later by comparison/matrix building. The two are
different concepts and are never conflated.
"""

from __future__ import annotations

import re

from ..models.caption import ExactTimeRange, SeedClaim
from ..models.review_intelligence import (
    ClaimImportance,
    ClaimReviewStatus,
    SeedClaimType,
    SeedDocument,
    SeedEntry,
    SeedFieldKind,
    SeedParseIssue,
    SeedParseSeverity,
    SeedSection,
    SeedSectionKind,
)
from .parser import _extract_ids

# Character/object definition lines look like "C1: description" / "O2 - ...".
_DEF_RE = re.compile(r"^\s*@?(?P<id>[CO]\d+)\s*[:\-–—)]", re.IGNORECASE)  # noqa: RUF001

# Protected / unsupported traits: captured so they can be flagged unsupported,
# NEVER visually inferred. Detection is conservative and additive.
_GENDER_WORDS = re.compile(
    r"\b(man|woman|male|female|boy|girl|lady|gentleman|guys?|men|women)\b", re.IGNORECASE
)
_AGE_WORDS = re.compile(
    r"\b(\d+\s*[-\s]?year[-\s]?old|aged?\s+\d+|young|old|elderly|middle[-\s]?aged|teen(?:ager)?|"
    r"child|adult|twenties|thirties|forties|fifties|sixties)\b",
    re.IGNORECASE,
)
_ACCENT_WORDS = re.compile(r"\baccent(?:ed)?\b", re.IGNORECASE)
_EXPLICIT_PROTECTED = re.compile(r"\b(nationality|ethnicity|ethnic|race|racial)\b", re.IGNORECASE)
#: A modest nationality/ethnicity adjective set (extend as needed). Deliberately
#: small — the real guarantee is that the CV side never *infers* these.
_NATIONALITY = re.compile(
    r"\b(American|British|English|Irish|Scottish|French|German|Spanish|Italian|"
    r"Portuguese|Russian|Chinese|Japanese|Korean|Indian|Pakistani|Nigerian|"
    r"Mexican|Brazilian|Canadian|Australian|Dutch|Swedish|Norwegian|Polish|"
    r"Greek|Turkish|Arab|Arabic|African|Asian|European|Latino|Latina|Hispanic|"
    r"Caucasian|Black|White)\b"
)


def _time_range(entry: SeedEntry) -> ExactTimeRange | None:
    start = entry.parsed_start_exact
    end = entry.parsed_end_exact
    if start is None and end is None:
        return None
    if end is None:
        end = start
    if start is None:
        start = end
    # Both are guaranteed non-None here.
    assert start is not None and end is not None
    return ExactTimeRange(start_seconds=start, end_seconds=end)


def _protected_trait_hits(text: str) -> list[str]:
    hits: list[str] = []
    if _EXPLICIT_PROTECTED.search(text):
        hits.append("nationality/ethnicity/race")
    if _NATIONALITY.search(text):
        hits.append("nationality/ethnicity")
    if _GENDER_WORDS.search(text):
        hits.append("gender")
    if _AGE_WORDS.search(text):
        hits.append("age")
    if _ACCENT_WORDS.search(text):
        hits.append("accent")
    return hits


class _Counter:
    def __init__(self) -> None:
        self.value = 0

    def next(self, prefix: str) -> str:
        self.value += 1
        return f"{prefix}-{self.value:04d}"


# --- attribute decomposition (item B) -------------------------------------

#: phrase-classifier: (regex, trait subtype). First match wins.
_CHAR_TRAIT_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bhair\b|\bbald\b|\bbeard\b|\bmoustache\b|\bmustache\b", re.I), "hair"),
    (re.compile(r"\bglasses\b|\bsunglasses\b|\beyewear\b|\bspectacles\b|\bmonocle\b", re.I),
     "eyewear"),
    (re.compile(r"\b(shirt|t-?shirt|jacket|coat|hoodie|sweater|jumper|dress|blouse|top|"
                r"vest|uniform|tunic|robe)\b", re.I), "upper_clothing"),
    (re.compile(r"\b(trousers|pants|jeans|shorts|skirt|leggings|shoes|boots|sneakers|"
                r"sandals|socks)\b", re.I), "lower_clothing"),
    (re.compile(r"\b(backpack|bag|rucksack|watch|necklace|bracelet|gloves|hat|cap|helmet|"
                r"scarf|tie|earrings?|ring)\b", re.I), "accessory"),
]
_VISIBILITY_RE = re.compile(
    r"not visible|never visible|off[-\s]?screen|lower body|out of frame|only.*visible", re.I
)
_OBJ_TRAIT_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bwheels?\b|\btyres?\b|\btires?\b", re.I), "wheels"),
    (re.compile(r"\blogo\b|\bmarking?s?\b|\bsticker\b|\blabel\b|\bemblem\b|\btext\b|\bnumber\b",
                re.I), "marking"),
    (re.compile(r"\b(red|orange|yellow|green|blue|purple|pink|black|white|grey|gray|brown|"
                r"silver|gold|chrome|beige|tan|navy)\b", re.I), "color"),
]


def _split_phrases(text: str) -> list[str]:
    """Split a description into candidate attribute phrases on commas / 'and' /
    'with'. Conservative: keeps each phrase's exact words."""
    # Remove a leading article-only head like "A man" is kept as the base claim;
    # split the remainder on separators.
    parts = re.split(r",|\band\b|\bwith\b|\bwearing\b", text, flags=re.IGNORECASE)
    return [p.strip(" .;") for p in parts if p.strip(" .;")]


def _character_attribute_claims(
    text: str, char_id: str, entry: SeedEntry, counter: _Counter
) -> list[SeedClaim]:
    claims: list[SeedClaim] = []
    seen_subtypes: set[str] = set()
    for phrase in _split_phrases(text):
        subtype: str | None = None
        claim_type = SeedClaimType.CHARACTER_TRAIT
        if _VISIBILITY_RE.search(phrase):
            subtype = "visibility"
            claim_type = SeedClaimType.CHARACTER_VISIBILITY
        else:
            for pattern, name in _CHAR_TRAIT_RULES:
                if pattern.search(phrase):
                    subtype = name
                    break
        if subtype is None or subtype in seen_subtypes:
            continue
        seen_subtypes.add(subtype)
        claims.append(
            SeedClaim(
                claim_id=counter.next("CLM"),
                source_field=f"Characters/{subtype}",
                text=phrase,
                claim_type=claim_type,
                subject_ids=[char_id],
                seed_source_line=entry.source_line,
                seed_entry_id=entry.entry_id,
                importance=ClaimImportance.LOCAL,
                review_status=ClaimReviewStatus.MACHINE_ONLY,
            )
        )
    return claims


def _object_attribute_claims(
    text: str, obj_id: str, entry: SeedEntry, counter: _Counter
) -> list[SeedClaim]:
    claims: list[SeedClaim] = []
    seen: set[str] = set()
    for phrase in _split_phrases(text):
        for pattern, name in _OBJ_TRAIT_RULES:
            if pattern.search(phrase) and name not in seen:
                seen.add(name)
                claims.append(
                    SeedClaim(
                        claim_id=counter.next("CLM"),
                        source_field=f"Objects/{name}",
                        text=phrase,
                        claim_type=SeedClaimType.OBJECT_TRAIT,
                        object_ids=[obj_id],
                        seed_source_line=entry.source_line,
                        seed_entry_id=entry.entry_id,
                        importance=ClaimImportance.LOCAL,
                        review_status=ClaimReviewStatus.MACHINE_ONLY,
                    )
                )
                break
    return claims


def extract_claims(doc: SeedDocument) -> list[SeedClaim]:
    """Extract atomic claims from a parsed seed document."""
    claims: list[SeedClaim] = []
    counter = _Counter()

    # --- MEDIA_ID (foundational) ---
    if doc.video_id is not None:
        claims.append(
            SeedClaim(
                claim_id=counter.next("CLM"),
                source_field="video_id",
                text=f"Video ID: {doc.video_id}",
                claim_type=SeedClaimType.MEDIA_ID,
                seed_source_line=_video_id_line(doc),
                importance=ClaimImportance.FOUNDATIONAL,
                review_status=ClaimReviewStatus.MACHINE_ONLY,
            )
        )

    # --- SHOT_COUNT (foundational) ---
    seed_shots = doc.seed_shot_count
    if seed_shots is not None:
        claims.append(
            SeedClaim(
                claim_id=counter.next("CLM"),
                source_field="shot_structure",
                text=f"Seed describes {seed_shots} shot(s).",
                claim_type=SeedClaimType.SHOT_COUNT,
                importance=ClaimImportance.FOUNDATIONAL,
                review_status=ClaimReviewStatus.MACHINE_ONLY,
            )
        )

    for section in doc.sections:
        if section.kind == SeedSectionKind.CAST:
            claims.extend(_cast_claims(section, counter))
        elif section.kind == SeedSectionKind.OBJECTS:
            claims.extend(_object_claims(section, counter))
        elif section.kind in (SeedSectionKind.SCENE, SeedSectionKind.OVERVIEW):
            claims.extend(_generic_claims(section, counter, SeedClaimType.SCENE_STATE))
        elif section.kind == SeedSectionKind.STYLE:
            claims.extend(_generic_claims(section, counter, SeedClaimType.STYLE_STATE))
        elif section.kind == SeedSectionKind.AUDIO:
            claims.extend(_generic_claims(section, counter, SeedClaimType.SOUND))
        elif section.kind == SeedSectionKind.VISUAL_CONCERNS:
            claims.extend(_generic_claims(section, counter, SeedClaimType.VISUAL_CONCERN))
        elif section.kind == SeedSectionKind.AUDIO_CONCERNS:
            claims.extend(_generic_claims(section, counter, SeedClaimType.AUDIO_CONCERN))
        elif section.kind == SeedSectionKind.SHOT:
            claims.extend(_shot_claims(section, counter))

    return claims


def _video_id_line(doc: SeedDocument) -> int | None:
    for section in doc.sections:
        if section.kind == SeedSectionKind.VIDEO_ID:
            return section.source_line
    return None


def _cast_claims(section: SeedSection, counter: _Counter) -> list[SeedClaim]:
    claims: list[SeedClaim] = []
    for entry in section.entries:
        text = entry.value_text or entry.raw_line
        match = _DEF_RE.match(text)
        char_id = match.group("id").upper().lstrip("@") if match else None
        subject = [char_id] if char_id else entry.referenced_character_ids
        if char_id is not None:
            claims.append(
                SeedClaim(
                    claim_id=counter.next("CLM"),
                    source_field="Characters",
                    text=text,
                    claim_type=SeedClaimType.CHARACTER_EXISTS,
                    subject_ids=[char_id],
                    shot_number=None,
                    seed_source_line=entry.source_line,
                    seed_entry_id=entry.entry_id,
                    importance=ClaimImportance.FOUNDATIONAL,
                    review_status=ClaimReviewStatus.MACHINE_ONLY,
                )
            )
            # Independent attribute claims (hair, eyewear, clothing, accessory,
            # visibility) — each independently reviewable.
            claims.extend(_character_attribute_claims(text, char_id, entry, counter))
        for trait in _protected_trait_hits(text):
            claims.append(
                SeedClaim(
                    claim_id=counter.next("CLM"),
                    source_field="Characters",
                    text=f"{subject or ['?']}: {trait} trait — {text}",
                    claim_type=SeedClaimType.PROTECTED_TRAIT,
                    subject_ids=list(subject),
                    seed_source_line=entry.source_line,
                    seed_entry_id=entry.entry_id,
                    importance=ClaimImportance.LOCAL,
                    review_status=ClaimReviewStatus.REVIEW_REQUIRED,
                )
            )
    return claims


def _object_claims(section: SeedSection, counter: _Counter) -> list[SeedClaim]:
    claims: list[SeedClaim] = []
    for entry in section.entries:
        text = entry.value_text or entry.raw_line
        match = _DEF_RE.match(text)
        obj_id = match.group("id").upper().lstrip("@") if match else None
        if obj_id is not None:
            claims.append(
                SeedClaim(
                    claim_id=counter.next("CLM"),
                    source_field="Objects",
                    text=text,
                    claim_type=SeedClaimType.OBJECT_EXISTS,
                    object_ids=[obj_id],
                    seed_source_line=entry.source_line,
                    seed_entry_id=entry.entry_id,
                    importance=ClaimImportance.FOUNDATIONAL,
                    review_status=ClaimReviewStatus.MACHINE_ONLY,
                )
            )
            claims.extend(_object_attribute_claims(text, obj_id, entry, counter))
    return claims


def _generic_claims(
    section: SeedSection, counter: _Counter, claim_type: SeedClaimType
) -> list[SeedClaim]:
    claims: list[SeedClaim] = []
    for entry in section.entries:
        text = entry.value_text or entry.raw_line
        if not text.strip():
            continue
        claims.append(
            SeedClaim(
                claim_id=counter.next("CLM"),
                source_field=section.kind.value,
                text=text,
                claim_type=claim_type,
                subject_ids=entry.referenced_character_ids,
                object_ids=entry.referenced_object_ids,
                seed_source_line=entry.source_line,
                seed_entry_id=entry.entry_id,
                importance=ClaimImportance.LOCAL,
                review_status=ClaimReviewStatus.MACHINE_ONLY,
            )
        )
    return claims


_SPEECH_RE = re.compile(r"\b(say|says|said|speak|speaks|shout|shouts|asks?|replies|whisper)\b",
                        re.IGNORECASE)
_TEXT_RE = re.compile(r"\b(on[-\s]?screen text|text reads?|caption reads?|overlay|subtitle)\b",
                      re.IGNORECASE)
_SOUND_RE = re.compile(
    r"\b(sounds?|music|noise|beeps?|thud|bang|ambient|sfx|chimes?|dings?|clicks?|"
    r"whooshe?s?|rings?|jingle|tone|plays?|rumble|crash|clang)\b",
    re.IGNORECASE,
)


_UI_CUE = re.compile(
    r"\b(popup|pop-?up|overlay|banner|notification|hud|toast|window appears|icon appears)\b",
    re.IGNORECASE,
)
_CONNECTIVE = re.compile(r"\s+\b(?:and|then|while|as)\b\s+", re.IGNORECASE)
_NEW_CLAUSE = re.compile(r"^(?:a|an|the)\s+\w+\s+\w+", re.IGNORECASE)


def _classify_clause(clause: str, has_quote: bool) -> SeedClaimType:
    if _TEXT_RE.search(clause) or _UI_CUE.search(clause):
        return SeedClaimType.ON_SCREEN_TEXT
    if has_quote and _SPEECH_RE.search(clause):
        return SeedClaimType.SPEECH
    if _SOUND_RE.search(clause):
        return SeedClaimType.SOUND
    return SeedClaimType.ACTION


def _clause_is_new_event(right: str) -> bool:
    """A connective introduces a separable event when the right clause carries a
    sound/UI/text cue, a quote, or a fresh '<article> <noun> <verb>' clause."""
    return bool(
        _SOUND_RE.search(right)
        or _UI_CUE.search(right)
        or _TEXT_RE.search(right)
        or '"' in right
        or _NEW_CLAUSE.match(right.strip())
    )


def _event_text(entry: SeedEntry) -> str:
    """The Action & Audio text with any leading timestamp line/prefix removed."""
    value = entry.value_text or entry.raw_line
    if entry.timestamp_text and value.startswith(entry.timestamp_text):
        value = value[len(entry.timestamp_text):]
    # Block form joins "1.0-2.0\nC1 moves." — take the text after the newline.
    if "\n" in value:
        value = value.split("\n", 1)[1]
    return value.strip(" :\t")


def _decompose_action(text: str) -> list[str]:
    """Split an Action & Audio line into independent candidate clauses at
    connectives — but only where the separation is grammatically clear. Never
    splits inside a quoted span; never splits an inseparable single clause."""
    clauses: list[str] = []
    remaining = text
    while True:
        cut = None
        for match in _CONNECTIVE.finditer(remaining):
            # Do not split inside a quoted span.
            if remaining[: match.start()].count('"') % 2 == 1:
                continue
            right = remaining[match.end():]
            if _clause_is_new_event(right):
                cut = (match.start(), match.end())
                break
        if cut is None:
            clauses.append(remaining.strip(" .;"))
            break
        clauses.append(remaining[: cut[0]].strip(" .;"))
        remaining = remaining[cut[1]:]
    return [c for c in clauses if c]


def _shot_claims(section: SeedSection, counter: _Counter) -> list[SeedClaim]:
    claims: list[SeedClaim] = []
    shot = section.shot_number

    # One boundary claim per shot, combining the `[Shot N: start-end]` header
    # with any explicit Start:/End: fields (so a correct Start:/End: pair is not
    # mis-scored against the header range, and vice versa).
    boundary = _shot_boundary_claim(section, shot, counter)
    if boundary is not None:
        claims.append(boundary)

    for entry in section.entries:
        if entry.field_label == "Shot header":
            continue
        if entry.field == SeedFieldKind.ACTION_AUDIO:
            claims.extend(_action_claims(entry, shot, counter))
            continue
        claim_type, importance = _shot_entry_type(entry)
        if claim_type is None:
            continue
        quoted = entry.quoted_strings[0] if entry.quoted_strings else None
        claims.append(
            SeedClaim(
                claim_id=counter.next("CLM"),
                source_field=f"Shot {shot} {entry.field.value}" if shot else entry.field.value,
                text=entry.value_text or entry.raw_line,
                claim_type=claim_type,
                subject_ids=entry.referenced_character_ids,
                object_ids=entry.referenced_object_ids,
                shot_number=shot,
                seed_source_line=entry.source_line,
                seed_entry_id=entry.entry_id,
                seed_time_range=_time_range(entry),
                quoted_text=quoted,
                importance=importance,
                review_status=ClaimReviewStatus.MACHINE_ONLY,
            )
        )
    return claims


def _action_claims(entry: SeedEntry, shot: int | None, counter: _Counter) -> list[SeedClaim]:
    """Decompose one Action & Audio entry into independent candidate claims,
    recording that they came from the same SeedEntry (seed_entry_id)."""
    event = _event_text(entry)
    has_quote = bool(entry.quoted_strings)
    clauses = _decompose_action(event) or [event]
    time_range = _time_range(entry)
    claims: list[SeedClaim] = []
    for clause in clauses:
        chars, objs = _extract_ids(clause)
        quoted = _first_quote(clause)
        claims.append(
            SeedClaim(
                claim_id=counter.next("CLM"),
                source_field=f"Shot {shot} Action & Audio" if shot else "Action & Audio",
                text=clause,
                claim_type=_classify_clause(clause, has_quote),
                subject_ids=chars or entry.referenced_character_ids,
                object_ids=objs,
                shot_number=shot,
                seed_source_line=entry.source_line,
                seed_entry_id=entry.entry_id,
                seed_time_range=time_range,
                quoted_text=quoted,
                importance=ClaimImportance.LOCAL,
                review_status=ClaimReviewStatus.MACHINE_ONLY,
            )
        )
    return claims


def _first_quote(text: str) -> str | None:
    m = re.search(r'"([^"]*)"', text)
    return m.group(1) if m else None


_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


def collect_seed_diagnostics(doc: SeedDocument) -> list[SeedParseIssue]:
    """Platform-semantic seed-review diagnostics for Action & Audio atomicity.

    These are seed-review evidence (leads for the reviewer), NOT final caption
    validation. Each is anchored to the seed source line.
    """
    diagnostics: list[SeedParseIssue] = []
    counter = 0

    def add(entry: SeedEntry, code: str, detail: str) -> None:
        nonlocal counter
        counter += 1
        diagnostics.append(
            SeedParseIssue(
                issue_id=f"SDX-{counter:04d}",
                source_line=entry.source_line,
                raw_text=entry.raw_line,
                message=f"{code}: {detail}",
                severity=SeedParseSeverity.INFO,
            )
        )

    for section in doc.shot_sections:
        for entry in section.entries:
            if entry.field != SeedFieldKind.ACTION_AUDIO:
                continue
            event = _event_text(entry)
            sentences = [s for s in _SENTENCE_END.split(event) if s.strip()]
            if len(sentences) >= 2:
                add(entry, "MULTIPLE_SENTENCES", f"{len(sentences)} sentences in one line")
            if len(entry.quoted_strings) >= 2:
                add(entry, "MULTIPLE_QUOTED_SPANS",
                    f"{len(entry.quoted_strings)} quoted spans in one line")
            clauses = _decompose_action(event)
            action_clauses = [
                c for c in clauses if _classify_clause(c, False) == SeedClaimType.ACTION
            ]
            if len(action_clauses) >= 2:
                add(entry, "MULTIPLE_ACTION_CLAUSES",
                    f"{len(action_clauses)} distinct finite action clauses")
            has_action = any(_classify_clause(c, bool(entry.quoted_strings)) == SeedClaimType.ACTION
                             for c in clauses)
            has_sound = bool(_SOUND_RE.search(event))
            has_overlay = bool(_TEXT_RE.search(event) or _UI_CUE.search(event))
            if has_action and has_sound:
                add(entry, "MIXED_VISUAL_AND_SOUND", "one line mixes a visual action and a sound")
            if has_action and has_overlay:
                add(entry, "MIXED_VISUAL_AND_OVERLAY",
                    "one line mixes a visual action and an on-screen text/UI event")
    return diagnostics


def _shot_boundary_claim(
    section: SeedSection, shot: int | None, counter: _Counter
) -> SeedClaim | None:
    """Build the single foundational SHOT_BOUNDARY claim for a shot.

    Prefers the header's start-end range; fills a missing endpoint from an
    explicit Start:/End: field. Returns None when no boundary time is stated.
    """
    header = next((e for e in section.entries if e.field_label == "Shot header"), None)
    start_entry = next((e for e in section.entries if e.field == SeedFieldKind.SHOT_START), None)
    end_entry = next((e for e in section.entries if e.field == SeedFieldKind.SHOT_END), None)

    b_start = header.parsed_start_exact if header is not None else None
    b_end = header.parsed_end_exact if header is not None else None
    if b_start is None and start_entry is not None:
        b_start = start_entry.parsed_start_exact
    if b_end is None and end_entry is not None:
        b_end = end_entry.parsed_start_exact  # End: field time is stored in start

    if b_start is None and b_end is None:
        return None
    if b_end is None:
        b_end = b_start
    if b_start is None:
        b_start = b_end
    assert b_start is not None and b_end is not None

    source = header or start_entry or end_entry
    assert source is not None
    return SeedClaim(
        claim_id=counter.next("CLM"),
        source_field="Shot boundary",
        text=source.value_text or source.raw_line,
        claim_type=SeedClaimType.SHOT_BOUNDARY,
        shot_number=shot,
        seed_source_line=source.source_line,
        seed_entry_id=source.entry_id,
        seed_time_range=ExactTimeRange(start_seconds=b_start, end_seconds=b_end),
        importance=ClaimImportance.FOUNDATIONAL,
        review_status=ClaimReviewStatus.MACHINE_ONLY,
    )


def _shot_entry_type(entry: SeedEntry) -> tuple[SeedClaimType | None, ClaimImportance]:
    field = entry.field
    if field == SeedFieldKind.TRANSITION:
        return SeedClaimType.TRANSITION, ClaimImportance.FOUNDATIONAL
    if field == SeedFieldKind.SHOT_START or field == SeedFieldKind.SHOT_END:
        # Folded into the single per-shot boundary claim above.
        return None, ClaimImportance.FOUNDATIONAL
    if field == SeedFieldKind.CAMERA:
        return SeedClaimType.CAMERA_FRAMING, ClaimImportance.LOCAL
    if field == SeedFieldKind.CAMERA_MOVEMENTS:
        return SeedClaimType.CAMERA_MOVEMENT, ClaimImportance.LOCAL
    if field == SeedFieldKind.SCENE:
        return SeedClaimType.SCENE_STATE, ClaimImportance.LOCAL
    if field == SeedFieldKind.PLAYBACK_SPEED or field == SeedFieldKind.SPEED_CHANGES:
        return SeedClaimType.PLAYBACK_SPEED, ClaimImportance.LOCAL
    # ACTION_AUDIO is decomposed separately (_action_claims); FREEFORM and any
    # future field produce no atomic claim on their own.
    return None, ClaimImportance.LOCAL
