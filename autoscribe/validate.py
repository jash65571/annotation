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

from manuscript_reviewer.caption.textcheck import (
    find_quote_spans,
    pronoun_hits,
    strip_quotes,
)

from .blockers import WARNING, BlockerLog
from .cuts import CUT_TYPES

#: Canonical section labels of the live tool's master template (SOT §26/§27).
REQUIRED_OVERVIEW_FIELDS = (
    "Cast:", "Scene:", "Style:", "Audio:", "Visual Concerns:", "Audio Concerns:",
)
CANONICAL_SHOT_FIELDS = ("Cut:", "Camera:", "Scene:", "Playback Speed:")

#: Field names AutoScribe used to emit that are NOT the canonical ones.
LEGACY_FIELDS = {
    "Characters:": "Cast:",
    "Camera movements:": "Camera Movements:",
    "Video playback speed:": "Playback Speed:",
    "Visual concerns:": "Visual Concerns:",
    "Audio concerns:": "Audio Concerns:",
}

#: Evaluator feedback (Aug 2026): per-shot Scenes ran 26-32 words against a
#: ~72-word norm, and read as object inventories. These are the minimum bars a
#: description must clear before a human even looks at it.
SCENE_MIN_WORDS = 45
SCENE_TARGET_WORDS = 72
STYLE_MIN_WORDS = 25

#: A scene that reconstructs a space names where things are relative to one
#: another, not merely that they exist.
_SPATIAL_TERMS = re.compile(
    r"\b(left of|right of|screen-left|screen-right|in front of|behind|beside|"
    r"next to|opposite|above|below|beneath|under|on top of|rests? on|stands? on|"
    r"between|across from|adjacent|along|toward the camera|away from the camera|"
    r"foreground|middle ground|midground|background)\b",
    re.IGNORECASE,
)
_DEPTH_PLANES = (
    re.compile(r"\bforeground\b", re.IGNORECASE),
    re.compile(r"\b(middle ?ground|midground)\b", re.IGNORECASE),
    re.compile(r"\bbackground\b", re.IGNORECASE),
)
_LIGHT_DIRECTION = re.compile(
    r"\b(from (?:the )?(?:left|right|above|below|behind|front|window|overhead)|"
    r"overhead|backlit|backlight|side-?lit|top-?lit|key light|fill light|rim light|"
    r"under-?lit|frontal)\b",
    re.IGNORECASE,
)
_SHADOW_QUALITY = re.compile(
    r"\b(shadow|shadows|shadowless|hard-?edged|soft-?edged|diffuse|diffused|"
    r"high-?contrast|low-?contrast)\b",
    re.IGNORECASE,
)
_COLOR_TEMPERATURE = re.compile(
    r"\b(warm|cool|tungsten|daylight|neutral|golden|amber|blue-?ish|"
    r"colou?r temperature|kelvin|\d{4}\s*K)\b",
    re.IGNORECASE,
)
_NO_CHANGES = re.compile(r"^no changes from overview\.?$", re.IGNORECASE)

#: A line that reports speech. Source of truth §10 Rule 4 requires each to carry
#: a supported audible tone.
_SPEECH_VERB = re.compile(
    r"\b(says?|said|sings?|shouts?|whispers?|asks?|replies|replied|narrates?|"
    r"calls? out|mutters?|yells?|screams?|chants?)\b",
    re.IGNORECASE,
)
#: The canonical delivery phrasing ("in a questioning tone", "in a low, strained
#: tone") plus the other supported ways a delivery attribute can appear.
_DELIVERY = re.compile(
    r"\bin an? [^,\"]{2,60}\b(tone|voice|delivery|pitch|register)\b"
    r"|\b(tone|pitch|pace|delivery)\b"
    r"|\b(quietly|loudly|softly|firmly|urgently|calmly|slowly|quickly|rapidly|"
    r"hesitantly|flatly|sharply|wearily|breathlessly)\b",
    re.IGNORECASE,
)
#: English is the default caption language; anything else must be declared.
_ENGLISH = re.compile(r"^\s*english\s*$", re.IGNORECASE)

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


def _words(text: str) -> int:
    return len([w for w in text.split() if any(c.isalnum() for c in w)])


def _extract_descriptions(
    lines: list[str],
) -> tuple[str, list[tuple[int, str]], str, str]:
    """(overview_scene, [(shot_index, scene)], style, audio) from a caption."""
    overview_scene = ""
    style = ""
    audio = ""
    shot_scenes: list[tuple[int, str]] = []
    shot_index = 0
    for raw in lines:
        line = raw.strip()
        header = _SHOT_HEADER.match(line)
        if header:
            shot_index = int(header.group(1))
            continue
        if line.startswith("Scene:"):
            body = line[len("Scene:"):].strip()
            if shot_index == 0:
                overview_scene = body
            else:
                shot_scenes.append((shot_index, body))
        elif line.startswith("Style:") and not style:
            style = line[len("Style:"):].strip()
        elif line.startswith("Audio:") and not audio:
            audio = line[len("Audio:"):].strip()
    return overview_scene, shot_scenes, style, audio


def check_scene_depth(
    scene: str, where: str, log: BlockerLog, *, is_overview: bool
) -> None:
    """Enforce the 3D-reconstruction standard on a Scene field.

    The decisive finding of the Aug 2026 evaluator audit was that Scenes read as
    object inventories at 26-32 words against a ~72-word norm. A validator that
    passes "A kitchen with a wooden table" enforces nothing.
    """
    body = scene.strip()
    if not body:
        log.add("SCENE_EMPTY", f"{where} has no Scene description.")
        return
    if not is_overview and _NO_CHANGES.match(body):
        # Legitimate ONLY when the space is genuinely identical. It is also a
        # complete bypass of every depth check below, which is exactly how the
        # evaluator's decisive Scene failure could return — so each use has to
        # be confirmed rather than accepted silently.
        log.add(
            "SCENE_UNCHANGED_UNCONFIRMED",
            f"{where} Scene says only 'No changes from overview.', which skips all "
            f"depth requirements. Confirm this shot's space is genuinely identical "
            f"to the Overview; if anything differs, describe it.",
            severity=WARNING,
        )
        return
    count = _words(body)
    if count < SCENE_MIN_WORDS:
        log.add(
            "SCENE_TOO_SHALLOW",
            f"{where} Scene is {count} words; the standard is about "
            f"{SCENE_TARGET_WORDS}. It must let a reader rebuild the space as a 3D "
            f"environment — spatial relationships, geometry, materials, and depth "
            f"planes — not list objects.",
        )
    if not _SPATIAL_TERMS.search(body):
        log.add(
            "SCENE_NO_SPATIAL_RELATIONSHIPS",
            f"{where} Scene names no spatial relationship (left of, behind, resting "
            f"on, foreground/background...). An object list is not a scene.",
        )
    planes = sum(1 for rx in _DEPTH_PLANES if rx.search(body))
    if planes < 2:
        log.add(
            "SCENE_NO_DEPTH_PLANES",
            f"{where} Scene does not separate foreground / middle ground / "
            f"background ({planes} of 3 named).",
            severity=WARNING,
        )


def check_style_depth(style: str, log: BlockerLog) -> None:
    """Style must name light sources and direction, shadow quality, and colour
    temperature — 'Natural light.' satisfies none of those."""
    body = style.strip()
    if not body:
        log.add("STYLE_EMPTY", "Overview has no Style description.")
        return
    if _words(body) < STYLE_MIN_WORDS:
        log.add(
            "STYLE_TOO_SHALLOW",
            f"Style is {_words(body)} words; it must cover light sources and "
            f"direction, shadow quality, colour temperature, depth of field, medium "
            f"and aspect ratio.",
        )
    missing = []
    if not _LIGHT_DIRECTION.search(body):
        missing.append("light source/direction")
    if not _SHADOW_QUALITY.search(body):
        missing.append("shadow quality")
    if not _COLOR_TEMPERATURE.search(body):
        missing.append("colour temperature")
    if missing:
        log.add(
            "STYLE_MISSING_LIGHTING_DETAIL",
            f"Style does not state: {', '.join(missing)}.",
        )


def check_language_declared(
    audio_field: str, detected_language: str, log: BlockerLog
) -> None:
    """Non-English speech must be named in the Audio field.

    Missed case from the Aug 2026 evaluator audit: Tagalog lines ("Diba? Diba?",
    "Arte-arte siya!") went out with no language declared anywhere. Requiring it
    in the prompt is not enforcement — the rendered caption is checked here.
    """
    language = detected_language.strip()
    if not language or _ENGLISH.match(language):
        return
    if language.lower() not in audio_field.lower():
        log.add(
            "LANGUAGE_NOT_DECLARED",
            f"Speech was detected as {language} but the Audio field never names the "
            f"language. Non-English speech must be declared.",
        )


def check_speech_delivery(lines: list[str], log: BlockerLog) -> None:
    """Every speech line needs a supported audible tone (SOT §10 Rule 4).

    A line such as `C1 says off-screen, "Hola."` previously passed with zero
    findings, which is the exact omission the evaluator feedback failed.
    """
    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        timed = _TIMED_LINE.match(line)
        if not timed:
            continue
        body = timed.group(3)
        if not _SPEECH_VERB.search(body) or not find_quote_spans(body):
            continue
        # Check outside the quoted words: the delivery describes the speech,
        # it is not part of it.
        outside = strip_quotes(body)
        if not _DELIVERY.search(outside):
            log.add(
                "SPEECH_NO_DELIVERY",
                f"Line {lineno} reports speech with no delivery attribute (tone, "
                f"pitch or pace). The standard requires a supported audible tone — "
                f"confirm the delivery by ear and add it: {body[:60]!r}",
            )


def validate_caption(
    text: str,
    blockers: BlockerLog | None = None,
    detected_language: str = "",
) -> BlockerLog:
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

    # Descriptive depth — the decisive gap in the Aug 2026 evaluator audit.
    overview_scene, shot_scenes, style, audio = _extract_descriptions(lines)
    check_scene_depth(overview_scene, "Overview", log, is_overview=True)
    check_style_depth(style, log)
    for index, scene in shot_scenes:
        check_scene_depth(scene, f"Shot {index}", log, is_overview=False)
    unchanged = [i for i, s in shot_scenes if _NO_CHANGES.match(s.strip())]
    if len(shot_scenes) > 1 and len(unchanged) == len(shot_scenes):
        log.add(
            "ALL_SCENES_UNCHANGED",
            f"All {len(shot_scenes)} shots say only 'No changes from overview.', so "
            f"no shot carries a description. Separate shots differ by definition — "
            f"describe what each one shows.",
        )
    check_language_declared(audio, detected_language, log)
    check_speech_delivery(lines, log)

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
