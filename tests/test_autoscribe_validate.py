"""The deterministic caption gate.

Every check here corresponds to a defect that previously reached a "final"
file unchallenged.
"""

from __future__ import annotations

import pytest

from autoscribe.validate import validate_caption

#: A caption that actually meets the standard: Scene reconstructs the space,
#: Style names light direction, shadow quality and colour temperature, and the
#: concern fields use the canonical capitalisation. The previous fixture — "A
#: kitchen with a wooden table." / "Natural light." — passed the old validator
#: and is exactly the shallowness the Aug 2026 evaluator audit failed.
SCENE = (
    "A domestic kitchen shot along its length. In the foreground a rectangular "
    "oak table with turned legs occupies the lower third, its surface scratched "
    "and unvarnished. In the middle ground C1 stands behind the table, facing "
    "camera, with open pine shelving on screen-left holding stacked ceramic "
    "bowls and a steel refrigerator on screen-right. The background is a "
    "plastered wall with a deep sash window above the counter run, beyond which "
    "a brick garden wall is visible."
)
STYLE = (
    "Daylight key from the sash window at screen-left rakes across the table, "
    "with soft overhead fill from a ceiling fixture; shadows are soft-edged and "
    "shallow. Colour temperature is cool and slightly blue toward the window, "
    "warming near the shelving. Shallow depth of field, digital capture, gentle "
    "contrast, no non-standard aspect ratio."
)

GOOD = f"""[Overview]
Cast:
C1: A person in a red jacket. Lower body and shoes are not visible.

Scene: {SCENE}

Style: {STYLE}

Audio: Music plays throughout.

Visual Concerns: None.
Audio Concerns: None.

[Shot 1: 0.0s–2.0s]
Cut: Opening shot.
Camera: Medium-wide, eye-level, handheld.
Scene: No changes from overview.
Action & Audio:
(0.0s–1.0s) C1 raises the right hand.
(1.0s–2.0s) C1 lowers the right hand.
Playback Speed: regular.
"""


def _codes(text: str) -> set[str]:
    return {b.code for b in validate_caption(text).entries}


def test_clean_caption_has_no_blocking_findings() -> None:
    log = validate_caption(GOOD)
    assert log.blocking == [], [b.describe() for b in log.blocking]


def test_legacy_field_names_are_rejected() -> None:
    text = GOOD.replace("Cast:", "Characters:").replace(
        "Playback Speed:", "Video playback speed:"
    )
    assert "FIELD_NAME_NOT_CANONICAL" in _codes(text)


def test_lowercase_concerns_fields_are_rejected() -> None:
    """The source-of-truth template capitalises both words."""
    text = GOOD.replace("Visual Concerns:", "Visual concerns:").replace(
        "Audio Concerns:", "Audio concerns:"
    )
    codes = _codes(text)
    assert "FIELD_NAME_NOT_CANONICAL" in codes


def test_shallow_scene_is_rejected() -> None:
    """The decisive Aug 2026 evaluator finding: scenes read as object lists."""
    text = GOOD.replace(SCENE, "A kitchen with a wooden table.")
    codes = _codes(text)
    assert "SCENE_TOO_SHALLOW" in codes
    assert "SCENE_NO_SPATIAL_RELATIONSHIPS" in codes


def test_scene_without_spatial_relationships_is_rejected() -> None:
    """Long enough, but still an inventory: no element is placed relative to
    any other."""
    inventory = (
        "The kitchen contains a table, several chairs, a refrigerator, open "
        "shelving, ceramic bowls, a kettle, a chopping board, two mugs, a "
        "window, a plastered wall, a ceiling fixture, a counter run, a sink, "
        "a tap, a bin, a rug, a radiator, a clock and a doorway nearby."
    )
    assert "SCENE_NO_SPATIAL_RELATIONSHIPS" in _codes(GOOD.replace(SCENE, inventory))


def test_shallow_style_is_rejected() -> None:
    text = GOOD.replace(STYLE, "Natural light, shallow depth of field.")
    codes = _codes(text)
    assert "STYLE_TOO_SHALLOW" in codes
    assert "STYLE_MISSING_LIGHTING_DETAIL" in codes


def test_style_missing_colour_temperature_is_rejected() -> None:
    partial = (
        "Key light from the window at screen-left with overhead fill; shadows "
        "are soft-edged and shallow across the room. Shallow depth of field, "
        "digital capture, no non-standard aspect ratio at all."
    )
    log = validate_caption(GOOD.replace(STYLE, partial))
    hits = [b for b in log.entries if b.code == "STYLE_MISSING_LIGHTING_DETAIL"]
    assert hits and "colour temperature" in hits[0].detail


def test_no_changes_from_overview_is_allowed_for_a_shot() -> None:
    """A shot whose space genuinely matches the Overview may say so."""
    assert "SCENE_TOO_SHALLOW" not in _codes(GOOD)


def test_shallow_per_shot_scene_is_rejected() -> None:
    text = GOOD.replace(
        "Scene: No changes from overview.", "Scene: A table and some shelves."
    )
    assert "SCENE_TOO_SHALLOW" in _codes(text)


def test_no_changes_bypass_is_flagged_for_confirmation() -> None:
    """It skips every depth check, so it cannot pass silently."""
    log = validate_caption(GOOD)
    assert any(b.code == "SCENE_UNCHANGED_UNCONFIRMED" for b in log.entries)


def test_all_shots_unchanged_is_blocking() -> None:
    """Separate shots differ by definition; if every one says 'no changes',
    the caption carries no shot description at all — the evaluator's decisive
    Scene failure, reachable straight through the bypass."""
    two_shots = GOOD + """
[Shot 2: 2.0s–4.0s]
Cut: Hard cut.
Camera: Close-up, eye-level, static.
Scene: No changes from overview.
Action & Audio:
(2.0s–3.0s) C1 turns toward screen-left.
Playback Speed: regular.
"""
    assert "ALL_SCENES_UNCHANGED" in _codes(two_shots)


# --------------------------------------------------------------------------
# speech delivery (source of truth §10 Rule 4)
# --------------------------------------------------------------------------
def _with_speech(line: str) -> str:
    return GOOD.replace("(1.0s–2.0s) C1 lowers the right hand.", line)


def test_speech_without_delivery_is_rejected() -> None:
    """Exactly the reported gap: `C1 says off-screen, "Hola."` returned zero
    findings while the standard requires a supported audible tone."""
    text = _with_speech('(1.0s–2.0s) C1 says off-screen, "Hola."')
    assert "SPEECH_NO_DELIVERY" in _codes(text)


def test_speech_with_canonical_tone_phrase_passes() -> None:
    text = _with_speech(
        '(1.0s–2.0s) C1 says off-screen in a questioning tone, "Hola."'
    )
    assert "SPEECH_NO_DELIVERY" not in _codes(text)


def test_speech_with_adverbial_delivery_passes() -> None:
    text = _with_speech('(1.0s–2.0s) C1 says off-screen quietly, "Hola."')
    assert "SPEECH_NO_DELIVERY" not in _codes(text)


def test_delivery_word_inside_the_quote_does_not_count() -> None:
    """The delivery describes the speech; it is not part of the words."""
    text = _with_speech('(1.0s–2.0s) C1 says off-screen, "watch your tone."')
    assert "SPEECH_NO_DELIVERY" in _codes(text)


def test_non_speech_line_needs_no_delivery() -> None:
    text = _with_speech("(1.0s–2.0s) A door slams somewhere off-screen.")
    assert "SPEECH_NO_DELIVERY" not in _codes(text)


def test_all_speech_verbs_require_delivery() -> None:
    """'speaks' and 'talks' were missing from the verb list, so those lines
    skipped the delivery requirement entirely."""
    for verb in ("speaks", "talks", "adds", "responds", "murmurs", "exclaims"):
        text = _with_speech(f'(1.0s–2.0s) C1 {verb} off-screen, "Hola."')
        assert "SPEECH_NO_DELIVERY" in _codes(text), f"{verb!r} bypassed the check"


def test_negated_delivery_does_not_satisfy_the_requirement() -> None:
    """A bare occurrence of the word 'tone' used to pass — including a phrase
    that explicitly denies having one."""
    for line in (
        '(1.0s–2.0s) C1 says off-screen without a supported tone, "Hola."',
        '(1.0s–2.0s) C1 says off-screen with no clear tone, "Hola."',
        '(1.0s–2.0s) C1 says off-screen, tone unclear, "Hola."',
        '(1.0s–2.0s) C1 says off-screen in a not supported tone, "Hola."',
        '(1.0s–2.0s) C1 says off-screen in an undetermined tone, "Hola."',
        '(1.0s–2.0s) C1 says off-screen in an unspecified voice, "Hola."',
    ):
        assert "SPEECH_NO_DELIVERY" in _codes(_with_speech(line)), line


# --------------------------------------------------------------------------
# language declaration
# --------------------------------------------------------------------------
FOREIGN_PREFIX = "The spoken language is a foreign language."
TAGALOG_PREFIX = "The spoken language is Tagalog."


def _audio(sentence: str) -> str:
    return GOOD.replace("Audio: Music plays throughout.", f"Audio: {sentence}")


def _declared(prefix: str, rest: str = "Music plays throughout.") -> str:
    return _audio(f"{prefix} {rest}")


def _language_codes(text: str, **kw: object) -> set[str]:
    log = validate_caption(text, **kw)  # type: ignore[arg-type]
    return {b.code for b in log.blocking if b.code.startswith("LANGUAGE_")}


def test_no_invented_overview_field() -> None:
    """§5/§26 list exactly seven Overview fields. A dedicated 'Spoken Language:'
    field was an eighth the live tool does not have."""
    from autoscribe.validate import REQUIRED_OVERVIEW_FIELDS

    assert "Spoken Language:" not in REQUIRED_OVERVIEW_FIELDS
    assert set(REQUIRED_OVERVIEW_FIELDS) == {
        "Cast:", "Scene:", "Style:", "Audio:", "Visual Concerns:", "Audio Concerns:",
    }


def test_declaration_renders_as_an_audio_prefix_not_a_field() -> None:
    from autoscribe import render
    from autoscribe.structured import Annotation, Globals

    ann = Annotation(
        video_name="c.mp4", duration=1.0,
        globals=Globals(audio="Music throughout.", spoken_language="Tagalog"),
        shots=[],
    )
    out = render.render(ann)
    assert "Spoken Language:" not in out
    assert f"Audio: {TAGALOG_PREFIX} Music throughout." in out


def test_canonical_value_is_built_from_measured_evidence() -> None:
    from autoscribe.validate import canonical_language_value as value

    assert value("tagalog", speech_present=True, language_confident=False) == (
        "a foreign language"
    )
    assert value("tagalog", speech_present=True, language_confident=True) == "Tagalog"
    # English needs no declaration; §17 is about foreign speech.
    assert value("english", speech_present=True, language_confident=True) == ""
    assert value("", speech_present=False, language_confident=True) == ""


def test_non_english_speech_must_declare_the_language() -> None:
    """Missed case from the evaluator audit: Tagalog lines shipped with no
    language named anywhere."""
    assert "LANGUAGE_NOT_DECLARED" in _language_codes(
        GOOD, detected_language="tagalog"
    )


def test_correct_prefix_satisfies_the_check() -> None:
    assert not _language_codes(
        _declared(TAGALOG_PREFIX), detected_language="tagalog"
    )


def test_correct_fallback_prefix_satisfies_the_check() -> None:
    assert not _language_codes(
        _declared(FOREIGN_PREFIX), speech_present=True, language_confident=False
    )


def test_english_needs_no_declaration() -> None:
    assert not _language_codes(GOOD, detected_language="english")


def test_nothing_is_demanded_without_speech() -> None:
    assert not _language_codes(GOOD, detected_language="")


def test_uncertain_language_still_requires_the_fallback() -> None:
    assert "LANGUAGE_NOT_DECLARED" in _language_codes(
        GOOD, speech_present=True, language_confident=False
    )


def test_declaration_without_evidence_is_rejected() -> None:
    """No expected value used to RETURN EARLY, so a caption could declare a
    language the evidence never supported and raise nothing at all."""
    spanish = "The spoken language is Spanish."
    assert "LANGUAGE_DECLARED_WITHOUT_EVIDENCE" in _language_codes(
        _declared(spanish), detected_language="english"
    )
    assert "LANGUAGE_DECLARED_WITHOUT_EVIDENCE" in _language_codes(_declared(spanish))


def test_wrong_value_is_rejected() -> None:
    assert "LANGUAGE_NOT_DECLARED" in _language_codes(
        _declared("The spoken language is Spanish."), detected_language="tagalog"
    )


def test_case_differences_are_rejected() -> None:
    """The value is generated, so any difference is an edit. Comparison used to
    ignore case, and 'tAgAlOg' passed."""
    assert "LANGUAGE_NOT_DECLARED" in _language_codes(
        _declared("The spoken language is tAgAlOg."), detected_language="tagalog"
    )


def _with_shot_declaration(base: str) -> str:
    return base.replace(
        "Cut: Opening shot.", "Cut: Opening shot.\n" + TAGALOG_PREFIX
    )


def test_declaration_inside_a_shot_does_not_satisfy_the_overview() -> None:
    """It was accepted anywhere, including inside a shot."""
    assert "LANGUAGE_NOT_DECLARED" in _language_codes(
        _with_shot_declaration(GOOD), detected_language="tagalog"
    )


def test_duplicate_declarations_are_rejected() -> None:
    """Only the FIRST match was read, so a second contradicting claim elsewhere
    was invisible."""
    assert "LANGUAGE_DECLARATION_COUNT" in _language_codes(
        _with_shot_declaration(_declared(TAGALOG_PREFIX)),
        detected_language="tagalog",
    )


def test_prefix_must_open_the_audio_field() -> None:
    """Mid-field it is prose again, and prose can be negated by what precedes it."""
    assert "LANGUAGE_NOT_DECLARED" in _language_codes(
        _audio(f"Music plays throughout. {TAGALOG_PREFIX}"),
        detected_language="tagalog",
    )


@pytest.mark.parametrize(
    "audio",
    [
        "It is false that The spoken language is Tagalog.",
        "Do not claim that The spoken language is a foreign language.",
        "A voice speaks in a foreign language.",
        "The speech is in Tagalog.",
    ],
)
def test_prose_can_never_satisfy_the_declaration(audio: str) -> None:
    """No sentence in the model's Audio prose — however phrased, negated or
    quoted — can stand in for the generated declaration."""
    assert _language_codes(
        _audio(audio), speech_present=True, language_confident=False
    ), audio


def test_speculative_audio_prose_is_advisory_not_blocking() -> None:
    """An unhedged guess in free prose ("The voice speaks Pashto.") cannot be
    detected without a language list, so this check must not pretend to be
    authoritative — it warns, and never decides the declaration."""
    log = validate_caption(
        _declared(FOREIGN_PREFIX, "The voice may be speaking Pashto."),
        speech_present=True, language_confident=False,
    )
    assert any(b.code == "AUDIO_PROSE_SPECULATIVE" for b in log.entries)
    assert all(b.code != "AUDIO_PROSE_SPECULATIVE" for b in log.blocking)


def test_ordinary_prose_words_do_not_trigger_the_advisory() -> None:
    """"Specifically, a door slams" is not a language guess."""
    log = validate_caption(
        _declared(FOREIGN_PREFIX, "Specifically, a door slams before the voice."),
        speech_present=True, language_confident=False,
    )
    assert not any(b.code == "AUDIO_PROSE_SPECULATIVE" for b in log.entries)


def test_absent_language_counts_as_unestablished() -> None:
    """A transcript with real speech but no language value reported
    language_confident=True, which skipped the fallback entirely."""
    from autoscribe.transcribe import Segment, Transcript

    good = Segment(0.0, 1.0, "Hola", avg_logprob=-0.2, no_speech_prob=0.01)
    t = Transcript(language="", text="Hola", segments=[good])
    assert t.has_speech is True
    assert t.language_confident is False


# --------------------------------------------------------------------------
# canonical shot fields (SOT §27)
# --------------------------------------------------------------------------
def test_missing_shot_fields_are_caught() -> None:
    """CANONICAL_SHOT_FIELDS was declared but never checked, so a shot with no
    Camera, Scene or Playback Speed produced no finding at all."""
    text = GOOD
    for field in (
        "Camera: Medium-wide, eye-level, handheld.",
        "Scene: No changes from overview.",
        "Playback Speed: regular.",
    ):
        text = text.replace(field, "")
    findings = [b for b in validate_caption(text).blocking
                if b.code == "SHOT_FIELD_MISSING"]
    assert len(findings) == 3, [b.detail for b in findings]


def test_complete_shot_reports_no_missing_fields() -> None:
    assert "SHOT_FIELD_MISSING" not in _codes(GOOD)


def test_missing_cut_field_is_caught() -> None:
    text = GOOD.replace("Cut: Opening shot.", "")
    assert "SHOT_FIELD_MISSING" in _codes(text)


def test_present_but_empty_shot_fields_are_caught() -> None:
    """A bare label carries no information; it used to satisfy the check."""
    text = GOOD.replace(
        "Camera: Medium-wide, eye-level, handheld.", "Camera:"
    ).replace("Playback Speed: regular.", "Playback Speed:")
    findings = [b for b in validate_caption(text).blocking
                if b.code == "SHOT_FIELD_EMPTY"]
    assert len(findings) == 2, [b.detail for b in findings]


def test_field_holding_only_a_full_stop_is_empty() -> None:
    text = GOOD.replace("Camera: Medium-wide, eye-level, handheld.", "Camera: .")
    assert "SHOT_FIELD_EMPTY" in _codes(text)


def test_ghost_character_id_is_caught() -> None:
    text = GOOD.replace("(1.0s–2.0s) C1 lowers", "(1.0s–2.0s) C4 lowers")
    assert "GHOST_ID" in _codes(text)


def test_pronoun_outside_quotes_is_caught() -> None:
    text = GOOD.replace("C1 raises the right hand.", "C1 raises his right hand.")
    assert "PRONOUN_OUTSIDE_QUOTES" in _codes(text)


def test_pronoun_inside_quoted_speech_is_allowed() -> None:
    text = GOOD.replace(
        "(1.0s–2.0s) C1 lowers the right hand.",
        '(1.0s–2.0s) C1 says: "he told me to wait."',
    )
    assert "PRONOUN_OUTSIDE_QUOTES" not in _codes(text)


def test_timestamp_outside_shot_is_caught() -> None:
    text = GOOD.replace("(1.0s–2.0s) C1 lowers", "(1.0s–9.0s) C1 lowers")
    assert "TIMESTAMP_OUTSIDE_SHOT" in _codes(text)


def test_reversed_timestamp_is_caught() -> None:
    text = GOOD.replace("(1.0s–2.0s) C1 lowers", "(2.0s–1.0s) C1 lowers")
    assert "TIMESTAMP_REVERSED" in _codes(text)


def test_unbalanced_quote_is_caught() -> None:
    text = GOOD.replace(
        "(1.0s–2.0s) C1 lowers the right hand.",
        '(1.0s–2.0s) C1 says: "hello.',
    )
    assert "QUOTES_UNBALANCED" in _codes(text)


def test_missing_terminal_punctuation_is_caught() -> None:
    text = GOOD.replace("C1 lowers the right hand.", "C1 lowers the right hand")
    assert "PUNCTUATION_MISSING" in _codes(text)


def test_filler_absence_line_is_caught() -> None:
    text = GOOD.replace(
        "(1.0s–2.0s) C1 lowers the right hand.", "(1.0s–2.0s) No speech."
    )
    assert "FILLER_LINE" in _codes(text)


def test_timing_filler_phrase_is_caught() -> None:
    text = GOOD.replace(
        "C1 lowers the right hand.", "C1 lowers the right hand near the end."
    )
    assert "TIMING_FILLER" in _codes(text)


def test_hedge_is_caught() -> None:
    text = GOOD.replace("A person in a red jacket.", "A person whose age cannot be determined.")
    assert "HEDGE_TEXT" in _codes(text)


def test_protected_trait_is_caught() -> None:
    text = GOOD.replace(
        "A person in a red jacket.", "A person of unspecified ethnicity in a red jacket."
    )
    assert "PROTECTED_TRAIT" in _codes(text)


def test_unknown_cut_type_is_caught() -> None:
    text = GOOD.replace("Cut: Opening shot.", "Cut: Sparkle transition.")
    assert "CUT_TYPE_UNKNOWN" in _codes(text)


def test_shot_numbering_gap_is_caught() -> None:
    text = GOOD.replace("[Shot 1: 0.0s–2.0s]", "[Shot 2: 0.0s–2.0s]")
    assert "SHOT_NUMBERING" in _codes(text)


def test_simultaneous_events_are_flagged_but_not_blocking() -> None:
    """Two events sharing a range is legitimate — it must be surfaced for
    confirmation, never merged and never treated as an error."""
    text = GOOD.replace(
        "(1.0s–2.0s) C1 lowers the right hand.",
        "(1.0s–2.0s) C1 lowers the right hand.\n(1.0s–2.0s) Music swells.",
    )
    log = validate_caption(text)
    codes = {b.code for b in log.entries}
    assert "SIMULTANEOUS_EVENTS_UNCONFIRMED" in codes
    assert all(b.code != "SIMULTANEOUS_EVENTS_UNCONFIRMED" for b in log.blocking)


def test_timeline_gap_between_shots_is_caught() -> None:
    text = GOOD + """
[Shot 2: 5.0s–7.0s]
Cut: Hard cut.
Camera: Close-up.
Scene: No changes from overview.
Action & Audio:
(5.0s–6.0s) C1 turns toward screen-left.
Playback Speed: regular.
"""
    assert "SHOT_TIMELINE_GAP" in _codes(text)
