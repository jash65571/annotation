"""The deterministic caption gate.

Every check here corresponds to a defect that previously reached a "final"
file unchallenged.
"""

from __future__ import annotations

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
    ):
        assert "SPEECH_NO_DELIVERY" in _codes(_with_speech(line)), line


# --------------------------------------------------------------------------
# language declaration
# --------------------------------------------------------------------------
def test_non_english_speech_must_declare_the_language() -> None:
    """Missed case from the evaluator audit: Tagalog lines shipped with no
    language named anywhere."""
    log = validate_caption(GOOD, detected_language="tagalog")
    assert any(b.code == "LANGUAGE_NOT_DECLARED" for b in log.blocking)


def test_declared_language_satisfies_the_check() -> None:
    text = GOOD.replace(
        "Audio: Music plays throughout.",
        "Audio: Music plays throughout. The speech is in Tagalog.",
    )
    log = validate_caption(text, detected_language="tagalog")
    assert not any(b.code == "LANGUAGE_NOT_DECLARED" for b in log.entries)


def test_english_needs_no_declaration() -> None:
    log = validate_caption(GOOD, detected_language="english")
    assert not any(b.code == "LANGUAGE_NOT_DECLARED" for b in log.entries)


def test_unknown_language_is_not_demanded_without_speech() -> None:
    """No speech at all means nothing to declare."""
    log = validate_caption(GOOD, detected_language="")
    assert not any(b.code == "LANGUAGE_NOT_DECLARED" for b in log.entries)


def test_uncertain_language_over_real_speech_still_requires_a_declaration() -> None:
    """SOT §17 gives a fallback ('a foreign language'), so an unconfident
    detection is not an exemption — it just must not name a language."""
    log = validate_caption(GOOD, speech_present=True, language_confident=False)
    assert any(b.code == "LANGUAGE_NOT_DECLARED" for b in log.blocking)


def test_foreign_language_fallback_phrase_satisfies_the_check() -> None:
    text = GOOD.replace(
        "Audio: Music plays throughout.",
        "Audio: Music plays throughout. A voice speaks in a foreign language.",
    )
    log = validate_caption(text, speech_present=True, language_confident=False)
    assert not any(b.code == "LANGUAGE_NOT_DECLARED" for b in log.entries)


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
