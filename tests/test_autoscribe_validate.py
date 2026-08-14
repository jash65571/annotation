"""The deterministic caption gate.

Every check here corresponds to a defect that previously reached a "final"
file unchallenged.
"""

from __future__ import annotations

from autoscribe.validate import validate_caption

GOOD = """[Overview]
Cast:
C1: A person in a red jacket. Lower body and shoes are not visible.

Scene: A kitchen with a wooden table.

Style: Natural light, shallow depth of field.

Audio: Music plays throughout.

Visual concerns: None.
Audio concerns: None.

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
