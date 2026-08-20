"""Pasted-back QA for Manuscript II audio (spec 36, 37, 39, 40).

This is a SEPARATE tool surface from the audio analysis pipeline. The analysis
pass never sees the filled UI fields or the generated caption, so it cannot
check them. Here the reviewer pastes back the edited state and the generated
text, and this tool predicts the blockers before anything enters Handshake.

Input JSON (all fields optional except events):

    {
      "events": [
        {
          "id": "e1",
          "type": "Speech",                 # or "Sound"
          "source": "C2",                   # C#, O#, Music, ambience, ...
          "transcript": "Good job, man!",   # Speech only
          "voice_status": "Observed",       # optional
          "recorded_level": "Moderate",     # or "" / null when blank
          "mix_role": "Foreground",
          "speaking_level": "Raised",
          "clarity": "Clear",
          "tone": "excited",
          "caption_sentence": "C2 says ... at a moderate recorded level ..."
        }
      ],
      "final_audio_text": "....",
      "overview_audio": "....",
      "cast_current": ["C1", "C2"],
      "objects_current": ["O1"],
      "silent_objects": ["O1"]
    }

Baseline (original cast/objects) is read from task_context.json seed_meta so
added-vs-original and 'unused added source' can be checked (spec 34, 35).

Usage:
    python manuscript_audio_qa.py path/to/state.json
"""

from pathlib import Path
import json
import re
import sys


ROOT = Path(__file__).resolve().parent
CONTEXT = ROOT / "task_context.json"
OUTPUT = ROOT / "analysis" / "manuscript_audio_qa.json"


# Controlled vocabularies (spec 11, 13, 31).
RECORDED_LEVELS = ["faint", "quiet", "moderate", "loud"]
MIX_ROLES = ["foreground", "supporting", "background"]
SPEAKING_LEVELS = ["whispered", "soft", "normal", "raised", "shouted"]
CLARITIES = ["clear", "partly unclear", "largely unclear"]
TONES = [
    "casual", "calm", "serious", "amused",
    "tense", "irritated", "excited", "playful",
]
NON_ENTITY_SOURCES = {
    "music",
    "room ambience",
    "traffic ambience",
    "outdoor ambience",
    "wind ambience",
    "ocean ambience",
    "water ambience",
    "crowd ambience",
    "machinery ambience",
    "unidentified sound",
}

# Past-tense reporting verbs that should not appear in present-tense final prose.
PAST_TENSE = re.compile(
    r"\b(said|asked|clapped|cheered|laughed|walked|shouted|whispered|"
    r"began|started|continued|was|were|had)\b",
    re.IGNORECASE,
)
# Numeric timestamps must never appear in final text.
TIMESTAMP = re.compile(r"\b\d+(?:\.\d+)?\s*s\b|\b\d+:\d{2}\b")


def issue(level, code, message, event=None):
    record = {"level": level, "code": code, "message": message}
    if event is not None:
        record["event"] = event
    return record


def _blank(value):
    return value is None or (isinstance(value, str) and not value.strip())


def _prose_value(text, vocabulary):
    """Return the vocabulary term explicitly mentioned in prose, or None."""
    if not text:
        return None

    lowered = text.lower()

    # Longer phrases first so "partly unclear" wins over "clear".
    for term in sorted(vocabulary, key=len, reverse=True):
        if re.search(r"\b" + re.escape(term) + r"\b", lowered):
            return term

    return None


def _check_field_vs_prose(event, field, vocabulary, label, problems):
    """Spec 37: prose claims a value while the structured field is blank/wrong."""
    prose = _prose_value(event.get("caption_sentence", ""), vocabulary)

    if prose is None:
        return

    structured = event.get(field)
    event_id = event.get("id", event.get("source", "?"))

    if _blank(structured):
        problems.append(
            issue(
                "blocker",
                "STRUCTURED_FIELD_MISSING",
                f"Caption for {event_id} states {label} "
                f"'{prose}', but the {label} field is blank.",
                event_id,
            )
        )
        return

    if structured.strip().lower() != prose:
        problems.append(
            issue(
                "blocker",
                "STRUCTURED_FIELD_MISMATCH",
                f"Caption for {event_id} states {label} '{prose}', "
                f"but the field is set to '{structured}'.",
                event_id,
            )
        )


def load_baseline(context_path=CONTEXT):
    context_path = Path(context_path)

    if not context_path.exists():
        return {"characters": [], "objects": []}

    with context_path.open("r", encoding="utf-8-sig") as f:
        context = json.load(f)

    meta = context.get("seed_meta", {})

    return {
        "characters": meta.get(
            "original_character_ids", context.get("characters", [])
        ),
        "objects": meta.get(
            "original_object_ids", context.get("objects", [])
        ),
    }


def run_qa(state, baseline=None):
    baseline = baseline or {"characters": [], "objects": []}

    events = state.get("events", [])
    final_text = state.get("final_audio_text", "") or ""
    overview = state.get("overview_audio", "") or ""

    cast_current = state.get("cast_current", [])
    objects_current = state.get("objects_current", [])
    silent_objects = {o.upper() for o in state.get("silent_objects", [])}

    problems = []

    used_sources = set()

    for event in events:
        event_id = event.get("id", event.get("source", "?"))
        etype = (event.get("type") or "").strip().lower()
        source = (event.get("source") or "").strip()

        if source:
            used_sources.add(source.upper())

        # Spec 37: structured field vs prose, per controlled field.
        _check_field_vs_prose(
            event, "recorded_level", RECORDED_LEVELS, "recorded level", problems
        )
        _check_field_vs_prose(
            event, "mix_role", MIX_ROLES, "mix role", problems
        )
        _check_field_vs_prose(
            event, "speaking_level", SPEAKING_LEVELS, "speaking level", problems
        )
        _check_field_vs_prose(
            event, "clarity", CLARITIES, "clarity", problems
        )

        # Spec 36: required structured fields for a Speech event.
        if etype == "speech":
            if _blank(event.get("transcript")):
                problems.append(
                    issue(
                        "blocker",
                        "missing_transcript",
                        f"Speech event {event_id} has no transcript.",
                        event_id,
                    )
                )
            for field, label in (
                ("recorded_level", "recorded level"),
                ("mix_role", "mix role"),
                ("clarity", "clarity"),
            ):
                if _blank(event.get(field)):
                    problems.append(
                        issue(
                            "review",
                            "missing_required_field",
                            f"Speech event {event_id} is missing {label}.",
                            event_id,
                        )
                    )

            # Spec 40: a 'Not observed' character must not be a Speech source.
            if (event.get("voice_status") or "").strip().lower() == "not observed":
                problems.append(
                    issue(
                        "blocker",
                        "not_observed_speaks",
                        f"{source} is marked 'Not observed' but is a Speech "
                        "source.",
                        event_id,
                    )
                )

            # Spec 37/39: Clear speech should carry a verbatim transcript.
            if (event.get("clarity") or "").strip().lower() == "clear" and _blank(
                event.get("transcript")
            ):
                problems.append(
                    issue(
                        "blocker",
                        "clear_without_transcript",
                        f"{event_id} is marked Clear but has no verbatim "
                        "transcript.",
                        event_id,
                    )
                )

        # Spec 16/40: a silent object must not be a sounding source.
        if source.upper() in silent_objects:
            problems.append(
                issue(
                    "blocker",
                    "silent_object_sounds",
                    f"{source} is listed silent but is used as a sound source.",
                    event_id,
                )
            )

        # Spec 31/39: source must be a known entity or supported category.
        if source:
            known = (
                source.upper() in {c.upper() for c in cast_current}
                or source.upper() in {o.upper() for o in objects_current}
                or source.lower() in NON_ENTITY_SOURCES
            )
            if not known:
                problems.append(
                    issue(
                        "review",
                        "unsupported_source",
                        f"Event {event_id} source '{source}' is not in the "
                        "current cast/objects or a supported sound category.",
                        event_id,
                    )
                )

        # Spec 40: past tense / timestamps in the caption sentence.
        sentence = event.get("caption_sentence", "") or ""
        if TIMESTAMP.search(sentence):
            problems.append(
                issue(
                    "blocker",
                    "timestamp_in_caption",
                    f"Caption for {event_id} contains a numeric timestamp.",
                    event_id,
                )
            )

    # --- Cast integrity (spec 34, 35) -----------------------------------
    original_cast = {c.upper() for c in baseline.get("characters", [])}
    current_cast = {c.upper() for c in cast_current}

    for removed in sorted(original_cast - current_cast):
        problems.append(
            issue(
                "blocker",
                "original_cast_changed",
                f"Original cast member {removed} is missing from the current "
                "task. Carried-over cast must not be deleted.",
            )
        )

    added_cast = sorted(current_cast - original_cast)
    added_objects = sorted(
        {o.upper() for o in objects_current}
        - {o.upper() for o in baseline.get("objects", [])}
    )

    for added in added_cast:
        if added not in used_sources:
            problems.append(
                issue(
                    "review",
                    "unused_added_speaker",
                    f"{added} was added but is never the source of an event.",
                )
            )

    for added in added_objects:
        if added not in used_sources:
            problems.append(
                issue(
                    "review",
                    "unused_added_object",
                    f"{added} was added but is never the source of an event.",
                )
            )

    # --- Final Audio Text semantic QA (spec 39, 40) ---------------------
    if final_text:
        if TIMESTAMP.search(final_text):
            problems.append(
                issue(
                    "blocker",
                    "timestamp_in_final",
                    "Final Audio Text contains a numeric timestamp.",
                )
            )

        if PAST_TENSE.search(final_text):
            problems.append(
                issue(
                    "review",
                    "possible_past_tense",
                    "Final Audio Text may use past tense; final prose should be "
                    "present tense.",
                )
            )

        # Every Speech transcript should be preserved verbatim in the final.
        for event in events:
            if (event.get("type") or "").lower() != "speech":
                continue
            transcript = (event.get("transcript") or "").strip()
            if transcript and transcript not in final_text:
                problems.append(
                    issue(
                        "review",
                        "event_missing_from_final",
                        f"Transcript of {event.get('id', event.get('source'))} "
                        f"(\"{transcript}\") does not appear verbatim in Final "
                        "Audio Text.",
                        event.get("id"),
                    )
                )

    # --- Overview integrity (spec 32) -----------------------------------
    if overview:
        for event in events:
            source = (event.get("source") or "").strip().lower()
            if source == "unidentified sound" and "unidentified" not in overview.lower():
                problems.append(
                    issue(
                        "review",
                        "unidentified_missing_from_overview",
                        "An unidentified sound event is not reflected in the "
                        "Overview Audio.",
                        event.get("id"),
                    )
                )
                break

    blockers = sum(1 for p in problems if p["level"] == "blocker")
    reviews = sum(1 for p in problems if p["level"] == "review")

    if blockers:
        status = "BLOCKED"
    elif reviews:
        status = "HUMAN_REVIEW_REQUIRED"
    else:
        status = "QA_CLEAR"

    return {
        "status": status,
        "blocker_count": blockers,
        "review_item_count": reviews,
        "issues": problems,
        "policy": {
            "qa_clear_is_not_final_approval": True,
            "actual_media_remains_ground_truth": True,
        },
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv:
        raise SystemExit(
            "Usage: python manuscript_audio_qa.py path/to/state.json"
        )

    with Path(argv[0]).open("r", encoding="utf-8-sig") as f:
        state = json.load(f)

    result = run_qa(state, load_baseline())

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("=== PASTED-BACK AUDIO QA ===")
    print("Status:", result["status"])
    print(
        "Blockers:", result["blocker_count"],
        "| Review items:", result["review_item_count"],
    )
    print()
    for p in result["issues"]:
        print(f"[{p['level'].upper()}] {p['code']}: {p['message']}")
    print()
    print("QA clearance never replaces the final human listen or export check.")

    return result


if __name__ == "__main__":
    main()
