from pathlib import Path
import json
import statistics


def _pitch_candidate(hz, median_hz):
    # Do not classify pitch against the whole clip.
    # Different speakers can have very different natural ranges.
    # Speaker grouping / human listening is required first.
    return None


def _speed_candidate(segment):
    if not segment.get("speed_measurement_reliable"):
        return None

    wpm = float(segment.get("raw_wpm", 0.0))

    if wpm < 110:
        return "Slow"

    if wpm > 190:
        return "Fast"

    return "Normal"


def _delivery_candidate(acoustic):
    if not acoustic.get(
        "voice_quality_measurement_reliable",
        False,
    ):
        return None

    pitch_range = float(
        acoustic.get("pitch_range_semitones", 0.0)
    )

    if pitch_range <= 3.0:
        return "Flat"

    if pitch_range >= 8.0:
        return "Dramatic"

    return "Ordinary"


def _recorded_level_candidate(db, median_db):
    if db is None or median_db is None:
        return None

    difference = float(db) - float(median_db)

    if difference >= 4.0:
        return "Loud"

    if difference >= -2.0:
        return "Moderate"

    if difference >= -7.0:
        return "Quiet"

    return "Faint"


def _clarity_candidate(segment):
    low_words = segment.get("low_confidence_words", [])
    word_count = int(segment.get("word_count", 0))
    mean_confidence = segment.get("mean_word_confidence")

    if word_count < 3 or mean_confidence is None:
        return None

    low_ratio = len(low_words) / word_count

    # ASR may support a strong "Clear" candidate when the
    # transcription evidence is exceptionally clean.
    #
    # It must NOT automatically produce Partly unclear or
    # Largely unclear. Those require actual human listening.
    if (
        float(mean_confidence) >= 0.85
        and low_ratio == 0
    ):
        return "Clear"

    return None


def build_ui_candidates(evidence):
    speech = evidence.get("whisperx_segments", [])
    acoustic = evidence.get("acoustic_segments", [])

    valid_pitch = [
        float(item["mean_pitch_hz"])
        for item in acoustic
        if item.get("mean_pitch_hz")
    ]

    valid_db = [
        float(item["equivalent_sound_level_db"])
        for item in acoustic
        if item.get("equivalent_sound_level_db") is not None
    ]

    median_pitch = (
        statistics.median(valid_pitch)
        if valid_pitch
        else None
    )

    median_db = (
        statistics.median(valid_db)
        if valid_db
        else None
    )

    results = []

    for index, speech_segment in enumerate(speech):
        acoustic_segment = acoustic[index]

        results.append({
            "segment": index,

            "pitch_candidate": _pitch_candidate(
                acoustic_segment.get("mean_pitch_hz"),
                median_pitch,
            ),

            "speed_candidate": _speed_candidate(
                speech_segment
            ),

            "delivery_candidate": _delivery_candidate(
                acoustic_segment
            ),

            "recorded_level_candidate":
                _recorded_level_candidate(
                    acoustic_segment.get(
                        "equivalent_sound_level_db"
                    ),
                    median_db,
                ),

            "clarity_candidate": _clarity_candidate(
                speech_segment
            ),

            # These must not be inferred from the
            # currently available evidence alone.
            "speaking_level_candidate": None,
            "mix_role_candidate": None,
            "confidence_candidate": None,
            "tone_candidate": None,
            "texture_candidate": None,
            "voice_status_candidate": None,

            "manual_confirmation_required": True,

            "warnings": [
                "Candidates are reviewer aids only.",
                "Pitch is relative to the clip, not speaker identity.",
                "Clarity uses ASR only as supporting evidence.",
                "ASR confidence is not Manuscript Confidence.",
                "Recorded level is not speaking level.",
                "Emotion labels are not mapped directly to Tone.",
            ],
        })

    return results


def enrich_evidence_with_ui_candidates(evidence_path):
    evidence_path = Path(evidence_path)

    with evidence_path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        evidence = json.load(f)

    evidence["manuscript_ui_candidates"] = (
        build_ui_candidates(evidence)
    )

    with evidence_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            evidence,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "MANUSCRIPT UI CANDIDATES: PASS |",
        len(evidence["manuscript_ui_candidates"]),
        "speech segments",
    )

    return evidence["manuscript_ui_candidates"]

