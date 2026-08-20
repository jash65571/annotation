from pathlib import Path
import json
import statistics
from collections import Counter


def load_speaker_map(
    path,
    current_video_sha256=None,
):
    path = Path(path)

    if not path.exists():
        return {}

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        data = json.load(f)

    # New safe format:
    #
    # {
    #   "video_sha256": "...",
    #   "segments": {
    #       "0": "C1"
    #   }
    # }
    if "segments" in data:
        if current_video_sha256 is not None:
            if (
                data.get("video_sha256")
                != current_video_sha256
            ):
                print(
                    "SPEAKER MAP: IGNORED | "
                    "belongs to a different video"
                )
                return {}

        data = data.get(
            "segments",
            {},
        )

    # Old flat files are no longer trusted when
    # a current-video fingerprint is available.
    elif current_video_sha256 is not None:
        print(
            "SPEAKER MAP: IGNORED | "
            "file is not bound to the current video"
        )
        return {}

    return {
        int(segment): character
        for segment, character in data.items()
    }


def predominant_candidate(
    values,
    reliable_duration,
    total_duration,
):
    if not values or total_duration <= 0:
        return None

    coverage = reliable_duration / total_duration

    counts = Counter(values)
    candidate, count = counts.most_common(1)[0]

    agreement = count / len(values)

    # Conservative gate:
    # enough usable speech must support the property,
    # and reliable segments must agree.
    if coverage < 0.60:
        return None

    if agreement < 0.75:
        return None

    return candidate


def build_character_voice_profiles(
    evidence,
    speaker_map,
):
    speech = evidence.get(
        "whisperx_segments",
        [],
    )

    acoustic = evidence.get(
        "acoustic_segments",
        [],
    )

    ui = evidence.get(
        "manuscript_ui_candidates",
        [],
    )

    grouped = {}

    for segment_index, character in speaker_map.items():
        if segment_index >= len(speech):
            continue

        grouped.setdefault(
            character,
            [],
        ).append(segment_index)

    profiles = {}

    for character, segment_ids in grouped.items():
        segment_ids = sorted(segment_ids)

        durations = []
        pitches = []
        levels = []

        speed_values = []
        delivery_values = []
        level_values = []

        speed_duration = 0.0
        delivery_duration = 0.0
        level_duration = 0.0

        low_confidence_words = []
        total_words = 0

        for segment_id in segment_ids:
            speech_item = speech[segment_id]
            acoustic_item = acoustic[segment_id]

            duration = float(
                speech_item["duration_sec"]
            )

            durations.append(duration)

            total_words += int(
                speech_item["word_count"]
            )

            low_confidence_words.extend(
                speech_item.get(
                    "low_confidence_words",
                    [],
                )
            )

            if acoustic_item.get("mean_pitch_hz"):
                pitches.append(
                    float(
                        acoustic_item[
                            "mean_pitch_hz"
                        ]
                    )
                )

            if acoustic_item.get(
                "equivalent_sound_level_db"
            ) is not None:
                levels.append(
                    float(
                        acoustic_item[
                            "equivalent_sound_level_db"
                        ]
                    )
                )

            if segment_id >= len(ui):
                continue

            candidate = ui[segment_id]

            speed = candidate.get(
                "speed_candidate"
            )

            if speed:
                speed_values.append(speed)
                speed_duration += duration

            delivery = candidate.get(
                "delivery_candidate"
            )

            if delivery:
                delivery_values.append(delivery)
                delivery_duration += duration

            level = candidate.get(
                "recorded_level_candidate"
            )

            if level:
                level_values.append(level)
                level_duration += duration

        total_duration = sum(durations)

        speed_coverage = (
            speed_duration / total_duration
            if total_duration
            else 0.0
        )

        delivery_coverage = (
            delivery_duration / total_duration
            if total_duration
            else 0.0
        )

        level_coverage = (
            level_duration / total_duration
            if total_duration
            else 0.0
        )

        profiles[character] = {
            "character": character,
            "confirmed_segment_ids": segment_ids,

            "usable_speech_duration_sec": round(
                total_duration,
                3,
            ),

            "total_word_count": total_words,

            "mean_pitch_hz_evidence": (
                round(
                    statistics.mean(pitches),
                    1,
                )
                if pitches
                else None
            ),

            "median_recorded_db_evidence": (
                round(
                    statistics.median(levels),
                    2,
                )
                if levels
                else None
            ),

            "speed_evidence_values": sorted(
                set(speed_values)
            ),

            "speed_evidence_coverage": round(
                speed_coverage,
                3,
            ),

            "speed_candidate": predominant_candidate(
                speed_values,
                speed_duration,
                total_duration,
            ),

            "delivery_evidence_values": sorted(
                set(delivery_values)
            ),

            "delivery_evidence_coverage": round(
                delivery_coverage,
                3,
            ),

            "delivery_candidate":
                predominant_candidate(
                    delivery_values,
                    delivery_duration,
                    total_duration,
                ),

            "recorded_level_evidence_values": sorted(
                set(level_values)
            ),

            "recorded_level_evidence_coverage": round(
                level_coverage,
                3,
            ),

            "recorded_level_candidate":
                predominant_candidate(
                    level_values,
                    level_duration,
                    total_duration,
                ),

            "low_confidence_word_count":
                len(low_confidence_words),

            # These remain human decisions.
            "voice_status_candidate": None,
            "pitch_candidate": None,
            "speaking_level_candidate": None,
            "clarity_candidate": None,
            "confidence_candidate": None,
            "texture_candidate": None,
            "mix_role_candidate": None,

            "manual_confirmation_required": True,

            "warnings": [
                "Speaker mapping must be human-confirmed.",
                "Candidates describe predominant evidence only.",
                "Pitch requires speaker-aware human listening.",
                "ASR confidence is not Manuscript Confidence.",
                "Clarity requires human listening.",
                "Speaking level is separate from recorded level.",
            ],
        }

    return profiles


def enrich_evidence_with_voice_profiles(
    evidence_path,
    speaker_map_path,
):
    evidence_path = Path(evidence_path)

    with evidence_path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        evidence = json.load(f)

    identity_path = (
        evidence_path.parent
        / "video_identity.json"
    )

    current_video_sha256 = None

    if identity_path.exists():
        with identity_path.open(
            "r",
            encoding="utf-8-sig",
        ) as f:
            identity = json.load(f)

        current_video_sha256 = identity.get(
            "video_sha256"
        )

    speaker_map = load_speaker_map(
        speaker_map_path,
        current_video_sha256,
    )

    profiles = build_character_voice_profiles(
        evidence,
        speaker_map,
    )

    evidence[
        "character_voice_profiles"
    ] = profiles

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
        "CHARACTER VOICE PROFILES: PASS |",
        len(profiles),
        "mapped characters",
    )

    return profiles



