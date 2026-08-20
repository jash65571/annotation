from pathlib import Path
import json


ROOT = Path(__file__).resolve().parent

EVIDENCE = (
    ROOT
    / "analysis"
    / "manuscript_audio_evidence.json"
)

DIARIZATION = (
    ROOT
    / "analysis"
    / "diarization_evidence.json"
)

SOUND_EVIDENCE = (
    ROOT
    / "analysis"
    / "sound_event_evidence.json"
)

OUTPUT = (
    ROOT
    / "analysis"
    / "masking_overlap_evidence.json"
)


def load_json(path):
    path = Path(path)

    if not path.exists():
        return None

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        return json.load(f)


def overlaps(a_start, a_end, b_start, b_end):
    return (
        float(a_start) < float(b_end)
        and float(a_end) > float(b_start)
    )


def build_speaker_overlap_regions(diarization):
    if not diarization:
        return []

    if diarization.get("status") != "complete":
        return []

    turns = diarization.get("turns", [])
    regions = []

    for i, left in enumerate(turns):
        for right in turns[i + 1:]:
            if (
                left.get("speaker")
                == right.get("speaker")
            ):
                continue

            start = max(
                float(left["start"]),
                float(right["start"]),
            )

            end = min(
                float(left["end"]),
                float(right["end"]),
            )

            if end <= start:
                continue

            regions.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "speakers": sorted([
                    left["speaker"],
                    right["speaker"],
                ]),
                "duration_sec": round(
                    end - start,
                    3,
                ),
                "masking_confirmed": False,
                "manual_listening_required": True,
            })

    return regions


def get_supported_sound_candidates(sound_data):
    if not sound_data:
        return []

    candidates = sound_data.get(
        "shot_assigned_candidates",
        sound_data.get(
            "candidate_events",
            [],
        ),
    )

    results = []

    for candidate in candidates:
        strength = candidate.get(
            "evidence_strength"
        )

        # Weak model output is not strong enough
        # even to create a masking-risk cue.
        if strength not in (
            "moderate",
            "strong",
        ):
            continue

        results.append({
            "label": candidate.get("label"),
            "start": float(candidate["start"]),
            "end": float(candidate["end"]),
            "strength": strength,
            "max_score": candidate.get(
                "max_score"
            ),
        })

    return results


def get_low_confidence_words(evidence):
    results = []

    if not evidence:
        return results

    for segment in evidence.get(
        "whisperx_segments",
        [],
    ):
        segment_id = int(
            segment["segment"]
        )

        for word in segment.get(
            "low_confidence_words",
            [],
        ):
            results.append({
                "segment": segment_id,
                "word": word.get("word", ""),
                "start": float(word["start"]),
                "end": float(word["end"]),
                "score": word.get("score"),
            })

    return results


def build_word_risks(
    words,
    speaker_regions,
    sound_candidates,
):
    results = []

    for word in words:
        speaker_matches = [
            region
            for region in speaker_regions
            if overlaps(
                word["start"],
                word["end"],
                region["start"],
                region["end"],
            )
        ]

        sound_matches = [
            sound
            for sound in sound_candidates
            if overlaps(
                word["start"],
                word["end"],
                sound["start"],
                sound["end"],
            )
        ]

        if not speaker_matches and not sound_matches:
            continue

        reasons = []

        if speaker_matches:
            reasons.append(
                "speaker_overlap_near_low_confidence_word"
            )

        if sound_matches:
            reasons.append(
                "non_speech_overlap_near_low_confidence_word"
            )

        results.append({
            **word,
            "risk_reasons": reasons,
            "speaker_overlap_regions":
                speaker_matches,
            "sound_overlap_candidates":
                sound_matches,

            # Critical Manuscript rule.
            "masking_confirmed": False,

            "manual_intelligibility_check_required":
                True,
        })

    return results


def build_review_windows(word_risks):
    windows = []

    for item in word_risks:
        start = max(
            0.0,
            float(item["start"]) - 0.40,
        )

        end = (
            float(item["end"])
            + 0.40
        )

        windows.append({
            "priority": "high",
            "type":
                "overlap_intelligibility_check",
            "start": round(start, 3),
            "end": round(end, 3),
            "description": (
                f"Verify whether overlap actually "
                f"reduces intelligibility around "
                f"'{item['word']}'. Do not call "
                f"this masking unless the audio "
                f"confirms reduced audibility."
            ),
            "word": item["word"],
            "segment": item["segment"],
        })

    return windows


def main():
    print(
        "=== MASKING / OVERLAP EVIDENCE ==="
    )

    evidence = load_json(
        EVIDENCE
    )

    diarization = load_json(
        DIARIZATION
    )

    sounds = load_json(
        SOUND_EVIDENCE
    )

    speaker_regions = (
        build_speaker_overlap_regions(
            diarization
        )
    )

    supported_sounds = (
        get_supported_sound_candidates(
            sounds
        )
    )

    low_words = (
        get_low_confidence_words(
            evidence
        )
    )

    word_risks = build_word_risks(
        low_words,
        speaker_regions,
        supported_sounds,
    )

    review_windows = (
        build_review_windows(
            word_risks
        )
    )

    result = {
        "speaker_overlap_regions":
            speaker_regions,

        "supported_sound_overlap_candidates":
            supported_sounds,

        "low_confidence_word_overlap_risks":
            word_risks,

        "review_windows":
            review_windows,

        "policy": {
            "automatic_masking_claims":
                False,

            "overlap_is_not_masking":
                True,

            "human_listening_required":
                True,

            "weak_sound_model_evidence_used_for_masking":
                False,
        },
    }

    with OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "MASKING / OVERLAP EVIDENCE: PASS |",
        len(speaker_regions),
        "speaker overlaps |",
        len(supported_sounds),
        "supported sound candidates |",
        len(word_risks),
        "word risks |",
        len(review_windows),
        "review cues",
    )

    print()

    for item in word_risks:
        print(
            "WORD",
            repr(item["word"]),
            "|",
            round(item["start"], 3),
            "-->",
            round(item["end"], 3),
            "| reasons=",
            item["risk_reasons"],
            "| masking=UNCONFIRMED",
        )


if __name__ == "__main__":
    main()
