from pathlib import Path
import json

from manuscript_audio_task_identity import load_bound_json
def validate_context(context):
    shots = context.get("shots", [])

    if not shots:
        raise ValueError(
            "task_context.json must contain at least one shot."
        )

    previous_end = None

    for expected_number, shot in enumerate(shots, 1):
        number = int(shot["shot"])
        start = float(shot["start"])
        end = float(shot["end"])

        if number != expected_number:
            raise ValueError(
                "Shots must be numbered consecutively from 1."
            )

        if end <= start:
            raise ValueError(
                f"Shot {number} has an invalid time range."
            )

        if (
            previous_end is not None
            and start < previous_end - 0.001
        ):
            raise ValueError(
                f"Shot {number} overlaps the previous shot."
            )

        previous_end = end


def build_shot_audio_evidence(context, whisperx_data):
    validate_context(context)

    segments = whisperx_data.get("segments", [])
    results = []

    for shot_index, shot in enumerate(context["shots"]):
        number = int(shot["shot"])
        start = float(shot["start"])
        end = float(shot["end"])

        segment_ids = []
        aligned_words = []

        for segment_index, segment in enumerate(segments):
            segment_start = float(segment["start"])
            segment_end = float(segment["end"])

            if segment_start < end and segment_end > start:
                segment_ids.append(segment_index)

            for word in segment.get("words", []):
                if "start" not in word or "end" not in word:
                    continue

                word_start = float(word["start"])
                word_end = float(word["end"])

                if word_start < end and word_end > start:
                    aligned_words.append(
                        word.get("word", "").strip()
                    )

        crosses_in = False
        crosses_out = False

        if shot_index > 0:
            crosses_in = any(
                float(segment["start"]) < start
                < float(segment["end"])
                for segment in segments
            )

        if shot_index < len(context["shots"]) - 1:
            crosses_out = any(
                float(segment["start"]) < end
                < float(segment["end"])
                for segment in segments
            )

        results.append({
            "shot": number,
            "start": round(start, 3),
            "end": round(end, 3),
            "speech_segment_ids": sorted(set(segment_ids)),
            "aligned_word_count": len(aligned_words),
            "first_aligned_word": (
                aligned_words[0]
                if aligned_words
                else None
            ),
            "last_aligned_word": (
                aligned_words[-1]
                if aligned_words
                else None
            ),
            "speech_crosses_into_shot": crosses_in,
            "speech_crosses_out_of_shot": crosses_out,
            "manual_boundary_review_required": (
                crosses_in or crosses_out
            ),
        })

    return results


def enrich_evidence_with_shots(
    evidence_path,
    whisperx_path,
    context_path,
):
    evidence_path = Path(evidence_path)
    whisperx_path = Path(whisperx_path)
    context_path = Path(context_path)

    if not context_path.exists():
        print(
            "SHOT CONTEXT: SKIPPED "
            "(task_context.json not found)"
        )
        return False

    identity_path = evidence_path.parent / "video_identity.json"

    if not identity_path.exists():
        print(
            "SHOT CONTEXT: SKIPPED "
            "(video identity missing)"
        )
        return False

    with identity_path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        identity = json.load(f)

    context = load_bound_json(
        context_path,
        identity["video_sha256"],
        "SHOT CONTEXT",
    )

    if context is None:
        return False

    with whisperx_path.open("r", encoding="utf-8-sig") as f:
        whisperx_data = json.load(f)

    with evidence_path.open("r", encoding="utf-8-sig") as f:
        evidence = json.load(f)

    evidence["task_context"] = {
        "characters": context.get("characters", []),
        "objects": context.get("objects", []),
        "shot_count": len(context["shots"]),
    }

    evidence["shot_audio_evidence"] = (
        build_shot_audio_evidence(
            context,
            whisperx_data,
        )
    )

    with evidence_path.open("w", encoding="utf-8") as f:
        json.dump(
            evidence,
            f,
            indent=2,
            ensure_ascii=False,
        )

    flagged = [
        shot["shot"]
        for shot in evidence["shot_audio_evidence"]
        if shot["manual_boundary_review_required"]
    ]

    print(
        "SHOT CONTEXT: PASS |",
        len(evidence["shot_audio_evidence"]),
        "shots",
    )

    if flagged:
        print(
            "Shot-boundary speech review:",
            ", ".join(str(x) for x in flagged),
        )
    else:
        print("Shot-boundary speech review: none")

    return True


