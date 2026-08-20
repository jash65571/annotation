from pathlib import Path
import json
import os

from whisperx.diarize import (
    DiarizationPipeline,
    assign_word_speakers,
)


ROOT = Path(__file__).resolve().parent

AUDIO = ROOT / "analysis" / "audio.wav"
TRANSCRIPT = ROOT / "output" / "VIDEO.json"

DIARIZATION_EVIDENCE = (
    ROOT
    / "analysis"
    / "diarization_evidence.json"
)

DIARIZED_TRANSCRIPT = (
    ROOT
    / "output"
    / "VIDEO.diarized.json"
)


def main():
    print("=== OPTIONAL SPEAKER DIARIZATION ===")

    token = os.environ.get("HF_TOKEN")

    if not token:
        result = {
            "status": "skipped",
            "reason": "HF_TOKEN is not set",
            "automatic_character_assignment": False,
        }

        DIARIZATION_EVIDENCE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with DIARIZATION_EVIDENCE.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(result, f, indent=2)

        print("DIARIZATION: SKIPPED | HF_TOKEN not set")
        return

    if not AUDIO.exists():
        raise FileNotFoundError(
            f"Missing analysis WAV: {AUDIO}"
        )

    if not TRANSCRIPT.exists():
        raise FileNotFoundError(
            f"Missing WhisperX transcript: {TRANSCRIPT}"
        )

    with TRANSCRIPT.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        transcript = json.load(f)

    print("Loading pyannote diarization model...")

    try:
        pipeline = DiarizationPipeline(
            token=token,
            device="cpu",
        )

        diarization = pipeline(
            str(AUDIO)
        )

    except Exception as exc:
        message = str(exc)

        result = {
            "status": "skipped",
            "reason": "diarization_model_unavailable",
            "error": message,
            "automatic_character_assignment": False,
            "human_mapping_required": True,
        }

        DIARIZATION_EVIDENCE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with DIARIZATION_EVIDENCE.open(
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
            "DIARIZATION: SKIPPED | "
            "pyannote model could not be loaded"
        )

        if "403" in message or "gated" in message.lower():
            print(
                "Reason: Hugging Face model access "
                "has not been granted for this token."
            )

        return

    labeled_transcript = assign_word_speakers(
        diarization,
        transcript,
        fill_nearest=False,
    )

    turns = []

    for _, row in diarization.iterrows():
        turns.append({
            "start": round(
                float(row["start"]),
                3,
            ),
            "end": round(
                float(row["end"]),
                3,
            ),
            "speaker": str(
                row["speaker"]
            ),
        })

    segment_speakers = []

    assigned_word_count = 0
    total_word_count = 0

    for index, segment in enumerate(
        labeled_transcript.get(
            "segments",
            [],
        )
    ):
        speaker = segment.get("speaker")

        words = segment.get("words", [])

        word_speakers = {}

        for word in words:
            total_word_count += 1

            word_speaker = word.get(
                "speaker"
            )

            if word_speaker:
                assigned_word_count += 1

                word_speakers[
                    word_speaker
                ] = (
                    word_speakers.get(
                        word_speaker,
                        0,
                    )
                    + 1
                )

        segment_speakers.append({
            "segment": index,
            "start": round(
                float(segment["start"]),
                3,
            ),
            "end": round(
                float(segment["end"]),
                3,
            ),
            "speaker": speaker,
            "word_speaker_counts":
                word_speakers,
        })

    speakers = sorted({
        turn["speaker"]
        for turn in turns
    })

    evidence = {
        "status": "complete",
        "speaker_labels": speakers,
        "speaker_count": len(speakers),
        "turns": turns,
        "segment_speakers":
            segment_speakers,
        "assigned_word_count":
            assigned_word_count,
        "total_word_count":
            total_word_count,
        "word_assignment_coverage": (
            round(
                assigned_word_count
                / total_word_count,
                3,
            )
            if total_word_count
            else 0.0
        ),

        # Critical Manuscript safety rule.
        "automatic_character_assignment":
            False,

        "human_mapping_required": True,

        "warning": (
            "SPEAKER_XX labels are diarization "
            "clusters, not Manuscript C# identities."
        ),
    }

    DIARIZATION_EVIDENCE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with DIARIZATION_EVIDENCE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            evidence,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with DIARIZED_TRANSCRIPT.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            labeled_transcript,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "DIARIZATION: PASS |",
        len(speakers),
        "speaker clusters |",
        f"{assigned_word_count}/{total_word_count}",
        "words assigned",
    )

    print(
        "Speaker labels:",
        ", ".join(speakers),
    )


if __name__ == "__main__":
    main()

