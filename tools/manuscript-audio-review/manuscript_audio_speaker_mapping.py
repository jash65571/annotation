from pathlib import Path
import json


ROOT = Path(__file__).resolve().parent

DIARIZATION = (
    ROOT
    / "analysis"
    / "diarization_evidence.json"
)

TRANSCRIPT = (
    ROOT
    / "output"
    / "VIDEO.json"
)

CLUSTER_MAP = (
    ROOT
    / "speaker_cluster_map.json"
)

GENERATED_SEGMENT_MAP = (
    ROOT
    / "speaker_map.generated.json"
)

REVIEW_REPORT = (
    ROOT
    / "analysis"
    / "speaker_mapping_review.json"
)


def load_json(path):
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def create_cluster_template(
    diarization,
    path,
):
    labels = diarization.get(
        "speaker_labels",
        [],
    )

    existing = {}

    path = Path(path)

    if path.exists():
        existing = load_json(path)

    result = {
        label: existing.get(label)
        for label in labels
    }

    write_json(
        path,
        result,
    )

    return result


def build_review_report(
    diarization,
    transcript,
):
    transcript_segments = transcript.get(
        "segments",
        [],
    )

    speaker_segments = {
        int(item["segment"]): item
        for item in diarization.get(
            "segment_speakers",
            [],
        )
    }

    results = []

    for index, segment in enumerate(
        transcript_segments
    ):
        diarized = speaker_segments.get(
            index,
            {},
        )

        results.append({
            "segment": index,

            "speaker_cluster":
                diarized.get("speaker"),

            "start": round(
                float(segment["start"]),
                3,
            ),

            "end": round(
                float(segment["end"]),
                3,
            ),

            "text":
                segment.get(
                    "text",
                    "",
                ).strip(),

            "word_speaker_counts":
                diarized.get(
                    "word_speaker_counts",
                    {},
                ),

            "human_character_mapping":
                None,
        })

    return results


def build_generated_segment_map(
    review_report,
    cluster_map,
):
    result = {}

    for item in review_report:
        cluster = item.get(
            "speaker_cluster"
        )

        character = cluster_map.get(
            cluster
        )

        if not character:
            continue

        result[
            str(item["segment"])
        ] = character

        item[
            "human_character_mapping"
        ] = character

    return result


def main():
    print(
        "=== SPEAKER CLUSTER MAPPING ==="
    )

    if not DIARIZATION.exists():
        print(
            "SPEAKER MAPPING: SKIPPED | "
            "no diarization evidence"
        )
        return

    diarization = load_json(
        DIARIZATION
    )

    if diarization.get("status") != "complete":
        print(
            "SPEAKER MAPPING: SKIPPED | "
            "diarization is not complete"
        )
        return

    transcript = load_json(
        TRANSCRIPT
    )

    cluster_map = create_cluster_template(
        diarization,
        CLUSTER_MAP,
    )

    review_report = build_review_report(
        diarization,
        transcript,
    )

    generated_map = (
        build_generated_segment_map(
            review_report,
            cluster_map,
        )
    )

    write_json(
        REVIEW_REPORT,
        review_report,
    )

    write_json(
        GENERATED_SEGMENT_MAP,
        generated_map,
    )

    print(
        "SPEAKER MAPPING: PASS |",
        len(cluster_map),
        "clusters |",
        len(generated_map),
        "segments mapped",
    )

    print()

    for label, character in cluster_map.items():
        print(
            label,
            "->",
            character
            if character
            else "UNMAPPED",
        )

    print()
    print("DIARIZED SEGMENTS")

    for item in review_report:
        print(
            f"SEG {item['segment']} | "
            f"{item['speaker_cluster']} | "
            f"{item['start']:.3f} --> "
            f"{item['end']:.3f} | "
            f"{item['text']}"
        )


if __name__ == "__main__":
    main()
