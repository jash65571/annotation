from pathlib import Path
import json


ROOT = Path(__file__).resolve().parent

DIARIZATION = (
    ROOT
    / "analysis"
    / "diarization_evidence.json"
)

OUTPUT = (
    ROOT
    / "analysis"
    / "diarization_cluster_review.json"
)


def load_json(path):
    with Path(path).open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        return json.load(f)


def merge_intervals(intervals):
    if not intervals:
        return []

    intervals = sorted(intervals)

    merged = [
        [
            intervals[0][0],
            intervals[0][1],
        ]
    ]

    for start, end in intervals[1:]:
        current = merged[-1]

        if start <= current[1]:
            current[1] = max(
                current[1],
                end,
            )
        else:
            merged.append(
                [start, end]
            )

    return merged


def interval_duration(intervals):
    return sum(
        end - start
        for start, end in intervals
    )


def overlap_intervals(
    start,
    end,
    other_turns,
):
    overlaps = []

    for turn in other_turns:
        other_start = float(
            turn["start"]
        )

        other_end = float(
            turn["end"]
        )

        overlap_start = max(
            start,
            other_start,
        )

        overlap_end = min(
            end,
            other_end,
        )

        if overlap_end > overlap_start:
            overlaps.append(
                (
                    overlap_start,
                    overlap_end,
                )
            )

    return merge_intervals(
        overlaps
    )


def build_cluster_review(
    diarization,
):
    turns = diarization.get(
        "turns",
        [],
    )

    segment_speakers = (
        diarization.get(
            "segment_speakers",
            [],
        )
    )

    labels = diarization.get(
        "speaker_labels",
        [],
    )

    results = []

    for label in labels:
        own_turns = [
            turn
            for turn in turns
            if turn["speaker"] == label
        ]

        other_turns = [
            turn
            for turn in turns
            if turn["speaker"] != label
        ]

        total_duration = sum(
            float(turn["end"])
            - float(turn["start"])
            for turn in own_turns
        )

        overlap_duration = 0.0

        for turn in own_turns:
            start = float(
                turn["start"]
            )

            end = float(
                turn["end"]
            )

            overlaps = overlap_intervals(
                start,
                end,
                other_turns,
            )

            overlap_duration += (
                interval_duration(
                    overlaps
                )
            )

        overlap_ratio = (
            overlap_duration
            / total_duration
            if total_duration
            else 0.0
        )

        word_count = 0
        dominant_segments = []

        for segment in segment_speakers:
            if (
                segment.get("speaker")
                == label
            ):
                dominant_segments.append(
                    int(
                        segment[
                            "segment"
                        ]
                    )
                )

            word_count += int(
                segment.get(
                    "word_speaker_counts",
                    {},
                ).get(
                    label,
                    0,
                )
            )

        zero_words = (
            word_count == 0
        )

        overlap_only = (
            total_duration >= 0.20
            and overlap_ratio >= 0.90
        )

        suspicious = (
            zero_words
            and (
                overlap_only
                or total_duration >= 0.50
            )
        )

        if suspicious:
            priority = "high"
        elif overlap_ratio >= 0.50:
            priority = "medium"
        else:
            priority = "normal"

        review_windows = []

        if suspicious:
            for turn in own_turns:
                review_windows.append({
                    "priority": "high",
                    "type":
                        "diarization_cluster_check",
                    "speaker_cluster":
                        label,
                    "start": round(
                        max(
                            0.0,
                            float(
                                turn[
                                    "start"
                                ]
                            )
                            - 0.50,
                        ),
                        3,
                    ),
                    "end": round(
                        float(
                            turn[
                                "end"
                            ]
                        )
                        + 0.50,
                        3,
                    ),
                    "description": (
                        f"{label} has a "
                        "diarization turn but "
                        "no assigned transcript "
                        "words; verify whether "
                        "this is another speaker, "
                        "overlapping vocal sound, "
                        "or diarization error."
                    ),
                })

        results.append({
            "speaker_cluster":
                label,

            "turn_count":
                len(own_turns),

            "total_duration_sec":
                round(
                    total_duration,
                    3,
                ),

            "assigned_word_count":
                word_count,

            "dominant_transcript_segments":
                dominant_segments,

            "overlap_with_other_speakers_sec":
                round(
                    overlap_duration,
                    3,
                ),

            "overlap_ratio":
                round(
                    overlap_ratio,
                    3,
                ),

            "zero_assigned_words":
                zero_words,

            "overlap_only_cluster":
                overlap_only,

            "suspicious_cluster":
                suspicious,

            "review_priority":
                priority,

            "automatic_character_assignment":
                False,

            "human_confirmation_required":
                True,

            "review_windows":
                review_windows,
        })

    return results


def main():
    print(
        "=== DIARIZATION CLUSTER QC ==="
    )

    if not DIARIZATION.exists():
        print(
            "DIARIZATION CLUSTER QC: "
            "SKIPPED | no evidence"
        )
        return

    diarization = load_json(
        DIARIZATION
    )

    if (
        diarization.get("status")
        != "complete"
    ):
        print(
            "DIARIZATION CLUSTER QC: "
            "SKIPPED | diarization incomplete"
        )
        return

    clusters = build_cluster_review(
        diarization
    )

    all_windows = []

    for cluster in clusters:
        all_windows.extend(
            cluster["review_windows"]
        )

    result = {
        "clusters": clusters,
        "review_windows":
            all_windows,
        "policy": {
            "automatic_character_assignment":
                False,
            "human_confirmation_required":
                True,
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
        "DIARIZATION CLUSTER QC: PASS |",
        len(clusters),
        "clusters |",
        len(all_windows),
        "review windows",
    )

    print()

    for cluster in clusters:
        print(
            cluster[
                "speaker_cluster"
            ],
            "| duration=",
            cluster[
                "total_duration_sec"
            ],
            "| words=",
            cluster[
                "assigned_word_count"
            ],
            "| overlap=",
            cluster[
                "overlap_ratio"
            ],
            "| suspicious=",
            cluster[
                "suspicious_cluster"
            ],
            "| priority=",
            cluster[
                "review_priority"
            ],
        )


if __name__ == "__main__":
    main()
