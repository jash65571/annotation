def merge_transcript_review_windows(
    windows,
    max_gap_sec=0.25,
):
    """
    Merge nearby transcript-word review windows.

    Every original issue is preserved inside `issues`.
    Non-transcript review windows are left unchanged.
    """

    transcript_windows = sorted(
        [
            item
            for item in windows
            if item.get("type") == "transcript_word_check"
        ],
        key=lambda item: (
            float(item["start"]),
            float(item["end"]),
        ),
    )

    other_windows = [
        item
        for item in windows
        if item.get("type") != "transcript_word_check"
    ]

    merged = []

    for item in transcript_windows:
        issue = {
            "start": item["start"],
            "end": item["end"],
            "description": item["description"],
        }

        if not merged:
            new_item = dict(item)
            new_item["issues"] = [issue]
            new_item["merged_issue_count"] = 1
            merged.append(new_item)
            continue

        current = merged[-1]

        gap = (
            float(item["start"])
            - float(current["end"])
        )

        if gap <= max_gap_sec:
            current["end"] = round(
                max(
                    float(current["end"]),
                    float(item["end"]),
                ),
                3,
            )

            current["issues"].append(issue)
            current["merged_issue_count"] = len(
                current["issues"]
            )

            current["description"] = (
                f"{current['merged_issue_count']} nearby "
                "low-confidence transcript checks; "
                "verify each flagged word by listening."
            )
        else:
            new_item = dict(item)
            new_item["issues"] = [issue]
            new_item["merged_issue_count"] = 1
            merged.append(new_item)

    combined = merged + other_windows

    priority_rank = {
        "high": 0,
        "medium": 1,
        "normal": 2,
    }

    combined.sort(
        key=lambda item: (
            float(item["start"]),
            priority_rank.get(
                item.get("priority", "normal"),
                9,
            ),
            item.get("type", ""),
        )
    )

    return combined
