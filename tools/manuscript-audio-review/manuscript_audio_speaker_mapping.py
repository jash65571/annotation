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

FACE_TRACKS = (
    ROOT
    / "analysis"
    / "face_track_evidence.json"
)

# Optional human-authored file, mirroring speaker_map.json's pattern: maps
# an anonymous face-track id to a character id ONLY when a reviewer has
# confirmed it. Never auto-populated.
FACE_CHARACTER_MAP = ROOT / "face_character_map.json"

SPEAKER_MAPPING_EVIDENCE = (
    ROOT
    / "analysis"
    / "speaker_mapping_evidence.json"
)


# ---------------------------------------------------------------------------
# Phase 3B: face-track / active-speaker fusion (spec 3B, non-negotiable
# rules 2/3). Pure logic, no models -- consumes manuscript_audio_face_worker's
# JSON output plus whatever independent speech-presence signal is available
# (diarization turns, else VAD regions).
#
# Scope reminder: the face worker produces a mouth-motion proxy (MAR), not
# genuine audiovisual sync. Every candidate here is therefore capped at
# MEDIUM -- STRONG is reserved for real sync evidence this pipeline does not
# have. This cap is enforced in one place (`_cap_tier`) so it cannot drift.
# ---------------------------------------------------------------------------

STRONG = "STRONG"
MEDIUM = "MEDIUM"
WEAK = "WEAK"
CONFLICT = "CONFLICT"
UNKNOWN = "UNKNOWN"

# A track must be visible for at least this fraction of a speech window's
# sampled frames to count as "visible throughout" rather than "partial".
VISIBILITY_STRONG_RATIO = 0.6
# Minimum mouth-motion (max-min MAR within the window) to count as active
# speaking motion rather than a static/closed mouth.
MOTION_ACTIVE_THRESHOLD = 0.08
# Two candidates are "similar" (-> CONFLICT) when neither clearly leads.
CONFLICT_SCORE_RATIO = 1.3
# Mouth-motion-only evidence is never allowed to reach STRONG.
_MAX_MOUTH_MOTION_TIER = MEDIUM
_TIER_RANK = {UNKNOWN: 0, WEAK: 1, CONFLICT: 1, MEDIUM: 2, STRONG: 3}


def _cap_tier(tier):
    if _TIER_RANK.get(tier, 0) > _TIER_RANK[_MAX_MOUTH_MOTION_TIER]:
        return _MAX_MOUTH_MOTION_TIER
    return tier


def merge_intervals(intervals, join_gap=0.15):
    intervals = sorted(
        (float(a), float(b)) for a, b in intervals if float(b) > float(a)
    )
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1] + join_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def independent_speech_windows(diarization, vad):
    """Same source-of-truth precedence as master.build_coverage: diarization
    turns when complete, else VAD regions, else nothing (honest UNKNOWN).
    Returns (windows, source_label).
    """
    if diarization and diarization.get("status") == "complete":
        turns = diarization.get("turns", [])
        if turns:
            return (
                merge_intervals((t["start"], t["end"]) for t in turns),
                "diarization_turns",
            )

    if vad and vad.get("status") == "complete":
        regions = vad.get("regions", [])
        if regions:
            return (
                merge_intervals((r["start"], r["end"]) for r in regions),
                "silero_vad",
            )

    return [], "unavailable"


def _track_points_in_window(track, start, end):
    return [p for p in track.get("points", []) if start <= p["time"] <= end]


def _visibility_ratio(track, start, end, sampled_fps):
    duration = end - start
    if duration <= 0 or not sampled_fps:
        return 0.0
    expected = max(1, round(duration * sampled_fps))
    actual = len(_track_points_in_window(track, start, end))
    return round(min(1.0, actual / expected), 3)


def _motion_score(track, start, end):
    points = _track_points_in_window(track, start, end)
    if len(points) < 2:
        return 0.0
    mars = [p["mar"] for p in points]
    return round(max(mars) - min(mars), 4)


def compute_active_speaker_windows(face_evidence, speech_windows, sampled_fps=None):
    """For each independent speech window, score which visible face track(s)
    are plausibly speaking, from mouth-motion + visibility alone.

    Never emits a character (C#) identity -- only face_id candidates. Never
    exceeds MEDIUM (mouth-motion-only cap, non-negotiable rule 3B-F).
    """
    tracks = face_evidence.get("face_tracks", []) if face_evidence else []
    sampled_fps = sampled_fps or (face_evidence or {}).get("media", {}).get("sampled_fps")

    # 3.6: distinguish "no face was actually detected" from "face evidence
    # was unavailable". When the face worker did not run (missing
    # .venv-vision), we did NOT prove no face is visible -- only that
    # visible-speaker evidence is missing.
    face_evidence_unavailable = (
        not face_evidence
        or face_evidence.get("status") != "complete"
    )

    results = []

    for start, end in speech_windows:
        overlapping = [
            t for t in tracks
            if t["last_seen"] >= start and t["first_seen"] <= end
        ]

        if not overlapping:
            if face_evidence_unavailable:
                reason = "face_evidence_unavailable"
                action = (
                    "Visible-speaker evidence was unavailable (face tracking "
                    "did not run or produced no usable tracks). Do not "
                    "assume the speaker is off-screen -- no face evidence "
                    "was collected."
                )
            else:
                reason = "no_visible_face_during_speech"
                action = (
                    "Speaker may be off-screen, or no face was detected. "
                    "Do not assume a visible character spoke this line."
                )
            results.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "candidates": [],
                "tier": UNKNOWN,
                "reason": reason,
                "action": action,
            })
            continue

        scored = []
        for track in overlapping:
            visibility = _visibility_ratio(track, start, end, sampled_fps)
            motion = _motion_score(track, start, end)
            scored.append({
                "face_id": track["face_id"],
                "visibility_ratio": visibility,
                "motion_score": motion,
                "active": motion >= MOTION_ACTIVE_THRESHOLD,
            })

        scored.sort(key=lambda s: s["motion_score"], reverse=True)
        top = scored[0]

        if len(scored) >= 2:
            second = scored[1]
            close = (
                second["motion_score"] > 0
                and top["motion_score"] > 0
                and top["motion_score"] / second["motion_score"] < CONFLICT_SCORE_RATIO
            )
            if close:
                results.append({
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "candidates": scored,
                    "tier": CONFLICT,
                    "reason": "multiple_visible_faces_similar_motion",
                    "action": (
                        "Two or more visible faces show similar mouth "
                        "motion. Listen and watch to determine the actual "
                        "speaker; do not guess from motion score alone."
                    ),
                })
                continue

        if not top["active"]:
            tier = WEAK
            reason = "mouth_motion_below_threshold"
            action = (
                "Best visible candidate shows little mouth motion. "
                "Treat as a weak lead only; confirm by watching."
            )
        elif top["visibility_ratio"] >= VISIBILITY_STRONG_RATIO:
            tier = _cap_tier(MEDIUM)
            reason = "single_visible_face_with_mouth_motion"
            action = (
                "Single plausible visible candidate with mouth motion "
                "during this window. Mouth-motion evidence only (no "
                "audiovisual sync model) -- confirm by watching before "
                "treating this as the speaker."
            )
        else:
            tier = WEAK
            reason = "single_candidate_partial_visibility"
            action = (
                "Only partially visible during this window. Confirm by "
                "watching before treating this as the speaker."
            )

        results.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "candidates": scored,
            "tier": tier,
            "reason": reason,
            "action": action,
        })

    return results


def build_cluster_to_face_candidates(diarization, active_speaker_windows):
    """Lead only: which face track most often coincides with each
    diarization speaker cluster's speech. Never a claim -- always paired
    with a confidence tier and a listen/watch action.
    """
    if not diarization or diarization.get("status") != "complete":
        return []

    turns = diarization.get("turns", [])
    if not turns:
        return []

    votes = {}  # speaker_cluster -> {face_id: count}

    for turn in turns:
        t_start, t_end, speaker = turn["start"], turn["end"], turn["speaker"]

        for window in active_speaker_windows:
            if window["tier"] not in (MEDIUM, WEAK):
                continue
            if not window["candidates"]:
                continue
            overlap = min(t_end, window["end"]) - max(t_start, window["start"])
            if overlap <= 0.1:
                continue

            top_candidate = max(
                window["candidates"], key=lambda c: c["motion_score"]
            )
            bucket = votes.setdefault(speaker, {})
            bucket[top_candidate["face_id"]] = bucket.get(top_candidate["face_id"], 0) + 1

    candidates = []
    for speaker, bucket in votes.items():
        total = sum(bucket.values())
        best_face, best_count = max(bucket.items(), key=lambda kv: kv[1])
        consistency = best_count / total

        tier = _cap_tier(MEDIUM) if consistency >= 0.7 else WEAK

        candidates.append({
            "speaker_cluster": speaker,
            "face_id": best_face,
            "tier": tier,
            "consistency_ratio": round(consistency, 3),
            "supporting_windows": total,
            "action": (
                "Lead only, from mouth-motion co-occurrence -- confirm by "
                "watching before mapping this cluster to a visible face."
            ),
        })

    candidates.sort(key=lambda c: c["speaker_cluster"])
    return candidates


def build_face_to_character_candidates():
    """Only returns entries a human has already confirmed in
    face_character_map.json (mirrors speaker_map.json's human-confirmation
    pattern). Never inferred automatically -- there is no reliable
    face-appearance-to-character-description matcher in this pipeline, and
    guessing one would violate the "never auto-assign identity" rule.
    """
    if not FACE_CHARACTER_MAP.exists():
        return []

    try:
        mapping = load_json(FACE_CHARACTER_MAP)
    except (json.JSONDecodeError, OSError):
        return []

    return [
        {
            "face_id": face_id,
            "character": character,
            "tier": STRONG,
            "action": "Human-confirmed mapping; safe default.",
        }
        for face_id, character in mapping.items()
        if character
    ]


def build_speaker_mapping_evidence(diarization, vad, face_evidence):
    speech_windows, speech_source = independent_speech_windows(diarization, vad)

    active_speaker_windows = compute_active_speaker_windows(
        face_evidence, speech_windows
    )

    result = {
        "status": "complete",
        "speech_presence_source": speech_source,
        "face_worker_status": (face_evidence or {}).get("status", "unavailable"),
        "speaker_clusters": (
            diarization.get("speaker_labels", [])
            if diarization and diarization.get("status") == "complete"
            else []
        ),
        "face_tracks": (face_evidence or {}).get("face_tracks", []),
        "active_speaker_windows": active_speaker_windows,
        "cluster_to_face_candidates": build_cluster_to_face_candidates(
            diarization, active_speaker_windows
        ),
        "face_to_character_candidates": build_face_to_character_candidates(),
        "policy": (
            "Face tracks are anonymous (F1, F2, ...). Diarization clusters "
            "are anonymous (SPEAKER_XX). Neither becomes a character (C#) "
            "identity without human confirmation in face_character_map.json "
            "/ speaker_map.json. Mouth-motion evidence never exceeds MEDIUM."
        ),
    }

    return result


def write_speaker_mapping_evidence(
    diarization_path=DIARIZATION,
    vad_path=None,
    face_tracks_path=FACE_TRACKS,
    output_path=SPEAKER_MAPPING_EVIDENCE,
):
    diarization = load_json(diarization_path) if Path(diarization_path).exists() else {}
    vad = load_json(vad_path) if vad_path and Path(vad_path).exists() else {}
    face_evidence = (
        load_json(face_tracks_path) if Path(face_tracks_path).exists() else {}
    )

    result = build_speaker_mapping_evidence(diarization, vad, face_evidence)

    write_json(output_path, result)

    print(
        "SPEAKER/FACE MAPPING: PASS |",
        f"speech_source={result['speech_presence_source']} |",
        f"face_tracks={len(result['face_tracks'])} |",
        f"active_speaker_windows={len(result['active_speaker_windows'])} |",
        f"cluster_to_face={len(result['cluster_to_face_candidates'])}",
    )

    return result


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
