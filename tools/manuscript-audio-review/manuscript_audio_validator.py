from pathlib import Path
import json


ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis"

EVIDENCE = ANALYSIS / "manuscript_audio_evidence.json"
DIARIZATION = ANALYSIS / "diarization_evidence.json"
DIARIZATION_QC = ANALYSIS / "diarization_cluster_review.json"
SOUND = ANALYSIS / "sound_event_evidence.json"
DEFECTS = ANALYSIS / "recording_defect_evidence.json"
MASKING = ANALYSIS / "masking_overlap_evidence.json"
QUEUE = ANALYSIS / "audio_review_queue.json"
CONTEXT = ROOT / "task_context.json"
CLUSTER_MAP = ROOT / "speaker_cluster_map.json"

OUTPUT = ANALYSIS / "manuscript_audio_validator.json"


def load_json(path, default):
    path = Path(path)

    if not path.exists():
        return default

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        return json.load(f)


def issue(level, code, message):
    return {
        "level": level,
        "code": code,
        "message": message,
    }


def main():
    print("=== MANUSCRIPT AUDIO V2 VALIDATOR ===")

    evidence = load_json(EVIDENCE, {})
    diarization = load_json(DIARIZATION, {})
    diarization_qc = load_json(
        DIARIZATION_QC,
        {},
    )
    sound = load_json(SOUND, {})
    defects = load_json(DEFECTS, {})
    masking = load_json(MASKING, {})
    queue = load_json(QUEUE, [])
    context = load_json(CONTEXT, {})
    cluster_map = load_json(
        CLUSTER_MAP,
        {},
    )

    problems = []

    # --------------------------------------------------
    # Core files
    # --------------------------------------------------

    if not evidence:
        problems.append(
            issue(
                "blocker",
                "missing_evidence",
                "Main audio evidence is missing.",
            )
        )

    shots = evidence.get(
        "shot_audio_evidence",
        [],
    )

    if not shots:
        problems.append(
            issue(
                "blocker",
                "missing_shot_evidence",
                "No shot-aware audio evidence exists.",
            )
        )

    # --------------------------------------------------
    # Transcript
    # --------------------------------------------------

    speech = evidence.get(
        "whisperx_segments",
        [],
    )

    low_word_count = sum(
        len(
            segment.get(
                "low_confidence_words",
                [],
            )
        )
        for segment in speech
    )

    if low_word_count:
        problems.append(
            issue(
                "review",
                "transcript_words_need_listening",
                f"{low_word_count} ASR word candidates "
                "still require human listening.",
            )
        )

    # --------------------------------------------------
    # Diarization
    # --------------------------------------------------

    if diarization.get("status") == "complete":
        clusters = diarization.get(
            "speaker_labels",
            [],
        )

        unmapped = [
            cluster
            for cluster in clusters
            if not cluster_map.get(cluster)
        ]

        if unmapped:
            problems.append(
                issue(
                    "review",
                    "unmapped_speaker_clusters",
                    "Unmapped diarization clusters: "
                    + ", ".join(unmapped)
                    + ". Do not infer C# identity automatically.",
                )
            )

    for cluster in diarization_qc.get(
        "clusters",
        [],
    ):
        if cluster.get(
            "suspicious_cluster"
        ):
            problems.append(
                issue(
                    "review",
                    "suspicious_diarization_cluster",
                    f"{cluster['speaker_cluster']} "
                    "is a suspicious diarization cluster "
                    "and requires listening.",
                )
            )

    # --------------------------------------------------
    # Character Voice
    # --------------------------------------------------

    characters = context.get(
        "characters",
        [],
    )

    profiles = evidence.get(
        "character_voice_profiles",
        {},
    )

    for character in characters:
        if character not in profiles:
            problems.append(
                issue(
                    "review",
                    "character_voice_unresolved",
                    f"{character} has no resolved Voice "
                    "evidence/status yet.",
                )
            )

    # --------------------------------------------------
    # Sound candidates
    # --------------------------------------------------

    sound_candidates = sound.get(
        "shot_assigned_candidates",
        sound.get(
            "candidate_events",
            [],
        ),
    )

    weak_sounds = [
        item
        for item in sound_candidates
        if item.get(
            "evidence_strength"
        ) == "weak"
    ]

    if weak_sounds:
        labels = sorted({
            item.get(
                "label",
                "unknown"
            )
            for item in weak_sounds
        })

        problems.append(
            issue(
                "review",
                "weak_sound_candidates",
                "Weak sound candidates require confirmation: "
                + ", ".join(labels)
                + ". Do not create Sound events from these alone.",
            )
        )

    # --------------------------------------------------
    # Recording defects
    # --------------------------------------------------

    defect_rows = defects.get(
        "recording_defects",
        {},
    )

    required_defects = [
        "Obvious clipping",
        "Distortion",
        "Excessive echo",
        "Wind noise",
        "Other recording defect",
    ]

    for name in required_defects:
        row = defect_rows.get(name)

        if not row:
            problems.append(
                issue(
                    "blocker",
                    "missing_recording_defect_row",
                    f"Recording-defect evidence missing: {name}.",
                )
            )
            continue

        if row.get(
            "recommended_ui_answer"
        ) is None:
            problems.append(
                issue(
                    "review",
                    "recording_defect_unanswered",
                    f"Final Yes/No still needs human choice: {name}.",
                )
            )

    # --------------------------------------------------
    # Masking
    # --------------------------------------------------

    risks = masking.get(
        "low_confidence_word_overlap_risks",
        [],
    )

    if risks:
        problems.append(
            issue(
                "review",
                "masking_unresolved",
                f"{len(risks)} overlap/intelligibility "
                "cue(s) require listening. "
                "Overlap alone is not masking.",
            )
        )

    # --------------------------------------------------
    # Shot boundaries
    # --------------------------------------------------

    boundary_shots = [
        shot["shot"]
        for shot in shots
        if shot.get(
            "manual_boundary_review_required"
        )
    ]

    if boundary_shots:
        problems.append(
            issue(
                "review",
                "shot_boundary_speech",
                "Speech crosses visual shot boundaries "
                "for shot(s): "
                + ", ".join(
                    str(x)
                    for x in boundary_shots
                )
                + ". Preserve continuity unless audio truly cuts.",
            )
        )

    # --------------------------------------------------
    # Review queue
    # --------------------------------------------------

    if queue:
        problems.append(
            issue(
                "review",
                "targeted_review_queue_open",
                f"{len(queue)} targeted listening "
                "window(s) remain available for review.",
            )
        )

    # --------------------------------------------------
    # Determine machine status
    # --------------------------------------------------

    blocker_count = sum(
        1
        for item in problems
        if item["level"] == "blocker"
    )

    review_count = sum(
        1
        for item in problems
        if item["level"] == "review"
    )

    if blocker_count:
        status = "BLOCKED"
    elif review_count:
        status = "HUMAN_REVIEW_REQUIRED"
    else:
        status = "MACHINE_PREFLIGHT_CLEAR"

    result = {
        "status": status,
        "blocker_count": blocker_count,
        "review_item_count": review_count,
        "issues": problems,

        "policy": {
            "machine_clearance_is_not_final_approval": True,
            "actual_media_remains_ground_truth": True,
            "final_adversarial_listen_required": True,
            "export_checker_still_required": True,
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
        "VALIDATOR STATUS:",
        status,
    )

    print(
        "Blockers:",
        blocker_count,
        "| Human-review items:",
        review_count,
    )

    print()

    for item in problems:
        print(
            f"[{item['level'].upper()}] "
            f"{item['code']}: "
            f"{item['message']}"
        )

    print()
    print(
        "Machine preflight never replaces "
        "the final human listen or platform export check."
    )

    return result


if __name__ == "__main__":
    main()

