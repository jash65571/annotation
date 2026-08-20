from pathlib import Path
import json


ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis"
OUTPUT = ANALYSIS / "manuscript_audio_review_report.md"

EVIDENCE = ANALYSIS / "manuscript_audio_evidence.json"
DIARIZATION = ANALYSIS / "diarization_evidence.json"
DIARIZATION_QC = ANALYSIS / "diarization_cluster_review.json"
SOUND = ANALYSIS / "sound_event_evidence.json"
DEFECTS = ANALYSIS / "recording_defect_evidence.json"
MASKING = ANALYSIS / "masking_overlap_evidence.json"
QUEUE = ANALYSIS / "audio_review_queue.json"


def load_json(path, default):
    path = Path(path)

    if not path.exists():
        return default

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        return json.load(f)


def yes_no(value):
    return "YES" if value else "NO"


def fmt_time(value):
    if value is None:
        return "?"

    return f"{float(value):.3f}s"


def add(lines, text=""):
    lines.append(text)


def build_report():
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

    speech = evidence.get(
        "whisperx_segments",
        [],
    )

    speech_by_id = {
        int(item["segment"]): item
        for item in speech
    }

    shots = evidence.get(
        "shot_audio_evidence",
        [],
    )

    voices = evidence.get(
        "character_voice_profiles",
        {},
    )

    ui_candidates = evidence.get(
        "manuscript_ui_candidates",
        [],
    )

    lines = []

    add(
        lines,
        "# Manuscript II Audio Review Report",
    )

    add(lines)
    add(
        lines,
        "> Reviewer evidence only. "
        "Actual audio and video remain the factual source of truth.",
    )
    add(
        lines,
        "> Numeric times in this report are internal review aids only. "
        "Do not copy them into Final Audio Text.",
    )

    # ---------------------------------------------------------
    # OVERVIEW
    # ---------------------------------------------------------

    add(lines)
    add(lines, "## 1. Overview Audio Evidence")
    add(lines)

    media = evidence.get("media", {})

    add(
        lines,
        f"- Audio duration: "
        f"{media.get('duration_sec', '?')} seconds",
    )

    add(
        lines,
        f"- WhisperX speech segments: "
        f"{len(speech)}",
    )

    total_words = sum(
        int(item.get("word_count", 0))
        for item in speech
    )

    add(
        lines,
        f"- Aligned transcript words: {total_words}",
    )

    low_words = sum(
        len(
            item.get(
                "low_confidence_words",
                [],
            )
        )
        for item in speech
    )

    add(
        lines,
        f"- Low-confidence ASR words requiring review: "
        f"{low_words}",
    )

    add(
        lines,
        f"- Locked shots represented in task context: "
        f"{len(shots)}",
    )

    add(
        lines,
        "- Final Overview Audio prose: HUMAN REVIEW REQUIRED",
    )

    # ---------------------------------------------------------
    # RECORDING DEFECTS
    # ---------------------------------------------------------

    add(lines)
    add(lines, "## 2. Recording Defect Checklist")
    add(lines)

    defect_rows = defects.get(
        "recording_defects",
        {},
    )

    if not defect_rows:
        add(
            lines,
            "- No recording-defect evidence file available.",
        )

    for name, row in defect_rows.items():
        add(
            lines,
            f"- **{name}:** "
            f"machine evidence = "
            f"`{row.get('evidence_candidate')}`; "
            f"final UI answer = **UNSET**; "
            f"human listening required = "
            f"{yes_no(row.get('human_listening_required'))}",
        )

    add(
        lines,
        "- `not_detected` does not mean the reviewer should automatically choose No.",
    )

    # ---------------------------------------------------------
    # DIARIZATION
    # ---------------------------------------------------------

    add(lines)
    add(lines, "## 3. Speaker / Diarization Evidence")
    add(lines)

    if diarization.get("status") == "complete":
        add(
            lines,
            f"- Diarization clusters: "
            f"{diarization.get('speaker_count', 0)}",
        )

        add(
            lines,
            f"- Word speaker assignment coverage: "
            f"{diarization.get('word_assignment_coverage', 0):.1%}",
        )

        add(
            lines,
            "- SPEAKER_XX labels are anonymous clusters, not C# identities.",
        )

    else:
        add(
            lines,
            "- Diarization unavailable or skipped.",
        )

    clusters = diarization_qc.get(
        "clusters",
        [],
    )

    for cluster in clusters:
        add(lines)
        add(
            lines,
            f"### {cluster['speaker_cluster']}",
        )

        add(
            lines,
            f"- Diarized duration: "
            f"{cluster['total_duration_sec']}s",
        )

        add(
            lines,
            f"- Assigned transcript words: "
            f"{cluster['assigned_word_count']}",
        )

        add(
            lines,
            f"- Dominant transcript segments: "
            f"{cluster['dominant_transcript_segments']}",
        )

        add(
            lines,
            f"- Overlap ratio: "
            f"{cluster['overlap_ratio']:.1%}",
        )

        add(
            lines,
            f"- Suspicious cluster: "
            f"{yes_no(cluster['suspicious_cluster'])}",
        )

        add(
            lines,
            f"- Review priority: "
            f"{cluster['review_priority'].upper()}",
        )

        add(
            lines,
            "- C# mapping: HUMAN CONFIRMATION REQUIRED",
        )

    # ---------------------------------------------------------
    # CHARACTER VOICE
    # ---------------------------------------------------------

    add(lines)
    add(lines, "## 4. Character Voice Review")
    add(lines)

    if not voices:
        add(
            lines,
            "- No human-confirmed speaker-to-character mappings available.",
        )

    for character, profile in sorted(
        voices.items()
    ):
        add(
            lines,
            f"### {character}",
        )

        add(
            lines,
            f"- Confirmed transcript segments: "
            f"{profile.get('confirmed_segment_ids')}",
        )

        add(
            lines,
            f"- Usable mapped speech: "
            f"{profile.get('usable_speech_duration_sec')}s",
        )

        add(
            lines,
            f"- Total words: "
            f"{profile.get('total_word_count')}",
        )

        add(
            lines,
            f"- Mean pitch evidence: "
            f"{profile.get('mean_pitch_hz_evidence')} Hz",
        )

        add(
            lines,
            f"- Speed candidate: "
            f"{profile.get('speed_candidate')}",
        )

        add(
            lines,
            f"- Speed evidence coverage: "
            f"{profile.get('speed_evidence_coverage', 0):.1%}",
        )

        add(
            lines,
            f"- Delivery candidate: "
            f"{profile.get('delivery_candidate')}",
        )

        add(
            lines,
            f"- Delivery evidence coverage: "
            f"{profile.get('delivery_evidence_coverage', 0):.1%}",
        )

        add(
            lines,
            f"- Recorded level candidate: "
            f"{profile.get('recorded_level_candidate')}",
        )

        add(
            lines,
            f"- Recorded-level evidence coverage: "
            f"{profile.get('recorded_level_evidence_coverage', 0):.1%}",
        )

        add(
            lines,
            "- Pitch final choice: HUMAN REVIEW REQUIRED",
        )
        add(
            lines,
            "- Speaking level: HUMAN REVIEW REQUIRED",
        )
        add(
            lines,
            "- Clarity: HUMAN REVIEW REQUIRED",
        )
        add(
            lines,
            "- Confidence: HUMAN REVIEW REQUIRED",
        )
        add(
            lines,
            "- Texture: HUMAN REVIEW REQUIRED",
        )
        add(
            lines,
            "- Mix role: HUMAN REVIEW REQUIRED",
        )
        add(
            lines,
            "- Tone: HUMAN REVIEW REQUIRED",
        )

    # ---------------------------------------------------------
    # SHOTS
    # ---------------------------------------------------------

    add(lines)
    add(lines, "## 5. Shot-by-Shot Audio Evidence")

    assigned_sounds = sound.get(
        "shot_assigned_candidates",
        [],
    )

    for shot in shots:
        number = int(shot["shot"])

        add(lines)
        add(
            lines,
            f"### Shot {number}",
        )

        add(
            lines,
            f"- Internal range: "
            f"{fmt_time(shot['start'])} "
            f"to {fmt_time(shot['end'])}",
        )

        add(
            lines,
            f"- Speech crosses into shot: "
            f"{yes_no(shot['speech_crosses_into_shot'])}",
        )

        add(
            lines,
            f"- Speech crosses out of shot: "
            f"{yes_no(shot['speech_crosses_out_of_shot'])}",
        )

        add(
            lines,
            f"- Boundary review required: "
            f"{yes_no(shot['manual_boundary_review_required'])}",
        )

        add(lines)
        add(lines, "**Speech candidates**")

        segment_ids = shot.get(
            "speech_segment_ids",
            [],
        )

        if not segment_ids:
            add(
                lines,
                "- No aligned speech segment overlaps this shot.",
            )

        for segment_id in segment_ids:
            segment = speech_by_id.get(
                int(segment_id)
            )

            if not segment:
                continue

            segment_start = float(
                segment["start"]
            )

            segment_end = float(
                segment["end"]
            )

            shot_start = float(
                shot["start"]
            )

            shot_end = float(
                shot["end"]
            )

            continuity_tags = []

            if segment_start < shot_start < segment_end:
                continuity_tags.append(
                    "continuation from previous shot"
                )

            if segment_start < shot_end < segment_end:
                continuity_tags.append(
                    "continues into next shot"
                )

            continuity_text = ""

            if continuity_tags:
                continuity_text = (
                    " ["
                    + "; ".join(continuity_tags)
                    + "]"
                )

            add(
                lines,
                f"- SEG {segment_id}"
                f"{continuity_text}: "
                f"\"{segment.get('text', '')}\"",
            )

            if int(segment_id) < len(
                ui_candidates
            ):
                candidate = ui_candidates[
                    int(segment_id)
                ]

                values = []

                for key, label in (
                    ("speed_candidate", "Speed"),
                    ("delivery_candidate", "Delivery"),
                    (
                        "recorded_level_candidate",
                        "Recorded level",
                    ),
                    ("clarity_candidate", "Clarity"),
                ):
                    value = candidate.get(key)

                    if value:
                        values.append(
                            f"{label}={value}"
                        )

                if values:
                    add(
                        lines,
                        "  - Machine-supported UI candidates: "
                        + ", ".join(values),
                    )

            low = segment.get(
                "low_confidence_words",
                [],
            )

            if low:
                words = ", ".join(
                    repr(item.get("word"))
                    for item in low
                )

                add(
                    lines,
                    f"  - Transcript words to verify: {words}",
                )

        add(lines)
        add(lines, "**Sound candidates**")

        shot_sounds = [
            item
            for item in assigned_sounds
            if any(
                int(assignment["shot"])
                == number
                for assignment
                in item.get(
                    "shot_assignments",
                    [],
                )
            )
        ]

        if not shot_sounds:
            add(
                lines,
                "- No model sound candidate overlaps this shot.",
            )

        for item in shot_sounds:
            add(
                lines,
                f"- {item['label']} | "
                f"strength={item['evidence_strength']} | "
                f"max score={item['max_score']} | "
                f"source candidate="
                f"{item.get('manuscript_source_candidate')}",
            )

            if (
                item.get(
                    "evidence_strength"
                )
                == "weak"
            ):
                add(
                    lines,
                    "  - LISTENING CUE ONLY. "
                    "Do not create a Sound event from this evidence alone.",
                )

            if item.get(
                "crosses_shot_boundary"
            ):
                add(
                    lines,
                    "  - Candidate continues across a shot boundary.",
                )

    # ---------------------------------------------------------
    # MASKING / OVERLAP
    # ---------------------------------------------------------

    add(lines)
    add(lines, "## 6. Masking / Overlap Review")
    add(lines)

    risks = masking.get(
        "low_confidence_word_overlap_risks",
        [],
    )

    if not risks:
        add(
            lines,
            "- No low-confidence transcript word currently intersects supported overlap evidence.",
        )

    for risk in risks:
        add(
            lines,
            f"- \"{risk['word']}\" "
            f"({fmt_time(risk['start'])} to "
            f"{fmt_time(risk['end'])})",
        )

        add(
            lines,
            f"  - Reasons: "
            f"{', '.join(risk['risk_reasons'])}",
        )

        add(
            lines,
            "  - Masking: UNCONFIRMED",
        )

        add(
            lines,
            "  - Reviewer must listen for actual reduction in intelligibility.",
        )

    # ---------------------------------------------------------
    # REVIEW QUEUE
    # ---------------------------------------------------------

    add(lines)
    add(lines, "## 7. Targeted Human Review Queue")
    add(lines)

    if not queue:
        add(
            lines,
            "- No targeted review windows.",
        )

    for index, item in enumerate(
        queue,
        1,
    ):
        add(
            lines,
            f"{index}. "
            f"**{item.get('type')}** | "
            f"{fmt_time(item.get('start'))} to "
            f"{fmt_time(item.get('end'))} | "
            f"{item.get('priority', '').upper()}",
        )

        add(
            lines,
            f"   - {item.get('description', '')}",
        )

    # ---------------------------------------------------------
    # OPEN ITEMS / PRE-FINAL
    # ---------------------------------------------------------

    add(lines)
    add(lines, "## 8. Open Human Review Items")
    add(lines)

    open_items = []

    for name in defect_rows:
        open_items.append(
            f"Choose final Yes/No for recording defect: {name}"
        )

    suspicious = [
        item
        for item in clusters
        if item.get(
            "suspicious_cluster"
        )
    ]

    if suspicious:
        open_items.append(
            "Resolve suspicious diarization cluster(s) by listening."
        )

    if risks:
        open_items.append(
            "Resolve overlap/intelligibility cue(s); "
            "only use masking language if audibility is truly reduced."
        )

    weak_sounds = [
        item
        for item in assigned_sounds
        if item.get(
            "evidence_strength"
        )
        == "weak"
    ]

    if weak_sounds:
        open_items.append(
            "Confirm or reject weak non-speech sound candidate(s)."
        )

    if any(
        shot.get(
            "manual_boundary_review_required"
        )
        for shot in shots
    ):
        open_items.append(
            "Confirm speech continuity across flagged visual shot boundary."
        )

    if not voices:
        open_items.append(
            "Map confirmed audible speakers to Manuscript C# characters."
        )

    if not open_items:
        add(
            lines,
            "- No machine-generated open items remain.",
        )
    else:
        for item in open_items:
            add(
                lines,
                f"- [ ] {item}",
            )

    # ---------------------------------------------------------
    # SAFETY
    # ---------------------------------------------------------

    add(lines)
    add(lines, "## 9. Export Safety Rules")
    add(lines)

    add(
        lines,
        "- Do not copy numeric timestamps from this report into Final Audio Text.",
    )
    add(
        lines,
        "- Do not map SPEAKER_XX directly to a C# without media confirmation.",
    )
    add(
        lines,
        "- Do not turn weak sound evidence into a Sound event.",
    )
    add(
        lines,
        "- Do not treat ASR confidence as Manuscript Confidence.",
    )
    add(
        lines,
        "- Do not call ordinary overlap masking unless intelligibility is actually reduced.",
    )
    add(
        lines,
        "- Do not treat emotion-model labels as final Tone values.",
    )
    add(
        lines,
        "- Actual audio remains the final source of truth.",
    )

    return "\n".join(lines) + "\n"


def main():
    report = build_report()

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        report,
        encoding="utf-8",
    )

    print(
        "MANUSCRIPT REVIEW REPORT: PASS"
    )

    print(
        "Report:",
        OUTPUT,
    )

    print(
        "Lines:",
        len(
            report.splitlines()
        ),
    )


if __name__ == "__main__":
    main()

