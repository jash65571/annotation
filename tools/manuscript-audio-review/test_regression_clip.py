"""Regression lock for the reference applause clip (spec section 47).

Runs standalone with no video and no models:

    python test_regression_clip.py

It reconstructs the reference clip's evidence structure in memory --
main ASR stops at 5.154s while diarization shows speech out to 10.44s --
and asserts the invariants that were paid for in real review:

- Late speech missed by ASR is surfaced as UNTRANSCRIBED_SPEECH, not lost.
- The overlap-only cluster does not become invented dialogue.
- Weak Music evidence does not become a Music finding above WEAK.
- Overlap is never auto-promoted to masking.
- Recording defects stay UNKNOWN (never an automatic No).
- Mapped C1 stays a STRONG default; late speech is only a MEDIUM lead.
"""

import manuscript_audio_master as m
import manuscript_audio_seed as seed
import manuscript_audio_qa as qa


# --- Reference clip fixtures (structure mirrors real analysis output) ------

EVIDENCE = {
    "media": {"sample_rate": 16000, "duration_sec": 10.5},
    "whisperx_segments": [
        {
            "segment": 0, "start": 0.131, "end": 2.552,
            "text": "Let's give a round of applause for Eddie!",
            "word_count": 8, "mean_word_confidence": 0.8,
            "low_confidence_words": [],
        },
        {
            "segment": 1, "start": 3.513, "end": 4.373,
            "text": "Good job Eddie!", "word_count": 3,
            "mean_word_confidence": 0.6,
            "low_confidence_words": [
                {"word": "Eddie!", "start": 4.073, "end": 4.373, "score": 0.377}
            ],
        },
        {
            "segment": 2, "start": 4.414, "end": 5.154,
            "text": "Good job man!", "word_count": 3,
            "mean_word_confidence": 0.7, "low_confidence_words": [],
        },
    ],
    "character_voice_profiles": {
        "C1": {
            "character": "C1", "confirmed_segment_ids": [0, 1, 2],
            "usable_speech_duration_sec": 4.021, "total_word_count": 14,
            "speed_candidate": "Fast", "speed_evidence_coverage": 0.602,
            "delivery_candidate": "Dramatic",
            "delivery_evidence_coverage": 0.602,
            "recorded_level_candidate": "Moderate",
            "recorded_level_evidence_coverage": 1.0,
        }
    },
    "shot_audio_evidence": [],
}

DIARIZATION = {
    "status": "complete",
    "speaker_labels": ["SPEAKER_00", "SPEAKER_01"],
    "speaker_count": 2,
    "turns": [
        {"start": 0.031, "end": 2.647, "speaker": "SPEAKER_00"},
        {"start": 1.212, "end": 2.174, "speaker": "SPEAKER_01"},
        {"start": 3.457, "end": 6.865, "speaker": "SPEAKER_00"},
        {"start": 7.54, "end": 9.346, "speaker": "SPEAKER_00"},
        {"start": 9.97, "end": 10.443, "speaker": "SPEAKER_00"},
    ],
    "segment_speakers": [
        {"segment": 0, "start": 0.131, "end": 2.552, "speaker": "SPEAKER_00"},
        {"segment": 1, "start": 3.513, "end": 4.373, "speaker": "SPEAKER_00"},
        {"segment": 2, "start": 4.414, "end": 5.154, "speaker": "SPEAKER_00"},
    ],
}

DIARIZATION_QC = {
    "clusters": [
        {"speaker_cluster": "SPEAKER_00", "overlap_only_cluster": False,
         "overlap_ratio": 0.18},
        {"speaker_cluster": "SPEAKER_01", "overlap_only_cluster": True,
         "overlap_ratio": 1.0, "suspicious_cluster": True},
    ]
}

SOUND = {
    "candidate_events": [
        {"label": "Music", "start": 1.0, "end": 5.0, "max_score": 0.3923,
         "evidence_strength": "weak", "manuscript_source_candidate": None},
    ]
}

DEFECTS = {
    "recording_defects": {
        "Obvious clipping": {"evidence_candidate": "not_detected"},
        "Wind noise": {"evidence_candidate": "not_detected"},
    }
}

MASKING = {
    "speaker_overlap_regions": [
        {"start": 1.212, "end": 2.174, "speakers": ["SPEAKER_00", "SPEAKER_01"],
         "duration_sec": 0.962},
    ],
    "low_confidence_word_overlap_risks": [
        {"segment": 1, "word": "Eddie!", "start": 4.073, "end": 4.373,
         "score": 0.377, "risk_reasons": ["speaker_overlap_near_low_confidence_word"]},
    ],
}

CONTEXT = {"characters": ["C1"], "objects": [], "shots": [
    {"shot": 1, "start": 0.0, "end": 4.0},
    {"shot": 2, "start": 4.0, "end": 10.5},
]}

SPEAKER_MAP = {"segments": {"0": "C1", "1": "C1", "2": "C1"}}


def check(name, condition):
    if not condition:
        raise AssertionError(f"REGRESSION FAILED: {name}")
    print(f"  ok: {name}")


def run():
    print("=== REFERENCE CLIP REGRESSION ===")

    coverage = m.build_coverage(EVIDENCE, DIARIZATION)
    regions = coverage["untranscribed_regions"]

    # 1. Late speech is recovered, not lost.
    check(
        "late speech after last ASR word (>5.15s) is flagged",
        any(r["start"] >= 5.0 for r in regions),
    )
    check(
        "the ~7.5-9.3s late speech window is present",
        any(abs(r["start"] - 7.54) < 0.2 for r in regions),
    )
    check(
        "coverage ratio is well under 100%",
        coverage["coverage_ratio"] is not None
        and coverage["coverage_ratio"] < 0.7,
    )

    # 2. Overlap-only cluster is a caution, not dialogue.
    clusters = m.build_speaker_clusters(DIARIZATION, DIARIZATION_QC)
    overlap_findings = [
        f for f in clusters["findings"]
        if "overlap-only" in f["claim"]
    ]
    check("SPEAKER_01 overlap-only cluster is flagged", overlap_findings)
    check(
        "overlap-only cluster is not STRONG",
        all(f["tier"] != m.STRONG for f in overlap_findings),
    )

    # 3. Weak music stays weak.
    sound = m.build_sound_events(SOUND)
    check(
        "Music is a WEAK finding, never promoted",
        all(f["tier"] == m.WEAK for f in sound["music_findings"]),
    )

    # 4. Overlap is not auto-masking.
    om = m.build_overlap_masking(MASKING)
    check(
        "no overlap finding claims confirmed masking",
        all("masking" not in f["claim"].lower()
            or "possible" in f["action"].lower()
            or "not masking" in f["action"].lower()
            for f in om["findings"]),
    )

    # 5. Defects stay UNKNOWN.
    defects = m.build_recording_defects(DEFECTS)
    check(
        "recording defects never auto-answered No",
        all(f["tier"] == m.UNKNOWN for f in defects["findings"]),
    )

    # 6. C1 STRONG default; late speech only a MEDIUM lead.
    mapping = m.build_character_mapping(
        EVIDENCE, coverage, DIARIZATION, SPEAKER_MAP
    )
    strong_c1 = [
        f for f in mapping["findings"]
        if f["tier"] == m.STRONG and "C1 speaks" in f["claim"]
    ]
    check("C1 mapped segments are a STRONG default", strong_c1)
    late_leads = [
        f for f in mapping["findings"]
        if "Untranscribed speech" in f["claim"]
    ]
    check(
        "late-speech->C1 stays a MEDIUM lead (never STRONG)",
        late_leads and all(f["tier"] == m.MEDIUM for f in late_leads),
    )

    # 7. UI suggestions stay sparse: no tone/pitch/clarity invented.
    ui = m.build_ui_suggestions(EVIDENCE)
    c1 = ui["characters"].get("C1", {})
    check("UI suggestions never invent tone", "tone" not in c1)
    check("UI suggestions never invent pitch", "pitch" not in c1)
    check("UI suggestions never invent clarity", "clarity" not in c1)

    # 8. Seed parser: ids, descriptions, shots, and edge handling.
    parsed = seed.parse_seed_text(
        "C1: a man in a red shirt\n"
        "O1 - a wooden podium\n"
        "Shot 1: 0.0 - 4.0\n"
        "Shot 2 4.00–10.50\n"
    )
    check("seed extracts character ids", parsed["characters"] == ["C1"])
    check("seed extracts object ids", parsed["objects"] == ["O1"])
    check("seed extracts two shots", len(parsed["shots"]) == 2)
    check(
        "seed records original baseline",
        parsed["seed_meta"]["original_character_ids"] == ["C1"],
    )
    check(
        "seed keeps descriptions",
        parsed["seed_meta"]["characters"][0]["description"]
        == "a man in a red shirt",
    )

    bad = seed.parse_seed_text("Shot 1: 5-5\nC1: x\nC1: y\n")
    check(
        "seed flags zero-length shot",
        any("end <= start" in i for i in bad["seed_meta"]["parse_issues"]),
    )
    check(
        "seed flags duplicate id",
        any("duplicate" in i for i in bad["seed_meta"]["parse_issues"]),
    )

    # 9. Pasted-back QA: the field-vs-prose and integrity blockers.
    qa_baseline = {"characters": ["C1", "C2", "C5"], "objects": ["O1"]}
    qa_state = {
        "events": [
            {"id": "e1", "type": "Speech", "source": "C2",
             "transcript": "Good job, man!", "recorded_level": "",
             "mix_role": "Foreground", "clarity": "Clear",
             "caption_sentence":
                 "C2 says \"Good job, man!\" at a moderate recorded level "
                 "in the foreground."},
            {"id": "e2", "type": "Speech", "source": "C3",
             "transcript": "Nice one", "voice_status": "Not observed",
             "recorded_level": "Loud", "mix_role": "Supporting",
             "clarity": "Clear", "caption_sentence": "C3 says \"Nice one\"."},
        ],
        "final_audio_text": "C2 said \"Good job, man!\" at 4.4s.",
        "cast_current": ["C1", "C2", "C3"],
        "objects_current": ["O1"],
    }
    result = qa.run_qa(qa_state, qa_baseline)
    codes = {p["code"] for p in result["issues"]}

    check("QA blocks on blockers", result["status"] == "BLOCKED")
    check(
        "QA catches STRUCTURED_FIELD_MISSING (prose vs blank field)",
        "STRUCTURED_FIELD_MISSING" in codes,
    )
    check(
        "QA catches a Not-observed character speaking",
        "not_observed_speaks" in codes,
    )
    check(
        "QA catches deletion of original cast (C5)",
        "original_cast_changed" in codes,
    )
    check(
        "QA catches a timestamp in Final Audio Text",
        "timestamp_in_final" in codes,
    )

    clean = qa.run_qa(
        {
            "events": [
                {"id": "e1", "type": "Speech", "source": "C1",
                 "transcript": "Hello there", "recorded_level": "Moderate",
                 "mix_role": "Foreground", "clarity": "Clear",
                 "caption_sentence":
                     "C1 says \"Hello there\" at a moderate recorded level "
                     "in the foreground."},
            ],
            "final_audio_text": "C1 says \"Hello there\".",
            "cast_current": ["C1"],
        },
        {"characters": ["C1"], "objects": []},
    )
    check("QA passes a clean state", clean["status"] == "QA_CLEAR")

    # 10. VAD fallback: coverage still works when diarization is absent.
    vad = {
        "status": "complete",
        "source": "silero_vad",
        "regions": [
            {"start": 0.05, "end": 2.6},
            {"start": 3.45, "end": 6.9},
            {"start": 7.5, "end": 9.4},
            {"start": 9.95, "end": 10.45},
        ],
    }
    vad_coverage = m.build_coverage(EVIDENCE, {}, vad)
    check(
        "VAD is used when diarization is absent",
        vad_coverage["speech_presence_source"] == "silero_vad",
    )
    check(
        "VAD fallback still flags the late untranscribed speech",
        any(r["start"] >= 7.0 for r in vad_coverage["untranscribed_regions"]),
    )
    check(
        "VAD-sourced regions carry no speaker identity",
        all(
            r["diarized_speakers"] == []
            for r in vad_coverage["untranscribed_regions"]
        ),
    )
    no_signal = m.build_coverage(EVIDENCE, {}, {})
    check(
        "coverage is honestly UNKNOWN with no diarization and no VAD",
        no_signal["speech_presence_source"] == "unavailable",
    )

    print("=== ALL REGRESSION CHECKS PASSED ===")


if __name__ == "__main__":
    run()
