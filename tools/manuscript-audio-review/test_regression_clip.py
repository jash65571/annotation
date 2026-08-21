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
import manuscript_audio_asr_consensus as asrc
import manuscript_audio_speaker_mapping as sm


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

    # 11. ASR consensus (Phase 3A): pure logic, no models, no video.
    primary_words = [
        {"word": "good", "start": 0.1, "end": 0.4, "score": 0.9, "segment": 0},
        {"word": "job", "start": 0.4, "end": 0.7, "score": 0.85, "segment": 0},
        {"word": "eddie", "start": 0.7, "end": 1.1, "score": 0.3, "segment": 0},
    ]
    secondary_words_agree = [
        {"word": "good", "start": 0.12, "end": 0.42, "score": 0.8},
        {"word": "job", "start": 0.41, "end": 0.72, "score": 0.75},
        {"word": "eddie", "start": 0.71, "end": 1.09, "score": 0.6},
    ]

    consensus, secondary_only = asrc.build_word_consensus(
        primary_words, secondary_words_agree
    )
    check(
        "high-confidence agreeing word is confirmed",
        consensus[0]["state"] == "confirmed",
    )
    check(
        "low primary-confidence agreeing word is probable, not confirmed",
        consensus[2]["state"] == "probable" and consensus[2]["needs_listen"],
    )
    check("no secondary-only words when full overlap", secondary_only == [])

    secondary_words_conflict = [
        {"word": "good", "start": 0.12, "end": 0.42, "score": 0.8},
        {"word": "jobe", "start": 0.41, "end": 0.72, "score": 0.75},
        {"word": "eddie", "start": 0.71, "end": 1.09, "score": 0.6},
    ]
    consensus2, _ = asrc.build_word_consensus(
        primary_words, secondary_words_conflict
    )
    check(
        "mismatched text between models becomes conflicting, not silently kept",
        consensus2[1]["state"] == "conflicting",
    )
    conflicts = asrc.build_conflicts(consensus2)
    check("build_conflicts extracts exactly the conflicting word", len(conflicts) == 1)

    # Regression mirror of the reference-clip scenario: secondary model
    # catches speech the primary transcript missed entirely (5.15s cutoff).
    secondary_words_late = secondary_words_agree + [
        {"word": "steve", "start": 7.5, "end": 7.9, "score": 0.7},
        {"word": "never", "start": 7.9, "end": 8.3, "score": 0.65},
    ]
    consensus3, late_only = asrc.build_word_consensus(
        primary_words, secondary_words_late
    )
    check(
        "secondary-only late speech is captured, not dropped",
        len(late_only) == 2
        and all(w["missing_from"] == "primary" for w in late_only),
    )

    coverage_stats = asrc.build_coverage_stats(
        primary_words, late_only, consensus3, duration_sec=10.0
    )
    check(
        "coverage stats surface the recovered late-speech gap",
        coverage_stats["uncovered_speech_duration_sec"] > 0,
    )

    rerun_windows = asrc.identify_rerun_windows(
        primary_words, late_only, consensus3, duration_sec=10.0
    )
    check(
        "late secondary-only speech becomes a rerun window",
        any("secondary_only_speech" in w["reasons"] for w in rerun_windows),
    )

    # Master-layer wrapping: disagreement must never become a silent fact,
    # and an unavailable secondary pass must not break the base packet.
    asr_wrapped = m.build_asr_consensus({
        "status": "complete",
        "coverage": coverage_stats,
        "word_consensus": consensus3,
        "secondary_only_words": late_only,
        "conflicts": asrc.build_conflicts(consensus2),
        "rerun_windows": rerun_windows,
        "reruns_executed": [],
        "reruns_skipped_count": 0,
        "primary_word_count": len(primary_words),
        "secondary_word_count": len(secondary_words_late),
        "secondary_model": "large-v3-turbo",
    })
    check(
        "recovered late speech is surfaced as a finding, not merged into transcript",
        any(
            "recovered speech the primary transcript missed" in f["claim"]
            for f in asr_wrapped["findings"]
        ),
    )
    check(
        "no asr_consensus finding auto-inserts bracketed uncertainty markers",
        all(
            "[uncertain" not in f["claim"].lower()
            and "[inaudible" not in f["claim"].lower()
            and "[unintelligible" not in f["claim"].lower()
            for f in asr_wrapped["findings"]
        ),
    )

    unavailable = m.build_asr_consensus({"status": "unavailable"})
    check(
        "unavailable secondary ASR degrades to a single UNKNOWN finding",
        len(unavailable["findings"]) == 1
        and unavailable["findings"][0]["tier"] == m.UNKNOWN,
    )
    check(
        "unavailable secondary ASR never raises (fail-soft)",
        unavailable["word_consensus"] == [],
    )

    # 12. 3A.1 hardening: stronger rerun triggers, merge-with-provenance,
    # and the two advisory risk signals. Pure logic, no models.

    # A tiny, uncorroborated secondary-only gap must NOT earn an automatic
    # rerun (was burning a full model reload for alignment jitter). It must
    # still remain in secondary_only_words -- recall is unaffected, only the
    # rerun trigger got stricter.
    tiny_gap_words = [
        {"word": "ah", "start": 3.0, "end": 3.12, "secondary_score": 0.4,
         "state": "missing_from_one_model", "missing_from": "primary",
         "needs_listen": True},
    ]
    tiny_rerun_windows = asrc.identify_rerun_windows(
        primary_words, tiny_gap_words, consensus, duration_sec=20.0
    )
    check(
        "tiny uncorroborated secondary-only gap does not trigger an auto-rerun",
        not any(
            "secondary_only_speech" in w["reasons"] for w in tiny_rerun_windows
        ),
    )

    # The same tiny gap DOES earn a rerun when an independent VAD/diarization
    # signal corroborates it -- a stronger trigger, per spec.
    corroborated_windows = asrc.identify_rerun_windows(
        primary_words, tiny_gap_words, consensus, duration_sec=20.0,
        independent_speech_regions=[(2.9, 3.3)],
    )
    check(
        "the same tiny gap triggers a rerun once VAD/diarization corroborates it",
        any(
            "secondary_only_speech" in w["reasons"] for w in corroborated_windows
        ),
    )

    # Two close conflicting-word candidates merge into one inference window,
    # but each original window survives as source_windows provenance.
    close_conflict_consensus = [
        {"word": "a", "start": 5.0, "end": 5.2, "state": "conflicting",
         "secondary_word": "x"},
        {"word": "b", "start": 5.3, "end": 5.5, "state": "conflicting",
         "secondary_word": "y"},
    ]
    merged_windows = asrc.identify_rerun_windows(
        [], [], close_conflict_consensus, duration_sec=20.0
    )
    check(
        "two close conflict candidates merge into a single rerun window",
        len(merged_windows) == 1,
    )
    check(
        "the merged window preserves both original source windows",
        len(merged_windows[0]["source_windows"]) == 2,
    )

    # map_rerun_to_source_windows attributes a merged transcription back to
    # each original sub-window instead of blurring which gap it resolved.
    fake_rerun_result = {
        "segments": [{
            "words": [
                {"word": "alpha", "start": 5.0, "end": 5.2},
                {"word": "beta", "start": 5.3, "end": 5.5},
            ],
        }],
    }
    mapped = asrc.map_rerun_to_source_windows(
        fake_rerun_result, merged_windows[0]["source_windows"]
    )
    check(
        "recovered text is attributed to the correct original sub-window",
        mapped[0]["recovered_text"] == "alpha"
        and mapped[1]["recovered_text"] == "beta",
    )

    # hallucination_risk: advisory only, capped at MEDIUM, never a fact.
    risky_word = {
        "word": "xyzzy", "start": 6.0, "end": 6.2, "segment": 0,
        "primary_score": 0.1, "state": "conflicting",
        "secondary_word": "something",
    }
    hallucination = asrc.compute_hallucination_risk(
        risky_word,
        primary_segment_avg_logprob=-0.9,
        secondary_segments=[{
            "start": 5.8, "end": 6.4,
            "no_speech_prob": 0.8, "compression_ratio": 3.0,
        }],
        independent_speech_regions=[],
    )
    check(
        "a maximally-suspicious word still gets only a WEAK/MEDIUM risk tier",
        hallucination is not None
        and hallucination["tier"] in ("WEAK", "MEDIUM"),
    )
    check(
        "hallucination risk carries its contributing reasons",
        len(hallucination["reasons"]) >= 3,
    )

    clean_word = {
        "word": "hello", "start": 6.0, "end": 6.2, "segment": 0,
        "primary_score": 0.95, "state": "confirmed",
        "secondary_word": "hello",
    }
    check(
        "a clean, agreeing, high-confidence word has no hallucination risk",
        asrc.compute_hallucination_risk(
            clean_word, None, [], None
        ) is None,
    )

    asr_wrapped_risk = m.build_asr_consensus({
        "status": "complete",
        "coverage": coverage_stats,
        "word_consensus": consensus3,
        "secondary_only_words": late_only,
        "conflicts": [],
        "rerun_windows": [],
        "reruns_executed": [],
        "reruns_skipped_count": 0,
        "primary_word_count": len(primary_words),
        "secondary_word_count": len(secondary_words_late),
        "secondary_model": "large-v3-turbo",
        "hallucination_risk_words": [hallucination],
    })
    check(
        "hallucination-risk finding never reaches STRONG or CONFLICT tier",
        all(
            f["tier"] not in (m.STRONG, m.CONFLICT)
            for f in asr_wrapped_risk["findings"]
            if "hallucination" in f["claim"].lower()
        ),
    )

    # proper_noun_risk: conservative -- common/stop words and clean matches
    # never get flagged; only disagreement or low confidence on a
    # non-trivial token does, and it never proposes a spelling.
    check(
        "a common stopword is never flagged as a proper-noun risk",
        asrc.compute_proper_noun_risk({
            "word": "the", "start": 0.0, "end": 0.1,
            "state": "conflicting", "primary_score": 0.1,
            "secondary_word": "a",
        }) is None,
    )
    check(
        "a clean, high-confidence, agreeing word is never flagged",
        asrc.compute_proper_noun_risk({
            "word": "chris", "start": 0.0, "end": 0.1,
            "state": "confirmed", "primary_score": 0.95,
            "secondary_word": "chris",
        }) is None,
    )
    # 3.6: proper-noun risk targets real name-like tokens only. Lowercase
    # common words -- even disagreeing, low-confidence ones -- are never
    # flagged (the old rule produced noise like 'percussion', 'Check',
    # 'him'). Only a mid-sentence capitalized token is name-like.
    name_risk = asrc.compute_proper_noun_risk({
        "word": "Chris", "start": 0.0, "end": 0.1,
        "sentence_initial": False,
    })
    check(
        "a mid-sentence capitalized name-like word is flagged for review",
        name_risk is not None and "Chris" == name_risk["word"],
    )
    check(
        "a lowercase disagreeing common word is never flagged as a name",
        asrc.compute_proper_noun_risk({
            "word": "chris", "start": 0.0, "end": 0.1,
            "sentence_initial": False,
        }) is None,
    )
    check(
        "proper_noun_risk never asserts a corrected spelling",
        "correct" not in " ".join(name_risk["reasons"]).lower(),
    )

    # 13. Face tracking / active-speaker mapping (Phase 3B): pure logic,
    # no video, no mediapipe.

    # A single visible face with strong mouth motion throughout a speech
    # window: the ONLY discipline invariant here is the mouth-motion cap --
    # it must never reach STRONG (that's reserved for real audiovisual sync
    # evidence this pipeline does not have).
    face_evidence_single = {
        "status": "complete",
        "media": {"sampled_fps": 5},
        "face_tracks": [{
            "face_id": "F1",
            "first_seen": 0.0,
            "last_seen": 4.0,
            "points": [
                {"time": t / 5, "bbox": [0, 0, 10, 10], "mar": 0.1 + 0.3 * (t % 2)}
                for t in range(21)
            ],
        }],
    }
    speech_windows = [(0.0, 4.0)]
    active = sm.compute_active_speaker_windows(face_evidence_single, speech_windows)
    check(
        "single visible face with mouth motion produces exactly one window",
        len(active) == 1,
    )
    check(
        "mouth-motion-only active-speaker evidence never reaches STRONG",
        active[0]["tier"] in (sm.MEDIUM, sm.WEAK),
    )
    check(
        "active-speaker candidate names the face_id, never a character",
        active[0]["candidates"]
        and active[0]["candidates"][0]["face_id"] == "F1",
    )

    # No face at all during a speech window -> UNKNOWN, never silently
    # dropped and never guessed at.
    off_screen = sm.compute_active_speaker_windows(
        {"status": "complete", "media": {"sampled_fps": 5}, "face_tracks": []},
        [(0.0, 2.0)],
    )
    check(
        "speech with zero visible faces stays UNKNOWN, not guessed",
        off_screen[0]["tier"] == sm.UNKNOWN
        and off_screen[0]["candidates"] == [],
    )

    # Two visible faces with near-identical motion -> CONFLICT, not a
    # coin-flip pick of one.
    face_evidence_conflict = {
        "status": "complete",
        "media": {"sampled_fps": 5},
        "face_tracks": [
            {
                "face_id": "F1", "first_seen": 0.0, "last_seen": 2.0,
                "points": [
                    {"time": t / 5, "bbox": [0, 0, 10, 10], "mar": 0.1 + 0.2 * (t % 2)}
                    for t in range(11)
                ],
            },
            {
                "face_id": "F2", "first_seen": 0.0, "last_seen": 2.0,
                "points": [
                    {"time": t / 5, "bbox": [50, 0, 10, 10], "mar": 0.1 + 0.19 * (t % 2)}
                    for t in range(11)
                ],
            },
        ],
    }
    conflict_windows = sm.compute_active_speaker_windows(
        face_evidence_conflict, [(0.0, 2.0)]
    )
    check(
        "two faces with similar motion scores produce CONFLICT, not a guess",
        conflict_windows[0]["tier"] == sm.CONFLICT,
    )

    # Diarization clusters never silently become face/character identities:
    # cluster_to_face_candidates only ever names anonymous ids, and stays
    # empty with no diarization.
    check(
        "no diarization -> no cluster-to-face candidates invented",
        sm.build_cluster_to_face_candidates({"status": "unavailable"}, active) == [],
    )

    diarization_fixture = {
        "status": "complete",
        "turns": [{"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"}],
    }
    cluster_candidates = sm.build_cluster_to_face_candidates(
        diarization_fixture, active
    )
    check(
        "cluster-to-face candidate names an anonymous cluster and face id, "
        "never a character",
        cluster_candidates
        and cluster_candidates[0]["speaker_cluster"] == "SPEAKER_00"
        and cluster_candidates[0]["face_id"] == "F1"
        and "character" not in cluster_candidates[0],
    )
    check(
        "cluster-to-face candidate is capped, never STRONG",
        all(c["tier"] != sm.STRONG for c in cluster_candidates),
    )

    # face_to_character_candidates never invents a mapping -- only a
    # human-confirmed face_character_map.json (which does not exist in this
    # test run) can produce one.
    check(
        "no human-confirmed face/character map -> zero candidates, never guessed",
        sm.build_face_to_character_candidates() == [],
    )

    # Master-layer wrapping: never claims a character identity; fails soft
    # when face tracking did not run at all.
    wrapped = m.build_speaker_face_mapping({
        "status": "complete",
        "face_worker_status": "complete",
        "face_tracks": face_evidence_single["face_tracks"],
        "active_speaker_windows": active,
        "cluster_to_face_candidates": cluster_candidates,
        "face_to_character_candidates": [],
    })
    check(
        "no speaker_face_mapping finding claims a C# character identity",
        all(
            "C1" not in f["claim"] and "C2" not in f["claim"]
            for f in wrapped["findings"]
        ),
    )
    check(
        "no speaker_face_mapping finding reaches STRONG from mouth motion alone",
        all(
            f["tier"] != m.STRONG
            for f in wrapped["findings"]
            if "mouth motion" in f["claim"].lower()
            or "mouth-motion" in " ".join(f["evidence"]).lower()
        ),
    )

    unavailable_mapping = m.build_speaker_face_mapping({"status": "unavailable"})
    check(
        "face tracking unavailable degrades to a single UNKNOWN finding",
        len(unavailable_mapping["findings"]) == 1
        and unavailable_mapping["findings"][0]["tier"] == m.UNKNOWN,
    )
    check(
        "face tracking unavailable never raises (fail-soft)",
        unavailable_mapping["active_speaker_windows"] == [],
    )

    print("=== ALL REGRESSION CHECKS PASSED ===")


if __name__ == "__main__":
    run()
