"""Phase 3.5 accuracy-hardening regression checks.

Runs standalone with no models, no audio, no network:

    python test_3d_accuracy_hardening.py

Locks the ten Phase 3.5 behaviors from the accuracy audit:

1. Auto-ingest the seed before analysis (pipeline arg wiring is verified by
   the pipeline itself; the parser is covered by manuscript_audio_seed.py).
2. Every finding is shot-aware.
3. Cross-model ASR matching is sequence-aware (the `Go!` drift case).
4. Targeted reruns are hypotheses, marked CONFLICT against surrounding
   evidence.
5. The transient/SFX detector surfaces strong unexplained transients as
   high-priority review windows.
6. Door classes are split (doorbell_chime / door_open_close /
   door_latch_click / door_knock).
7. Masking is far stricter (real overlap + intelligibility loss only).
8. The review report is deduplicated by time window.
9. Music mix_role is left blank for human review.
10. Media metadata reports both source_sample_rate and analysis_sample_rate.
"""

import json

import manuscript_audio_asr_consensus as a
import manuscript_audio_master as m
import manuscript_audio_shots as ms
import manuscript_audio_sound_events as se
import manuscript_audio_sound_fusion as sf
from manuscript_audio_sound_vocabulary import (
    map_raw_label, CLAP_PROMPTS,
)


def check(name, condition, details=""):
    if not condition:
        msg = f"\n  FAILED: {name}"
        if details:
            msg += f"\n    {details}"
        print(msg)
        raise AssertionError(f"3D VERIFICATION FAILED: {name}")
    print(f"  ok: {name}")


def word(text, start, end, score=0.9, segment=0):
    return {"word": text, "start": start, "end": end,
            "score": score, "segment": segment}


def run():
    print("=== 3.5 ACCURACY HARDENING VERIFICATION ===\n")

    # ------------------------------------------------------------------
    # 3. Cross-model ASR matching is sequence-aware (`Go!` regression).
    # ------------------------------------------------------------------
    primary = [
        word("I", 8.0, 8.2, 0.9),
        word("Go", 8.6, 8.8, 0.45),
        word("Go", 8.9, 9.0, 0.3),
    ]
    secondary = [
        word("I", 8.05, 8.25, 0.9),
        word("go", 8.3, 8.5, 0.6),    # drifted ~300ms earlier
        word("Go", 9.2, 9.3, 0.7),    # drifted ~300ms later
    ]
    consensus, secondary_only = a.build_word_consensus(primary, secondary)
    go_states = [c["state"] for c in consensus if c["word"] == "Go"]
    check("drifted 'Go!' counts as cross-model agreement (probable), not a gap",
          len(go_states) == 2 and all(s in ("confirmed", "probable") for s in go_states)
          and go_states == ["probable", "probable"],
          json.dumps(consensus))
    check("drifted same-word matches leave no secondary-only gaps",
          secondary_only == [], json.dumps(secondary_only))
    check("aligned words record their center-time distance",
          all(c.get("center_distance_sec") is not None for c in consensus),
          json.dumps(consensus))

    # Different text at the same time is still a conflict.
    conflict_primary = [word("need", 13.0, 13.2, 0.9)]
    conflict_secondary = [word("I", 13.05, 13.15, 0.8)]
    c2, _ = a.build_word_consensus(conflict_primary, conflict_secondary)
    check("different text at the same time stays a model conflict",
          c2[0]["state"] == "conflicting" and c2[0]["secondary_word"] == "I",
          json.dumps(c2))

    # A far-away unrelated secondary word never gets stretched into a match.
    far_primary = [word("hello", 1.0, 1.3, 0.9)]
    far_secondary = [word("goodbye", 8.0, 8.3, 0.9)]
    c3, only3 = a.build_word_consensus(far_primary, far_secondary)
    check("far-apart unrelated words are not aligned",
          c3[0]["state"] == "missing_from_one_model"
          and len(only3) == 1 and only3[0]["word"] == "goodbye",
          json.dumps({"consensus": c3, "only": only3}))

    # ------------------------------------------------------------------
    # 4. Targeted reruns are hypotheses, CONFLICT vs surrounding evidence.
    # ------------------------------------------------------------------
    rerun_no_corroboration = {
        "status": "complete",
        "word_consensus": [],
        "conflicts": [],
        "secondary_only_words": [],
        "reruns_executed": [
            {"window": [9.0, 10.0], "recovered_text": "Daddy?",
             "reasons": ["secondary_only_speech"]},
        ],
        "rerun_windows": [],
        "hallucination_risk_words": [],
        "proper_noun_risk_words": [],
    }
    sec = m.build_asr_consensus(rerun_no_corroboration, independent_speech_regions=[])
    rerun_findings = [f for f in sec["findings"] if "rerun" in f["claim"].lower()]
    check("rerun without corroboration is a CONFLICT hypothesis, not truth",
          rerun_findings and rerun_findings[0]["tier"] == m.CONFLICT
          and "hypothesis" in rerun_findings[0]["claim"].lower(),
          json.dumps(rerun_findings))

    sec_ok = m.build_asr_consensus(
        rerun_no_corroboration,
        independent_speech_regions=[(9.1, 9.9)],
    )
    rerun_ok = [f for f in sec_ok["findings"] if "rerun" in f["claim"].lower()]
    check("rerun with independent speech corroboration stays MEDIUM hypothesis",
          rerun_ok and rerun_ok[0]["tier"] == m.MEDIUM,
          json.dumps(rerun_ok))

    # ------------------------------------------------------------------
    # 5. Transient / SFX detector.
    # ------------------------------------------------------------------
    quiet = [
        {"start": i * 0.5, "end": i * 0.5 + 1.5, "rms_db": -40.0,
         "crest_factor": 3.0, "spectral_flux": 0.001,
         "onset_strength": 1e-5, "energy_change_db": 0.0}
        for i in range(20)
    ]
    spike = [{"start": 8.5, "end": 10.0, "rms_db": -12.0, "crest_factor": 14.0,
              "spectral_flux": 0.09, "onset_strength": 0.004,
              "energy_change_db": 28.0}]
    events = sf.build_transient_events(quiet + spike)
    check("strong unexplained transient becomes a STRONG event",
          len(events) == 1 and events[0]["tier"] == sf.STRONG
          and len(events[0]["signals"]) >= 4,
          json.dumps(events))

    speech_like = [
        {"start": 0.0, "end": 2.0, "rms_db": -25.0, "crest_factor": 5.0,
         "spectral_flux": 0.02, "onset_strength": 0.0005, "energy_change_db": 2.0},
        {"start": 2.0, "end": 4.0, "rms_db": -26.0, "crest_factor": 5.5,
         "spectral_flux": 0.018, "onset_strength": 0.0004, "energy_change_db": 1.0},
        {"start": 4.0, "end": 6.0, "rms_db": -25.5, "crest_factor": 5.0,
         "spectral_flux": 0.02, "onset_strength": 0.0005, "energy_change_db": 2.0},
    ]
    check("steady speech-like windows do not fire the transient detector",
          sf.build_transient_events(speech_like) == [],
          json.dumps(sf.build_transient_events(speech_like)))

    raw = {
        "status": "complete",
        "runtime_sec": 2.0,
        "media": {"duration_sec": 12.0, "sample_rate": 16000},
        "panns_windows": [],
        "clap_windows": [],
        "transient_feature_windows": quiet + spike,
        "provenance": {"transient": {"detector": "test", "runtime_sec": 0.1}},
    }
    fused = se.build_sound_fusion_evidence(raw, {})
    check("fusion surfaces transient events and review windows",
          fused["transients"]["events"]
          and any(w["type"] == "transient_sfx_check" and w["tier"] == sf.STRONG
                  for w in fused["review_windows"]),
          json.dumps(fused["transients"]))

    # A transient during speech is NOT explained by the speech -- a punch
    # during dialogue is still a punch the models cannot name. It stays
    # STRONG and records the speech overlap as context.
    speech_raw = dict(raw)
    speech_raw["transient_feature_windows"] = quiet + spike
    fused_sp = se.build_sound_fusion_evidence(
        speech_raw, {"diarization": {"status": "complete", "turns": [
            {"start": 8.6, "end": 9.9, "speaker": "SPEAKER_00"},
        ]}},
    )
    during_speech = [e for e in fused_sp["transients"]["events"] if e["unexplained"]]
    check("transient during speech stays STRONG and is not explained by speech",
          during_speech and during_speech[0]["tier"] == sf.STRONG
          and "speech" in during_speech[0]["explained_by"],
          json.dumps(during_speech))
    check("transient carries explicit reason-why-retained metadata",
          during_speech
          and during_speech[0].get("overlaps_speech") is True
          and during_speech[0].get("explained_by_named_sound") is False,
          json.dumps(during_speech))

    # A transient overlapping a NAMED sound candidate IS explained -> demoted.
    named_raw = dict(raw)
    named_raw["transient_feature_windows"] = quiet + spike
    named_raw["panns_windows"] = [
        {"start": 8.0, "end": 10.5, "top_labels": [
            {"raw_label": "Thud", "score": 0.8},
        ]},
    ]
    fused_named = se.build_sound_fusion_evidence(named_raw, {})
    named_ev = [
        e for e in fused_named["transients"]["events"]
        if e["unexplained"] is False
    ]
    check("transient overlapping a named sound is demoted below STRONG",
          named_ev and named_ev[0]["tier"] == sf.MEDIUM
          and "impact" in named_ev[0]["explained_by"],
          json.dumps(named_ev))
    check("named-sound transient carries explained_by_named_sound=true",
          named_ev
          and named_ev[0].get("explained_by_named_sound") is True
          and named_ev[0].get("overlaps_speech") is False,
          json.dumps(named_ev))

    # ------------------------------------------------------------------
    # 6. Door classes are split.
    # ------------------------------------------------------------------
    check("doorbell maps to doorbell_chime, never 'door'",
          map_raw_label("Doorbell") == ("object_sfx", "doorbell_chime"))
    check("knock maps to door_knock",
          map_raw_label("Knock") == ("object_sfx", "door_knock"))
    check("generic door maps to door_open_close",
          map_raw_label("Door") == ("object_sfx", "door_open_close"))
    check("slam maps to door_open_close",
          map_raw_label("Slam") == ("object_sfx", "door_open_close"))
    door_prompts = [p for p in CLAP_PROMPTS if "door" in p["prompt"].lower()]
    prompt_classes = {p["candidate_class"] for p in door_prompts}
    check("CLAP prompts cover the four split door classes",
          {"door_open_close", "doorbell_chime", "door_latch_click", "door_knock"}
          <= prompt_classes,
          json.dumps(sorted(prompt_classes)))

    # ------------------------------------------------------------------
    # 7. Masking is far stricter.
    # ------------------------------------------------------------------
    masking = {
        "speaker_overlap_regions": [
            {"start": 1.0, "end": 1.5, "speakers": ["SPEAKER_00", "SPEAKER_01"],
             "duration_sec": 0.5},
        ],
        "low_confidence_word_overlap_risks": [
            # Real overlapping source (another speaker).
            {"word": "Mrs.", "start": 5.2, "end": 5.4,
             "risk_reasons": ["speaker_overlap_near_low_confidence_word"],
             "speaker_overlap_regions": [{"start": 5.0, "end": 5.6}],
             "sound_overlap_candidates": []},
            # Real overlapping source (supported sound).
            {"word": "bad", "start": 6.0, "end": 6.2,
             "risk_reasons": ["non_speech_overlap_near_low_confidence_word"],
             "speaker_overlap_regions": [],
             "sound_overlap_candidates": [{"label": "Impact"}]},
            # No overlapping source at all -> NOT a masking candidate.
            {"word": "nope", "start": 7.0, "end": 7.2,
             "risk_reasons": [], "speaker_overlap_regions": [],
             "sound_overlap_candidates": []},
        ],
    }
    mask_sec = m.build_overlap_masking(masking)
    check("plain speaker overlap alone is NOT a masking finding",
          not any("Speaker overlap" in f["claim"] for f in mask_sec["findings"]))
    check("word with sound overlap is a masking candidate",
          any("non-speech sound" in f["claim"] for f in mask_sec["findings"]),
          json.dumps(mask_sec["findings"]))
    check("word with speaker overlap is a masking candidate",
          any("speaker overlap" in f["claim"] for f in mask_sec["findings"]))
    check("word without any overlapping source is never a masking warning",
          all("nope" not in f["claim"] for f in mask_sec["findings"]))

    # Sound-fusion masking: only possible_masking becomes a finding/window.
    mask_ev = sf.evaluate_masking(
        {"start": 0.0, "end": 2.0}, [(0.5, 1.5)], asr_words_lost=False
    )
    check("overlap with intact ASR agreement is not surfaced as masking",
          mask_ev["masking"] == "masking_not_supported")

    # ------------------------------------------------------------------
    # 8. Review report deduplication by time window.
    # ------------------------------------------------------------------
    raw_findings = [
        {"section": "asr_consensus", "tier": m.CONFLICT, "claim": "conflict A",
         "action": "listen", "window": [8.5, 8.7], "shot": 2},
        {"section": "asr_consensus", "tier": m.MEDIUM, "claim": "hallucination B",
         "action": "listen", "window": [8.6, 9.0], "shot": 2},
        {"section": "overlap_masking", "tier": m.MEDIUM, "claim": "masking C",
         "action": "listen", "window": [8.5, 8.9], "shot": 2},
        {"section": "coverage_gaps", "tier": m.STRONG, "claim": "untranscribed D",
         "action": "transcribe", "window": [2.0, 2.4], "shot": 1},
        {"section": "media", "tier": m.UNKNOWN, "claim": "no window", "action": "x",
         "window": None},
    ]
    deduped = m.deduplicate_findings(raw_findings)
    grouped_items = [f for f in deduped if f.get("grouped")]
    check("three signals in one window collapse into one grouped item",
          len(deduped) == 3 and len(grouped_items) == 1
          and len(grouped_items[0]["items"]) == 3
          and grouped_items[0]["tier"] == m.CONFLICT,
          json.dumps(deduped))
    check("grouped item keeps the merged window and shot",
          grouped_items[0]["window"][0] <= 8.5 and grouped_items[0]["window"][1] >= 9.0
          and grouped_items[0]["shot"] == 2,
          json.dumps(grouped_items[0]))
    check("windowless findings survive dedup untouched",
          any(not f.get("window") for f in deduped))

    summary = m.build_confidence_summary(deduped, raw_count=len(raw_findings))
    check("confidence summary reports grouped and raw counts",
          summary["total_findings"] == 3 and summary["raw_finding_count"] == 5,
          json.dumps(summary))

    # ------------------------------------------------------------------
    # 2. Shot-aware findings.
    # ------------------------------------------------------------------
    shots = [
        {"shot": 1, "start": 0.0, "end": 4.0},
        {"shot": 2, "start": 4.0, "end": 10.0},
    ]
    check("shot_for_window maps a window to its locked shot",
          ms.shot_for_window(8.5, 9.0, shots) == 2
          and ms.shot_for_window(1.0, 1.5, shots) == 1)
    collected = m.collect_all_findings(
        {"test_section": {"findings": [
            {"tier": m.MEDIUM, "claim": "x", "action": "y", "window": [8.5, 9.0]},
        ]}},
        context_shots=shots,
    )
    check("collected findings carry their locked shot",
          collected[0]["shot"] == 2, json.dumps(collected))

    # ------------------------------------------------------------------
    # 9. Music mix_role is left blank.
    # ------------------------------------------------------------------
    region = {"start": 0.0, "end": 3.0, "duration_sec": 3.0, "tier": sf.STRONG,
              "signals": [{"source": "panns", "score": 0.9}],
              "raw_labels": ["Music"], "reasons": [],
              "reviewer_action": "Confirm by listening."}
    supporting = {"clip_baseline_rms_dbfs": -30.0, "clip_duration_sec": 10.0,
                  "shots": []}
    enriched = se._enrich_music_region(
        region, 1, supporting,
        [{"start": 0.0, "end": 3.0, "rms_dbfs": -20.0}],
    )
    check("music mix_role is blank for human review",
          enriched["mix_role_candidate"] is None,
          json.dumps(enriched.get("mix_role_candidate")))
    check("music recorded level is still estimated",
          enriched["recorded_level_candidate"] is not None,
          json.dumps(enriched.get("recorded_level_candidate")))

    # ==================================================================
    # PHASE 3.6 CHECKS
    # ==================================================================

    # ------------------------------------------------------------------
    # 3.6a. Multi-stream ASR divergence gate.
    # ------------------------------------------------------------------
    run_words = []
    t = 4.0
    for i, (pw, sw) in enumerate([
        ("Harry", "but"), ("Lindsey", "we're"), ("percussion", "wasted"),
        ("check", "up"), ("him", "now"), ("out", "today"),
    ]):
        run_words.append({
            "word": pw, "secondary_word": sw, "start": t, "end": t + 0.4,
            "primary_score": 0.9, "secondary_score": 0.7,
            "state": "conflicting", "needs_listen": True,
        })
        t += 0.5
    divergence_regions, flagged = a.detect_stream_divergence(run_words)
    check("consecutive all-conflict run becomes ONE divergence region",
          len(divergence_regions) == 1
          and divergence_regions[0]["assessment"] == "MULTI_STREAM_ASR_DIVERGENCE"
          and divergence_regions[0]["word_count"] == 6,
          json.dumps(divergence_regions))
    check("divergence words are excluded from per-word conflicts",
          a.build_conflicts(run_words, excluded_indices=flagged) == [],
          json.dumps(a.build_conflicts(run_words, excluded_indices=flagged)))

    # Scattered conflicts (with confirmed words in between) are NOT a stream
    # divergence -- they stay per-word conflicts.
    scattered = [
        {"word": "a", "secondary_word": "a", "start": 0.0, "end": 0.3,
         "primary_score": 0.9, "secondary_score": 0.9, "state": "confirmed"},
        {"word": "b", "secondary_word": "x", "start": 0.5, "end": 0.8,
         "primary_score": 0.9, "secondary_score": 0.7, "state": "conflicting"},
        {"word": "c", "secondary_word": "c", "start": 1.0, "end": 1.3,
         "primary_score": 0.9, "secondary_score": 0.9, "state": "confirmed"},
    ]
    check("scattered conflicts are not a divergence region",
          a.detect_stream_divergence(scattered)[0] == [],
          json.dumps(a.detect_stream_divergence(scattered)[0]))

    # Master surfaces divergence as ONE high-priority finding.
    consensus_with_divergence = {
        "status": "complete",
        "coverage": {"model_agreement_pct": 0.4},
        "word_consensus": run_words,
        "conflicts": [],
        "divergence_regions": divergence_regions,
        "secondary_only_words": [],
        "reruns_executed": [],
        "rerun_windows": [],
        "hallucination_risk_words": [],
        "proper_noun_risk_words": [],
    }
    asr_sec = m.build_asr_consensus(consensus_with_divergence)
    divergence_findings = [
        f for f in asr_sec["findings"]
        if "MULTI_STREAM_ASR_DIVERGENCE" in f["claim"]
    ]
    check("divergence region is ONE STRONG finding, not N conflicts",
          len(divergence_findings) == 1
          and divergence_findings[0]["tier"] == m.STRONG,
          json.dumps(divergence_findings))

    # Coverage stats must not re-inflate the per-word conflict count that
    # the divergence gate just collapsed.
    cov = a.build_coverage_stats(
        [{"start": 4.0, "end": 6.4, "word": "Harry"}],
        [],
        run_words,
        duration_sec=10.0,
        divergence_flags=flagged,
    )
    check("divergence words are excluded from per-word disagreement count",
          cov["word_disagreement_count"] == 0
          and cov["words_in_divergence_regions"] == 6,
          json.dumps(cov))

    # ------------------------------------------------------------------
    # 3.6b. Proper-noun filter: name-like tokens only.
    # ------------------------------------------------------------------
    check("mid-sentence capitalized token is a proper-noun candidate",
          a.compute_proper_noun_risk({
              "word": "Lindsey", "start": 1.0, "end": 1.4,
              "sentence_initial": False,
          }) is not None)
    check("sentence-start capitalization is formatting, not a name",
          a.compute_proper_noun_risk({
              "word": "Check", "start": 1.0, "end": 1.4,
              "sentence_initial": True,
          }) is None)
    check("lowercase common word is never a proper-noun candidate",
          a.compute_proper_noun_risk({
              "word": "percussion", "start": 1.0, "end": 1.4,
              "sentence_initial": False,
          }) is None)
    check("lowercase pronoun-like common word is never flagged",
          a.compute_proper_noun_risk({
              "word": "him", "start": 1.0, "end": 1.4,
              "sentence_initial": False,
          }) is None)

    # ------------------------------------------------------------------
    # 3.6c. Masking: fusion never emits masking_check windows; WEAK sound
    # can't drive masking.
    # ------------------------------------------------------------------
    masking_raw = dict(raw)
    masking_raw["panns_windows"] = [
        {"start": 1.0, "end": 3.0, "top_labels": [
            {"raw_label": "Cheering", "score": 0.24},   # WEAK
        ]},
    ]
    masking_raw["transient_feature_windows"] = []
    fused_mask = se.build_sound_fusion_evidence(
        masking_raw,
        {
            "diarization": {"status": "complete", "turns": [
                {"start": 1.2, "end": 2.8, "speaker": "SPEAKER_00"},
            ]},
            "asr_consensus": {"status": "complete", "word_consensus": [
                {"word": "lost", "start": 1.5, "end": 1.7,
                 "state": "conflicting"},
            ], "secondary_only_words": []},
        },
    )
    check("WEAK sound never becomes a masking candidate",
          fused_mask["masking_evidence"]["candidates"] == [],
          json.dumps(fused_mask["masking_evidence"]))
    check("fusion never emits masking_check review windows",
          all(w["type"] != "masking_check"
              for w in fused_mask["review_windows"]),
          json.dumps(fused_mask["review_windows"]))
    check("fusion never emits masking findings",
          all("masking" not in f["claim"].lower()
              for f in fused_mask["findings"]),
          json.dumps(fused_mask["findings"]))

    # ------------------------------------------------------------------
    # 3.6d. WEAK named sound never explains or demotes a transient.
    # ------------------------------------------------------------------
    weak_explain_raw = dict(raw)  # spike at 8.5-10.0s
    weak_explain_raw["panns_windows"] = [
        {"start": 8.0, "end": 10.5, "top_labels": [
            {"raw_label": "Cheering", "score": 0.24},   # WEAK
        ]},
    ]
    fused_weak = se.build_sound_fusion_evidence(weak_explain_raw, {})
    weak_ev = [
        e for e in fused_weak["transients"]["events"]
        if e["unexplained"]
    ]
    check("WEAK named sound leaves the transient unexplained",
          weak_ev and weak_ev[0]["tier"] == sf.STRONG
          and weak_ev[0]["explained_by_named_sound"] is False,
          json.dumps(weak_ev))

    # ------------------------------------------------------------------
    # 3.6e. Cross-shot evidence carries shots: [...] not one forced shot.
    # ------------------------------------------------------------------
    three_shots = [
        {"shot": 1, "start": 0.0, "end": 5.4},
        {"shot": 2, "start": 5.4, "end": 12.8},
        {"shot": 3, "start": 12.8, "end": 15.3},
    ]
    whole_music = {"start": 0.0, "end": 15.3, "duration_sec": 15.3,
                   "tier": sf.MEDIUM, "signals": [{"source": "panns", "score": 0.6}],
                   "raw_labels": ["Music"], "reasons": [],
                   "reviewer_action": "Confirm by listening."}
    enriched_music = se._enrich_music_region(
        whole_music, 1,
        {"clip_baseline_rms_dbfs": -30.0, "clip_duration_sec": 15.3,
         "shots": three_shots},
        [],
    )
    check("whole-clip music gets shots [1,2,3] and no forced single shot",
          enriched_music["shots"] == [1, 2, 3]
          and enriched_music["shot"] is None
          and enriched_music["scope"] == "whole_clip",
          json.dumps(enriched_music))
    check("master shots_for_window lists all crossed shots",
          m.shots_for_window(4.9, 15.3, three_shots) == [1, 2, 3],
          json.dumps(m.shots_for_window(4.9, 15.3, three_shots)))
    cross = m.collect_all_findings(
        {"asr_consensus": {"findings": [
            {"tier": m.CONFLICT, "claim": "rerun cross shot", "action": "listen",
             "window": [4.987, 15.312]},
        ]}},
        context_shots=three_shots,
    )
    check("cross-shot finding gets shots list and no single forced shot",
          cross[0]["shots"] == [1, 2, 3] and cross[0]["shot"] is None,
          json.dumps(cross[0]))

    # ------------------------------------------------------------------
    # 3.6f. Transient regions > 1s are high_energy_acoustic_region.
    # ------------------------------------------------------------------
    long_quiet = [
        {"start": i * 0.5, "end": i * 0.5 + 1.5, "rms_db": -40.0,
         "crest_factor": 3.0, "spectral_flux": 0.001,
         "onset_strength": 1e-5, "energy_change_db": 0.0}
        for i in range(20)
    ]
    two_spikes = [
        {"start": 8.0, "end": 9.0, "rms_db": -12.0, "crest_factor": 14.0,
         "spectral_flux": 0.09, "onset_strength": 0.004, "energy_change_db": 28.0},
        {"start": 9.2, "end": 10.0, "rms_db": -13.0, "crest_factor": 12.0,
         "spectral_flux": 0.08, "onset_strength": 0.003, "energy_change_db": 24.0},
    ]
    merged_events = sf.build_transient_events(long_quiet + two_spikes)
    check("merged >1s event is a high_energy_acoustic_region with peaks",
          len(merged_events) == 1
          and merged_events[0]["kind"] == "high_energy_acoustic_region"
          and len(merged_events[0]["peaks"]) == 2,
          json.dumps(merged_events))
    short_spike = [{"start": 8.5, "end": 9.0, "rms_db": -12.0, "crest_factor": 14.0,
                    "spectral_flux": 0.09, "onset_strength": 0.004,
                    "energy_change_db": 28.0}]
    check("sub-1s event stays a transient",
          sf.build_transient_events(long_quiet + short_spike)[0]["kind"]
          == "transient")

    # ------------------------------------------------------------------
    # 3.6g. Face wording: unavailable evidence != no visible face.
    # ------------------------------------------------------------------
    face_map = {
        "status": "complete",
        "face_worker_status": "unavailable",
        "face_tracks": [],
        "active_speaker_windows": [{
            "start": 1.0, "end": 2.0, "candidates": [], "tier": m.UNKNOWN,
            "reason": "face_evidence_unavailable", "action": "listen",
        }],
        "cluster_to_face_candidates": [],
        "face_to_character_candidates": [],
    }
    face_sec = m.build_speaker_face_mapping(face_map)
    check("unavailable face evidence says evidence unavailable, not 'no face'",
          any("Visible-speaker evidence unavailable" in f["claim"]
              for f in face_sec["findings"])
          and not any("No visible face" in f["claim"] for f in face_sec["findings"]),
          json.dumps(face_sec["findings"]))

    # ------------------------------------------------------------------
    # 3.6h. REVIEW_ME: STRONG is 'HIGH PRIORITY -- strong evidence'.
    # ------------------------------------------------------------------
    from manuscript_audio_speaker_mapping import (
        compute_active_speaker_windows,
    )
    face_windows = compute_active_speaker_windows(
        {"status": "unavailable", "face_tracks": []},
        [(1.0, 2.0)],
    )
    check("speaker mapping marks unavailable face evidence distinctly",
          face_windows and face_windows[0]["reason"] == "face_evidence_unavailable",
          json.dumps(face_windows))

    # ------------------------------------------------------------------
    # 3.6h2. REVIEW_ME heading: STRONG is HIGH PRIORITY, never "safe".
    # ------------------------------------------------------------------
    sections, evidence = m.build_packet()
    sections["ranked_findings"] = [
        {"tier": m.STRONG, "claim": "strong untranscribed speech",
         "action": "listen", "window": [1.0, 2.0], "shot": 1, "shots": [1],
         "section": "coverage_gaps"},
        {"tier": m.MEDIUM, "claim": "medium item", "action": "listen",
         "window": [3.0, 3.5], "shot": 1, "shots": [1], "section": "x"},
    ]
    sections["confidence_summary"] = m.build_confidence_summary(
        sections["ranked_findings"], raw_count=2
    )
    review_me = m.build_review_me(sections, evidence)
    check("REVIEW_ME uses 'HIGH PRIORITY -- strong evidence' heading",
          "HIGH PRIORITY — strong evidence" in review_me
          and "STRONG — safe defaults" not in review_me,
          review_me[:400])

    # ------------------------------------------------------------------
    # 3.6i. setup_vision_windows.ps1 exists for fresh-install face tracking.
    # ------------------------------------------------------------------
    import os
    check("setup_vision_windows.ps1 ships with the repo",
          os.path.exists("setup_vision_windows.ps1"))

    # ------------------------------------------------------------------
    # 10. Media metadata: both sample rates reported.
    # ------------------------------------------------------------------
    media_sec = m.build_media(
        {"media": {"analysis_sample_rate": 16000, "sample_rate": 16000,
                   "duration_sec": 16.0}},
        {"video_name": "v.mp4", "video_sha256": "x",
         "source_sample_rate": 48000, "audio_channels": 2, "audio_codec": "aac"},
    )
    check("both source_sample_rate and analysis_sample_rate are reported",
          media_sec["present"]["analysis_sample_rate"] == 16000
          and media_sec["present"]["source_sample_rate"] == 48000
          and media_sec["present"]["audio_channels"] == 2
          and media_sec["present"]["audio_codec"] == "aac",
          json.dumps(media_sec))

    print("\n=== ALL 3.5 ACCURACY HARDENING CHECKS PASSED ===\n")


if __name__ == "__main__":
    run()
