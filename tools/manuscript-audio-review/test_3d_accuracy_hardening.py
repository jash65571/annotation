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
