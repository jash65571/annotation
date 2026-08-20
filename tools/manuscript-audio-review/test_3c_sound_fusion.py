"""Phase 3C regression: PANNs + CLAP sound/music/ambience fusion.

Runs standalone with no models, no audio, no network:

    python test_3c_sound_fusion.py

Locks the 15 non-negotiable 3C behaviors plus the extra merge / provenance /
fail-soft / music-conflict invariants the spec calls out. Uses the real
fusion, orchestrator, and master modules against synthetic worker fixtures so
every threshold and tier rule is exercised exactly as it runs in production.
"""

import json

import manuscript_audio_sound_fusion as sf
import manuscript_audio_sound_events as se
import manuscript_audio_master as m


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def panns_window(start, end, labels, rms=None):
    w = {"start": start, "end": end, "top_labels": labels}
    if rms is not None:
        w["rms_dbfs"] = rms
    return w


def clap_window(start, end, prompts):
    return {"start": start, "end": end, "prompt_scores": prompts}


def check(name, condition, details=""):
    if not condition:
        msg = f"\n  FAILED: {name}"
        if details:
            msg += f"\n    {details}"
        print(msg)
        raise AssertionError(f"3C VERIFICATION FAILED: {name}")
    print(f"  ok: {name}")


def run():
    print("=== 3C SOUND / MUSIC / AMBIENCE FUSION VERIFICATION ===\n")

    # 1. PANNs strong applause + CLAP strong applause -> STRONG applause.
    panns = [panns_window(1.0, 2.5, [{"raw_label": "Applause", "score": 0.85}])]
    clap = [clap_window(1.0, 2.5, [{"prompt": "an audience applauding", "score": 0.9}])]
    cands = sf.build_sound_event_candidates(panns, clap)
    applause = [c for c in cands if c["candidate_class"] == "applause"]
    check("strong applause (panns+clap) becomes STRONG applause evidence",
          applause and applause[0]["tier"] == sf.STRONG,
          json.dumps(applause))

    # 2. weak PANNs music + weak CLAP music -> WEAK music, absent from UI.
    weak_panns = [panns_window(0.0, 2.0, [{"raw_label": "Music", "score": 0.3}])]
    weak_clap = [clap_window(0.0, 2.0, [{"prompt": "background music", "score": 0.3}])]
    weak_music = sf.build_music_candidates(weak_panns, weak_clap)
    check("weak music stays WEAK",
          weak_music["overall_confidence"] == sf.WEAK,
          json.dumps(weak_music["overall_confidence"]))
    check("weak music carries the explicit do-not-create action",
          weak_music["regions"]
          and "DO NOT CREATE MUSIC EVENT WITHOUT LISTENING"
          in weak_music["regions"][0]["reviewer_action"])

    # 3. strong rhythmicity alone -> cannot create Music.
    rhythmic_only = sf.compute_music_tier(
        None, None, None, duration_sec=3.0, rhythmicity_compatible=True
    )
    check("strong rhythmicity alone cannot create Music",
          rhythmic_only["tier"] == sf.UNKNOWN,
          json.dumps(rhythmic_only))

    # 4. rhythmic clapping evidence -> cannot become Music solely due to rhythm.
    clap_panns = [panns_window(0.0, 2.0, [{"raw_label": "Clapping", "score": 0.9}])]
    clap_clap = [clap_window(0.0, 2.0, [{"prompt": "people clapping", "score": 0.9}])]
    clap_music = sf.build_music_candidates(clap_panns, clap_clap)
    clap_sounds = sf.build_sound_event_candidates(clap_panns, clap_clap)
    check("rhythmic clapping produces no Music region",
          clap_music["regions"] == [])
    check("clapping stays a human-nonverbal Sound candidate, not Music",
          any(c["candidate_class"] == "clapping" and c["group"] == "human_nonverbal"
              for c in clap_sounds))

    # 5. outdoor water ambience -> never maps automatically to Room ambience.
    water_panns = [panns_window(0.0, 3.0, [{"raw_label": "Ocean", "score": 0.8}])]
    water_clap = [clap_window(0.0, 3.0, [{"prompt": "ocean waves and water", "score": 0.8}])]
    water_cands = sf.build_sound_event_candidates(water_panns, water_clap)
    water = [c for c in water_cands if c["group"] == "ambience"]
    check("outdoor water ambience is detected",
          water, json.dumps(water_cands))
    if water:
        amb = sf.build_ambience_candidate(water[0])
        check("outdoor water ambience never maps to Room ambience",
              amb["ui_source_candidate"] != "Room ambience"
              and amb["ui_source_candidate"] == "Unidentified sound",
              json.dumps(amb))

    # 6. sound overlaps clear speech -> masking not automatically asserted.
    mask_ok = sf.evaluate_masking(
        {"start": 0.0, "end": 2.0}, [(0.5, 1.5)], asr_words_lost=False
    )
    check("overlap with intact speech is not auto-asserted as masking",
          mask_ok["masking"] == "masking_not_supported"
          and mask_ok["tier"] == sf.MEDIUM,
          json.dumps(mask_ok))
    mask_unknown = sf.evaluate_masking(
        {"start": 0.0, "end": 2.0}, [(0.5, 1.5)], asr_words_lost=None
    )
    check("overlap with unknown ASR outcome stays UNKNOWN, not 'masked'",
          mask_unknown["masking"] == "UNKNOWN",
          json.dumps(mask_unknown))

    # 7. group cheering -> no automatic single C# source.
    cheer_attrib = sf.attribute_human_source(
        {"start": 0.0, "end": 2.0, "tier": "STRONG"}, visual_candidates=[]
    )
    check("group cheering gets no automatic single C# source",
          cheer_attrib["source_candidate"] == "Unidentified sound",
          json.dumps(cheer_attrib))

    # 8. nonverbal human reaction -> never converted to Speech.
    laugh_panns = [panns_window(0.0, 1.5, [{"raw_label": "Laughter", "score": 0.8}])]
    laugh_clap = [clap_window(0.0, 1.5, [{"prompt": "a person laughing", "score": 0.8}])]
    laugh_cands = sf.build_sound_event_candidates(laugh_panns, laugh_clap)
    laughter = [c for c in laugh_cands if c["candidate_class"] == "laughter"]
    check("laughter is detected as a nonverbal Sound",
          laughter and laughter[0]["is_nonverbal"]
          and laughter[0]["group"] == "human_nonverbal",
          json.dumps(laugh_cands))
    check("laughter is never labeled Speech",
          all(c["group"] != "speech" for c in laugh_cands))

    # 9. zero-word diarization cluster + laughter evidence -> Sound only.
    cluster_claim = sf.classify_vocal_cluster(0, laughter)
    check("zero-word cluster + laughter yields a Sound claim, not dialogue",
          cluster_claim is not None
          and "laughter" in cluster_claim["candidate_classes"]
          and "not dialogue" in cluster_claim["action"],
          json.dumps(cluster_claim))
    check("a cluster with confirmed words is not hijacked into a Sound claim",
          sf.classify_vocal_cluster(5, laughter) is None)

    # 10. visible object movement without acoustic event -> no O# suggestion.
    obj_motion_only = sf.attribute_object_source(
        {"start": 0.0, "end": 1.0, "tier": "MEDIUM"},
        object_interactions=[{
            "object_id": "O1", "start": 0.0, "end": 1.0,
            "acoustic_class_match": False, "temporal_overlap": True,
        }],
    )
    check("object movement without acoustic event yields no O# source",
          obj_motion_only["source_candidate"] == "Unidentified sound",
          json.dumps(obj_motion_only))

    # 11. acoustic click without matching visible object -> Unidentified sound.
    click_attrib = sf.attribute_object_source(
        {"start": 0.0, "end": 1.0, "tier": "MEDIUM"}, object_interactions=[]
    )
    check("acoustic click without visible object stays Unidentified sound",
          click_attrib["source_candidate"] == "Unidentified sound",
          json.dumps(click_attrib))

    # 12. PANNs/CLAP unavailable -> valid UNKNOWN evidence, pipeline viable.
    unavail = se.build_sound_fusion_evidence(
        {"status": "unavailable", "error": "no .venv-audio-events"}
    )
    check("unavailable worker yields a valid UNKNOWN contract",
          unavail["status"] == "unavailable"
          and unavail["music"]["overall_confidence"] == "UNKNOWN"
          and unavail["sound_events"]["candidates"] == [],
          json.dumps(unavail["status"]))
    m_sound = m.build_fusion_sound_events(unavail)
    check("master degrades to a single UNKNOWN sound finding",
          len(m_sound["findings"]) == 1
          and m_sound["findings"][0]["tier"] == m.UNKNOWN)
    m_music = m.build_music(unavail)
    check("master music reports 'No supported music candidate' as UNKNOWN",
          any("No supported music" in f["claim"] and f["tier"] == m.UNKNOWN
              for f in m_music["findings"]))

    # 13/14/15. UI-suggestion gating and evidence-id preservation.
    ui_fusion = {
        "status": "complete",
        "sound_events": {"candidates": [
            {"id": "snd-001", "semantic_label": "clapping", "tier": "STRONG",
             "ui_source_candidate": "Unidentified sound",
             "recorded_level_candidate": "Loud",
             "mix_role_candidate": "Foreground",
             "description_candidate": "clapping detected",
             "relationship_candidates": ["throughout the shot"]},
            {"id": "snd-002", "semantic_label": "click", "tier": "MEDIUM",
             "ui_source_candidate": "Unidentified sound",
             "recorded_level_candidate": "Quiet",
             "mix_role_candidate": "Supporting",
             "description_candidate": "click detected",
             "relationship_candidates": []},
            {"id": "snd-003", "semantic_label": "whoop", "tier": "WEAK",
             "ui_source_candidate": "Unidentified sound",
             "description_candidate": None,
             "relationship_candidates": []},
            {"id": "snd-004", "semantic_label": "cheering", "tier": "CONFLICT",
             "ui_source_candidate": "Unidentified sound",
             "description_candidate": None,
             "relationship_candidates": []},
        ]},
        "ambience": {"candidates": []},
        "music": {"regions": [
            {"id": "mus-001", "tier": "WEAK", "recorded_level_candidate": None,
             "mix_role_candidate": None},
        ]},
    }
    ui = m.build_ui_suggestions({}, ui_fusion)
    sound_tiers = {s["confidence"] for s in ui["sounds"]}
    sound_ids = {s["evidence_ids"][0] for s in ui["sounds"]}
    check("WEAK candidates are absent from UI suggestions",
          "WEAK" not in sound_tiers)
    check("CONFLICT candidates are absent from UI suggestions",
          "CONFLICT" not in sound_tiers)
    check("MEDIUM/STRONG candidates are allowed in UI suggestions",
          {"STRONG", "MEDIUM"} <= sound_tiers,
          json.dumps(sorted(sound_tiers)))
    check("UI suggestions preserve evidence ids",
          sound_ids == {"snd-001", "snd-002"},
          json.dumps(sorted(sound_ids)))
    check("weak music never reaches UI music suggestions",
          ui["music"] == [])
    check("UI sound suggestions never produce caption/final prose",
          all("caption_sentence" not in s and "final_audio_text" not in s
              for s in ui["sounds"]))

    # --- extra: adjacent-window merging / smoothing / non-merge ---
    merged = sf.merge_intervals([(0.0, 1.0), (1.1, 2.0)], join_gap=0.35)
    check("adjacent windows within join gap are merged",
          merged == [(0.0, 2.0)], json.dumps(merged))
    apart = sf.merge_intervals([(0.0, 1.0), (3.0, 4.0)], join_gap=0.35)
    check("separate sound bursts are not incorrectly merged",
          apart == [(0.0, 1.0), (3.0, 4.0)], json.dumps(apart))

    cont_panns = [
        panns_window(0.0, 1.5, [{"raw_label": "Wind", "score": 0.6}]),
        panns_window(0.5, 2.0, [{"raw_label": "Wind", "score": 0.6}]),
    ]
    cont_cands = sf.build_sound_event_candidates(cont_panns, [])
    wind = [c for c in cont_cands if c["candidate_class"] == "wind"]
    check("continuous overlapping wind windows are smoothed into one event",
          len(wind) == 1 and wind[0]["end"] == 2.0,
          json.dumps(cont_cands))

    # --- extra: raw provenance preserved ---
    raw = {
        "status": "complete",
        "runtime_sec": 2.0,
        "media": {"duration_sec": 10.0, "sample_rate": 16000},
        "panns_windows": panns,
        "clap_windows": clap,
        "provenance": {
            "panns": {"model": "Cnn14", "runtime_sec": 1.0},
            "clap": {"model": "clap-htsat-unfused", "runtime_sec": 1.0},
        },
    }
    fused = se.build_sound_fusion_evidence(raw, {})
    check("fused evidence preserves worker provenance",
          fused["worker"]["panns"].get("model") == "Cnn14"
          and fused["worker"]["clap"].get("model") == "clap-htsat-unfused",
          json.dumps(fused["worker"]))
    check("fused candidate preserves raw labels",
          fused["sound_events"]["candidates"]
          and fused["sound_events"]["candidates"][0]["raw_labels"],
          json.dumps(fused["sound_events"]["candidates"]))
    check("fused evidence never produces final-caption prose",
          "caption_sentence" not in json.dumps(fused)
          and "final_audio_text" not in json.dumps(fused))

    # --- extra: worker malformed output -> fail-soft ---
    malformed = se.build_sound_fusion_evidence("not-a-dict")
    check("malformed worker output degrades to failed contract",
          malformed["status"] == "failed", json.dumps(malformed["status"]))
    failed = se.build_sound_fusion_evidence(
        {"status": "failed", "error": "model load failed"}
    )
    check("failed worker output degrades to failed contract, never raises",
          failed["status"] == "failed" and failed["error"] == "model load failed")

    # --- extra: music model disagreement -> CONFLICT ---
    conflict = sf.compute_music_tier(
        0.8, None, 0.8, duration_sec=3.0
    )
    check("music vs non-music model disagreement becomes CONFLICT",
          conflict["tier"] == sf.CONFLICT, json.dumps(conflict))

    print("\n=== ALL 3C REGRESSION CHECKS PASSED ===\n")


if __name__ == "__main__":
    run()
