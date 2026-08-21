"""Phase 3F run-provenance and observed-output regressions.

These checks lock failures found in the 16-second snow-scene audit. They use
only the standard library and do not require media models or network access.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import manuscript_audio_master as master
import manuscript_audio_pipeline as pipeline
import manuscript_audio_sound_events as sound_events
import manuscript_audio_sound_fusion as sound_fusion
from manuscript_audio_accuracy_gate import build_accuracy_gate
from manuscript_audio_face_worker import _dedupe_face_detections
from manuscript_audio_sound_vocabulary import AMBIENCE, map_raw_label
from manuscript_audio_voice import load_speaker_map


def check(name, condition, details=""):
    if not condition:
        raise AssertionError(f"FAILED: {name}\n{details}")
    print(f"  ok: {name}")


def run():
    print("=== 3F RUN PROVENANCE VERIFICATION ===\n")

    with TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        old_context = pipeline.CONTEXT
        try:
            pipeline.CONTEXT = temp / "task_context.json"
            pipeline.CONTEXT.write_text(
                json.dumps({"video_sha256": "old-video", "shots": []}),
                encoding="utf-8",
            )
            try:
                pipeline.require_current_task_context("new-video")
                stale_rejected = False
            except RuntimeError:
                stale_rejected = True
            check("a previous clip's task seed fails closed", stale_rejected)

            pipeline.CONTEXT.write_text(
                json.dumps({"video_sha256": "new-video", "shots": []}),
                encoding="utf-8",
            )
            current = pipeline.require_current_task_context("new-video")
            check("a hash-matched task seed is accepted",
                  current["video_sha256"] == "new-video")
        finally:
            pipeline.CONTEXT = old_context

        speaker_map = temp / "speaker_map.json"
        speaker_map.write_text(
            json.dumps({
                "video_sha256": "old-video",
                "segments": {"0": "C1"},
            }),
            encoding="utf-8",
        )
        check("a stale speaker map cannot create Cast attribution",
              load_speaker_map(speaker_map, "new-video") == {})
        check("a current-video speaker map remains usable",
              load_speaker_map(speaker_map, "old-video") == {0: "C1"})

    evidence = {
        "media": {"duration_sec": 16.0},
        "whisperx_segments": [{"start": 6.38, "end": 15.92}],
    }
    early_gap = {
        "asr_last_end_sec": 15.92,
        "untranscribed_regions": [{"start": 4.891, "end": 5.566}],
    }
    check("an early ASR gap is not mislabeled as unfinished tail speech",
          master.build_clip_boundaries(evidence, early_gap)["findings"] == [])

    late_gap = {
        "asr_last_end_sec": 15.7,
        "untranscribed_regions": [{"start": 15.72, "end": 15.98}],
    }
    late_findings = master.build_clip_boundaries(evidence, late_gap)["findings"]
    check("independently detected speech after the last word is still flagged",
          len(late_findings) == 1 and late_findings[0]["tier"] == "STRONG")

    check("AudioSet 'Train' no longer becomes rain", map_raw_label("Train") is None)
    check("whole-token rain mapping still works",
          map_raw_label("Rain") == (AMBIENCE, "rain"))
    check("the token 'human' no longer becomes humming",
          map_raw_label("Human voice") is None)

    source_media = pipeline.parse_source_streams([
        {
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "avg_frame_rate": "30/1",
            "nb_frames": "480",
        },
        {
            "codec_type": "audio",
            "codec_name": "aac",
            "sample_rate": "48000",
            "channels": 2,
        },
    ])
    check("original video integrity metadata is preserved",
          source_media["resolution"] == "1920x1080"
          and source_media["fps"] == 30.0
          and source_media["frame_count"] == 480
          and source_media["video_codec"] == "h264")
    check("original audio metadata remains separate from analysis audio",
          source_media["source_sample_rate"] == 48000
          and source_media["audio_channels"] == 2
          and source_media["audio_codec"] == "aac")

    sections = {
        "locked_task_structure": {"characters": ["C1"]},
        "raw_findings": [
            {
                "section": "asr_consensus",
                "tier": "STRONG",
                "claim": "Lexical agreement between ASR models: 90%.",
                "window": None,
                "evidence": ["engineering metric"],
                "action": "Review conflicts.",
            },
            {
                "section": "coverage_gaps",
                "tier": "STRONG",
                "claim": "UNTRANSCRIBED_SPEECH: speech present but no words.",
                "window": [4.891, 5.566],
                "evidence": ["diarization"],
                "action": "Listen and transcribe.",
            },
            {
                "section": "coverage_gaps",
                "tier": "STRONG",
                "claim": "UNTRANSCRIBED_SPEECH: speech present but no words.",
                "window": [4.891, 5.566],
                "evidence": ["diarization"],
                "action": "Listen and transcribe.",
            },
        ],
        "ranked_findings": [{
            "section": "coverage_gaps",
            "tier": "STRONG",
            "claim": "9 related signals in one window.",
            "window": [2.0, 6.0],
        }],
        "review_queue": [{
            "priority": "medium",
            "type": "tone_delivery_check",
            "start": 6.38,
            "end": 12.54,
            "description": "Listen for actual tone and delivery.",
        }],
    }
    gate = build_accuracy_gate(sections, video_sha256="new-video")
    ledger = gate["claim_ledger"]
    claims = ledger["rows"]
    check("the ledger is bound to the analyzed video",
          ledger["video_sha256"] == "new-video")
    check("the ledger uses atomic claims, deduplicates them, and omits metrics",
          len(claims) == 2
          and any("UNTRANSCRIBED_SPEECH" in row["claim_candidate"] for row in claims)
          and all("related signals" not in row["claim_candidate"] for row in claims)
          and all("Lexical agreement" not in row["claim_candidate"] for row in claims),
          json.dumps(claims, indent=2))
    check("the Cast audit is bound to the same video",
          gate["cast_vocalization_audit"]["video_sha256"] == "new-video")

    # Browser-grid run: two independent classifiers called 7.5-10.0s a
    # cough, but most of the region is independently covered by speech.
    cough = sound_fusion.build_sound_event_candidates(
        [{"start": 7.5, "end": 10.0, "top_labels": [
            {"raw_label": "Cough", "score": 0.5301},
        ]}],
        [{"start": 7.5, "end": 10.0, "prompt_scores": [
            {"prompt": "a person coughing", "score": 0.6482},
        ]}],
        speech_windows=[(8.54, 11.08)],
    )
    cough = [c for c in cough if c["candidate_class"] == "cough"]
    check("long vocal classifications dominated by speech become conflicts",
          cough and cough[0]["tier"] == sound_fusion.CONFLICT
          and cough[0]["speech_contamination_conflict"] is True
          and cough[0]["speech_overlap_ratio"] >= 0.55,
          json.dumps(cough, indent=2))

    cough_raw = {
        "status": "complete",
        "media": {"duration_sec": 16.0},
        "panns_windows": [{"start": 7.5, "end": 10.0, "top_labels": [
            {"raw_label": "Cough", "score": 0.5301},
        ]}],
        "clap_windows": [{"start": 7.5, "end": 10.0, "prompt_scores": [
            {"prompt": "a person coughing", "score": 0.6482},
        ]}],
    }
    cough_fused = sound_events.build_sound_fusion_evidence(
        cough_raw,
        {"diarization": {"status": "complete", "turns": [
            {"start": 8.54, "end": 11.08, "speaker": "SPEAKER_00"},
        ]}},
    )
    cough_ui = master.build_ui_suggestions({}, cough_fused)
    check("speech-contaminated cough evidence cannot reach UI suggestions",
          cough_ui["sounds"] == []
          and cough_fused["sound_events"]["candidates"][0]["tier"]
          == sound_fusion.CONFLICT,
          json.dumps(cough_fused["sound_events"], indent=2))

    quiet = [
        {"start": i * 0.5, "end": i * 0.5 + 1.5, "rms_db": -40.0,
         "crest_factor": 3.0, "spectral_flux": 0.001,
         "onset_strength": 1e-5, "energy_change_db": 0.0}
        for i in range(20)
    ]
    speech_energy_raw = {
        "status": "complete",
        "media": {"duration_sec": 12.0},
        "panns_windows": [],
        "clap_windows": [],
        "transient_feature_windows": quiet + [{
            "start": 8.5, "end": 10.0, "rms_db": -20.0,
            "crest_factor": 5.0, "spectral_flux": 0.04,
            "onset_strength": 0.001, "energy_change_db": 8.0,
        }],
    }
    speech_energy = sound_events.build_sound_fusion_evidence(
        speech_energy_raw,
        {"diarization": {"status": "complete", "turns": [
            {"start": 8.4, "end": 10.5, "speaker": "SPEAKER_00"},
        ]}},
    )
    speech_events = [
        event for event in speech_energy["transients"]["events"]
        if event.get("speech_associated")
    ]
    check("long non-impact energy dominated by speech is not an SFX lead",
          speech_events
          and speech_events[0]["kind"] == "speech_associated_energy"
          and not any(w["type"] == "transient_sfx_check"
                      for w in speech_energy["review_windows"]),
          json.dumps(speech_energy, indent=2))

    faces = _dedupe_face_detections([
        {"bbox": [100.0, 100.0, 200.0, 200.0], "mar": 0.1},
        {"bbox": [105.0, 105.0, 195.0, 195.0], "mar": 0.2},
        {"bbox": [600.0, 100.0, 180.0, 180.0], "mar": 0.3},
    ])
    check("overlapping tile detections dedupe without losing distinct faces",
          len(faces) == 2)

    print("\n3F RUN PROVENANCE: PASS")


if __name__ == "__main__":
    run()
