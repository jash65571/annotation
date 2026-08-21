"""Regression tests for the 2026-08-21 reviewed 16-second run."""

import json

import manuscript_audio_accuracy_gate as gate
import manuscript_audio_face_worker as face_worker
import manuscript_audio_master as master
import manuscript_audio_seed as seed
import manuscript_audio_sound_events as sounds
import manuscript_audio_sound_vocabulary as vocabulary
from manuscript_audio_speaker_mapping import build_speaker_mapping_evidence


def check(label, condition, details=""):
    if not condition:
        raise AssertionError(f"{label}: {details}")
    print("PASS |", label)


def face_track(face_id, times):
    return {
        "face_id": face_id,
        "first_seen": min(times),
        "last_seen": max(times),
        "frame_count": len(times),
        "points": [
            {"time": time, "mar": 0.1 + index * 0.04}
            for index, time in enumerate(times)
        ],
    }


def main():
    verbose_seed = """
Characters
C1
Voice type
Male
Female
Off-screen male speaker; visual appearance is not available.
Voice
Observed
C2
Voice type
Male
Off-screen male speaker; visual appearance is not available.
Voice
Observed
Objects
O1
Off-screen sound source; visual appearance is not available.
Sound
Makes a sound
Bottles being sat on the table
Audio
Obvious wind sound. Obvious sound of chewing food.
Shot 1of 1
00:00:00.0–00:00:16.0
"""
    parsed = seed.parse_seed_text(verbose_seed)
    check(
        "verbose UI seed preserves C1/C2/O1 locked descriptions",
        all(
            item["description"].startswith("Off-screen")
            for item in parsed["seed_meta"]["characters"]
            + parsed["seed_meta"]["objects"]
        ),
        json.dumps(parsed["seed_meta"], indent=2),
    )
    target_classes = {
        item["class"]
        for item in parsed["seed_meta"]["human_listening_targets"]
    }
    check(
        "seed preserves chewing, wind, and bottle-contact listen targets",
        target_classes == {"chewing", "wind_noise", "bottle_table_contact"},
        str(target_classes),
    )
    no_wind = seed.parse_seed_text("""
C1: narrator
Wind noise
no
No wind noise is audible.
Shot 1: 0-1
""")
    check(
        "wind UI label and explicit no do not create a wind target",
        no_wind["seed_meta"]["human_listening_targets"] == [],
        json.dumps(no_wind["seed_meta"], indent=2),
    )

    rendered = master.build_master_md(
        {}, {}, ui={"characters": {}, "sounds": [], "music": [], "policy": []}
    )
    check(
        "MASTER accepts current-run UI directly and cannot inherit cough",
        "cough detected" not in rendered.lower(),
    )

    face_evidence = {
        "status": "complete",
        "media": {"sampled_fps": 5},
        "face_tracks": [
            face_track("F1", [0.0, 0.2, 0.4, 0.6, 0.8]),
            face_track("F5", [0.2, 0.4]),
        ],
    }
    mapping = build_speaker_mapping_evidence(
        {"status": "complete", "turns": [{
            "start": 0.0, "end": 0.8, "speaker": "SPEAKER_00"
        }], "speaker_labels": ["SPEAKER_00"]},
        {},
        face_evidence,
    )
    candidate_ids = {
        item["face_id"]
        for window in mapping["active_speaker_windows"]
        for item in window.get("candidates", [])
    }
    check(
        "two-frame face flicker is excluded from speaker leads",
        candidate_ids == {"F1"}
        and mapping["face_track_summary"]["discarded_short_track_count"] == 1,
        json.dumps(mapping, indent=2),
    )
    tracked = face_worker._track_faces([
        {"time": 0.0, "faces": [
            {"bbox": [0, 0, 100, 100], "mar": 0.10},
            {"bbox": [300, 0, 100, 100], "mar": 0.10},
        ]},
        {"time": 0.2, "faces": [
            {"bbox": [0, 0, 100, 100], "mar": 0.14},
            {"bbox": [300, 0, 100, 100], "mar": 0.12},
        ]},
        {"time": 0.4, "faces": [
            {"bbox": [0, 0, 100, 100], "mar": 0.18},
        ]},
        {"time": 0.6, "faces": [
            {"bbox": [0, 0, 100, 100], "mar": 0.12},
        ]},
    ])
    check(
        "face worker itself emits only stable track segments",
        len(tracked) == 1 and tracked[0]["frame_count"] == 4,
        json.dumps(tracked, indent=2),
    )

    speech_windows = [(8.75, 9.903), (12.029, 13.143), (13.514, 15.0)]
    for start, end in (
        (9.903, 10.0),
        (11.75, 12.029),
        (13.143, 13.514),
    ):
        event = {
            "start": start,
            "end": end,
            "peaks": [{
                "start": start,
                "end": end,
                "raw_features": {"crest_factor": 5.0, "energy_change_db": 8.0},
            }],
        }
        check(
            f"speech-edge sliver {start:.3f}-{end:.3f} is contextualized",
            sounds._speech_boundary_energy_match(event, speech_windows),
        )
    check(
        "impact-shaped edge remains an SFX candidate",
        not sounds._speech_boundary_energy_match({
            "start": 9.903,
            "end": 10.0,
            "peaks": [{
                "start": 9.903,
                "end": 10.0,
                "raw_features": {"crest_factor": 14.0, "energy_change_db": 28.0},
            }],
        }, speech_windows),
    )

    ledger = gate.build_claim_ledger({
        "raw_findings": [
            {"section": "transients", "tier": "MEDIUM", "window": [1, 1.2],
             "claim": "Unidentified transient at 1-1.2s."},
            {"section": "transients", "tier": "MEDIUM", "window": [2, 2.2],
             "claim": "Speech-associated energy at 2-2.2s; not an independent SFX finding."},
            {"section": "speaker_face_mapping", "tier": "MEDIUM", "window": [3, 4],
             "claim": "F1 is the best visible active-speaker candidate."},
        ],
        "review_queue": [{
            "type": "transient_sfx_check", "priority": "medium",
            "start": 1.0, "end": 1.2,
            "description": "unidentified transient; listen and identify it",
        }],
    })
    check(
        "gate deduplicates queue playback from its atomic transient claim",
        len(ledger) == 3,
        json.dumps(ledger, indent=2),
    )
    check(
        "support-only speech energy and face leads are not stop-ship",
        sum(bool(row["stop_ship"]) for row in ledger) == 1,
        json.dumps(ledger, indent=2),
    )

    check(
        "vocabulary covers chewing and bottle contact",
        vocabulary.map_raw_label("Chewing, mastication")
        == (vocabulary.HUMAN_NONVERBAL, "chewing")
        and vocabulary.map_raw_label("Bottle clink")
        == (vocabulary.OBJECT_SFX, "bottle_contact"),
    )

    print("\nALL 3G REPORTED-REGRESSION TESTS PASSED")


if __name__ == "__main__":
    main()
