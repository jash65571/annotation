"""Phase 3E human accuracy-gate regression checks.

Runs with the base interpreter. No media, models, or network are required.
"""

import copy
import json

from manuscript_audio_accuracy_gate import (
    VOCALIZATION_CATEGORIES,
    build_accuracy_gate,
)
from manuscript_audio_qa import run_qa


def check(name, condition, details=""):
    if not condition:
        raise AssertionError(f"FAILED: {name}\n{details}")
    print(f"  ok: {name}")


def complete_checks(heard="speech"):
    return {
        category: {
            "status": (
                "confirmed_heard" if category == heard else "checked_not_heard"
            ),
            "candidate_claim_ids": [],
            "notes": "",
        }
        for category in VOCALIZATION_CATEGORIES
    }


def run():
    print("=== 3E HUMAN ACCURACY GATE VERIFICATION ===\n")
    sections = {
        "locked_task_structure": {
            "characters": ["C1", "C2"],
            "objects": ["O1"],
            "shots": [{"shot": 1, "start": 0.0, "end": 4.0}],
        },
        "ranked_findings": [
            {
                "section": "coverage_gaps",
                "tier": "STRONG",
                "claim": 'C1 speech may include "Hello there".',
                "window": [1.0, 1.8],
                "shot": 1,
                "shots": [1],
                "evidence": ["VAD", "ASR"],
                "action": "Listen and transcribe exactly.",
            },
            {
                "section": "sound_events",
                "tier": "MEDIUM",
                "claim": "C2 may laugh briefly.",
                "window": [2.0, 2.3],
                "shot": 1,
                "shots": [1],
                "evidence": ["sound fusion"],
                "action": "Confirm the source by listening.",
            },
        ],
        "review_queue": [{
            "priority": "high",
            "type": "shot_boundary_speech_check",
            "start": 1.5,
            "end": 2.5,
            "shot": 1,
            "description": "Check whether C1 continues across the cut.",
        }],
    }

    gate = build_accuracy_gate(sections)
    claims = gate["claim_ledger"]["rows"]
    cast = gate["cast_vocalization_audit"]["characters"]
    check("gate blocks unresolved evidence", gate["status"] == "HUMAN_REVIEW_REQUIRED")
    check("ledger includes findings and review windows", len(claims) == 3)
    speech = next(row for row in claims if row["event_type"] == "speech_candidate")
    check("ledger extracts source and transcript candidates",
          speech["possible_sources"] == ["C1"]
          and speech["transcript_candidates"] == ["Hello there"],
          json.dumps(speech))
    c2 = next(row for row in cast if row["character"] == "C2")
    check("Cast audit links laughter evidence to C2",
          c2["checks"]["laughter"]["candidate_claim_ids"],
          json.dumps(c2))
    check("all Cast sound classes start pending",
          all(value["status"] == "pending" for value in c2["checks"].values()))

    previous_claims = copy.deepcopy(gate["claim_ledger"])
    previous_claims["rows"][0]["manual_verification_status"] = "checked"
    previous_claims["rows"][0]["final_decision"] = "confirmed_included"
    previous_claims["rows"][0]["reviewer_notes"] = "Verified in original audio."
    previous_cast = copy.deepcopy(gate["cast_vocalization_audit"])
    previous_cast["characters"][0]["checks"] = complete_checks("speech")
    rerun = build_accuracy_gate(sections, previous_claims, previous_cast)
    preserved = rerun["claim_ledger"]["rows"][0]
    check("stable rows preserve human decisions after rerun",
          preserved["manual_verification_status"] == "checked"
          and preserved["final_decision"] == "confirmed_included"
          and preserved["reviewer_notes"] == "Verified in original audio.")
    check("Cast decisions survive a rerun",
          rerun["cast_vocalization_audit"]["characters"][0]["completion_status"]
          == "complete")

    strict_incomplete = {
        "require_accuracy_gate": True,
        "events": [{
            "id": "e1", "type": "Speech", "source": "C1",
            "transcript": "Hi", "recorded_level": "Moderate",
            "mix_role": "Foreground", "clarity": "Clear",
            "caption_sentence": 'C1 says "Hi".',
        }],
        "final_audio_text": 'C1 says "Hi".',
        "overview_audio": "C1 is the principal speaker.",
        "cast_current": ["C1"],
        "objects_current": [],
        "cast_voice_states": {"C1": "Observed"},
    }
    result = run_qa(strict_incomplete, {"characters": ["C1"], "objects": []})
    codes = {item["code"] for item in result["issues"]}
    check("strict QA blocks unverified events and unreviewed Tone",
          {"event_not_manually_verified", "tone_not_reviewed"}.issubset(codes),
          json.dumps(result))
    check("strict QA blocks a missing Cast audit", "cast_audit_missing" in codes)

    strict_complete = copy.deepcopy(strict_incomplete)
    strict_complete["events"][0]["manual_verified"] = True
    strict_complete["events"][0]["tone_reviewed"] = True
    strict_complete["cast_vocalization_audit"] = {"characters": [{
        "character": "C1",
        "checks": complete_checks("speech"),
    }]}
    result = run_qa(strict_complete, {"characters": ["C1"], "objects": []})
    check("completed strict state clears the new accuracy gate",
          result["status"] == "QA_CLEAR", json.dumps(result))

    not_observed = copy.deepcopy(strict_complete)
    not_observed["cast_voice_states"] = {"C1": "Not observed"}
    result = run_qa(not_observed, {"characters": ["C1"], "objects": []})
    check("confirmed vocalization blocks Not observed",
          any(i["code"] == "not_observed_has_vocalization" for i in result["issues"]),
          json.dumps(result))

    print("\n3E ACCURACY GATE: PASS")


if __name__ == "__main__":
    run()
