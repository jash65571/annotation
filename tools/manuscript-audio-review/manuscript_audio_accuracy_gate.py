"""Human-verification gate for Manuscript II audio evidence.

The analysis pipeline produces leads, not final facts. This module turns the
normalized master packet into two editable, stable artifacts:

* a claim ledger with one decision row per machine lead or review window;
* a per-character audit that forces speech and nonverbal checks separately.

Rows have stable IDs. Manual decisions from an older artifact survive a rerun
when the underlying claim remains unchanged.
"""

from hashlib import sha1
import json
import re


SOURCE_ID = re.compile(r"\b[CO]\d+\b", re.IGNORECASE)
QUOTED_TEXT = re.compile(r'["“]([^"”]+)["”]')

VOCALIZATION_CATEGORIES = (
    "speech",
    "laughter",
    "chuckle",
    "gasp",
    "sigh",
    "scream",
    "cheer",
    "cough",
    "humming",
    "audible_breath",
    "wordless_reaction",
    "other_vocalization",
    "physical_sound",
)

DECISIONS = {
    None,
    "confirmed_included",
    "heard_uncertain_marked",
    "checked_rejected",
    "not_applicable",
}
VERIFICATION_STATES = {"pending", "checked"}
AUDIT_STATES = {
    "pending",
    "checked_not_heard",
    "confirmed_heard",
    "not_applicable",
}

EVENT_SECTIONS = {
    "coverage_gaps": "speech_candidate",
    "asr_consensus": "transcript_candidate",
    "speaker_face_mapping": "speaker_attribution_candidate",
    "speaker_clusters": "speaker_attribution_candidate",
    "character_mapping": "speaker_attribution_candidate",
    "voice_profiles": "voice_profile_candidate",
    "sound_events": "sound_candidate",
    "sound_events_ast": "sound_candidate",
    "music": "music_candidate",
    "ambience": "ambience_candidate",
    "transients": "transient_candidate",
    "recording_defects": "recording_defect_candidate",
    "overlap_masking": "overlap_masking_candidate",
    "clip_boundaries": "clip_boundary_candidate",
}

NON_MEDIA_METRIC_PREFIXES = (
    "lexical agreement between asr models:",
    "high-confidence cross-model confirmation:",
)


def _stable_id(prefix, payload):
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return f"{prefix}_{sha1(raw.encode('utf-8')).hexdigest()[:12]}"


def _window(value):
    if not value or len(value) != 2:
        return None
    try:
        return [round(float(value[0]), 3), round(float(value[1]), 3)]
    except (TypeError, ValueError):
        return None


def _sources(text):
    return sorted({match.upper() for match in SOURCE_ID.findall(text or "")})


def _transcripts(text):
    return [item.strip() for item in QUOTED_TEXT.findall(text or "") if item.strip()]


def _prior_rows(value, key="rows"):
    if isinstance(value, dict):
        value = value.get(key, [])
    if not isinstance(value, list):
        return {}
    return {row.get("id"): row for row in value if isinstance(row, dict) and row.get("id")}


def _preserve_decision(row, prior):
    old = prior.get(row["id"], {})
    verification = old.get("manual_verification_status")
    decision = old.get("final_decision")
    if verification in VERIFICATION_STATES:
        row["manual_verification_status"] = verification
    if decision in DECISIONS:
        row["final_decision"] = decision
    if isinstance(old.get("reviewer_notes"), str):
        row["reviewer_notes"] = old["reviewer_notes"]
    row["stop_ship"] = not (
        row["manual_verification_status"] == "checked"
        and row["final_decision"] is not None
    ) and row["review_priority"] in {"high", "medium"}
    return row


def build_claim_ledger(sections, previous=None):
    """Build stable claim rows from normalized findings and review windows."""
    prior = _prior_rows(previous)
    rows = []
    seen = set()

    # The ranked report deliberately groups nearby signals for readability.
    # A grouped sentence such as "9 related signals in one window" is not a
    # fact a reviewer can confirm or reject. The ledger therefore consumes
    # the atomic raw findings when available and only falls back to ranked
    # findings for older packets/tests.
    findings = sections.get("raw_findings")
    if not isinstance(findings, list):
        findings = sections.get("ranked_findings", []) or []

    for finding in findings:
        section = finding.get("section", "unknown")
        claim = str(finding.get("claim") or "")
        window = _window(finding.get("window"))
        if (
            section == "asr_consensus"
            and window is None
            and claim.strip().lower().startswith(NON_MEDIA_METRIC_PREFIXES)
        ):
            # Useful report diagnostics, but not media claims and therefore
            # never human-verification blockers.
            continue
        dedupe_key = (
            section,
            claim,
            tuple(window or []),
            finding.get("shot"),
            tuple(finding.get("shots", []) or []),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        payload = {
            "section": section,
            "claim": claim,
            "window": window,
            "shot": finding.get("shot"),
            "shots": finding.get("shots", []),
        }
        tier = str(finding.get("tier") or "UNKNOWN").upper()
        priority = "high" if tier in {"STRONG", "CONFLICT"} else (
            "medium" if tier == "MEDIUM" else "low"
        )
        row = {
            "id": _stable_id("claim", payload),
            "origin": "ranked_finding",
            "section": section,
            "event_type": EVENT_SECTIONS.get(section, "global_check"),
            "shot": finding.get("shot"),
            "shots": finding.get("shots", []) or [],
            "private_window": window,
            "possible_sources": _sources(claim),
            "transcript_candidates": _transcripts(claim),
            "claim_candidate": claim,
            "confidence_tier": tier,
            "review_priority": priority,
            "evidence": finding.get("evidence", []) or [],
            "review_action": finding.get("action"),
            "manual_verification_status": "pending",
            "final_decision": None,
            "reviewer_notes": "",
            "stop_ship": priority in {"high", "medium"},
        }
        rows.append(_preserve_decision(row, prior))

    for item in sections.get("review_queue", []) or []:
        window = _window([item.get("start"), item.get("end")])
        description = str(item.get("description") or "")
        payload = {
            "type": item.get("type"),
            "description": description,
            "window": window,
            "shot": item.get("shot"),
            "shots": item.get("shots", []),
        }
        priority = str(item.get("priority") or "medium").lower()
        row = {
            "id": _stable_id("queue", payload),
            "origin": "review_queue",
            "section": "review_queue",
            "event_type": item.get("type") or "manual_listen",
            "shot": item.get("shot"),
            "shots": item.get("shots", []) or [],
            "private_window": window,
            "possible_sources": _sources(description),
            "transcript_candidates": _transcripts(description),
            "claim_candidate": description,
            "confidence_tier": "UNKNOWN",
            "review_priority": priority,
            "evidence": ["targeted review queue"],
            "review_action": "Loop the original audio and record a final decision.",
            "manual_verification_status": "pending",
            "final_decision": None,
            "reviewer_notes": "",
            "stop_ship": priority in {"high", "medium"},
        }
        rows.append(_preserve_decision(row, prior))

    rows.sort(key=lambda row: (
        {"high": 0, "medium": 1, "low": 2}.get(row["review_priority"], 3),
        (row["private_window"] or [float("inf")])[0],
        row["id"],
    ))
    return rows


def _candidate_categories(claim):
    text = (claim or "").lower()
    mapping = {
        "speech": ("speech", "says", "word", "transcript", "voice"),
        "laughter": ("laugh",),
        "chuckle": ("chuckle",),
        "gasp": ("gasp",),
        "sigh": ("sigh",),
        "scream": ("scream", "yell"),
        "cheer": ("cheer",),
        "cough": ("cough",),
        "humming": ("hum",),
        "audible_breath": ("breath", "exhale", "inhale"),
        "wordless_reaction": ("wordless", "reaction"),
        "physical_sound": ("clap", "shuffle", "handling", "footstep"),
    }
    return {
        category
        for category, terms in mapping.items()
        if any(term in text for term in terms)
    }


def build_cast_audit(sections, claim_rows, previous=None):
    """Require every carried-over character to be checked by sound class."""
    if isinstance(previous, dict):
        previous = previous.get("characters", [])
    previous = previous if isinstance(previous, list) else []
    prior = {
        row.get("character"): row
        for row in previous
        if isinstance(row, dict) and row.get("character")
    }
    characters = sections.get("locked_task_structure", {}).get("characters", []) or []
    output = []

    for character in characters:
        cid = str(character).upper()
        related = [
            row for row in claim_rows
            if cid in row.get("possible_sources", [])
            or cid in str(row.get("claim_candidate") or "").upper()
        ]
        category_leads = {category: [] for category in VOCALIZATION_CATEGORIES}
        for row in related:
            for category in _candidate_categories(row.get("claim_candidate")):
                category_leads[category].append(row["id"])

        old_checks = prior.get(cid, {}).get("checks", {})
        checks = {}
        for category in VOCALIZATION_CATEGORIES:
            old = old_checks.get(category, {}) if isinstance(old_checks, dict) else {}
            state = old.get("status", "pending")
            if state not in AUDIT_STATES:
                state = "pending"
            checks[category] = {
                "status": state,
                "candidate_claim_ids": sorted(set(category_leads[category])),
                "notes": old.get("notes", "") if isinstance(old, dict) else "",
            }

        pending = [name for name, value in checks.items() if value["status"] == "pending"]
        confirmed_vocal = [
            name for name, value in checks.items()
            if name != "physical_sound" and value["status"] == "confirmed_heard"
        ]
        physical = checks["physical_sound"]["status"] == "confirmed_heard"
        if pending:
            voice_guidance = "Do not finalize the Voice state until every category is checked."
        elif confirmed_vocal:
            voice_guidance = "Use the briefest truthful Observed state; add Sound events when required."
        elif physical:
            voice_guidance = "Voice may be Not observed; add a separate Character Sound statement/event."
        else:
            voice_guidance = "Not observed is supportable only after the final full-clip listen."

        output.append({
            "character": cid,
            "checks": checks,
            "completion_status": "pending" if pending else "complete",
            "pending_categories": pending,
            "related_claim_ids": [row["id"] for row in related],
            "voice_state_guidance": voice_guidance,
        })
    return output


def build_accuracy_gate(
    sections,
    previous_claims=None,
    previous_cast=None,
    video_sha256=None,
):
    claims = build_claim_ledger(sections, previous_claims)
    cast = build_cast_audit(sections, claims, previous_cast)
    unresolved_claims = [row["id"] for row in claims if row.get("stop_ship")]
    incomplete_cast = [
        row["character"] for row in cast if row["completion_status"] != "complete"
    ]
    return {
        "status": (
            "HUMAN_REVIEW_REQUIRED"
            if unresolved_claims or incomplete_cast
            else "HUMAN_VERIFICATION_COMPLETE"
        ),
        "policy": {
            "machine_evidence_is_not_final_truth": True,
            "numeric_windows_are_private_review_aids": True,
            "delivery_blocked_by_unresolved_high_or_medium_claims": True,
            "delivery_blocked_by_incomplete_cast_audit": True,
        },
        "claim_ledger": {
            "schema_version": 2,
            "video_sha256": video_sha256,
            "rows": claims,
        },
        "cast_vocalization_audit": {
            "schema_version": 2,
            "video_sha256": video_sha256,
            "characters": cast,
        },
        "unresolved_stop_ship_claim_ids": unresolved_claims,
        "incomplete_cast_characters": incomplete_cast,
        "summary": {
            "claim_count": len(claims),
            "stop_ship_claim_count": len(unresolved_claims),
            "character_count": len(cast),
            "incomplete_character_count": len(incomplete_cast),
        },
    }
