"""Phase 3C orchestrator: PANNs + CLAP sound/music/ambience evidence.

Runs the isolated PANNs/CLAP worker under `.venv-audio-events`, loads the raw
per-window scores, fuses them against existing supporting evidence
(diarization/VAD speech windows, ASR consensus, face/speaker mapping, locked
shot ranges), and writes ONE normalized evidence artifact:

    analysis/sound_fusion_evidence.json

Design rules (carried over from 3A/3B, non-negotiable):

- Fail-soft. A missing environment, missing worker, model-load failure,
  inference failure, or malformed output must never break the main pipeline.
  It degrades to a well-formed `status: "unavailable"/"failed"` artifact and
  the base packet keeps generating.
- Pure stdlib under `.venv-review`. This module never imports
  torch/PANNs/CLAP; the heavy models live only in the worker
  (`.venv-audio-events`). It only *shells out* to the worker.
- This module fuses and normalizes; it never turns a score into a Manuscript
  fact. WEAK/CONFLICT/UNKNOWN candidates never reach UI suggestions, overlap
  never becomes automatic masking, and sound alone never selects a character
  or object source.

The per-window score fusion lives in manuscript_audio_sound_fusion.py (pure
logic, unit-testable with no models). This module is only responsible for:
run the worker -> load raw scores -> feed fusion -> enrich with supporting
evidence -> write the normalized artifact.
"""

from pathlib import Path
import json
import statistics
import subprocess
import time

from manuscript_audio_sound_fusion import (
    STRONG, MEDIUM, WEAK, CONFLICT, UNKNOWN,
    build_sound_event_candidates,
    build_music_candidates,
    build_ambience_candidate,
    attribute_human_source,
    attribute_object_source,
    evaluate_masking,
    estimate_recorded_level,
    estimate_mix_role,
    describe_relationships,
    build_transient_events,
)
from manuscript_audio_sound_vocabulary import (
    HUMAN_NONVERBAL, AMBIENCE, OBJECT_SFX,
    UI_SOURCE_MAP, DEFAULT_UI_SOURCE,
)


ROOT = Path(__file__).resolve().parent

AUDIO_EVENTS_PYTHON = ROOT / ".venv-audio-events" / "Scripts" / "python.exe"
SOUND_EVENTS_WORKER = ROOT / "manuscript_audio_sound_events_worker.py"
SOUND_EVENTS_RAW = ROOT / "analysis" / "sound_events_raw.json"
SOUND_FUSION_EVIDENCE = ROOT / "analysis" / "sound_fusion_evidence.json"

# Inputs (all optional -- each degrades to an empty default).
AUDIO = ROOT / "analysis" / "audio.wav"
DIARIZATION = ROOT / "analysis" / "diarization_evidence.json"
VAD = ROOT / "analysis" / "vad_speech_regions.json"
ASR_CONSENSUS = ROOT / "analysis" / "asr_consensus_evidence.json"
SPEAKER_MAPPING = ROOT / "analysis" / "speaker_mapping_evidence.json"
FACE_TRACKS = ROOT / "analysis" / "face_track_evidence.json"
EVIDENCE = ROOT / "analysis" / "manuscript_audio_evidence.json"
CONTEXT = ROOT / "task_context.json"

SCHEMA_VERSION = 1

_TIER_RANK = {UNKNOWN: 0, WEAK: 1, CONFLICT: 1, MEDIUM: 2, STRONG: 3}


# ---------------------------------------------------------------------------
# Loading helpers (all fail-soft)
# ---------------------------------------------------------------------------

def load_json(path, default=None):
    path = Path(path)

    if not path.exists():
        return default if default is not None else {}

    try:
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default if default is not None else {}


def _independent_speech_windows(diarization, vad):
    """Same precedence as the master aggregator: diarization turns when
    complete, else VAD regions, else nothing."""
    if diarization and diarization.get("status") == "complete":
        turns = diarization.get("turns", [])
        if turns:
            return [(float(t["start"]), float(t["end"])) for t in turns]
    if vad and vad.get("status") == "complete":
        regions = vad.get("regions", [])
        if regions:
            return [(float(r["start"]), float(r["end"])) for r in regions]
    return []


def _overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _asr_words_lost_in_window(asr_consensus, start, end):
    """None (unknown / not evaluated), False (coverage stayed intact), or
    True (words disappeared/disagreed during the overlap)."""
    if not asr_consensus or asr_consensus.get("status") != "complete":
        return None

    for w in asr_consensus.get("word_consensus", []):
        if w.get("state") in ("conflicting", "missing_from_one_model", "unrecoverable"):
            if _overlap(start, end, w.get("start", 0), w.get("end", 0)) > 0.05:
                return True

    for w in asr_consensus.get("secondary_only_words", []):
        if _overlap(start, end, w.get("start", 0), w.get("end", 0)) > 0.05:
            return True

    return False


def _shot_containing(event_start, event_end, shots):
    best = None
    best_overlap = 0.0
    for shot in shots:
        s = float(shot.get("start", 0.0))
        e = float(shot.get("end", 0.0))
        ov = _overlap(event_start, event_end, s, e)
        if ov > best_overlap:
            best_overlap = ov
            best = {"start": s, "end": e, "shot": shot.get("shot")}
    return best


def _shots_overlapped(event_start, event_end, shots):
    """ALL locked shots a window touches, sorted. 3.6: evidence that
    crosses shot boundaries carries a `shots` list instead of being forced
    onto one misleading `shot` (e.g. whole-clip music labeled Shot 2)."""
    out = []
    for shot in shots:
        s = float(shot.get("start", 0.0))
        e = float(shot.get("end", 0.0))
        if _overlap(event_start, event_end, s, e) > 0:
            out.append(shot.get("shot"))
    return sorted(s for s in out if s is not None)


def _assign_shots(event_start, event_end, shots):
    """Return (primary_shot, shots_list). A single-shot window keeps its
    primary `shot`; anything crossing boundaries gets `shot: None` plus a
    `shots` list (3.6 -- never force one wrong shot on cross-shot
    evidence)."""
    best = _shot_containing(event_start, event_end, shots)
    shots_list = _shots_overlapped(event_start, event_end, shots)
    primary = best["shot"] if best and len(shots_list) <= 1 else None
    return primary, shots_list


# ---------------------------------------------------------------------------
# Fail-soft skeleton
# ---------------------------------------------------------------------------

def _empty_contract(status, error=None):
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "error": error,
        "worker": {
            "status": status,
            "panns": {},
            "clap": {},
            "runtime": {},
        },
        "sound_events": {"candidates": []},
        "music": {
            "regions": [],
            "overall_confidence": UNKNOWN,
            "findings": [],
        },
        "ambience": {"candidates": [], "findings": []},
        "source_attribution": {
            "character_candidates": [],
            "object_candidates": [],
        },
        "masking_evidence": {"candidates": []},
        "transients": {"events": [], "findings": []},
        "review_windows": [],
        "findings": [],
    }


# ---------------------------------------------------------------------------
# Worker invocation (shell-out only)
# ---------------------------------------------------------------------------

def run_worker(audio_path, raw_path):
    """Shell out to the isolated worker. Returns the worker's result dict, or
    an unavailable/failed dict if the worker cannot run at all."""
    audio_path = Path(audio_path)
    raw_path = Path(raw_path)

    if not audio_path.exists():
        return {"status": "unavailable", "error": f"analysis WAV missing: {audio_path}"}

    if not AUDIO_EVENTS_PYTHON.exists():
        return {
            "status": "unavailable",
            "error": f"audio-events environment missing: {AUDIO_EVENTS_PYTHON}",
        }

    if not SOUND_EVENTS_WORKER.exists():
        return {
            "status": "unavailable",
            "error": f"sound-events worker missing: {SOUND_EVENTS_WORKER}",
        }

    try:
        proc = subprocess.run(
            [
                str(AUDIO_EVENTS_PYTHON),
                str(SOUND_EVENTS_WORKER),
                str(audio_path),
                str(raw_path),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    if not raw_path.exists():
        return {
            "status": "failed",
            "error": (
                f"worker exited {proc.returncode} without writing {raw_path}: "
                f"{(proc.stderr or '')[-500:]}"
            ),
        }

    try:
        result = json.loads(raw_path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError) as exc:
        return {"status": "failed", "error": f"worker output malformed: {exc}"}

    if not isinstance(result, dict):
        return {"status": "failed", "error": "worker output is not a JSON object"}

    return result


# ---------------------------------------------------------------------------
# Enrichment helpers (turn fusion output into the normalized contract)
# ---------------------------------------------------------------------------

def _reviewer_action_for(tier, source_candidate=None):
    if tier == STRONG:
        return (
            f"Strongly supported by independent models. Confirm by listening; "
            f"source candidate: {source_candidate or 'Unidentified sound'}."
        )
    if tier == MEDIUM:
        return (
            f"Supported. Confirm by listening before creating a Sound event; "
            f"source candidate: {source_candidate or 'Unidentified sound'}."
        )
    if tier == CONFLICT:
        return "Models disagree. Listen and decide manually; do not auto-create an event."
    if tier == WEAK:
        return "Listening cue only. Do NOT create a Manuscript event from this evidence alone."
    return "Insufficient evidence. Listen to determine whether this sound exists."


def _semantic_label(candidate_class):
    return candidate_class.replace("_", " ")


def _window_rms_dbfs(panns_windows, start, end):
    values = [
        float(w.get("rms_dbfs"))
        for w in panns_windows
        if w.get("rms_dbfs") is not None
        and _overlap(start, end, w.get("start", 0), w.get("end", 0)) > 0
    ]
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def _clip_baseline_rms_dbfs(panns_windows):
    values = [
        float(w.get("rms_dbfs"))
        for w in panns_windows
        if w.get("rms_dbfs") is not None
    ]
    if not values:
        return None
    return round(statistics.median(values), 2)


def _enrich_sound_candidate(cand, index, supporting, panns_windows):
    """Add id / semantic label / action / source / description / level / role /
    relationship / provenance to a fusion sound candidate."""
    cand_id = f"snd-{index:03d}"
    tier = cand["tier"]
    candidate_class = cand["candidate_class"]

    baseline = supporting.get("clip_baseline_rms_dbfs")
    event_rms = _window_rms_dbfs(panns_windows, cand["start"], cand["end"])
    level = estimate_recorded_level(event_rms, baseline)
    clip_duration = supporting.get("clip_duration_sec") or 0.0

    mix_role = estimate_mix_role(
        candidate_class, cand.get("duration_sec", 0.0),
        clip_duration, cand.get("overlaps_speech", False),
    )

    primary_shot, shots_list = _assign_shots(
        cand["start"], cand["end"], supporting.get("shots", [])
    )
    shot = _shot_containing(
        cand["start"], cand["end"], supporting.get("shots", [])
    )
    relationships = describe_relationships(
        cand,
        shot=shot,
        other_events=supporting.get("other_events", []),
        speech_windows=supporting.get("speech_windows", []),
    )

    # Ambience uses the semantic/UI-source split; every other group defaults
    # to Unidentified sound unless a (future) tracker provides support.
    if candidate_class in UI_SOURCE_MAP:
        ui_source = UI_SOURCE_MAP[candidate_class]
    else:
        ui_source = DEFAULT_UI_SOURCE

    description = None
    if candidate_class in UI_SOURCE_MAP:
        description = f"Soft {_semantic_label(candidate_class)}."
    elif tier in (STRONG, MEDIUM):
        description = f"{_semantic_label(candidate_class)} detected."

    source_candidate = ui_source
    if candidate_class not in UI_SOURCE_MAP and ui_source == DEFAULT_UI_SOURCE:
        source_candidate = DEFAULT_UI_SOURCE

    return {
        "id": cand_id,
        "semantic_label": _semantic_label(candidate_class),
        "group": cand["group"],
        "candidate_class": candidate_class,
        "start": cand["start"],
        "end": cand["end"],
        "duration_sec": cand["duration_sec"],
        "shot": primary_shot,
        "shots": shots_list,
        "tier": tier,
        "signals": cand["signals"],
        "evidence": [
            f"{s['source']} {candidate_class} score {s['score']}"
            for s in cand["signals"]
        ],
        "raw_labels": cand["raw_labels"],
        "provenance": {
            "panns": [
                {"score": s["score"]}
                for s in cand["signals"] if s["source"] == "panns"
            ],
            "clap": [
                {"score": s["score"]}
                for s in cand["signals"] if s["source"] == "clap"
            ],
            "raw_source_file": "analysis/sound_events_raw.json",
        },
        "reviewer_action": _reviewer_action_for(tier, source_candidate),
        "ui_source_candidate": ui_source,
        "description_candidate": description,
        "recorded_level_candidate": level["level"] if level else None,
        "recorded_level_relative_db": level["relative_db"] if level else None,
        "mix_role_candidate": mix_role,
        "relationship_candidates": relationships,
        "overlaps_speech": cand.get("overlaps_speech", False),
        "is_nonverbal": cand.get("is_nonverbal", False),
    }


def _enrich_music_region(region, index, supporting, panns_windows):
    region_id = f"mus-{index:03d}"
    baseline = supporting.get("clip_baseline_rms_dbfs")
    event_rms = _window_rms_dbfs(panns_windows, region["start"], region["end"])
    level = estimate_recorded_level(event_rms, baseline)

    # 3.6: continuous whole-clip evidence (music running under every shot)
    # is marked as overview evidence with a `shots` list -- never forced
    # onto one misleading shot.
    primary_shot, shots_list = _assign_shots(
        region["start"], region["end"], supporting.get("shots", [])
    )
    clip_duration = supporting.get("clip_duration_sec") or 0.0
    if clip_duration and (region["end"] - region["start"]) >= 0.9 * clip_duration:
        scope = "whole_clip"
    else:
        scope = "cross_shot" if len(shots_list) > 1 else "localized"

    # 3.5: mix_role is deliberately NOT auto-filled for Music. The recorded
    # level (e.g. "Quiet") may be reasonable, but a mix role like "Foreground"
    # requires knowing whether dialogue is the focus -- which the models
    # cannot establish. Leave it blank for human review.
    mix_role = None

    evidence_lines = []
    for s in region["signals"]:
        if s["source"] == "clap_exclusion":
            evidence_lines.append(f"clap non-music exclusion score {s['score']}")
        else:
            evidence_lines.append(f"{s['source']} music score {s['score']}")

    return {
        "id": region_id,
        "semantic_label": "music",
        "group": "music",
        "start": region["start"],
        "end": region["end"],
        "duration_sec": region["duration_sec"],
        "shot": primary_shot,
        "shots": shots_list,
        "scope": scope,
        "tier": region["tier"],
        "signals": region["signals"],
        "evidence": evidence_lines,
        "raw_labels": region["raw_labels"],
        "provenance": {
            "panns": [
                {"score": s["score"]}
                for s in region["signals"] if s["source"] == "panns"
            ],
            "clap": [
                {"score": s["score"]}
                for s in region["signals"] if s["source"] == "clap"
            ],
            "raw_source_file": "analysis/sound_events_raw.json",
        },
        "reviewer_action": region["reviewer_action"],
        "ui_source_candidate": "Music",
        "description_candidate": (
            "Music" if region["tier"] in (STRONG, MEDIUM) else None
        ),
        "recorded_level_candidate": level["level"] if level else None,
        "recorded_level_relative_db": level["relative_db"] if level else None,
        "mix_role_candidate": mix_role,
        "relationship_candidates": [],
        "reasons": region["reasons"],
    }


def _build_ambience(candidates, supporting, panns_windows):
    """Ambience contract: semantic vs UI-source split (3C.7), never maps
    outdoor environmental sound to a named indoor category."""
    out = []
    findings = []

    # co-occurring ambience classes by time (for composite descriptions)
    by_start = sorted(candidates, key=lambda c: c["start"])

    for i, cand in enumerate(by_start, 1):
        enriched = _enrich_sound_candidate(cand, i, supporting, panns_windows)

        co = [
            c["candidate_class"]
            for c in by_start
            if c is not cand
            and _overlap(
                cand["start"], cand["end"], c["start"], c["end"]
            ) > 0.5
        ]
        amb = build_ambience_candidate(cand, co_occurring_classes=co)

        enriched["semantic_candidate"] = amb["semantic_candidate"]
        enriched["ui_source_candidate"] = amb["ui_source_candidate"]
        enriched["description_candidate"] = amb["description_candidate"]

        out.append(enriched)

        if cand["tier"] in (STRONG, MEDIUM):
            findings.append({
                "claim": (
                    f"Ambience: {amb['semantic_candidate']} "
                    f"({cand['tier']})."
                ),
                "tier": cand["tier"],
                "evidence": enriched["evidence"],
                "action": (
                    "Confirm by listening. Do not force a named indoor "
                    "category unless the media is genuinely indoors."
                ),
                "window": [cand["start"], cand["end"]],
            })

    return {"candidates": out, "findings": findings}


# ---------------------------------------------------------------------------
# Main contract assembly (pure -- callable with fixtures, no subprocess)
# ---------------------------------------------------------------------------

def build_sound_fusion_evidence(raw, supporting=None):
    """Fuse a raw worker result + supporting evidence into the normalized
    `sound_fusion_evidence.json` contract.

    raw: the worker's output dict (or an unavailable/failed skeleton).
    supporting: optional dict of preloaded evidence:
        diarization, vad, asr_consensus, speaker_mapping, face_tracks,
        task_context (shots/characters/objects), evidence (shot ranges).
    """
    supporting = supporting or {}

    if not isinstance(raw, dict):
        return _empty_contract("failed", "worker output is not a JSON object")

    status = raw.get("status")
    if status not in ("complete", "failed", "unavailable"):
        return _empty_contract("failed", f"unknown worker status: {status}")

    if status != "complete":
        return _empty_contract(status, raw.get("error"))

    panns_windows = raw.get("panns_windows", []) or []
    clap_windows = raw.get("clap_windows", []) or []
    transient_features = raw.get("transient_feature_windows", []) or []

    # 3.5: a complete worker may have NO model windows (PANNs/CLAP missed
    # the sound) but still carry transient/SFX detector features -- the
    # exact "models cannot name the punch" case. Require at least one
    # evidence source, not specifically a model one.
    if not panns_windows and not clap_windows and not transient_features:
        return _empty_contract(
            "failed", "worker completed but produced no evidence windows"
        )

    provenance = raw.get("provenance", {}) or {}
    panns_prov = provenance.get("panns", {}) or {}
    clap_prov = provenance.get("clap", {}) or {}

    speech_windows = _independent_speech_windows(
        supporting.get("diarization"), supporting.get("vad")
    )
    asr_consensus = supporting.get("asr_consensus", {}) or {}
    shots = (supporting.get("task_context", {}) or {}).get("shots", []) or []
    clip_duration = (raw.get("media", {}) or {}).get("duration_sec")

    baseline = _clip_baseline_rms_dbfs(panns_windows)

    ctx = {
        "speech_windows": speech_windows,
        "shots": shots,
        "clip_duration_sec": clip_duration,
        "clip_baseline_rms_dbfs": baseline,
    }

    # 1. Generic sound candidates (human_nonverbal + object_sfx + ambience).
    all_sound = build_sound_event_candidates(
        panns_windows, clap_windows, speech_windows=speech_windows
    )

    # 2. Music (dedicated conservative pipeline).
    music_result = build_music_candidates(panns_windows, clap_windows)

    # Split ambience out of generic sound candidates.
    ambience_candidates = [c for c in all_sound if c["group"] == AMBIENCE]
    sound_candidates = [
        c for c in all_sound if c["group"] != AMBIENCE
    ]

    # Enrich non-ambience sound candidates (ambience gets its own pass).
    enriched_sound = []
    for i, cand in enumerate(sound_candidates, 1):
        enriched_sound.append(
            _enrich_sound_candidate(cand, i, ctx, panns_windows)
        )

    # Ambience contract.
    ambience = _build_ambience(ambience_candidates, ctx, panns_windows)

    # Music contract.
    music_regions = [
        _enrich_music_region(r, i, ctx, panns_windows)
        for i, r in enumerate(music_result.get("regions", []), 1)
    ]
    music = {
        "regions": music_regions,
        "overall_confidence": music_result.get("overall_confidence", UNKNOWN),
        "findings": [],
    }

    # Source attribution (conservative: no hand/gesture or object tracker in
    # this pipeline, so sound alone never selects a character or object).
    char_candidates = [
        {
            "candidate_id": c["id"],
            "candidate_class": c["candidate_class"],
            **attribute_human_source(c, visual_candidates=[]),
        }
        for c in enriched_sound
        if c["group"] == HUMAN_NONVERBAL
    ]
    obj_candidates = [
        {
            "candidate_id": c["id"],
            "candidate_class": c["candidate_class"],
            **attribute_object_source(c, object_interactions=[]),
        }
        for c in enriched_sound
        if c["group"] == OBJECT_SFX
    ]

    # Masking evidence: overlap is never automatically masking (3.5: a
    # masking finding only surfaces when intelligibility actually drops).
    # 3.6: only MEDIUM/STRONG sound candidates can even be masking
    # candidates -- WEAK model evidence must never drive a masking claim.
    # And the fusion layer never emits masking_check review windows: the
    # review queue only gets masking_check from the FINAL masking evidence
    # (masking_overlap_evidence.json / overlap_masking), so it can never
    # contradict the packet's masking section.
    masking_candidates = []
    for c in enriched_sound:
        if c.get("tier") not in (STRONG, MEDIUM):
            continue
        if not c.get("overlaps_speech"):
            continue
        asr_words_lost = _asr_words_lost_in_window(
            asr_consensus, c["start"], c["end"]
        )
        mask = evaluate_masking(
            c, speech_windows, asr_words_lost=asr_words_lost
        )
        primary_shot, shots_list = _assign_shots(c["start"], c["end"], shots)
        masking_candidates.append({
            "candidate_id": c["id"],
            "candidate_class": c["candidate_class"],
            "start": c["start"],
            "end": c["end"],
            "shot": primary_shot,
            "shots": shots_list,
            **mask,
        })

    # 3.5: independent transient/SFX detector evidence.
    transients = _build_transients(
        raw, speech_windows, enriched_sound, shots
    )

    # Review windows: only meaningful uncertainty, never every STRONG event.
    review_windows = _build_review_windows(
        enriched_sound, music_regions, masking_candidates,
        transient_events=transients["events"], shots=shots,
    )

    findings = _build_findings(
        enriched_sound, music, ambience, masking_candidates,
        transient_findings=transients["findings"],
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "error": None,
        "worker": {
            "status": "complete",
            "panns": panns_prov,
            "clap": clap_prov,
            "transient": (raw.get("provenance", {}) or {}).get("transient", {}),
            "runtime": {
                "total_sec": raw.get("runtime_sec"),
                "panns_sec": panns_prov.get("runtime_sec"),
                "clap_sec": clap_prov.get("runtime_sec"),
            },
        },
        "sound_events": {"candidates": enriched_sound},
        "music": music,
        "ambience": ambience,
        "source_attribution": {
            "character_candidates": char_candidates,
            "object_candidates": obj_candidates,
        },
        "masking_evidence": {"candidates": masking_candidates},
        "transients": transients,
        "review_windows": review_windows,
        "findings": findings,
    }


def _build_transients(raw, speech_windows, sound_candidates, shots):
    """Turn worker low-level features into reviewer-facing transient events.

    Unexplained STRONG transients (no speech, no named sound overlapping)
    are the high-priority review windows the 3.5 audit asked for: the models
    may not be able to NAME the sound, but a strong spike across the five
    low-level indicators is still evidence a Sound event exists.
    """
    feature_windows = raw.get("transient_feature_windows", []) or []

    if not feature_windows:
        return {"events": [], "findings": [], "status": "no_features"}

    events = build_transient_events(feature_windows)
    out = []
    findings = []

    for e in events:
        # 3.6: ONLY a MEDIUM/STRONG named sound can explain a transient.
        # WEAK/CONFLICT/UNKNOWN sound evidence must never explain or demote
        # a transient -- a WEAK cheering guess (CLAP 0.218) must not turn a
        # real impact into "explained". Speech overlap remains pure context.
        named_overlaps = [
            c for c in sound_candidates
            if c["tier"] in (STRONG, MEDIUM)
            and _overlap(e["start"], e["end"], c["start"], c["end"]) > 0.05
        ]
        has_speech = any(
            _overlap(e["start"], e["end"], s, t) > 0.05
            for s, t in speech_windows
        )

        # 3.5: only a NAMED sound candidate truly explains a transient.
        # Co-occurring speech is context, not an explanation -- a punch
        # during dialogue is still a punch, and the models could not name
        # it (the audit's exact case). Speech overlap is recorded so the
        # reviewer knows to separate a real SFX from speech emphasis.
        explained_by = sorted({c["semantic_label"] for c in named_overlaps})
        if has_speech:
            explained_by.append("speech")

        explained = bool(named_overlaps)
        tier = e["tier"]
        if explained and tier == STRONG:
            tier = MEDIUM

        primary_shot, shots_list = _assign_shots(e["start"], e["end"], shots)

        # Explicit reason-why-retained fields: speech overlap alone never
        # demotes a transient (a punch during dialogue is still a punch), so
        # `overlaps_speech` is pure context and `explained_by_named_sound`
        # is the only thing that can lower the tier.
        event = {
            **e,
            "tier": tier,
            "explained_by": explained_by,
            "unexplained": not explained,
            "overlaps_speech": has_speech,
            "explained_by_named_sound": explained,
            "shot": primary_shot,
            "shots": shots_list,
        }
        out.append(event)

        if tier in (STRONG, MEDIUM):
            kind_label = e.get("kind", "transient").replace("_", " ")
            peak_times = ", ".join(
                f"{p['start']:.2f}-{p['end']:.2f}s"
                for p in e.get("peaks", [])
            )

            if not explained:
                if tier == STRONG:
                    action = (
                        "High-priority listen: identify the sound. Even "
                        "unnamed, strong transients may support a Sound event."
                    )
                else:
                    action = (
                        "Listen and identify the sound; even unnamed, a "
                        "transient may support a Sound event."
                    )
                if has_speech:
                    action = (
                        "A transient occurs DURING speech -- separate a "
                        "real SFX from speech emphasis, and identify it. "
                        "Even unnamed, it may support a Sound event."
                    )
                    if tier == STRONG:
                        action = (
                            "High-priority listen: a strong transient occurs "
                            "DURING speech -- separate a real SFX from "
                            "speech emphasis, and identify it. Even unnamed, "
                            "it may support a Sound event."
                        )
                claim = f"Unidentified {kind_label} at {e['start']}-{e['end']}s"
                if peak_times:
                    claim += f" (peaks: {peak_times})"
                claim += f" ({tier} acoustic evidence)."
                findings.append({
                    "claim": claim,
                    "tier": tier,
                    "evidence": [
                        "transient detector: " + ", ".join(e["signals"]),
                        "no named sound source explains it",
                    ]
                    + (["occurs during speech"] if has_speech else []),
                    "action": action,
                    "window": [e["start"], e["end"]],
                })
            else:
                claim = f"{kind_label.capitalize()} at {e['start']}-{e['end']}s coincides "
                if peak_times:
                    claim += f"(peaks: {peak_times}) "
                claim += f"with {', '.join(explained_by)}."
                findings.append({
                    "claim": claim,
                    "tier": MEDIUM,
                    "evidence": ["transient detector: " + ", ".join(e["signals"])],
                    "action": (
                        "Likely a byproduct of the named source; confirm by "
                        "listening if you need to separate it."
                    ),
                    "window": [e["start"], e["end"]],
                })

    return {"events": out, "findings": findings, "status": "complete"}


def _build_review_windows(
    sound, music_regions, masking_candidates, transient_events=None, shots=None
):
    windows = []
    shots = shots or []

    def add(start, end, kind, tier, description):
        primary_shot, shots_list = _assign_shots(start, end, shots)
        windows.append({
            "type": kind,
            "tier": tier,
            "start": round(float(start), 3),
            "end": round(float(end), 3),
            "description": description,
            "shot": primary_shot,
            "shots": shots_list,
        })

    for c in sound:
        if c["tier"] == CONFLICT:
            add(c["start"], c["end"], "sound_identity_check", CONFLICT,
                f"{c['semantic_label']}: models disagree; listen and decide.")
        elif c["tier"] == MEDIUM:
            add(c["start"], c["end"], "sound_identity_check", MEDIUM,
                f"{c['semantic_label']}: confirm identity by listening.")

    for r in music_regions:
        if r["tier"] in (CONFLICT, MEDIUM):
            add(r["start"], r["end"], "music_check", r["tier"],
                "Music evidence: listen before creating a Music event.")
        elif r["tier"] == WEAK:
            add(r["start"], r["end"], "music_check", WEAK,
                "Music evidence is weak; do NOT create a Music event without listening.")

    # 3.6: masking_check review windows are deliberately NOT emitted here.
    # The review queue only creates masking_check from the FINAL masking
    # evidence (masking_overlap_evidence.json / the packet's overlap_masking
    # section). This fusion layer must never independently infer masking --
    # that is how the queue contradicted the packet's (empty) masking
    # section in the Phase 3.6 run.

    # 3.5: strong unexplained transients are high-priority review windows
    # even though no model can name them.
    for t in transient_events or []:
        if t["tier"] not in (STRONG, MEDIUM):
            continue
        if t.get("unexplained"):
            strength = "Strong " if t["tier"] == STRONG else ""
            add(
                t["start"], t["end"], "transient_sfx_check", t["tier"],
                f"{strength}unidentified {t.get('kind', 'transient')}; "
                "listen and identify it (may support a Sound event even "
                "unnamed).",
            )
        else:
            add(
                t["start"], t["end"], "transient_sfx_check", MEDIUM,
                t.get("kind", "transient").replace("_", " ")
                + " coincides with "
                + ", ".join(t.get("explained_by", []))
                + "; confirm it is a byproduct.",
            )

    # De-duplicate by (type, start, end) and sort.
    seen = set()
    unique = []
    for w in sorted(windows, key=lambda w: (w["start"], w["end"], w["type"])):
        key = (w["type"], w["start"], w["end"])
        if key not in seen:
            seen.add(key)
            unique.append(w)
    return unique


def _build_findings(sound, music, ambience, masking_candidates, transient_findings=None):
    findings = []
    transient_findings = transient_findings or []

    for c in sound:
        tier = c["tier"]
        if tier not in (STRONG, MEDIUM, CONFLICT, WEAK):
            tier = UNKNOWN
        findings.append({
            "claim": f"{c['semantic_label']}: {tier} sound evidence.",
            "tier": tier,
            "evidence": c["evidence"],
            "action": c["reviewer_action"],
            "window": [c["start"], c["end"]],
        })

    findings.extend(ambience["findings"])

    for r in music["regions"]:
        findings.append({
            "claim": f"Music: {r['tier']} evidence.",
            "tier": r["tier"],
            "evidence": r["evidence"],
            "action": r["reviewer_action"],
            "window": [r["start"], r["end"]],
        })

    if music["overall_confidence"] in (UNKNOWN, WEAK, CONFLICT):
        findings.append({
            "claim": f"Music overall: {music['overall_confidence']}.",
            "tier": music["overall_confidence"],
            "evidence": ["panns + clap music fusion"],
            "action": "Do NOT create a Music event without listening.",
        })

    # 3.6: masking findings are NOT emitted from this fusion layer either.
    # The packet's official masking section (overlap_masking) is the only
    # place masking claims live; fusion records overlap data in
    # `masking_evidence.candidates` but never asserts masking itself.

    findings.extend(transient_findings)

    findings.sort(
        key=lambda f: (
            -_TIER_RANK.get(f.get("tier"), 0),
            tuple(f.get("window") or [0.0, 0.0]),
        )
    )
    return findings


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def write_sound_fusion_evidence(
    audio_path=AUDIO,
    raw_path=SOUND_EVENTS_RAW,
    output_path=SOUND_FUSION_EVIDENCE,
    diarization_path=DIARIZATION,
    vad_path=VAD,
    asr_consensus_path=ASR_CONSENSUS,
    speaker_mapping_path=SPEAKER_MAPPING,
    face_tracks_path=FACE_TRACKS,
    evidence_path=EVIDENCE,
    context_path=CONTEXT,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()

    raw = run_worker(audio_path, raw_path)

    supporting = {
        "diarization": load_json(diarization_path, {}),
        "vad": load_json(vad_path, {}),
        "asr_consensus": load_json(asr_consensus_path, {}),
        "speaker_mapping": load_json(speaker_mapping_path, {}),
        "face_tracks": load_json(face_tracks_path, {}),
        "task_context": load_json(context_path, {}),
        "evidence": load_json(evidence_path, {}),
    }

    result = build_sound_fusion_evidence(raw, supporting)

    result["runtime_sec"] = round(time.time() - started, 2)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    status = result["status"]
    if status == "complete":
        print(
            "SOUND FUSION: PASS |",
            f"sound_events={len(result['sound_events']['candidates'])} |",
            f"music_regions={len(result['music']['regions'])} |",
            f"ambience={len(result['ambience']['candidates'])} |",
            f"masking={len(result['masking_evidence']['candidates'])} |",
            f"review_windows={len(result['review_windows'])}",
        )
    else:
        print(f"SOUND FUSION: {status.upper()} | {result.get('error')}")

    return result


def main():
    return write_sound_fusion_evidence()


if __name__ == "__main__":
    main()
