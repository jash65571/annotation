"""Phase 3C: non-speech sound evidence fusion.

Pure standard library. No PANNs/CLAP/torch dependency here -- this module
consumes whatever raw per-window scores the model workers produced (or
synthetic fixtures in tests) and turns them into reviewer-facing evidence
using the shared STRONG/MEDIUM/WEAK/CONFLICT/UNKNOWN vocabulary.

Phase 3.5 addition: the transient/SFX detector (short-time RMS, spectral
flux, onset strength, broadband energy change, crest factor) -- pure
stdlib, numpy lives only in the worker.

Design rules carried over from 3A/3B and restated for 3C:

- A classifier score is never a Manuscript fact by itself. Independent
  model agreement raises confidence; a single decimal score never does.
- WEAK/CONFLICT candidates never populate UI suggestions.
- Sound overlapping speech is never automatically "masking".
- A diarization cluster with vocal energy but no confirmed words supports a
  nonverbal Sound candidate, never invented dialogue.
- Visible motion alone never selects a human or object sound source.
"""

import statistics

from manuscript_audio_sound_vocabulary import (
    HUMAN_NONVERBAL, AMBIENCE, OBJECT_SFX, MUSIC,
    NONVERBAL_CLASSES, MUSIC_CLASSES,
    UI_SOURCE_MAP, DEFAULT_UI_SOURCE,
    map_raw_label,
    CLAP_PROMPTS, NON_MUSIC_EXCLUSION_CLASSES,
)


STRONG = "STRONG"
MEDIUM = "MEDIUM"
WEAK = "WEAK"
CONFLICT = "CONFLICT"
UNKNOWN = "UNKNOWN"

# Per-model score thresholds. Below WEAK_THRESHOLD a score is not evidence
# at all and is dropped before it ever reaches fusion.
STRONG_THRESHOLD = 0.70
MEDIUM_THRESHOLD = 0.45
WEAK_THRESHOLD = 0.20

# Below this many seconds, evidence for a class is treated as too brief to
# be more than a passing/incidental detection (relevant to music: 3C.6
# explicitly forbids a short rhythmic burst from becoming Music).
MIN_STRONG_DURATION_SEC = 1.5

JOIN_GAP_SEC = 0.35


# ---------------------------------------------------------------------------
# Windowing / merge (3C.4)
# ---------------------------------------------------------------------------

def merge_intervals(intervals, join_gap=JOIN_GAP_SEC):
    intervals = sorted(
        (float(a), float(b)) for a, b in intervals if float(b) > float(a)
    )
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1] + join_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


# ---------------------------------------------------------------------------
# Raw window -> mapped per-class score entries (3C.1 + 3C.2/3C.3)
# ---------------------------------------------------------------------------

def map_panns_window(window):
    """window: {"start","end","top_labels":[{"raw_label","score"}, ...]}
    Returns a list of {group, candidate_class, raw_label, score, start, end,
    source} entries for every top label that maps into the controlled
    vocabulary. Unmapped raw labels are silently dropped (they stay in the
    raw provenance file, never in fused evidence).
    """
    start, end = float(window["start"]), float(window["end"])
    out = []
    for item in window.get("top_labels", []):
        score = float(item.get("score", 0.0))
        if score < WEAK_THRESHOLD:
            continue
        mapped = map_raw_label(item.get("raw_label", ""))
        if not mapped:
            continue
        group, candidate_class = mapped
        out.append({
            "group": group, "candidate_class": candidate_class,
            "raw_label": item["raw_label"], "score": round(score, 4),
            "start": start, "end": end, "source": "panns",
        })
    return out


_CLAP_PROMPT_LOOKUP = {p["prompt"]: p for p in CLAP_PROMPTS}


def map_clap_window(window):
    """window: {"start","end","prompt_scores":[{"prompt","score"}, ...]}
    Same contract as map_panns_window, but keyed off the fixed prompt list
    (spec 3C.3) rather than free-form labels. Exclusion prompts ("speech
    without music", "silence") are preserved with group=None -- they are
    evidence AGAINST a class, consumed by compute_music_tier, and never
    become a standalone sound candidate.
    """
    start, end = float(window["start"]), float(window["end"])
    out = []
    for item in window.get("prompt_scores", []):
        score = float(item.get("score", 0.0))
        prompt_def = _CLAP_PROMPT_LOOKUP.get(item.get("prompt"))
        if not prompt_def:
            continue
        if score < WEAK_THRESHOLD and prompt_def["candidate_class"] not in NON_MUSIC_EXCLUSION_CLASSES:
            continue
        out.append({
            "group": prompt_def["group"],
            "candidate_class": prompt_def["candidate_class"],
            "raw_label": item["prompt"], "score": round(score, 4),
            "start": start, "end": end, "source": "clap",
        })
    return out


# ---------------------------------------------------------------------------
# Track collection + merge into candidate events (3C.4 + 3C.11)
# ---------------------------------------------------------------------------

def collect_candidate_tracks(panns_entries, clap_entries):
    """Group mapped per-window entries by candidate_class."""
    tracks = {}
    for entry in list(panns_entries) + list(clap_entries):
        cls = entry["candidate_class"]
        tracks.setdefault(cls, []).append(entry)
    return tracks


def merge_track_to_events(entries, join_gap=JOIN_GAP_SEC):
    """Merge one candidate_class's scattered window hits (from either
    model) into contiguous event windows, preserving per-source score lists
    so fusion can require independent agreement rather than a raw max.
    """
    entries = sorted(entries, key=lambda e: (e["start"], e["end"]))
    groups = []

    for e in entries:
        if groups and e["start"] <= groups[-1]["end"] + join_gap:
            g = groups[-1]
            g["end"] = max(g["end"], e["end"])
            g["by_source"].setdefault(e["source"], []).append(e["score"])
            g["raw_labels"].add(e["raw_label"])
        else:
            groups.append({
                "start": e["start"], "end": e["end"],
                "by_source": {e["source"]: [e["score"]]},
                "raw_labels": {e["raw_label"]},
            })

    events = []
    for g in groups:
        events.append({
            "start": round(g["start"], 3),
            "end": round(g["end"], 3),
            "panns_max": round(max(g["by_source"]["panns"]), 4) if "panns" in g["by_source"] else None,
            "clap_max": round(max(g["by_source"]["clap"]), 4) if "clap" in g["by_source"] else None,
            "raw_labels": sorted(g["raw_labels"]),
        })
    return events


# ---------------------------------------------------------------------------
# Generic confidence fusion (3C.11) -- everything except music, which has
# its own dedicated function below because the spec gives it its own rules.
# ---------------------------------------------------------------------------

def fuse_confidence(panns_max, clap_max):
    """Independent-agreement-first confidence. A lone strong score from one
    model is capped at MEDIUM -- STRONG requires two independent models to
    agree, per 3C.11 ("independent evidence agreement matters more than one
    decimal score").
    """
    has_panns = panns_max is not None
    has_clap = clap_max is not None

    if not has_panns and not has_clap:
        return UNKNOWN

    p = panns_max or 0.0
    c = clap_max or 0.0

    if has_panns and has_clap:
        if p >= STRONG_THRESHOLD and c >= STRONG_THRESHOLD:
            return STRONG
        if p >= MEDIUM_THRESHOLD and c >= MEDIUM_THRESHOLD:
            return MEDIUM
        return WEAK

    # Only one model produced a score for this class -- never STRONG alone.
    lone = p if has_panns else c
    if lone >= MEDIUM_THRESHOLD:
        return MEDIUM if lone >= STRONG_THRESHOLD else WEAK
    return WEAK


def build_sound_event_candidates(panns_windows, clap_windows, speech_windows=None):
    """Full 3C.4/3C.11 pipeline for the HUMAN_NONVERBAL / OBJECT_SFX /
    AMBIENCE groups (music is handled separately -- see
    build_music_candidates). speech_windows is an optional list of
    (start, end) independent speech-presence intervals (diarization/VAD),
    used only to record overlap -- never to infer masking (3C.12 handled by
    evaluate_masking, called by the caller with real ASR evidence).
    """
    speech_windows = speech_windows or []

    panns_entries = [
        e for w in panns_windows for e in map_panns_window(w)
        if e["group"] in (HUMAN_NONVERBAL, OBJECT_SFX, AMBIENCE)
    ]
    clap_entries = [
        e for w in clap_windows for e in map_clap_window(w)
        if e["group"] in (HUMAN_NONVERBAL, OBJECT_SFX, AMBIENCE)
    ]

    tracks = collect_candidate_tracks(panns_entries, clap_entries)

    # Long vocal-class windows that mostly coincide with known speech are a
    # common classifier failure: loud or rough speech can look like coughing,
    # throat clearing, laughter, or another vocal reaction to both models.
    # Preserve the lead as CONFLICT, but never let it become a UI default.
    speech_confusable_vocals = {
        "laughter", "chuckle", "giggle", "gasp", "sigh", "groan",
        "scream", "cough", "throat_clearing", "humming",
        "wordless_vocalization", "speech_babble",
    }

    def speech_overlap_ratio(start, end):
        duration = max(0.0, float(end) - float(start))
        if duration <= 0:
            return 0.0
        clipped = sorted(
            (max(float(start), float(s)), min(float(end), float(e)))
            for s, e in speech_windows
            if min(float(end), float(e)) > max(float(start), float(s))
        )
        merged = []
        for left, right in clipped:
            if merged and left <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], right))
            else:
                merged.append((left, right))
        return min(1.0, sum(right - left for left, right in merged) / duration)

    candidates = []
    for candidate_class, entries in tracks.items():
        group = entries[0]["group"]
        for event in merge_track_to_events(entries):
            tier = fuse_confidence(event["panns_max"], event["clap_max"])
            overlap_ratio = speech_overlap_ratio(event["start"], event["end"])
            speech_contaminated = (
                candidate_class in speech_confusable_vocals
                and event["end"] - event["start"] >= 1.0
                and overlap_ratio >= 0.55
                and tier in (STRONG, MEDIUM)
            )
            if speech_contaminated:
                tier = CONFLICT

            overlaps_speech = any(
                _overlap(event["start"], event["end"], s, e) > 0.05
                for s, e in speech_windows
            )

            signals = []
            if event["panns_max"] is not None:
                signals.append({"source": "panns", "score": event["panns_max"]})
            if event["clap_max"] is not None:
                signals.append({"source": "clap", "score": event["clap_max"]})

            candidates.append({
                "group": group,
                "candidate_class": candidate_class,
                "start": event["start"],
                "end": event["end"],
                "duration_sec": round(event["end"] - event["start"], 3),
                "tier": tier,
                "signals": signals,
                "raw_labels": event["raw_labels"],
                "overlaps_speech": overlaps_speech,
                "speech_overlap_ratio": round(overlap_ratio, 3),
                "speech_contamination_conflict": speech_contaminated,
                "is_nonverbal": candidate_class in NONVERBAL_CLASSES,
            })

    candidates.sort(key=lambda c: (c["start"], -_TIER_RANK.get(c["tier"], 9)))
    return candidates


_TIER_RANK = {STRONG: 3, MEDIUM: 2, WEAK: 1, CONFLICT: 1, UNKNOWN: 0}


# ---------------------------------------------------------------------------
# Music consensus (3C.6) -- deliberately its own function: conservative,
# rejection-critical. Rhythmicity may only ever demote a tier, never
# promote one on its own (a rhythmic clapping burst must never become
# Music -- spec 3C.6 / regression test #3).
# ---------------------------------------------------------------------------

def compute_music_tier(
    panns_music_score, clap_music_score, clap_exclusion_score,
    duration_sec, rhythmicity_compatible=None,
):
    has_panns = panns_music_score is not None
    has_clap = clap_music_score is not None

    if not has_panns and not has_clap:
        return {
            "tier": UNKNOWN,
            "reasons": ["no music-capable model produced a score"],
        }

    p = panns_music_score or 0.0
    c = clap_music_score or 0.0
    ex = clap_exclusion_score or 0.0

    # CONFLICT: an exclusion prompt (speech/silence) strongly opposes music
    # while at least one music signal is itself strong -- genuine model
    # disagreement, not a decision either way.
    if ex >= STRONG_THRESHOLD and max(p, c) >= STRONG_THRESHOLD:
        return {
            "tier": CONFLICT,
            "reasons": [
                f"clap non-music prompt scored {ex} while a music signal "
                f"scored {max(p, c)} -- models disagree",
            ],
        }

    if has_panns and has_clap and p >= STRONG_THRESHOLD and c >= STRONG_THRESHOLD:
        if duration_sec < MIN_STRONG_DURATION_SEC:
            return {
                "tier": WEAK,
                "reasons": [
                    f"both models strongly support music but duration "
                    f"{duration_sec}s is too short to trust",
                ],
            }
        if rhythmicity_compatible is False:
            return {
                "tier": MEDIUM,
                "reasons": [
                    "both models strongly support music, but rhythmicity "
                    "evidence is incompatible with music -- demoted",
                ],
            }
        reasons = ["panns music score >= strong threshold", "clap music score >= strong threshold"]
        if rhythmicity_compatible is True:
            reasons.append("rhythmicity evidence is compatible")
        return {"tier": STRONG, "reasons": reasons}

    if has_panns and has_clap and p >= MEDIUM_THRESHOLD and c >= MEDIUM_THRESHOLD:
        return {
            "tier": MEDIUM,
            "reasons": ["panns and clap both moderately support music"],
        }

    # Only one useful signal, or both marginal.
    lone_source = "panns" if has_panns and p >= c else "clap" if has_clap else None
    return {
        "tier": WEAK,
        "reasons": [
            f"only {lone_source or 'one'} model supports music, or scores "
            "are marginal -- do not create a Music event without listening",
        ],
    }


def build_music_candidates(panns_windows, clap_windows, rhythmicity_by_window=None):
    """Runs the full panns+clap windows through the music-specific pipeline.
    rhythmicity_by_window is optional: a callable(start, end) -> True/False/
    None from an existing rhythmicity feeder. When absent, rhythmicity never
    contributes (honest UNKNOWN contribution, never invented).
    """
    panns_music = [
        e for w in panns_windows for e in map_panns_window(w)
        if e["group"] == MUSIC
    ]
    clap_music = [
        e for w in clap_windows for e in map_clap_window(w)
        if e["group"] == MUSIC
    ]
    clap_exclusion = [
        e for w in clap_windows for e in map_clap_window(w)
        if e["candidate_class"] in NON_MUSIC_EXCLUSION_CLASSES
    ]

    tracks = collect_candidate_tracks(panns_music, clap_music)
    all_music_entries = [e for entries in tracks.values() for e in entries]
    events = merge_track_to_events(all_music_entries) if all_music_entries else []

    candidates = []
    for event in events:
        exclusion_here = [
            e["score"] for e in clap_exclusion
            if _overlap(event["start"], event["end"], e["start"], e["end"]) > 0
        ]
        exclusion_score = max(exclusion_here) if exclusion_here else None

        rhythmicity = None
        if rhythmicity_by_window:
            rhythmicity = rhythmicity_by_window(event["start"], event["end"])

        result = compute_music_tier(
            event["panns_max"], event["clap_max"], exclusion_score,
            duration_sec=round(event["end"] - event["start"], 3),
            rhythmicity_compatible=rhythmicity,
        )

        signals = []
        if event["panns_max"] is not None:
            signals.append({"source": "panns", "score": event["panns_max"]})
        if event["clap_max"] is not None:
            signals.append({"source": "clap", "score": event["clap_max"]})
        if exclusion_score is not None:
            signals.append({"source": "clap_exclusion", "score": exclusion_score})

        candidates.append({
            "start": event["start"],
            "end": event["end"],
            "duration_sec": round(event["end"] - event["start"], 3),
            "tier": result["tier"],
            "reasons": result["reasons"],
            "signals": signals,
            "raw_labels": event["raw_labels"],
            "reviewer_action": (
                "DO NOT CREATE MUSIC EVENT WITHOUT LISTENING"
                if result["tier"] in (WEAK, UNKNOWN, CONFLICT)
                else "Confirm by listening before creating a Music event."
            ),
        })

    candidates.sort(key=lambda c: c["start"])

    if not candidates:
        overall = UNKNOWN if not (panns_music or clap_music) else WEAK
    else:
        overall = candidates[0]["tier"]
        for c in candidates[1:]:
            if _TIER_RANK.get(c["tier"], 0) > _TIER_RANK.get(overall, 0):
                overall = c["tier"]

    return {
        "regions": candidates,
        "overall_confidence": overall,
    }


# ---------------------------------------------------------------------------
# Transient / SFX detector (3.5) -- pure stdlib, no numpy here.
#
# The model workers (PANNs/CLAP) miss short, loud, unnamed sounds (the
# punch at ~8.77s in the audit clip). This independent detector runs on
# low-level features -- short-time RMS, spectral flux, onset strength,
# broadband energy change, and crest factor -- computed in the worker
# (numpy) and turned into events here so the logic stays unit-testable
# without any audio dependency.
# ---------------------------------------------------------------------------

TRANSIENT_SCORE_THRESHOLD = 0.5
TRANSIENT_STRONG_SCORE = 0.7
TRANSIENT_JOIN_GAP_SEC = 0.4
TRANSIENT_SIGNAL_BAR = 0.5
# A merged transient event longer than this is not really a "transient" --
# it is a sustained high-energy acoustic region (speech emphasis, a musical
# crescendo, applause). Keep the individual peaks underneath it (3.6).
TRANSIENT_KIND_MAX_SEC = 1.0
TRANSIENT_KIND_TRANSIENT = "transient"
TRANSIENT_KIND_REGION = "high_energy_acoustic_region"


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _robust_z(values):
    """Median/MAD z-score (pure stdlib). Constant input -> all zeros."""
    values = [float(v) for v in values]
    n = len(values)
    if n == 0:
        return []

    median = statistics.median(values)
    deviations = [abs(v - median) for v in values]
    mad = statistics.median(deviations) if n > 1 else 0.0
    scale = mad * 1.4826 if mad > 0 else (statistics.pstdev(values) if n > 1 else 0.0)

    if scale <= 1e-9:
        return [0.0] * n

    return [(v - median) / scale for v in values]


def build_transient_events(feature_windows, join_gap=TRANSIENT_JOIN_GAP_SEC):
    """Turn per-window low-level detector features into transient events.

    feature_windows: list of {"start", "end", "rms_db", "crest_factor",
    "spectral_flux", "onset_strength", "energy_change_db"}. A window is a
    transient candidate when several independent indicators spike together
    (RMS above baseline, broadband energy change, spectral flux, onset
    strength, crest factor). Adjacent candidates merge. Windows without a
    usable rms_db are ignored.

    Tier is acoustic strength only: STRONG >= 0.7, MEDIUM >= 0.5. Whether a
    transient is *explained* by speech or a named sound is the caller's job
    (it needs speech windows + sound candidates); that demotion decides
    review priority.
    """
    feature_windows = [
        w for w in (feature_windows or [])
        if w.get("rms_db") is not None
    ]

    if not feature_windows:
        return []

    feature_windows = sorted(
        feature_windows, key=lambda w: (float(w["start"]), float(w["end"]))
    )

    rms = [float(w["rms_db"]) for w in feature_windows]
    flux = [float(w.get("spectral_flux", 0.0)) for w in feature_windows]
    onset = [float(w.get("onset_strength", 0.0)) for w in feature_windows]
    energy = [float(w.get("energy_change_db", 0.0)) for w in feature_windows]
    crest = [float(w.get("crest_factor", 0.0)) for w in feature_windows]

    rms_z = _robust_z(rms)
    flux_z = _robust_z(flux)
    onset_z = _robust_z(onset)
    energy_z = _robust_z(energy)

    events = []

    for i, w in enumerate(feature_windows):
        indicators = {
            "short_time_rms": _clamp01(rms_z[i] / 2.0),
            "broadband_energy_change": _clamp01(energy_z[i] / 2.0),
            "spectral_flux": _clamp01(flux_z[i] / 2.0),
            "onset_strength": _clamp01(onset_z[i] / 2.0),
            "crest_factor": _clamp01((crest[i] - 3.0) / 7.0),
        }

        score = round(
            sum(indicators.values()) / len(indicators), 3
        )

        if score < TRANSIENT_SCORE_THRESHOLD:
            continue

        events.append({
            "start": w["start"],
            "end": w["end"],
            "score": score,
            "tier": (
                STRONG if score >= TRANSIENT_STRONG_SCORE else MEDIUM
            ),
            "signals": [
                k for k, v in indicators.items()
                if v >= TRANSIENT_SIGNAL_BAR
            ],
            "features": {
                k: round(v, 4) for k, v in indicators.items()
            },
            # Preserve the low-level measurements so downstream gates can
            # distinguish a voiced speech onset from a broadband impact.
            "raw_features": {
                key: w.get(key)
                for key in (
                    "rms_db", "crest_factor", "spectral_flux",
                    "onset_strength", "energy_change_db",
                )
                if w.get(key) is not None
            },
        })

    merged = []

    for e in events:
        if merged and e["start"] <= merged[-1]["end"] + join_gap:
            g = merged[-1]
            g["end"] = max(g["end"], e["end"])
            g["score"] = round(max(g["score"], e["score"]), 3)
            g["tier"] = STRONG if g["score"] >= TRANSIENT_STRONG_SCORE else MEDIUM
            for signal in e["signals"]:
                if signal not in g["signals"]:
                    g["signals"].append(signal)
            g["peaks"].append({
                "start": e["start"],
                "end": e["end"],
                "score": e["score"],
                "tier": e["tier"],
                "raw_features": e.get("raw_features", {}),
            })
        else:
            merged.append({
                **e,
                "peaks": [{
                    "start": e["start"],
                    "end": e["end"],
                    "score": e["score"],
                    "tier": e["tier"],
                    "raw_features": e.get("raw_features", {}),
                }],
            })

    # 3.6: merged spans > TRANSIENT_KIND_MAX_SEC are high-energy acoustic
    # REGIONS (a musical crescendo or strong speech can look like one giant
    # transient otherwise). The individual peak times stay underneath.
    for g in merged:
        duration = g["end"] - g["start"]
        g["kind"] = (
            TRANSIENT_KIND_REGION
            if duration > TRANSIENT_KIND_MAX_SEC
            else TRANSIENT_KIND_TRANSIENT
        )

    return merged


# ---------------------------------------------------------------------------
# Ambience: semantic vs UI-source split (3C.7)
# ---------------------------------------------------------------------------

_AMBIENCE_PHRASES = {
    "room_ambience": "indoor room tone",
    "outdoor_ambience": "outdoor ambience",
    "traffic": "traffic noise",
    "wind": "wind noise",
    "ocean": "ocean ambience",
    "waves": "ocean/water ambience",
    "water": "water ambience",
    "splashing": "splashing water",
    "rain": "rain",
    "birds": "birdsong",
    "crowd_ambience": "crowd ambience",
    "boat_engine": "boat engine ambience",
    "vehicle_engine": "vehicle engine ambience",
    "machinery": "machinery noise",
}


def build_ambience_candidate(candidate_event, co_occurring_classes=None):
    """Turns one AMBIENCE-group candidate event into the semantic vs
    UI-source split required by 3C.7. co_occurring_classes lets callers
    combine adjacent/overlapping ambience candidates into one composite
    description (e.g. "outdoor boat and ocean ambience") without ever
    forcing a named UI category that doesn't fit.
    """
    classes = [candidate_event["candidate_class"]] + list(co_occurring_classes or [])
    seen = []
    for c in classes:
        if c not in seen:
            seen.append(c)

    phrases = [_AMBIENCE_PHRASES.get(c, c.replace("_", " ")) for c in seen]

    if len(phrases) == 1:
        description = f"Soft {phrases[0]}."
    else:
        description = "Soft " + " and ".join(phrases) + "."

    # Only a genuine room-tone class maps to a named UI source; every other
    # ambience class -- including composites -- stays Unidentified sound.
    ui_source = UI_SOURCE_MAP.get(candidate_event["candidate_class"], DEFAULT_UI_SOURCE)
    if len(seen) > 1:
        ui_source = DEFAULT_UI_SOURCE

    return {
        "semantic_candidate": "+".join(seen),
        "ui_source_candidate": ui_source,
        "description_candidate": description,
    }


# ---------------------------------------------------------------------------
# Human sound-source attribution (3C.8)
# ---------------------------------------------------------------------------

def attribute_human_source(sound_event, visual_candidates=None):
    """visual_candidates: [{"character": "C4", "start", "end", "confidence",
    "evidence": "..."}]. This pipeline currently has no hand/gesture
    tracker (only mediapipe face-mesh mouth motion, Phase 3B) -- callers
    will typically pass an empty list until such a feeder exists. That is a
    deliberate, honest gap: sound alone must never select a character.
    """
    visual_candidates = visual_candidates or []

    overlapping = [
        v for v in visual_candidates
        if _overlap(sound_event["start"], sound_event["end"], v["start"], v["end"]) > 0
    ]

    if not overlapping:
        return {
            "source_candidate": "Unidentified sound",
            "confidence": None,
            "reasons": ["no visible action evidence overlaps this sound"],
        }

    if len(overlapping) > 1:
        return {
            "source_candidate": "Unidentified sound",
            "confidence": None,
            "reasons": [
                f"{len(overlapping)} characters show plausible visible "
                "action; do not force one person to own a group reaction",
            ],
        }

    top = overlapping[0]
    confidence = (
        STRONG if sound_event["tier"] in (STRONG, MEDIUM) and top.get("confidence") == STRONG
        else MEDIUM
    )
    return {
        "source_candidate": top["character"],
        "confidence": confidence,
        "reasons": [top.get("evidence", "visible action support")],
    }


# ---------------------------------------------------------------------------
# Object sound-source attribution (3C.9) -- highly conservative
# ---------------------------------------------------------------------------

def attribute_object_source(sound_event, object_interactions=None):
    """object_interactions: [{"object_id": "O1", "start", "end",
    "acoustic_class_match": bool, "temporal_overlap": bool}]. As with
    attribute_human_source, no real object-interaction tracker exists yet
    in this pipeline -- callers pass an empty list by default. Visible
    movement alone (temporal_overlap True, acoustic_class_match False) is
    never sufficient.
    """
    object_interactions = object_interactions or []

    qualifying = [
        o for o in object_interactions
        if o.get("acoustic_class_match") and o.get("temporal_overlap")
        and _overlap(sound_event["start"], sound_event["end"], o["start"], o["end"]) > 0
    ]

    if len(qualifying) == 1:
        return {
            "source_candidate": qualifying[0]["object_id"],
            "confidence": MEDIUM,
            "reasons": [
                "matching acoustic event class and matching temporal "
                "visible interaction, no competing source",
            ],
        }

    if len(qualifying) > 1:
        return {
            "source_candidate": "Unidentified sound",
            "confidence": None,
            "reasons": [
                f"{len(qualifying)} objects have matching acoustic+visible "
                "evidence; no single competing source wins",
            ],
        }

    return {
        "source_candidate": "Unidentified sound",
        "confidence": None,
        "reasons": [
            "visible motion alone (without a matching acoustic event) is "
            "never sufficient to select an object source",
        ],
    }


# ---------------------------------------------------------------------------
# Nonverbal-vs-speech guard (3C.10)
# ---------------------------------------------------------------------------

def classify_vocal_cluster(diarization_cluster_word_count, sound_candidates_in_window):
    """A diarization cluster with vocal energy but zero confirmed ASR words,
    where PANNs/CLAP support a nonverbal class in the same window, supports
    a nonverbal Sound candidate -- never invented dialogue.
    """
    if diarization_cluster_word_count and diarization_cluster_word_count > 0:
        return None  # confirmed words exist; this is speech, not this module's concern

    nonverbal_here = [
        c for c in sound_candidates_in_window
        if c.get("is_nonverbal") and c["tier"] in (STRONG, MEDIUM)
    ]
    if not nonverbal_here:
        return None

    return {
        "claim": "vocal energy with zero confirmed words; nonverbal sound "
                 "evidence present in the same window",
        "candidate_classes": [c["candidate_class"] for c in nonverbal_here],
        "action": "This supports a nonverbal Sound event, not dialogue. "
                  "Do not invent Speech text for this cluster.",
    }


# ---------------------------------------------------------------------------
# Masking (3C.12) -- overlap is never automatically masking
# ---------------------------------------------------------------------------

def evaluate_masking(sound_event, speech_windows, asr_words_lost=None):
    """asr_words_lost: None (unknown / not evaluated), False (ASR coverage
    stayed intact through the overlap -- masking not supported), or True
    (words disappeared during the overlap across multiple ASR passes).
    """
    overlaps = any(
        _overlap(sound_event["start"], sound_event["end"], s, e) > 0.05
        for s, e in speech_windows
    )

    if not overlaps:
        return {"overlap": False, "masking": "not_applicable", "tier": None}

    if asr_words_lost is None:
        return {
            "overlap": True, "masking": "UNKNOWN", "tier": UNKNOWN,
            "action": "Overlaps speech. Listen before assuming any "
                      "intelligibility impact.",
        }

    if asr_words_lost is False:
        return {
            "overlap": True, "masking": "masking_not_supported", "tier": MEDIUM,
            "action": "Speech transcript agreement remained high through "
                      "the overlap; masking is not supported by ASR "
                      "evidence. Confirm by listening.",
        }

    return {
        "overlap": True, "masking": "possible_masking", "tier": MEDIUM,
        "action": "Words disappeared during this overlap across multiple "
                  "ASR passes. Possible masking -- still requires human "
                  "listening before calling it true masking.",
    }


# ---------------------------------------------------------------------------
# Recorded level (3C.13) and mix role (3C.14)
# ---------------------------------------------------------------------------

def estimate_recorded_level(event_rms_db, clip_baseline_rms_db):
    if event_rms_db is None or clip_baseline_rms_db is None:
        return None

    diff = round(event_rms_db - clip_baseline_rms_db, 2)

    if diff >= 6:
        level = "Loud"
    elif diff >= 0:
        level = "Moderate"
    elif diff >= -10:
        level = "Quiet"
    else:
        level = "Faint"

    return {"level": level, "relative_db": diff}


def estimate_mix_role(candidate_class, duration_sec, clip_duration_sec, overlaps_speech):
    if not clip_duration_sec:
        return None

    coverage_ratio = duration_sec / clip_duration_sec if clip_duration_sec else 0.0

    if candidate_class in ("room_ambience", "outdoor_ambience", "wind", "ocean",
                            "waves", "water", "traffic", "vehicle_engine",
                            "boat_engine", "machinery", "crowd_ambience", "rain"):
        if coverage_ratio >= 0.5:
            return "Background"
        return "Supporting" if overlaps_speech else "Background"

    if overlaps_speech and coverage_ratio < 0.3:
        return "Supporting"

    return "Foreground"


# ---------------------------------------------------------------------------
# Event relationships (3C.15) -- phrases only, timestamps stay internal
# ---------------------------------------------------------------------------

def describe_relationships(event, shot=None, other_events=None, speech_windows=None):
    other_events = other_events or []
    speech_windows = speech_windows or []
    phrases = []

    if shot:
        shot_start, shot_end = float(shot["start"]), float(shot["end"])
        shot_len = shot_end - shot_start
        if shot_len > 0:
            if event["start"] - shot_start < 0.5:
                phrases.append("at the beginning of the shot")
            if shot_end - event["end"] < 0.5:
                phrases.append("near the end of the shot")
            if (event["end"] - event["start"]) >= 0.9 * shot_len:
                phrases.append("throughout the shot")

    for s, e in speech_windows:
        if event["start"] <= s < event["end"]:
            phrases.append("begins during speech")
        if event["end"] > e and _overlap(event["start"], event["end"], s, e) > 0:
            phrases.append("continues briefly after speech ends")

    for other in other_events:
        if other is event:
            continue
        if _overlap(event["start"], event["end"], other["start"], other["end"]) > 0:
            phrases.append("overlaps another sound")
        elif 0 <= event["start"] - other["end"] < 0.5:
            phrases.append("follows another sound event")

    seen = []
    for p in phrases:
        if p not in seen:
            seen.append(p)
    return seen
