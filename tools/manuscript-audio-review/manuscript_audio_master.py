"""Manuscript II audio master aggregator.

Consolidates every per-stage evidence file into a single review packet,
a human-readable REVIEW_ME.md, and a sparse UI-suggestions file.

Design rules (from the Manuscript II reviewer spec):

- One shared confidence vocabulary: STRONG / MEDIUM / WEAK / CONFLICT / UNKNOWN.
  Never hide uncertainty behind a raw decimal score.
- Every machine conclusion carries its provenance (which signals produced it).
- UI suggestions stay sparse: a field is emitted only when evidence is at
  least MEDIUM. Weak evidence never becomes a default UI value.
- The tool produces evidence, not decisions. Actual media remains ground truth.

This module is pure standard library so it can run under the base interpreter
without either analysis virtual environment.
"""

from pathlib import Path
import json


ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis"

# Inputs (per-stage evidence).
EVIDENCE = ANALYSIS / "manuscript_audio_evidence.json"
DIARIZATION = ANALYSIS / "diarization_evidence.json"
DIARIZATION_QC = ANALYSIS / "diarization_cluster_review.json"
SPEAKER_MAPPING = ANALYSIS / "speaker_mapping_review.json"
SOUND = ANALYSIS / "sound_event_evidence.json"
DEFECTS = ANALYSIS / "recording_defect_evidence.json"
MASKING = ANALYSIS / "masking_overlap_evidence.json"
QUEUE = ANALYSIS / "audio_review_queue.json"
VAD = ANALYSIS / "vad_speech_regions.json"
VALIDATOR = ANALYSIS / "manuscript_audio_validator.json"
VIDEO_IDENTITY = ANALYSIS / "video_identity.json"
CONTEXT = ROOT / "task_context.json"
SPEAKER_MAP = ROOT / "speaker_map.json"

# Outputs.
PACKET = ANALYSIS / "manuscript_audio_review_packet.json"
REVIEW_ME = ANALYSIS / "REVIEW_ME.md"
UI_SUGGESTIONS = ANALYSIS / "manuscript_audio_ui_suggestions.json"


# ---------------------------------------------------------------------------
# Confidence vocabulary (shared contract for every feeder)
# ---------------------------------------------------------------------------

STRONG = "STRONG"
MEDIUM = "MEDIUM"
WEAK = "WEAK"
CONFLICT = "CONFLICT"
UNKNOWN = "UNKNOWN"

TIER_ORDER = {
    STRONG: 0,
    CONFLICT: 1,
    MEDIUM: 2,
    WEAK: 3,
    UNKNOWN: 4,
}


def finding(claim, tier, evidence, action, window=None):
    """Build one provenance-carrying conclusion.

    evidence is a list of short strings, each naming the signal it came from.
    action tells the reviewer what to do (or explicitly what NOT to do).
    """
    return {
        "claim": claim,
        "tier": tier,
        "evidence": list(evidence),
        "action": action,
        "window": (
            [round(float(window[0]), 3), round(float(window[1]), 3)]
            if window
            else None
        ),
    }


def load(path, default):
    path = Path(path)

    if not path.exists():
        return default

    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Interval helpers (for ASR-vs-diarization coverage)
# ---------------------------------------------------------------------------

def merge_intervals(intervals, join_gap=0.12):
    intervals = sorted(
        (float(a), float(b))
        for a, b in intervals
        if float(b) > float(a)
    )

    merged = []

    for start, end in intervals:
        if merged and start <= merged[-1][1] + join_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [(a, b) for a, b in merged]


def subtract_intervals(base, holes):
    """Return the parts of base not covered by any hole."""
    result = []

    for start, end in base:
        cursor = start

        for hole_start, hole_end in sorted(holes):
            if hole_end <= cursor or hole_start >= end:
                continue

            if hole_start > cursor:
                result.append((cursor, min(hole_start, end)))

            cursor = max(cursor, hole_end)

            if cursor >= end:
                break

        if cursor < end:
            result.append((cursor, end))

    return result


def total_span(intervals):
    return round(sum(b - a for a, b in intervals), 3)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def build_media(evidence, identity):
    media = dict(evidence.get("media", {}))

    # Surface what the current analyzer captures, and honestly mark the
    # identity/integrity fields the spec wants but the pipeline does not
    # yet extract, so nothing looks silently "covered".
    present = {
        "video_name": identity.get("video_name"),
        "video_sha256": identity.get("video_sha256"),
        "duration_sec": media.get("duration_sec"),
        "sample_rate": media.get("sample_rate"),
    }

    not_yet = [
        key
        for key in (
            "resolution",
            "fps",
            "frame_count",
            "audio_channels",
            "audio_codec",
        )
        if key not in media
    ]

    return {
        "present": present,
        "not_yet_extracted": not_yet,
    }


def build_coverage(evidence, diarization, vad=None):
    """Detect speech that the main ASR did not transcribe (spec 3 and 5).

    The main ASR (WhisperX) can stop early or miss late speech. An independent
    speech-presence signal is needed to notice it. Diarization turns are the
    primary such signal; when diarization is unavailable, Silero VAD regions are
    the fallback. Where independent speech exists with no overlapping ASR
    segment, that is UNTRANSCRIBED_SPEECH.
    """
    duration = float(evidence.get("media", {}).get("duration_sec", 0.0))

    asr_spans = merge_intervals(
        (seg["start"], seg["end"])
        for seg in evidence.get("whisperx_segments", [])
    )

    section = {
        "asr_covered_span_sec": total_span(asr_spans),
        "asr_last_end_sec": round(asr_spans[-1][1], 3) if asr_spans else None,
        "media_duration_sec": duration,
        "speech_presence_source": None,
        "speech_span_sec": None,
        "coverage_ratio": None,
        "untranscribed_regions": [],
        "findings": [],
    }

    turns = diarization.get("turns", []) if diarization else []
    vad_regions = vad.get("regions", []) if vad else []

    # cluster_lookup maps a speech window to diarization speaker(s); empty when
    # VAD is the source (VAD has no speaker identity).
    cluster_lookup = None

    if diarization.get("status") == "complete" and turns:
        speech_spans = merge_intervals(
            (t["start"], t["end"]) for t in turns
        )
        section["speech_presence_source"] = "diarization_turns"
        cluster_lookup = turns
    elif vad and vad.get("status") == "complete" and vad_regions:
        speech_spans = merge_intervals(
            (r["start"], r["end"]) for r in vad_regions
        )
        section["speech_presence_source"] = "silero_vad"
    else:
        # No independent speech signal available at all.
        section["speech_presence_source"] = "unavailable"
        section["findings"].append(
            finding(
                "Speech coverage could not be verified independently.",
                UNKNOWN,
                ["diarization: unavailable", "vad: unavailable"],
                "Listen fully before trusting ASR coverage; no independent "
                "speech-presence signal was available.",
            )
        )
        return section

    speech_total = total_span(speech_spans)
    section["speech_span_sec"] = speech_total

    asr_of_speech = total_span(
        [
            (a, b)
            for a, b in subtract_intervals(
                speech_spans,
                subtract_intervals(speech_spans, asr_spans),
            )
        ]
    )

    section["coverage_ratio"] = (
        round(asr_of_speech / speech_total, 3) if speech_total else None
    )

    # Uncovered = diarized speech minus (padded) ASR spans.
    padded_asr = merge_intervals(
        (a - 0.15, b + 0.15) for a, b in asr_spans
    )

    gaps = [
        (a, b)
        for a, b in subtract_intervals(speech_spans, padded_asr)
        if (b - a) >= 0.35
    ]

    for start, end in gaps:
        # Which cluster(s) speak in this gap? Empty when VAD is the source.
        clusters = sorted({
            t["speaker"]
            for t in (cluster_lookup or [])
            if t["end"] > start and t["start"] < end
        })

        length = end - start
        tier = STRONG if length >= 0.6 else MEDIUM

        section["untranscribed_regions"].append({
            "start": round(start, 3),
            "end": round(end, 3),
            "duration_sec": round(length, 3),
            "diarized_speakers": clusters,
        })

        presence = (
            "diarization: speech turn present"
            if cluster_lookup is not None
            else "silero VAD: speech region present"
        )
        evidence_lines = [
            presence,
            "whisperx: no segment overlaps this window",
        ]
        if clusters:
            evidence_lines.append("cluster(s): " + ", ".join(clusters))

        section["findings"].append(
            finding(
                "UNTRANSCRIBED_SPEECH: speech present but the main ASR "
                "produced no words here.",
                tier,
                evidence_lines,
                "Listen and transcribe manually. Do not treat this window "
                "as silent just because ASR is empty.",
                window=(start, end),
            )
        )

    return section


def build_speaker_clusters(diarization, diarization_qc):
    clusters = diarization_qc.get("clusters", [])

    findings = []

    for cluster in clusters:
        name = cluster.get("speaker_cluster")

        if cluster.get("overlap_only_cluster"):
            findings.append(
                finding(
                    f"{name} is an overlap-only cluster with zero assigned "
                    "words.",
                    MEDIUM,
                    [
                        "diarization QC: zero_assigned_words",
                        f"overlap_ratio: {cluster.get('overlap_ratio')}",
                    ],
                    "Treat as a possible second/overlapping voice. Do NOT "
                    "invent dialogue for it and do NOT map it to a C# on "
                    "diarization alone.",
                )
            )

    return {
        "status": diarization.get("status", "unavailable"),
        "cluster_count": diarization.get("speaker_count"),
        "labels": diarization.get("speaker_labels", []),
        "clusters": clusters,
        "findings": findings,
        "policy": "SPEAKER_XX labels are anonymous clusters, never C# identities.",
    }


def build_character_mapping(evidence, coverage, diarization, speaker_map):
    """Suggest speaker->character links only where a signal chain supports it.

    Never auto-assign. Strong existing mappings come from the human speaker
    map; late untranscribed speech gets a MEDIUM lead when it shares a cluster
    with an already-mapped character.
    """
    findings = []

    profiles = evidence.get("character_voice_profiles", {})
    mapping = speaker_map.get("segments", {}) if speaker_map else {}

    for character, profile in sorted(profiles.items()):
        findings.append(
            finding(
                f"{character} speaks the mapped transcript segments "
                f"{profile.get('confirmed_segment_ids')}.",
                STRONG,
                [
                    "human speaker map: confirmed",
                    f"mapped speech: {profile.get('usable_speech_duration_sec')}s",
                    f"words: {profile.get('total_word_count')}",
                ],
                "Safe reviewer default for these segments.",
            )
        )

    # Which cluster does each mapped character belong to?
    segment_speakers = {
        int(item["segment"]): item.get("speaker")
        for item in diarization.get("segment_speakers", [])
    } if diarization else {}

    character_cluster = {}

    for seg_str, character in mapping.items():
        cluster = segment_speakers.get(int(seg_str))
        if cluster:
            character_cluster.setdefault(character, set()).add(cluster)

    # Late untranscribed speech: if it is the same cluster as a mapped
    # character, surface a MEDIUM lead (needs listening to confirm content).
    for region in coverage.get("untranscribed_regions", []):
        region_clusters = set(region.get("diarized_speakers", []))

        matches = [
            character
            for character, clusters in character_cluster.items()
            if region_clusters & clusters
        ]

        if len(matches) == 1:
            findings.append(
                finding(
                    f"Untranscribed speech at "
                    f"{region['start']}-{region['end']}s is the same "
                    f"diarization cluster as {matches[0]}.",
                    MEDIUM,
                    [
                        "diarization: shared cluster "
                        + ", ".join(sorted(region_clusters)),
                        f"already mapped: {matches[0]}",
                    ],
                    f"Likely {matches[0]}, but listen to confirm before "
                    "assigning. Do not auto-assign from cluster identity.",
                    window=(region["start"], region["end"]),
                )
            )
        elif len(matches) > 1:
            findings.append(
                finding(
                    f"Untranscribed speech at "
                    f"{region['start']}-{region['end']}s matches multiple "
                    f"mapped characters.",
                    CONFLICT,
                    ["diarization: ambiguous cluster overlap"],
                    "Resolve by listening; do not guess the speaker.",
                    window=(region["start"], region["end"]),
                )
            )

    return {
        "human_confirmed": {
            character: profile.get("confirmed_segment_ids")
            for character, profile in profiles.items()
        },
        "findings": findings,
        "policy": "Diarization suggests; only human-confirmed maps are defaults.",
    }


def build_sound_events(sound):
    candidates = sound.get(
        "shot_assigned_candidates",
        sound.get("candidate_events", []),
    )

    findings = []
    music_findings = []

    for item in candidates:
        label = item.get("label", "unknown")
        strength = item.get("evidence_strength", "weak")
        is_music = "music" in label.lower()

        if strength == "weak":
            f = finding(
                f"{label}: weak model evidence only "
                f"(max score {item.get('max_score')}).",
                WEAK,
                [f"AST audioset: {item.get('max_score')}"],
                "Listening cue only. Do NOT create a Manuscript event from "
                "this evidence alone.",
                window=(item.get("start"), item.get("end")),
            )
        else:
            tier = STRONG if strength == "strong" else MEDIUM
            f = finding(
                f"{label}: {strength} model evidence "
                f"(max score {item.get('max_score')}).",
                tier,
                [f"AST audioset: {item.get('max_score')}"],
                "Confirm by listening; then it may support a Sound event "
                f"(source candidate: {item.get('manuscript_source_candidate')}).",
                window=(item.get("start"), item.get("end")),
            )

        if is_music:
            music_findings.append(f)
        else:
            findings.append(f)

    return {
        "sound_findings": findings,
        "music_findings": music_findings,
        "policy": "Weak evidence never becomes an event. Visible motion is "
                  "not evidence of audible sound.",
    }


def build_object_status(context, sound):
    objects = context.get("objects", [])

    if not objects:
        return {
            "objects_in_task": [],
            "note": "No objects (O#) in the locked task; nothing to profile.",
            "findings": [],
        }

    def object_id(obj):
        return obj.get("id", "?") if isinstance(obj, dict) else str(obj)

    # No per-object acoustic attribution feeder yet; be honest about it.
    findings = [
        finding(
            f"{object_id(obj)}: no acoustic evidence attributed.",
            UNKNOWN,
            ["object-sound attribution: not_yet_implemented"],
            "Default to Silent unless you hear it. Visible movement is not "
            "audible sound.",
        )
        for obj in objects
    ]

    return {
        "objects_in_task": objects,
        "findings": findings,
    }


def build_recording_defects(defects):
    rows = defects.get("recording_defects", {})

    findings = []

    for name, row in rows.items():
        candidate = row.get("evidence_candidate")

        findings.append(
            finding(
                f"{name}: machine evidence = {candidate}.",
                UNKNOWN,
                [f"defect analyzer: {candidate}"],
                "Reviewer owns the final Yes/No. 'not_detected' does not "
                "mean choose No automatically.",
            )
        )

    return {
        "rows": rows,
        "findings": findings,
    }


def build_overlap_masking(masking):
    regions = masking.get("speaker_overlap_regions", [])
    risks = masking.get("low_confidence_word_overlap_risks", [])

    findings = []

    for region in regions:
        findings.append(
            finding(
                "Speaker overlap "
                f"{region['start']}-{region['end']}s "
                f"({', '.join(region.get('speakers', []))}).",
                MEDIUM,
                [f"diarization overlap: {region.get('duration_sec')}s"],
                "Overlap is not masking. Listen: does intelligibility "
                "actually drop?",
                window=(region["start"], region["end"]),
            )
        )

    for risk in risks:
        findings.append(
            finding(
                f"Low-confidence word \"{risk.get('word')}\" sits inside a "
                "speaker overlap.",
                MEDIUM,
                risk.get("risk_reasons", []),
                "Possible masking. Confirm by listening before using masking "
                "language.",
                window=(risk.get("start"), risk.get("end")),
            )
        )

    return {
        "overlap_regions": regions,
        "findings": findings,
        "policy": "overlap != masking until intelligibility loss is heard.",
    }


def build_clip_boundaries(evidence, coverage):
    duration = float(evidence.get("media", {}).get("duration_sec", 0.0))
    segments = evidence.get("whisperx_segments", [])

    findings = []

    if segments:
        first_start = float(segments[0]["start"])
        last_end = coverage.get("asr_last_end_sec") or float(
            segments[-1]["end"]
        )

        if first_start <= 0.3:
            findings.append(
                finding(
                    "Clip begins with speech already in progress or right at "
                    "the start.",
                    MEDIUM,
                    [f"first ASR word at {first_start}s"],
                    "Check whether the clip starts mid-sentence.",
                )
            )

        trailing = duration - last_end

        if coverage.get("untranscribed_regions"):
            findings.append(
                finding(
                    "Speech continues after the last ASR word; the clip does "
                    "not end in silence.",
                    STRONG,
                    [
                        f"last ASR end {last_end}s",
                        f"media duration {duration}s",
                        "untranscribed speech regions present",
                    ],
                    "Transcribe the late speech. A visual cut is not an audio "
                    "cutoff.",
                )
            )
        elif trailing > 0.5:
            findings.append(
                finding(
                    f"{round(trailing, 2)}s of tail after the last ASR word.",
                    MEDIUM,
                    [f"last ASR end {last_end}s", f"duration {duration}s"],
                    "Confirm the tail is genuinely non-speech.",
                )
            )

    return {"findings": findings}


# ---------------------------------------------------------------------------
# UI suggestions (spec 46): sparse, MEDIUM-or-better only
# ---------------------------------------------------------------------------

def build_ui_suggestions(evidence):
    profiles = evidence.get("character_voice_profiles", {})

    characters = {}

    for character, profile in sorted(profiles.items()):
        fields = {}

        # Voice status: confirmed multi-word mapped speech supports "Observed".
        if (
            profile.get("total_word_count", 0) >= 3
            and float(profile.get("usable_speech_duration_sec", 0.0)) >= 1.0
        ):
            fields["voice_status"] = {
                "value": "Observed",
                "confidence": STRONG,
                "evidence": "human-mapped multi-word clear speech",
            }

        gated = [
            ("speed", "speed_candidate", "speed_evidence_coverage"),
            ("delivery", "delivery_candidate", "delivery_evidence_coverage"),
            (
                "recorded_level",
                "recorded_level_candidate",
                "recorded_level_evidence_coverage",
            ),
        ]

        for ui_field, candidate_key, coverage_key in gated:
            value = profile.get(candidate_key)

            if not value:
                continue

            coverage = float(profile.get(coverage_key, 0.0))
            tier = STRONG if coverage >= 0.85 else MEDIUM

            fields[ui_field] = {
                "value": value,
                "confidence": tier,
                "evidence": f"predominant across {coverage:.0%} of mapped speech",
            }

        if fields:
            characters[character] = fields

    return {
        "characters": characters,
        "policy": [
            "Only fields with MEDIUM+ evidence appear here.",
            "Blank fields are deliberate: pitch, speaking level, clarity, "
            "tone, texture, mix role, and confidence need human listening.",
            "These are suggestions to confirm in the live UI, not decisions.",
        ],
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def collect_all_findings(sections):
    all_findings = []

    for name, section in sections.items():
        if not isinstance(section, dict):
            continue

        for key in ("findings", "sound_findings", "music_findings"):
            for f in section.get(key, []):
                all_findings.append({"section": name, **f})

    all_findings.sort(key=lambda f: TIER_ORDER.get(f["tier"], 9))
    return all_findings


def build_confidence_summary(all_findings):
    counts = {STRONG: 0, MEDIUM: 0, WEAK: 0, CONFLICT: 0, UNKNOWN: 0}

    for f in all_findings:
        counts[f["tier"]] = counts.get(f["tier"], 0) + 1

    needs_human = [
        f
        for f in all_findings
        if f["tier"] in (MEDIUM, CONFLICT, UNKNOWN)
    ]

    return {
        "total_findings": len(all_findings),
        "by_tier": counts,
        "human_review_finding_count": len(needs_human),
    }


def build_packet():
    evidence = load(EVIDENCE, {})
    diarization = load(DIARIZATION, {})
    diarization_qc = load(DIARIZATION_QC, {})
    sound = load(SOUND, {})
    defects = load(DEFECTS, {})
    masking = load(MASKING, {})
    queue = load(QUEUE, [])
    validator = load(VALIDATOR, {})
    identity = load(VIDEO_IDENTITY, {})
    context = load(CONTEXT, {})
    speaker_map = load(SPEAKER_MAP, {})
    vad = load(VAD, {})

    coverage = build_coverage(evidence, diarization, vad)

    sections = {
        "media": build_media(evidence, identity),
        "locked_task_structure": {
            "characters": context.get("characters", []),
            "objects": context.get("objects", []),
            "shots": context.get("shots", []),
            "note": "Live locked task wins over machine shot detection.",
        },
        "coverage_gaps": coverage,
        "speaker_clusters": build_speaker_clusters(
            diarization, diarization_qc
        ),
        "character_mapping": build_character_mapping(
            evidence, coverage, diarization, speaker_map
        ),
        "voice_profiles": evidence.get("character_voice_profiles", {}),
        "sound_events": build_sound_events(sound),
        "object_sound_status": build_object_status(context, sound),
        "recording_defects": build_recording_defects(defects),
        "overlap_masking": build_overlap_masking(masking),
        "clip_boundaries": build_clip_boundaries(evidence, coverage),
        "shot_events": evidence.get("shot_audio_evidence", []),
        "review_queue": queue,
        "validator_predictions": validator,
        "not_yet_implemented": [
            "asr_consensus (second ASR model / word agreement)",
            "active_speaker_detection (lip/audio sync, needs video frames)",
            "ambience_classification (indoor/outdoor/traffic/ocean)",
            "object_sound_attribution",
        ],
        "separate_tools": {
            "pasted_back_qa": "manuscript_audio_qa.py (field-vs-prose, cast "
                              "integrity, final-text QA)",
            "task_seed_parser": "manuscript_audio_seed.py",
        },
    }

    all_findings = collect_all_findings(sections)
    sections["confidence_summary"] = build_confidence_summary(all_findings)
    sections["ranked_findings"] = all_findings

    return sections, evidence


# ---------------------------------------------------------------------------
# REVIEW_ME.md
# ---------------------------------------------------------------------------

def build_review_me(sections, evidence):
    lines = []

    def add(text=""):
        lines.append(text)

    media = sections["media"]["present"]
    summary = sections["confidence_summary"]
    coverage = sections["coverage_gaps"]

    add("# REVIEW_ME")
    add()
    add("> Machine evidence only. Actual audio and video remain the truth.")
    add("> Numeric times are review aids. Never copy them into Final Audio Text.")
    add()

    # Headline
    counts = summary["by_tier"]
    add(
        f"**{summary['total_findings']} findings** — "
        f"{counts[STRONG]} strong, {counts[MEDIUM]} medium, "
        f"{counts[WEAK]} weak, {counts[CONFLICT]} conflict, "
        f"{counts[UNKNOWN]} unknown."
    )

    if coverage.get("coverage_ratio") is not None:
        add(
            f"Speech coverage by ASR: **{coverage['coverage_ratio']:.0%}** "
            f"(source: {coverage['speech_presence_source']})."
        )

    add()
    add("## MEDIA")
    add(
        f"{media.get('duration_sec')}s, "
        f"{len(sections['locked_task_structure']['shots'])} locked shot(s)."
    )
    if sections["media"]["not_yet_extracted"]:
        add(
            "Not yet extracted: "
            + ", ".join(sections["media"]["not_yet_extracted"])
            + "."
        )

    def dump(title, findings):
        if not findings:
            return
        add()
        add(f"## {title}")
        for f in findings:
            window = ""
            if f.get("window"):
                window = f" [{f['window'][0]}-{f['window'][1]}s]"
            add(f"- **{f['tier']}**{window}: {f['claim']}")
            add(f"    - {f['action']}")

    ranked = sections["ranked_findings"]

    dump("STRONG — safe defaults", [f for f in ranked if f["tier"] == STRONG])
    dump(
        "NEEDS REVIEW — medium / conflict",
        [f for f in ranked if f["tier"] in (MEDIUM, CONFLICT)],
    )
    dump(
        "DO NOT AUTO-ASSERT — weak / unknown",
        [f for f in ranked if f["tier"] in (WEAK, UNKNOWN)],
    )

    # Untranscribed speech gets its own explicit callout.
    if coverage.get("untranscribed_regions"):
        add()
        add("## UNTRANSCRIBED SPEECH (listen + transcribe)")
        for region in coverage["untranscribed_regions"]:
            add(
                f"- {region['start']}-{region['end']}s "
                f"({region['duration_sec']}s), "
                f"cluster {', '.join(region['diarized_speakers'])}"
            )

    add()
    return "\n".join(lines) + "\n"


def main():
    print("=== MANUSCRIPT AUDIO MASTER AGGREGATOR ===")

    if not EVIDENCE.exists():
        raise FileNotFoundError(
            f"Main evidence missing: {EVIDENCE}. Run the pipeline first."
        )

    sections, evidence = build_packet()

    ANALYSIS.mkdir(parents=True, exist_ok=True)

    with PACKET.open("w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)

    review_me = build_review_me(sections, evidence)
    REVIEW_ME.write_text(review_me, encoding="utf-8")

    ui = build_ui_suggestions(evidence)

    with UI_SUGGESTIONS.open("w", encoding="utf-8") as f:
        json.dump(ui, f, indent=2, ensure_ascii=False)

    summary = sections["confidence_summary"]
    counts = summary["by_tier"]

    print("MASTER PACKET: PASS")
    print("Packet:", PACKET)
    print("Human summary:", REVIEW_ME)
    print("UI suggestions:", UI_SUGGESTIONS)
    print(
        f"Findings: {summary['total_findings']} "
        f"(STRONG {counts[STRONG]}, MEDIUM {counts[MEDIUM]}, "
        f"WEAK {counts[WEAK]}, CONFLICT {counts[CONFLICT]}, "
        f"UNKNOWN {counts[UNKNOWN]})"
    )

    untranscribed = sections["coverage_gaps"].get("untranscribed_regions", [])
    if untranscribed:
        print(f"UNTRANSCRIBED_SPEECH regions: {len(untranscribed)}")

    return sections


if __name__ == "__main__":
    main()
