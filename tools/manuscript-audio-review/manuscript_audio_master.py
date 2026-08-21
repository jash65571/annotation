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
import re


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
ASR_CONSENSUS = ANALYSIS / "asr_consensus_evidence.json"
SPEAKER_FACE_MAPPING = ANALYSIS / "speaker_mapping_evidence.json"
SOUND_FUSION = ANALYSIS / "sound_fusion_evidence.json"
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
    #
    # 3.5: sample rates are now explicit. `analysis_sample_rate` is the
    # resampled 16 kHz WAV the pipeline actually analyzed; `source_sample_rate`
    # is the ORIGINAL video's audio rate (probed from the source by ffprobe).
    # The two must never be conflated.
    present = {
        "video_name": identity.get("video_name"),
        "video_sha256": identity.get("video_sha256"),
        "duration_sec": media.get("duration_sec"),
        "analysis_sample_rate": (
            media.get("analysis_sample_rate")
            or media.get("sample_rate")
        ),
        "source_sample_rate": identity.get("source_sample_rate"),
        "audio_channels": identity.get("audio_channels"),
        "audio_codec": identity.get("audio_codec"),
    }

    not_yet = [
        key
        for key in (
            "resolution",
            "fps",
            "frame_count",
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


def build_asr_consensus(asr_consensus, independent_speech_regions=None):
    """Wrap the Phase 3A secondary-ASR comparison into shared-schema findings.

    Never turns disagreement into a transcript edit -- it only tells the
    reviewer which windows a second independent model does not corroborate.

    3.5: targeted reruns are demoted to *hypotheses*, never recovered truth.
    A rerun that conflicts with surrounding evidence (no independent speech
    signal, or content that disagrees with the primary transcript in the
    same window) is marked CONFLICT and requires listening.
    """
    findings = []

    independent_speech_regions = independent_speech_regions or []

    status = asr_consensus.get("status", "unavailable")

    if status != "complete":
        findings.append(
            finding(
                "Secondary ASR pass did not run; transcript was not "
                "cross-checked by a second model.",
                UNKNOWN,
                [f"asr_consensus status: {status}"]
                + ([asr_consensus["error"]] if asr_consensus.get("error") else []),
                "Apply the same listening discipline as before Phase 3A; "
                "no cross-model corroboration is available.",
            )
        )
        return {
            "coverage": None,
            "word_consensus": [],
            "conflicts": [],
            "rerun_windows": [],
            "findings": findings,
            "policy": "Fails soft: base packet is unaffected when the "
                      "secondary model cannot run.",
        }

    coverage = asr_consensus.get("coverage", {}) or {}

    for c in asr_consensus.get("conflicts", []):
        findings.append(
            finding(
                f"Two ASR models disagree: primary heard "
                f"\"{c['primary_word']}\", secondary heard "
                f"\"{c['secondary_word']}\".",
                CONFLICT,
                [
                    f"primary score: {c.get('primary_score')}",
                    f"secondary score: {c.get('secondary_score')}",
                ],
                "Listen and decide the word manually; do not average or "
                "pick one model automatically.",
                window=(c["start"], c["end"]),
            )
        )

    for a, b in merge_intervals(
        (w["start"], w["end"])
        for w in asr_consensus.get("secondary_only_words", [])
    ):
        length = b - a
        tier = STRONG if length >= 0.6 else MEDIUM
        findings.append(
            finding(
                "Secondary ASR model recovered speech the primary "
                "transcript missed entirely.",
                tier,
                ["secondary model produced words; primary produced none"],
                "Listen and transcribe manually. Do not trust primary-only "
                "coverage as complete.",
                window=(a, b),
            )
        )

    # 3.5: reruns are hypotheses, not recovered truth. If a rerun suggests
    # words where no independent speech signal exists AND the primary
    # transcript has no words, or its content disagrees with the primary
    # words in the same window, it CONFLICTS with surrounding evidence and
    # must be listened to before it can be trusted at all.
    for rerun in asr_consensus.get("reruns_executed", []):
        if not rerun.get("recovered_text"):
            continue

        start, end = rerun["window"]

        corroborated_by_speech = any(
            min(float(end), float(r_end)) - max(float(start), float(r_start)) > 0.05
            for r_start, r_end in independent_speech_regions
        )

        primary_words_in_window = [
            w for w in asr_consensus.get("word_consensus", [])
            if min(float(end), float(w.get("end", end))) - max(float(start), float(w.get("start", start))) > 0.05
        ]

        conflicts = not corroborated_by_speech and not primary_words_in_window

        if primary_words_in_window:
            rerun_tokens = {
                re.sub(r"[^\w']", "", t.lower())
                for t in rerun["recovered_text"].split()
                if re.sub(r"[^\w']", "", t.lower())
            }
            primary_tokens = {
                re.sub(r"[^\w']", "", str(w.get("word", "")).lower())
                for w in primary_words_in_window
            }
            primary_tokens.discard("")
            if primary_tokens:
                overlap = len(rerun_tokens & primary_tokens) / len(primary_tokens)
                if overlap < 0.5:
                    conflicts = True

        tier = CONFLICT if conflicts else MEDIUM
        evidence_lines = [
            f"rerun reason(s): {', '.join(rerun.get('reasons', []))}"
        ]

        if not corroborated_by_speech and not primary_words_in_window:
            evidence_lines.append(
                "no independent VAD/diarization speech signal overlaps this "
                "window and the primary transcript has no words here"
            )
        elif conflicts:
            evidence_lines.append(
                "rerun content disagrees with the primary transcript in "
                "the same window"
            )

        action = (
            "Hypothesis only, not recovered truth. It conflicts with "
            "surrounding evidence; do NOT add it to the transcript without "
            "listening."
            if conflicts
            else "Hypothesis only, not recovered truth. Confirm by listening "
                 "before adding this to the transcript."
        )

        findings.append(
            finding(
                "Targeted rerun hypothesis: a rerun pass suggested "
                f"\"{rerun['recovered_text']}\" in a window the primary "
                "transcript left empty.",
                tier,
                evidence_lines,
                action,
                window=tuple(rerun["window"]),
            )
        )

    # Advisory-only signals (3A.1-4/5): never above MEDIUM, always a
    # listening cue, never a transcript edit or a name correction.
    for risk in asr_consensus.get("hallucination_risk_words", []):
        tier = MEDIUM if risk["tier"] == "MEDIUM" else WEAK
        findings.append(
            finding(
                f"Possible ASR hallucination: \"{risk['word']}\" "
                f"(advisory score {risk['score']}).",
                tier,
                risk["reasons"],
                "Advisory only. Listen before deciding whether this word "
                "was actually spoken.",
                window=(risk["start"], risk["end"]),
            )
        )

    for risk in asr_consensus.get("proper_noun_risk_words", []):
        findings.append(
            finding(
                f"\"{risk['word']}\" may be a name/proper noun worth "
                "double-checking.",
                WEAK,
                risk["reasons"],
                "Do not auto-correct the spelling; confirm by listening "
                "and/or against the locked task cast.",
                window=(risk["start"], risk["end"]),
            )
        )

    if asr_consensus.get("reruns_skipped_count", 0) > 0:
        findings.append(
            finding(
                f"{asr_consensus['reruns_skipped_count']} rerun window(s) "
                "were identified but not executed (rerun cap reached).",
                UNKNOWN,
                ["rerun_windows list is complete; reruns_executed is capped"],
                "Review the remaining rerun_windows manually if time allows.",
            )
        )

    agreement = coverage.get("model_agreement_pct")

    if agreement is not None:
        tier = STRONG if agreement >= 0.85 else MEDIUM if agreement >= 0.6 else WEAK
        findings.append(
            finding(
                f"Model agreement on matched words: {agreement:.0%}.",
                tier,
                [
                    f"primary words: {asr_consensus.get('primary_word_count')}",
                    f"secondary words: {asr_consensus.get('secondary_word_count')}",
                ],
                "High agreement supports the transcript; it does not "
                "replace listening to flagged windows.",
            )
        )

    return {
        "coverage": coverage,
        "word_consensus": asr_consensus.get("word_consensus", []),
        "secondary_only_words": asr_consensus.get("secondary_only_words", []),
        "conflicts": asr_consensus.get("conflicts", []),
        "rerun_windows": asr_consensus.get("rerun_windows", []),
        "reruns_executed": asr_consensus.get("reruns_executed", []),
        "hallucination_risk_words": asr_consensus.get("hallucination_risk_words", []),
        "proper_noun_risk_words": asr_consensus.get("proper_noun_risk_words", []),
        "secondary_model": asr_consensus.get("secondary_model"),
        "findings": findings,
        "policy": "Cross-model comparison is evidence, not a transcript "
                  "edit. Never auto-insert [uncertain]/[unintelligible]/"
                  "[inaudible].",
    }


def build_speaker_face_mapping(mapping):
    """Wrap Phase 3B face-track / active-speaker evidence into shared-schema
    findings. Never turns a face track or diarization cluster into a
    character (C#) identity -- only ever a labeled candidate for the
    reviewer to confirm by watching.
    """
    findings = []

    status = mapping.get("status", "unavailable") if mapping else "unavailable"

    if status != "complete":
        findings.append(
            finding(
                "Face tracking / active-speaker mapping did not run.",
                UNKNOWN,
                [f"speaker_mapping status: {status}"],
                "No visual speaker evidence available; rely on diarization "
                "clusters and audio-only listening.",
            )
        )
        return {
            "face_tracks": [],
            "active_speaker_windows": [],
            "cluster_to_face_candidates": [],
            "face_to_character_candidates": [],
            "findings": findings,
            "policy": "Fails soft: base packet is unaffected when face "
                      "tracking cannot run.",
        }

    if mapping.get("face_worker_status") != "complete":
        findings.append(
            finding(
                "Face detection did not complete "
                f"(status: {mapping.get('face_worker_status')}).",
                UNKNOWN,
                ["face worker did not produce usable tracks"],
                "Treat every speech window as possibly off-screen.",
            )
        )
    elif not mapping.get("face_tracks"):
        findings.append(
            finding(
                "No face was detected anywhere in the sampled frames.",
                UNKNOWN,
                ["face_tracks: empty"],
                "All speech in this clip is likely off-screen narration, "
                "or faces are too small/occluded to detect. Confirm by "
                "watching before assuming a visible speaker exists.",
            )
        )

    for window in mapping.get("active_speaker_windows", []):
        tier = window["tier"]

        if tier == UNKNOWN:
            findings.append(
                finding(
                    "No visible face during this speech window.",
                    UNKNOWN,
                    [window["reason"]],
                    window["action"],
                    window=(window["start"], window["end"]),
                )
            )
        elif tier == CONFLICT:
            candidate_ids = ", ".join(c["face_id"] for c in window["candidates"])
            findings.append(
                finding(
                    f"Multiple visible faces ({candidate_ids}) show similar "
                    "mouth motion during this speech window.",
                    CONFLICT,
                    [
                        f"{c['face_id']}: motion={c['motion_score']}, "
                        f"visibility={c['visibility_ratio']}"
                        for c in window["candidates"]
                    ],
                    window["action"],
                    window=(window["start"], window["end"]),
                )
            )
        elif window["candidates"]:
            top = max(window["candidates"], key=lambda c: c["motion_score"])
            findings.append(
                finding(
                    f"{top['face_id']} is the best visible active-speaker "
                    "candidate for this window (mouth-motion evidence "
                    "only).",
                    tier,
                    [
                        f"motion_score={top['motion_score']}",
                        f"visibility_ratio={top['visibility_ratio']}",
                        "signal: mouth-aspect-ratio motion, not verified "
                        "audiovisual sync",
                    ],
                    window["action"],
                    window=(window["start"], window["end"]),
                )
            )

    for candidate in mapping.get("cluster_to_face_candidates", []):
        findings.append(
            finding(
                f"{candidate['speaker_cluster']} co-occurs with "
                f"{candidate['face_id']}'s mouth motion in "
                f"{candidate['supporting_windows']} window(s) "
                f"({candidate['consistency_ratio']:.0%} consistency).",
                candidate["tier"],
                [f"consistency_ratio: {candidate['consistency_ratio']}"],
                candidate["action"],
            )
        )

    for candidate in mapping.get("face_to_character_candidates", []):
        findings.append(
            finding(
                f"{candidate['face_id']} is human-confirmed as "
                f"{candidate['character']}.",
                candidate["tier"],
                ["face_character_map.json: human-confirmed"],
                candidate["action"],
            )
        )

    return {
        "face_tracks": mapping.get("face_tracks", []),
        "active_speaker_windows": mapping.get("active_speaker_windows", []),
        "cluster_to_face_candidates": mapping.get("cluster_to_face_candidates", []),
        "face_to_character_candidates": mapping.get("face_to_character_candidates", []),
        "findings": findings,
        "policy": "Face-track ids (F#) and diarization clusters (SPEAKER_XX) "
                  "never become a character (C#) identity without human "
                  "confirmation. Mouth-motion evidence never exceeds MEDIUM.",
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


def _normalize_tier(tier):
    if tier in (STRONG, MEDIUM, WEAK, CONFLICT, UNKNOWN):
        return tier
    return UNKNOWN


def build_transients(sound_fusion):
    """Wrap Phase 3.5 transient/SFX detector evidence into shared-schema
    findings. A strong unexplained transient is a high-priority listening
    window -- evidence a Sound event exists even when no model can name it.
    It never becomes an automatic Manuscript event.
    """
    sound_fusion = sound_fusion or {}
    transients = sound_fusion.get("transients", {}) or {}
    events = transients.get("events", []) or []
    findings = [dict(f) for f in (transients.get("findings") or [])]

    return {
        "status": (
            transients.get("status")
            if transients.get("status") in ("complete", "failed", "unavailable")
            else ("complete" if events else "unavailable")
        ),
        "events": events,
        "findings": findings,
        "policy": (
            "Transients are acoustic evidence, not names. Strong unexplained "
            "transients become high-priority review windows; they never "
            "auto-create a Sound event."
        ),
    }


def build_fusion_sound_events(sound_fusion):
    """Wrap Phase 3C fused sound evidence (PANNs+CLAP human-nonverbal /
    object-SFX candidates) into shared-schema findings.

    Never turns a candidate into an automatic Manuscript event. WEAK /
    CONFLICT / UNKNOWN stay out of UI suggestions (enforced in
    build_ui_suggestions). Overlap is never asserted as masking here.
    """
    sound_fusion = sound_fusion or {}
    candidates = sound_fusion.get("sound_events", {}).get("candidates", [])
    status = sound_fusion.get("status", "unavailable")
    findings = []

    if status != "complete":
        findings.append(
            finding(
                "Sound / music / ambience fusion did not run.",
                UNKNOWN,
                [f"sound_fusion status: {status}"]
                + ([sound_fusion["error"]] if sound_fusion.get("error") else []),
                "No PANNs/CLAP sound evidence available; apply the same "
                "listening discipline as before Phase 3C.",
            )
        )
        return {
            "status": status,
            "candidates": [],
            "findings": findings,
            "policy": "Fails soft: base packet is unaffected when 3C cannot "
                      "run.",
        }

    for c in candidates:
        tier = _normalize_tier(c.get("tier"))
        findings.append(
            finding(
                f"{c['semantic_label']}: {tier} sound evidence.",
                tier,
                c.get("evidence", []),
                c.get("reviewer_action", "Confirm by listening."),
                window=(c["start"], c["end"]),
            )
        )

    return {
        "status": status,
        "candidates": candidates,
        "findings": findings,
        "policy": "PANNs/CLAP fusion is evidence, not a decision. WEAK / "
                  "CONFLICT / UNKNOWN never reach UI suggestions.",
    }


def build_music(sound_fusion):
    """Wrap Phase 3C fused music evidence into shared-schema findings.

    Music is deliberately conservative: one PANNs score, one CLAP score, or
    rhythmicity alone is never enough. WEAK music must say DO NOT CREATE
    MUSIC EVENT WITHOUT LISTENING.
    """
    sound_fusion = sound_fusion or {}
    music = sound_fusion.get("music", {})
    regions = music.get("regions", [])
    overall = _normalize_tier(music.get("overall_confidence"))
    findings = []

    for r in regions:
        tier = _normalize_tier(r.get("tier"))
        findings.append(
            finding(
                f"Music: {tier} evidence.",
                tier,
                r.get("evidence", []),
                r.get("reviewer_action",
                      "DO NOT CREATE MUSIC EVENT WITHOUT LISTENING"),
                window=(r["start"], r["end"]),
            )
        )

    if not regions:
        findings.append(
            finding(
                "No supported music candidate.",
                UNKNOWN,
                ["panns music scores", "clap music prompts"],
                "No music event unless you hear music. Silence is not "
                "evidence of music.",
            )
        )

    return {
        "regions": regions,
        "overall_confidence": overall,
        "findings": findings,
        "policy": "One model score, one CLAP score, or rhythmicity alone is "
                  "never enough. Only MEDIUM+ may reach UI suggestions.",
    }


def build_ambience(sound_fusion):
    """Wrap Phase 3C fused ambience evidence into shared-schema findings.

    Enforces the semantic-vs-UI-source split: outdoor environmental sound
    never maps to a named indoor category, and only genuine room tone maps
    to Room ambience.
    """
    sound_fusion = sound_fusion or {}
    ambience = sound_fusion.get("ambience", {})
    candidates = ambience.get("candidates", [])
    findings = []

    for c in candidates:
        tier = _normalize_tier(c.get("tier"))
        label = c.get("semantic_candidate") or c.get("semantic_label")
        findings.append(
            finding(
                f"Ambience: {label} ({tier}).",
                tier,
                c.get("evidence", []),
                "Confirm by listening. Do not force a named indoor category "
                "unless the media is genuinely indoors.",
                window=(c["start"], c["end"]),
            )
        )

    return {
        "candidates": candidates,
        "findings": findings,
        "policy": "Semantic class and UI source are separate. Outdoor "
                  "environmental sound never becomes Room ambience.",
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
    """3.5 stricter masking: a masking-style finding only exists when a
    low-confidence word really overlaps ANOTHER source -- a supported
    non-speech sound or another speaker. A plain diarization overlap with no
    word evidence, or a low-confidence word with no overlapping source, is
    not a masking warning: it is just "Low-confidence speech: re-listen",
    which the review queue already handles.
    """
    regions = masking.get("speaker_overlap_regions", [])
    risks = masking.get("low_confidence_word_overlap_risks", [])

    findings = []

    for risk in risks:
        reasons = list(risk.get("risk_reasons", []))

        if risk.get("sound_overlap_candidates"):
            labels = ", ".join(
                str(c.get("label", "?"))
                for c in risk["sound_overlap_candidates"]
            )
            findings.append(
                finding(
                    f"Possible masking: low-confidence word "
                    f"\"{risk.get('word')}\" overlaps a supported "
                    f"non-speech sound ({labels}).",
                    MEDIUM,
                    reasons,
                    "Another source is actually present over this word. "
                    "Listen: call it masking only if intelligibility really "
                    "drops.",
                    window=(risk.get("start"), risk.get("end")),
                )
            )
        elif risk.get("speaker_overlap_regions"):
            findings.append(
                finding(
                    f"Possible masking: low-confidence word "
                    f"\"{risk.get('word')}\" sits inside a speaker overlap.",
                    MEDIUM,
                    reasons,
                    "Another speaker is present over this word. Listen: "
                    "call it masking only if intelligibility really drops.",
                    window=(risk.get("start"), risk.get("end")),
                )
            )
        # No real overlapping source -> not a masking candidate. The word
        # stays a plain low-confidence-speech listening item.

    return {
        "overlap_regions": regions,
        "word_risks": risks,
        "findings": findings,
        "policy": (
            "Masking requires a real overlapping source AND intelligibility "
            "loss. Plain low-confidence speech or overlap alone is never a "
            "masking warning."
        ),
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

def build_ui_suggestions(evidence, sound_fusion=None):
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

    # Phase 3C: sparse Sound/Music suggestions. Only MEDIUM+ may appear;
    # WEAK / CONFLICT / UNKNOWN are deliberately excluded. These are live-UI
    # suggestions, not automatic events, and never produce Caption Sentence
    # or Final Audio Text.
    sounds = []
    music = []

    if sound_fusion and sound_fusion.get("status") == "complete":
        def _sound_suggestion(c):
            return {
                "event_type": "Sound",
                "source": c.get("ui_source_candidate") or "Unidentified sound",
                "recorded_level": c.get("recorded_level_candidate"),
                "mix_role": c.get("mix_role_candidate"),
                "description": c.get("description_candidate"),
                "relationship": (
                    " ".join(c.get("relationship_candidates", [])) or None
                ),
                "confidence": c["tier"],
                "evidence_ids": [c.get("id", c.get("semantic_label"))],
            }

        for c in sound_fusion.get("sound_events", {}).get("candidates", []):
            if c.get("tier") in (STRONG, MEDIUM):
                sounds.append(_sound_suggestion(c))

        for c in sound_fusion.get("ambience", {}).get("candidates", []):
            if c.get("tier") in (STRONG, MEDIUM):
                sounds.append(_sound_suggestion(c))

        for r in sound_fusion.get("music", {}).get("regions", []):
            if r.get("tier") not in (STRONG, MEDIUM):
                continue
            music.append({
                "event_type": "Music",
                "source": "Music",
                "recorded_level": r.get("recorded_level_candidate"),
                "mix_role": r.get("mix_role_candidate"),
                "description": "Music",
                "relationship": None,
                "confidence": r["tier"],
                "evidence_ids": [r.get("id", "music")],
            })

    return {
        "characters": characters,
        "sounds": sounds,
        "music": music,
        "policy": [
            "Only fields with MEDIUM+ evidence appear here.",
            "Blank fields are deliberate: pitch, speaking level, clarity, "
            "tone, texture, mix role, and confidence need human listening.",
            "Sound suggestions are never automatic events and never produce "
            "Caption Sentence or Final Audio Text.",
            "These are suggestions to confirm in the live UI, not decisions.",
        ],
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def shot_for_window(start, end, shots):
    """Best-overlap locked shot for a window, or None (3.5: every finding
    carries its shot when the seed defines one)."""
    best = None
    best_overlap = 0.0

    for shot in shots:
        shot_start = float(shot.get("start", 0.0))
        shot_end = float(shot.get("end", 0.0))
        overlap = max(
            0.0,
            min(float(end), shot_end) - max(float(start), shot_start),
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best = shot.get("shot")

    return best


def collect_all_findings(sections, context_shots=None):
    all_findings = []
    context_shots = context_shots or []

    for name, section in sections.items():
        if not isinstance(section, dict):
            continue

        for key in ("findings", "sound_findings", "music_findings"):
            for f in section.get(key, []):
                item = {"section": name, **f}
                if item.get("window") and context_shots:
                    item["shot"] = shot_for_window(
                        item["window"][0], item["window"][1], context_shots
                    )
                else:
                    item["shot"] = None
                all_findings.append(item)

    all_findings.sort(key=lambda f: TIER_ORDER.get(f["tier"], 9))
    return all_findings


GROUP_TOLERANCE_SEC = 0.5
# Windows longer than this are range-level findings (whole-clip music,
# long untranscribed regions), not localized events -- they must never
# absorb every nearby finding into one giant cluster.
MAX_GROUP_WINDOW_SEC = 4.0


def _cluster_windows(windowed):
    """Cluster findings whose windows overlap or are within
    GROUP_TOLERANCE_SEC. One questionable word padded by different stages
    (ASR conflict, hallucination risk, masking check) lands in ONE cluster
    instead of five separate findings (3.5 dedup). Findings spanning more
    than MAX_GROUP_WINDOW_SEC are kept standalone -- a whole-clip music
    candidate is a global condition, not a localized event."""
    clusters = []
    standalone = []

    for f in sorted(windowed, key=lambda f: (f["window"][0], f["window"][1])):
        span = f["window"][1] - f["window"][0]
        if span > MAX_GROUP_WINDOW_SEC:
            standalone.append(f)
            continue

        if clusters and f["window"][0] <= clusters[-1]["end"] + GROUP_TOLERANCE_SEC:
            merged_end = max(clusters[-1]["end"], f["window"][1])
            # Cap the merged span so adjacent-but-distinct moments do not
            # chain-merge into one giant cluster across the whole clip.
            if merged_end - clusters[-1]["start"] <= MAX_GROUP_WINDOW_SEC:
                clusters[-1]["end"] = merged_end
                clusters[-1]["items"].append(f)
                continue
        clusters.append({
            "start": f["window"][0],
            "end": f["window"][1],
            "items": [f],
        })

    return clusters, standalone


def deduplicate_findings(all_findings):
    """Group findings by time window so one questionable moment does not
    create five repetitive review items. Each cluster keeps its highest-tier
    signal as the headline and lists the individual signals as sub-items;
    the full raw list stays in the packet for provenance.
    """
    windowed = [f for f in all_findings if f.get("window")]
    windowless = [f for f in all_findings if not f.get("window")]

    grouped = []

    clusters, standalone = _cluster_windows(windowed)
    grouped.extend(standalone)

    for cluster in clusters:
        items = sorted(
            cluster["items"],
            key=lambda f: (TIER_ORDER.get(f["tier"], 9), f["window"]),
        )

        if len(items) == 1:
            grouped.append(items[0])
            continue

        top = items[0]
        merged = dict(top)
        merged["claim"] = (
            f"{len(items)} related signals in one window "
            f"({', '.join(sorted({i['section'] for i in items}))})."
        )
        merged["window"] = [
            round(cluster["start"], 3),
            round(cluster["end"], 3),
        ]
        merged["grouped"] = True
        merged["action"] = (
            "Resolve these signals together by listening once; do not treat "
            "them as separate issues."
        )

        # Collapse identical claims inside the cluster (e.g. five
        # "secondary-only gap" signals become one item with x5) so the
        # grouped item does not re-introduce the repetition it exists to
        # remove. Different claims (per-word conflicts etc.) stay distinct.
        collapsed = []
        claim_index = {}
        for i in items:
            key = (i["section"], i["claim"])
            if key in claim_index:
                collapsed[claim_index[key]]["count"] += 1
                continue
            claim_index[key] = len(collapsed)
            collapsed.append({
                "tier": i["tier"],
                "section": i["section"],
                "claim": i["claim"],
                "action": i.get("action"),
                "count": 1,
            })
        merged["items"] = collapsed
        grouped.append(merged)

    grouped.extend(windowless)
    grouped.sort(key=lambda f: TIER_ORDER.get(f["tier"], 9))
    return grouped


def build_confidence_summary(all_findings, raw_count=None):
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
        "raw_finding_count": raw_count if raw_count is not None else len(all_findings),
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
    asr_consensus = load(ASR_CONSENSUS, {"status": "unavailable"})
    speaker_face_mapping = load(SPEAKER_FACE_MAPPING, {"status": "unavailable"})
    sound_fusion = load(SOUND_FUSION, {"status": "unavailable"})

    coverage = build_coverage(evidence, diarization, vad)

    # Same precedence as build_coverage: diarization turns when complete,
    # else VAD regions. Used to judge whether a rerun hypothesis has any
    # independent speech corroboration.
    independent_speech_regions = []
    if diarization and diarization.get("status") == "complete":
        independent_speech_regions = [
            (float(t["start"]), float(t["end"]))
            for t in diarization.get("turns", [])
        ]
    elif vad and vad.get("status") == "complete":
        independent_speech_regions = [
            (float(r["start"]), float(r["end"]))
            for r in vad.get("regions", [])
        ]

    sections = {
        "media": build_media(evidence, identity),
        "locked_task_structure": {
            "characters": context.get("characters", []),
            "objects": context.get("objects", []),
            "shots": context.get("shots", []),
            "note": "Live locked task wins over machine shot detection.",
        },
        "coverage_gaps": coverage,
        "asr_consensus": build_asr_consensus(
            asr_consensus, independent_speech_regions
        ),
        "speaker_face_mapping": build_speaker_face_mapping(speaker_face_mapping),
        "speaker_clusters": build_speaker_clusters(
            diarization, diarization_qc
        ),
        "character_mapping": build_character_mapping(
            evidence, coverage, diarization, speaker_map
        ),
        "voice_profiles": evidence.get("character_voice_profiles", {}),
        "sound_events": build_fusion_sound_events(sound_fusion),
        "music": build_music(sound_fusion),
        "ambience": build_ambience(sound_fusion),
        "transients": build_transients(sound_fusion),
        "sound_events_ast": build_sound_events(sound),
        "object_sound_status": build_object_status(context, sound),
        "recording_defects": build_recording_defects(defects),
        "overlap_masking": build_overlap_masking(masking),
        "clip_boundaries": build_clip_boundaries(evidence, coverage),
        "shot_events": evidence.get("shot_audio_evidence", []),
        "review_queue": queue,
        "validator_predictions": validator,
        "not_yet_implemented": [
            "object_sound_attribution (no object-interaction tracker yet; "
            "conservative fallback stays Unidentified sound)",
            "visual contact sheets for ambiguous speaker windows",
        ],
        "separate_tools": {
            "pasted_back_qa": "manuscript_audio_qa.py (field-vs-prose, cast "
                              "integrity, final-text QA)",
            "task_seed_parser": "manuscript_audio_seed.py",
        },
    }

    all_findings = collect_all_findings(
        sections, context_shots=context.get("shots", [])
    )

    # 3.5: deduplicate by time window so one questionable moment does not
    # generate five repetitive review items. The raw list stays in the
    # packet for full provenance; the ranked list is what a human reads.
    deduped_findings = deduplicate_findings(all_findings)

    sections["confidence_summary"] = build_confidence_summary(
        deduped_findings, raw_count=len(all_findings)
    )
    sections["ranked_findings"] = deduped_findings
    sections["raw_findings"] = all_findings

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
    if summary.get("raw_finding_count") and summary["raw_finding_count"] != summary["total_findings"]:
        add(
            f"**{summary['total_findings']} review items** "
            f"(grouped from {summary['raw_finding_count']} raw signals) — "
            f"{counts[STRONG]} strong, {counts[MEDIUM]} medium, "
            f"{counts[WEAK]} weak, {counts[CONFLICT]} conflict, "
            f"{counts[UNKNOWN]} unknown."
        )
    else:
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

    # 3.5: report BOTH sample rates -- the resampled analysis WAV rate and
    # the original source rate -- and never present the 16 kHz analysis rate
    # as the source's.
    analysis_rate = media.get("analysis_sample_rate")
    source_rate = media.get("source_sample_rate")
    if analysis_rate is not None:
        rates = f"Analysis sample rate: {analysis_rate} Hz"
        if source_rate is not None and source_rate != analysis_rate:
            rates += f" (source audio: {source_rate} Hz)"
        elif source_rate is not None:
            rates += " (matches source)"
        add(rates)
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
                if f.get("shot"):
                    window += f", Shot {f['shot']}"
            add(f"- **{f['tier']}**{window}: {f['claim']}")
            if f.get("grouped"):
                for item in f["items"]:
                    count = f" x{item['count']}" if item.get("count", 1) > 1 else ""
                    add(f"    - [{item['section']}]{count} {item['claim']}")
                    if item.get("action"):
                        add(f"        - {item['action']}")
            else:
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

    # ASR consensus (Phase 3A): separate STRONG / NEEDS LISTENING /
    # COVERAGE GAPS / CONFLICTING WORDS per spec 3A-I.
    asr = sections.get("asr_consensus", {})
    asr_coverage = asr.get("coverage") or {}

    if asr_coverage:
        add()
        add("## ASR CONSENSUS (secondary model cross-check)")
        add(
            f"Model agreement on matched words: "
            f"{asr_coverage.get('model_agreement_pct')}. "
            f"Word disagreements: {asr_coverage.get('word_disagreement_count')}. "
            f"Recovered gap (speech secondary caught, primary missed): "
            f"{asr_coverage.get('uncovered_speech_duration_sec')}s "
            f"(longest single region "
            f"{asr_coverage.get('longest_uncovered_region_sec')}s)."
        )

        strong_words = [
            w for w in asr.get("word_consensus", [])
            if w["state"] == "confirmed"
        ]
        if strong_words:
            add()
            add(
                f"**STRONG TRANSCRIPT**: {len(strong_words)} word(s) "
                "confirmed by both models."
            )

        listen_words = [
            w for w in asr.get("word_consensus", [])
            if w.get("needs_listen")
        ]
        if listen_words:
            add()
            add(f"**NEEDS LISTENING**: {len(listen_words)} word(s) flagged.")
            for w in listen_words[:15]:
                add(
                    f"- {w['start']}-{w['end']}s \"{w['word']}\" "
                    f"[{w['state']}]"
                )
            if len(listen_words) > 15:
                add(f"- ... and {len(listen_words) - 15} more")

        gaps = asr.get("secondary_only_words", [])
        if gaps:
            add()
            add("**ASR COVERAGE GAPS** (secondary-only speech, primary missed it):")
            for a, b in merge_intervals((g["start"], g["end"]) for g in gaps):
                add(f"- {round(a, 3)}-{round(b, 3)}s")

        conflicts = asr.get("conflicts", [])
        if conflicts:
            add()
            add("**CONFLICTING WORDS**:")
            for c in conflicts:
                add(
                    f"- {c['start']}-{c['end']}s: primary "
                    f"\"{c['primary_word']}\" vs secondary "
                    f"\"{c['secondary_word']}\""
                )
    elif asr.get("findings"):
        add()
        add("## ASR CONSENSUS (secondary model cross-check)")
        add("Secondary ASR did not run this session; see findings above.")

    # Face tracking / active-speaker mapping (Phase 3B).
    speaker_mapping = sections.get("speaker_face_mapping", {})
    active_windows = speaker_mapping.get("active_speaker_windows", [])

    if active_windows:
        add()
        add("## VISIBLE SPEAKER CANDIDATES (mouth-motion evidence only)")
        add(
            f"{len(speaker_mapping.get('face_tracks', []))} face track(s) "
            "detected. Mouth-motion evidence is capped at MEDIUM -- it is "
            "not a verified audiovisual sync score. Confirm by watching."
        )
        for w in active_windows:
            if w["tier"] == UNKNOWN:
                continue
            top = (
                max(w["candidates"], key=lambda c: c["motion_score"])
                if w["candidates"] else None
            )
            label = top["face_id"] if top else "?"
            add(f"- **{w['tier']}** [{w['start']}-{w['end']}s]: {label} ({w['reason']})")

        off_screen = [w for w in active_windows if w["tier"] == UNKNOWN]
        if off_screen:
            add()
            add(f"{len(off_screen)} speech window(s) had no visible face candidate.")

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

    # Phase 3C: sound / music / ambience (compact, no raw PANNs/CLAP dump).
    sound_sec = sections.get("sound_events", {})
    music_sec = sections.get("music", {})
    ambience_sec = sections.get("ambience", {})

    sound_candidates = sound_sec.get("candidates", [])
    music_regions = music_sec.get("regions", [])
    ambience_candidates = ambience_sec.get("candidates", [])

    if sound_candidates or music_regions or ambience_candidates:
        add()
        add("## SOUND EVENTS")

        strong = [c for c in sound_candidates if c["tier"] == STRONG]
        needs = [c for c in sound_candidates if c["tier"] in (MEDIUM, CONFLICT)]

        if strong:
            add()
            add("### STRONG")
            for c in strong:
                add(f"- {c['semantic_label']} is strongly supported.")

        if needs:
            add()
            add("### NEEDS REVIEW")
            for c in needs:
                add(f"- {c['semantic_label']} is a candidate; confirm by listening.")

        add()
        add("## MUSIC")
        if music_regions:
            for r in music_regions:
                if r["tier"] in (STRONG, MEDIUM):
                    add(f"- {r['tier']} music candidate; confirm by listening.")
                else:
                    add(
                        "- Music evidence is not strong enough to auto-create. "
                        "DO NOT CREATE MUSIC EVENT WITHOUT LISTENING."
                    )
        else:
            add("- No supported music candidate.")

        add()
        add("## AMBIENCE")
        if ambience_candidates:
            for c in ambience_candidates:
                label = c.get("semantic_candidate") or c.get("semantic_label")
                add(f"- {label} ({c['tier']}).")
        else:
            add("- No ambience candidate.")

        weak_sound = [
            c for c in sound_candidates + ambience_candidates
            if c["tier"] in (WEAK, UNKNOWN)
        ]
        weak_music = [
            r for r in music_regions
            if r["tier"] in (WEAK, CONFLICT, UNKNOWN)
        ]
        if weak_sound or weak_music:
            add()
            add("## DO NOT AUTO-ASSERT")
            for c in weak_sound:
                add(
                    f"- {c['semantic_label']} is weak/unknown; "
                    "do not auto-create an event."
                )
            for r in weak_music:
                add(
                    "- Music evidence is weak/conflicting; "
                    "DO NOT CREATE MUSIC EVENT WITHOUT LISTENING."
                )

    # Phase 3.5: unnamed transient / SFX evidence (RMS, spectral flux,
    # onset, energy change, crest factor). Strong unexplained transients are
    # high-priority listening windows even though no model can name them.
    transients = sections.get("transients", {})
    transient_events = transients.get("events", [])

    if transient_events:
        add()
        add("## UNNAMED TRANSIENTS (listen + identify)")
        for t in transient_events:
            shot = f", Shot {t['shot']}" if t.get("shot") else ""
            if t.get("unexplained"):
                strength = "strong " if t["tier"] == STRONG else ""
                add(
                    f"- **{t['tier']}** [{t['start']}-{t['end']}s{shot}]: "
                    f"{strength}unidentified transient; no named sound "
                    "explains it. Listen and identify."
                )
            else:
                add(
                    f"- **{t['tier']}** [{t['start']}-{t['end']}s{shot}]: "
                    "transient coincides with "
                    + ", ".join(t.get("explained_by", []))
                    + "."
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
    sound_fusion = load(SOUND_FUSION, {"status": "unavailable"})

    ANALYSIS.mkdir(parents=True, exist_ok=True)

    with PACKET.open("w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)

    review_me = build_review_me(sections, evidence)
    REVIEW_ME.write_text(review_me, encoding="utf-8")

    ui = build_ui_suggestions(evidence, sound_fusion)

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
