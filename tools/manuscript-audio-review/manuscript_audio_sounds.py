from pathlib import Path
import json

import soundfile as sf
import torch
from transformers import (
    AutoFeatureExtractor,
    ASTForAudioClassification,
)


MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"


def classify_source_candidate(label):
    name = label.lower()

    if "music" in name:
        return "Music"

    if any(
        token in name
        for token in (
            "traffic",
            "roadway",
            "vehicle",
            "car",
            "truck",
            "motorcycle",
            "bus",
        )
    ):
        return "Traffic ambience"

    if any(
        token in name
        for token in (
            "inside, small room",
            "inside, large room",
            "room tone",
        )
    ):
        return "Room ambience"

    return "Unidentified sound"


def evidence_strength(score):
    score = float(score)

    if score >= 0.70:
        return "strong"

    if score >= 0.50:
        return "moderate"

    return "weak"


def is_useful_sound_label(label):
    name = label.lower()

    useful_terms = (
        "applause",
        "clapping",
        "cheering",
        "crowd",
        "laughter",
        "music",
        "traffic",
        "roadway",
        "vehicle",
        "car",
        "truck",
        "motorcycle",
        "bus",
        "footstep",
        "door",
        "click",
        "impact",
        "spray",
        "wind",
        "inside, small room",
        "inside, large room",
        "room tone",
    )

    return any(
        token in name
        for token in useful_terms
    )


def analyze_sound_windows(
    audio_path,
    window_sec=2.0,
    hop_sec=1.0,
    minimum_score=0.20,
    top_k=8,
):
    audio_path = Path(audio_path)

    audio, sample_rate = sf.read(audio_path)

    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)

    extractor = AutoFeatureExtractor.from_pretrained(
        MODEL_NAME
    )

    model = ASTForAudioClassification.from_pretrained(
        MODEL_NAME
    )
    model.eval()

    window_samples = max(
        1,
        int(window_sec * sample_rate),
    )

    hop_samples = max(
        1,
        int(hop_sec * sample_rate),
    )

    results = []
    start_sample = 0

    while start_sample < len(audio):
        end_sample = min(
            len(audio),
            start_sample + window_samples,
        )

        clip = audio[start_sample:end_sample]

        if len(clip) < int(0.5 * sample_rate):
            break

        inputs = extractor(
            clip,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )

        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = torch.sigmoid(logits)[0]

        values, indices = torch.topk(
            probabilities,
            min(top_k, len(probabilities)),
        )

        detections = []

        for score, index in zip(
            values.tolist(),
            indices.tolist(),
        ):
            label = model.config.id2label[index]

            if score < minimum_score:
                continue

            if not is_useful_sound_label(label):
                continue

            detections.append({
                "label": label,
                "score": round(float(score), 4),
            })

        if detections:
            results.append({
                "start": round(
                    start_sample / sample_rate,
                    3,
                ),
                "end": round(
                    end_sample / sample_rate,
                    3,
                ),
                "detections": detections,
            })

        start_sample += hop_samples

    return results


def merge_sound_windows(windows):
    by_label = {}

    for window in windows:
        for detection in window["detections"]:
            label = detection["label"]

            by_label.setdefault(label, []).append({
                "start": float(window["start"]),
                "end": float(window["end"]),
                "score": float(detection["score"]),
            })

    merged = []

    for label, items in by_label.items():
        items.sort(
            key=lambda x: (
                x["start"],
                x["end"],
            )
        )

        groups = []

        for item in items:
            if not groups:
                groups.append({
                    "start": item["start"],
                    "end": item["end"],
                    "scores": [item["score"]],
                })
                continue

            current = groups[-1]

            if item["start"] <= current["end"] + 0.05:
                current["end"] = max(
                    current["end"],
                    item["end"],
                )
                current["scores"].append(
                    item["score"]
                )
            else:
                groups.append({
                    "start": item["start"],
                    "end": item["end"],
                    "scores": [item["score"]],
                })

        for group in groups:
            maximum = max(group["scores"])
            average = (
                sum(group["scores"])
                / len(group["scores"])
            )

            strength = evidence_strength(maximum)

            # Only moderate/strong evidence receives
            # a proposed Manuscript source category.
            #
            # Weak evidence remains a listening cue only.
            source_candidate = (
                classify_source_candidate(label)
                if strength in ("moderate", "strong")
                else None
            )

            merged.append({
                "label": label,
                "start": round(group["start"], 3),
                "end": round(group["end"], 3),
                "max_score": round(maximum, 4),
                "mean_score": round(average, 4),
                "supporting_windows": len(
                    group["scores"]
                ),
                "evidence_strength": strength,
                "manuscript_source_candidate":
                    source_candidate,
                "manual_audio_confirmation_required":
                    True,
            })

    merged.sort(
        key=lambda x: (
            x["start"],
            -x["max_score"],
            x["label"],
        )
    )

    return merged


def write_sound_evidence(
    audio_path,
    output_path,
):
    raw_windows = analyze_sound_windows(
        audio_path
    )

    merged_events = merge_sound_windows(
        raw_windows
    )

    result = {
        "raw_windows": raw_windows,
        "candidate_events": merged_events,
        "policy": {
            "automatic_sound_event_creation": False,
            "human_listening_required": True,
            "weak_evidence_gets_source_candidate": False,
        },
    }

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "NON-SPEECH SOUND EVIDENCE: PASS |",
        len(raw_windows),
        "raw windows |",
        len(merged_events),
        "merged candidates",
    )

    return result


def assign_sound_candidates_to_shots(
    sound_result,
    task_context,
):
    shots = task_context.get("shots", [])
    candidates = sound_result.get(
        "candidate_events",
        [],
    )

    results = []

    for candidate in candidates:
        sound_start = float(candidate["start"])
        sound_end = float(candidate["end"])

        overlaps = []

        for shot in shots:
            shot_start = float(shot["start"])
            shot_end = float(shot["end"])

            overlap_start = max(
                sound_start,
                shot_start,
            )

            overlap_end = min(
                sound_end,
                shot_end,
            )

            overlap = max(
                0.0,
                overlap_end - overlap_start,
            )

            if overlap <= 0:
                continue

            overlaps.append({
                "shot": int(shot["shot"]),
                "overlap_sec": round(
                    overlap,
                    3,
                ),
            })

        item = dict(candidate)
        item["shot_assignments"] = overlaps

        if len(overlaps) == 1:
            item["primary_shot"] = overlaps[0][
                "shot"
            ]

        elif len(overlaps) > 1:
            strongest = max(
                overlaps,
                key=lambda x: x["overlap_sec"],
            )

            item["primary_shot"] = strongest[
                "shot"
            ]

        else:
            item["primary_shot"] = None

        item[
            "crosses_shot_boundary"
        ] = len(overlaps) > 1

        results.append(item)

    return results


def enrich_sound_evidence_with_shots(
    sound_path,
    context_path,
):
    sound_path = Path(sound_path)
    context_path = Path(context_path)

    with sound_path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        sound_result = json.load(f)

    with context_path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        task_context = json.load(f)

    assigned = assign_sound_candidates_to_shots(
        sound_result,
        task_context,
    )

    sound_result[
        "shot_assigned_candidates"
    ] = assigned

    with sound_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            sound_result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "SOUND SHOT ASSIGNMENT: PASS |",
        len(assigned),
        "candidate events",
    )

    return assigned
