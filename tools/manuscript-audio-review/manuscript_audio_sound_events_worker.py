"""Phase 3C.2/3C.3: PANNs + CLAP sound-event worker.

Runs under the isolated `.venv-audio-events` environment so the base
`.venv-review` interpreter never needs torch/panns/clap (same isolation
rule as manuscript_audio_face_worker.py for `.venv-vision`). JSON on stdout
only via the output file; every failure path degrades to a well-formed
status:"failed"/"unavailable" result instead of crashing the pipeline.

What this worker does:
- Loads the analysis WAV once.
- Slides overlapping windows across it (3C.4).
- Runs PANNs (AudioSet-527 CNN14) per window -> top-k raw labels + scores.
- Runs CLAP zero-shot against the fixed, versioned prompt list (3C.3).
- Records raw window-level RMS (dBFS) for downstream recorded-level
  estimation (3C.13).
- Writes raw per-model, per-window scores plus provenance. It does NOT
  fuse across models, map to the controlled vocabulary, or assign
  confidence tiers -- that is manuscript_audio_sound_fusion.py's job,
  which is pure logic and testable without either model installed.

Usage:
    python manuscript_audio_sound_events_worker.py AUDIO.wav OUTPUT.json
        [--window 1.5] [--hop 0.5] [--top-k 8]
"""

import argparse
import json
import sys
import time
from pathlib import Path

from manuscript_audio_sound_vocabulary import CLAP_PROMPTS, CLAP_PROMPT_SET_VERSION


PANNS_MODEL_NAME = "Cnn14"
PANNS_CHECKPOINT = "Cnn14_mAP=0.431.pth"  # panns_inference default checkpoint
CLAP_MODEL_NAME = "laion/clap-htsat-unfused"


def _rms_dbfs(samples):
    import numpy as np
    if len(samples) == 0:
        return None
    rms = float(np.sqrt(np.mean(np.square(samples.astype("float64")))))
    if rms <= 0:
        return -120.0
    import math
    return round(20.0 * math.log10(rms), 2)


def _slide_windows(duration_sec, window_sec, hop_sec):
    windows = []
    start = 0.0
    while start < duration_sec:
        end = min(duration_sec, start + window_sec)
        if end - start < min(0.5, window_sec * 0.5):
            break
        windows.append((round(start, 3), round(end, 3)))
        start += hop_sec
    return windows


def _resample_for_panns(audio, sample_rate):
    """PANNs Cnn14 is fixed at 32 kHz. The analysis WAV is 16 kHz, so
    resample only for PANNs. CLAP's processor resamples internally and keeps
    the original audio + sample_rate."""
    if sample_rate == 32000:
        return audio, 32000
    import librosa
    resampled = librosa.resample(
        audio.astype("float32"), orig_sr=sample_rate, target_sr=32000
    )
    return resampled, 32000


def _run_panns(audio, sample_rate, windows, top_k, device):
    """Returns list of {"start","end","top_labels":[{"raw_label","score"}]}"""
    from panns_inference import AudioTagging, labels as panns_labels

    tagger = AudioTagging(checkpoint_path=None, device=device)

    results = []
    for start, end in windows:
        s = int(start * sample_rate)
        e = int(end * sample_rate)
        clip = audio[s:e]
        if len(clip) == 0:
            continue

        clipped = clip.reshape(1, -1)
        clipwise_output, _ = tagger.inference(clipped)
        scores = clipwise_output[0]

        top_indices = scores.argsort()[::-1][:top_k]
        top_labels = [
            {"raw_label": panns_labels[i], "score": round(float(scores[i]), 4)}
            for i in top_indices
        ]

        results.append({
            "start": start, "end": end,
            "top_labels": top_labels,
            "rms_dbfs": _rms_dbfs(clip),
        })

    return results, {
        "model": PANNS_MODEL_NAME,
        "checkpoint": PANNS_CHECKPOINT,
        "device": device,
    }


def _resample_for_clap(audio, sample_rate):
    """laion/clap-htsat-unfused is trained at 48 kHz and its feature
    extractor does NOT resample automatically -- it rejects any other rate.
    Resample here so CLAP receives correctly-sampled audio."""
    if sample_rate == 48000:
        return audio, 48000
    import librosa
    resampled = librosa.resample(
        audio.astype("float32"), orig_sr=sample_rate, target_sr=48000
    )
    return resampled, 48000


def _run_clap(audio, sample_rate, windows, device):
    """Returns list of {"start","end","prompt_scores":[{"prompt","score"}]}"""
    import torch
    from transformers import ClapModel, ClapProcessor

    model = ClapModel.from_pretrained(CLAP_MODEL_NAME).to(device)
    model.eval()
    processor = ClapProcessor.from_pretrained(CLAP_MODEL_NAME)

    prompts = [p["prompt"] for p in CLAP_PROMPTS]
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)

    with torch.inference_mode():
        text_embeds = model.get_text_features(**text_inputs)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    results = []
    for start, end in windows:
        s = int(start * sample_rate)
        e = int(end * sample_rate)
        clip = audio[s:e]
        if len(clip) == 0:
            continue

        audio_inputs = processor(
            audio=clip, sampling_rate=sample_rate, return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            audio_embed = model.get_audio_features(**audio_inputs)
            audio_embed = audio_embed / audio_embed.norm(p=2, dim=-1, keepdim=True)
            similarity = (audio_embed @ text_embeds.T)[0]
            # CLAP cosine similarity is not a probability. Rescale to
            # [0, 1] with a temperature-scaled softmax so downstream fusion
            # has a bounded, comparable score -- explicitly documented as a
            # rescaling, never presented as a calibrated probability.
            probs = torch.softmax(similarity * 10.0, dim=-1)

        prompt_scores = [
            {"prompt": prompts[i], "score": round(float(probs[i]), 4)}
            for i in range(len(prompts))
        ]

        results.append({"start": start, "end": end, "prompt_scores": prompt_scores})

    return results, {
        "model": CLAP_MODEL_NAME,
        "prompt_set_version": CLAP_PROMPT_SET_VERSION,
        "device": device,
        "similarity_note": "cosine similarity, softmax-rescaled -- not a "
                            "calibrated probability",
    }


def write_sound_events_raw_evidence(
    audio_path, output_path, window_sec=1.5, hop_sec=0.5, top_k=8,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    result = {"status": "unavailable"}

    try:
        import soundfile as sf
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        audio, sample_rate = sf.read(str(audio_path))
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)

        duration_sec = len(audio) / sample_rate
        windows = _slide_windows(duration_sec, window_sec, hop_sec)

        if not windows:
            result = {
                "status": "failed",
                "error": "clip too short for any analysis window",
                "panns_windows": [], "clap_windows": [],
            }
        else:
            panns_started = time.time()
            try:
                panns_audio, panns_rate = _resample_for_panns(
                    audio, sample_rate
                )
                panns_windows, panns_provenance = _run_panns(
                    panns_audio, panns_rate, windows, top_k, device,
                )
                panns_runtime = round(time.time() - panns_started, 2)
                panns_status = "complete"
            except Exception as exc:  # noqa: BLE001 -- fail soft per model
                panns_windows, panns_provenance = [], {"error": f"{type(exc).__name__}: {exc}"}
                panns_runtime = round(time.time() - panns_started, 2)
                panns_status = "failed"

            clap_started = time.time()
            try:
                clap_audio, clap_rate = _resample_for_clap(
                    audio, sample_rate
                )
                clap_windows, clap_provenance = _run_clap(
                    clap_audio, clap_rate, windows, device,
                )
                clap_runtime = round(time.time() - clap_started, 2)
                clap_status = "complete"
            except Exception as exc:  # noqa: BLE001 -- fail soft per model
                clap_windows, clap_provenance = [], {"error": f"{type(exc).__name__}: {exc}"}
                clap_runtime = round(time.time() - clap_started, 2)
                clap_status = "failed"

            overall_status = (
                "complete" if panns_status == "complete" or clap_status == "complete"
                else "failed"
            )

            result = {
                "status": overall_status,
                "runtime_sec": round(time.time() - started, 2),
                "window_config": {"window_sec": window_sec, "hop_sec": hop_sec, "top_k": top_k},
                "media": {"duration_sec": round(duration_sec, 3), "sample_rate": sample_rate},
                "panns_status": panns_status,
                "clap_status": clap_status,
                "panns_windows": panns_windows,
                "clap_windows": clap_windows,
                "provenance": {
                    "panns": {**panns_provenance, "runtime_sec": panns_runtime},
                    "clap": {**clap_provenance, "runtime_sec": clap_runtime},
                },
            }

    except Exception as exc:  # noqa: BLE001 -- fail soft (design rule 4)
        result = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_sec": round(time.time() - started, 2),
            "panns_windows": [], "clap_windows": [],
        }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if result["status"] == "complete":
        print(
            "SOUND EVENTS WORKER: PASS |",
            f"panns={result['panns_status']} clap={result['clap_status']} |",
            f"{len(result['panns_windows'])} panns windows, "
            f"{len(result['clap_windows'])} clap windows |",
            f"{result['runtime_sec']}s",
        )
    else:
        print("SOUND EVENTS WORKER: FAILED |", result.get("error"))

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path")
    parser.add_argument("output_path")
    parser.add_argument("--window", type=float, default=1.5)
    parser.add_argument("--hop", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    result = write_sound_events_raw_evidence(
        args.audio_path, args.output_path,
        window_sec=args.window, hop_sec=args.hop, top_k=args.top_k,
    )
    sys.exit(0)  # never fail the pipeline -- caller checks status in JSON


if __name__ == "__main__":
    main()
