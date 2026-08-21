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
import math
import sys
import time
from pathlib import Path

from manuscript_audio_sound_vocabulary import CLAP_PROMPTS, CLAP_PROMPT_SET_VERSION


PANNS_MODEL_NAME = "Cnn14"
PANNS_CHECKPOINT = "Cnn14_mAP=0.431.pth"  # panns_inference default checkpoint
CLAP_MODEL_NAME = "laion/clap-htsat-unfused"

# 3.5: the transient/SFX detector is independent of the model windows and
# uses its OWN finer resolution -- transients are short events (a punch,
# a slam) that a 1.5s model window blurs. 0.75s windows / 0.25s hop
# localize them tightly for review clips.
TRANSIENT_WINDOW_SEC = 0.75
TRANSIENT_HOP_SEC = 0.25


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


def _transient_feature_windows(audio, sample_rate, window_sec, hop_sec):
    """3.5 transient/SFX detector features (numpy only -- no librosa).

    Per sliding window, computes the five independent low-level signals the
    models cannot name: short-time RMS (dBFS), crest factor (peak/RMS),
    spectral flux (mean positive spectral delta across 20ms frames),
    onset strength (mean positive energy-envelope delta), and broadband
    energy change (window RMS vs the clip's median RMS). The raw features
    are merged into events by manuscript_audio_sound_fusion.build_transient_events
    (pure stdlib, unit-testable without numpy).
    """
    import numpy as np

    x = np.asarray(audio, dtype="float64")
    n = len(x)
    win = max(1, int(window_sec * sample_rate))
    hop = max(1, int(hop_sec * sample_rate))
    frame_size = max(1, int(0.02 * sample_rate))
    frame_hop = max(1, frame_size // 2)

    if n < win:
        return []

    starts = list(range(0, n - win + 1, hop))

    # First pass: window RMS values so broadband energy change can be
    # measured against the clip baseline (median), not an arbitrary floor.
    rms_values = []
    for start in starts:
        seg = x[start:start + win]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        rms_values.append(rms)

    median_rms = float(np.median(rms_values)) if rms_values else 0.0

    window = np.hanning(frame_size)
    features = []

    for start in starts:
        seg = x[start:start + win]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        peak = float(np.max(np.abs(seg)))

        mags = []
        energies = []
        i = 0
        while i + frame_size <= win:
            frame = seg[i:i + frame_size] * window
            mags.append(np.abs(np.fft.rfft(frame)))
            energies.append(float(np.mean(frame ** 2)))
            i += frame_hop

        if len(mags) >= 2:
            mag_array = np.array(mags)
            diff = np.maximum(0.0, mag_array[1:] - mag_array[:-1])
            flux = float(np.sqrt(np.sum(diff ** 2))) / (len(mags) - 1)
            onset = float(np.mean(np.maximum(0.0, np.diff(energies))))
        else:
            flux = 0.0
            onset = 0.0

        rms_db = _rms_dbfs(seg)
        crest = (peak / rms) if rms > 1e-12 else 0.0
        baseline_db = (
            20.0 * math.log10(median_rms) if median_rms > 0 else -120.0
        )
        energy_change_db = rms_db - baseline_db

        features.append({
            "start": round(start / sample_rate, 3),
            "end": round((start + win) / sample_rate, 3),
            "rms_db": rms_db,
            "crest_factor": round(crest, 3),
            "spectral_flux": round(flux, 6),
            "onset_strength": round(onset, 8),
            "energy_change_db": round(energy_change_db, 2),
        })

    return features


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

            # 3.5: independent transient/SFX detector (numpy only). It runs
            # even when both models fail, so a strong unnamed punch/impact
            # still becomes a high-priority review window.
            transient_started = time.time()
            try:
                transient_features = _transient_feature_windows(
                    audio, sample_rate, TRANSIENT_WINDOW_SEC, TRANSIENT_HOP_SEC
                )
                transient_runtime = round(time.time() - transient_started, 2)
                transient_status = "complete"
            except Exception as exc:  # noqa: BLE001 -- fail soft per model
                transient_features = []
                transient_runtime = round(time.time() - transient_started, 2)
                transient_status = "failed"

            # 3.5: the worker is complete if ANY evidence source produced
            # output -- including the transient/SFX detector when both
            # models fail, so an unnamed punch still becomes a review window.
            overall_status = (
                "complete"
                if panns_status == "complete"
                or clap_status == "complete"
                or transient_status == "complete"
                else "failed"
            )

            result = {
                "status": overall_status,
                "runtime_sec": round(time.time() - started, 2),
                "window_config": {"window_sec": window_sec, "hop_sec": hop_sec, "top_k": top_k},
                "media": {"duration_sec": round(duration_sec, 3), "sample_rate": sample_rate},
                "panns_status": panns_status,
                "clap_status": clap_status,
                "transient_status": transient_status,
                "panns_windows": panns_windows,
                "clap_windows": clap_windows,
                "transient_feature_windows": transient_features,
                "provenance": {
                    "panns": {**panns_provenance, "runtime_sec": panns_runtime},
                    "clap": {**clap_provenance, "runtime_sec": clap_runtime},
                    "transient": {
                        "detector": "short_time_rms+spectral_flux+onset+energy_change+crest",
                        "runtime_sec": transient_runtime,
                    },
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
