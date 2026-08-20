"""Voice-activity detection fallback for speech-presence (spec 3).

Diarization is the primary independent speech signal, but it is optional and can
be skipped. When it is unavailable, ASR-vs-diarization coverage cannot flag late
or dropped speech. This module runs Silero VAD on the analysis WAV to produce an
independent list of speech regions so UNTRANSCRIBED_SPEECH still works.

It runs in the review environment (torch is already present). Silero is small
and CPU-friendly. If the dependency or model is unavailable, it writes a
status="unavailable" record and the pipeline continues -- this is a fallback,
never a hard requirement.

Output: analysis/vad_speech_regions.json
    {
      "status": "complete" | "unavailable",
      "source": "silero_vad",
      "regions": [{"start": 7.54, "end": 9.35}, ...]
    }
"""

from pathlib import Path
import json


ROOT = Path(__file__).resolve().parent
AUDIO = ROOT / "analysis" / "audio.wav"
OUTPUT = ROOT / "analysis" / "vad_speech_regions.json"

SAMPLE_RATE = 16000


def _soundfile_reader():
    """Read a 16 kHz mono WAV to a float32 torch tensor via soundfile.

    Silero ships its own read_audio, but that path pulls in torchaudio, whose
    backend API can mismatch the pinned torch build. Reading with soundfile
    (already a dependency) keeps VAD independent of torchaudio.
    """
    import torch
    import soundfile as sf

    def read_audio(path, sampling_rate=SAMPLE_RATE):
        audio, sr = sf.read(path, dtype="float32")
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        # The analysis WAV is already 16 kHz mono; if not, VAD timing on a
        # differing rate would be wrong, so refuse rather than mislead.
        if sr != sampling_rate:
            raise ValueError(
                f"expected {sampling_rate} Hz audio, got {sr} Hz"
            )
        return torch.as_tensor(audio, dtype=torch.float32)

    return read_audio


def _load_silero():
    """Return (get_speech_timestamps, model, read_audio) or None."""
    read_audio = _soundfile_reader()

    # Preferred: the standalone silero-vad package (v5+).
    try:
        from silero_vad import load_silero_vad, get_speech_timestamps

        model = load_silero_vad()
        return get_speech_timestamps, model, read_audio
    except Exception:
        pass

    # Fallback: torch.hub.
    try:
        import torch

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        get_speech_timestamps = utils[0]
        return get_speech_timestamps, model, read_audio
    except Exception:
        return None


def detect_speech_regions(audio_path=AUDIO):
    audio_path = Path(audio_path)

    if not audio_path.exists():
        return {
            "status": "unavailable",
            "source": "silero_vad",
            "reason": "analysis WAV missing",
            "regions": [],
        }

    loaded = _load_silero()

    if loaded is None:
        return {
            "status": "unavailable",
            "source": "silero_vad",
            "reason": "silero-vad not installed",
            "regions": [],
        }

    get_speech_timestamps, model, read_audio = loaded

    try:
        wav = read_audio(str(audio_path), sampling_rate=SAMPLE_RATE)

        stamps = get_speech_timestamps(
            wav,
            model,
            sampling_rate=SAMPLE_RATE,
            return_seconds=True,
        )
    except Exception as exc:
        return {
            "status": "unavailable",
            "source": "silero_vad",
            "reason": f"vad run failed: {exc}",
            "regions": [],
        }

    regions = [
        {
            "start": round(float(s["start"]), 3),
            "end": round(float(s["end"]), 3),
        }
        for s in stamps
        if float(s["end"]) > float(s["start"])
    ]

    return {
        "status": "complete",
        "source": "silero_vad",
        "regions": regions,
    }


def write_vad_regions(audio_path=AUDIO, output_path=OUTPUT):
    result = detect_speech_regions(audio_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if result["status"] == "complete":
        print(
            "VAD SPEECH REGIONS: PASS |",
            len(result["regions"]),
            "regions",
        )
    else:
        print(
            "VAD SPEECH REGIONS: SKIPPED |",
            result.get("reason", "unavailable"),
        )

    return result


if __name__ == "__main__":
    write_vad_regions()
