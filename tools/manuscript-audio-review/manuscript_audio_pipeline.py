from pathlib import Path
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parent

ANALYZER = ROOT / "manuscript_audio_review.py"
EVIDENCE = ROOT / "analysis" / "manuscript_audio_evidence.json"
QUEUE = ROOT / "analysis" / "audio_review_queue.json"
AUDIO = ROOT / "analysis" / "audio.wav"
CLIP_DIR = ROOT / "analysis" / "review_clips"
MANIFEST = CLIP_DIR / "review_clips_manifest.json"


def run(cmd):
    subprocess.run(cmd, cwd=ROOT, check=True)


def preprocess_source_video():
    print("\n=== PHASE 0: SOURCE PREPROCESSING ===\n")

    if len(sys.argv) >= 2:
        video = Path(sys.argv[1]).expanduser().resolve()
    else:
        video = ROOT / "VIDEO.mp4"

    output_dir = ROOT / "output"
    analysis_dir = ROOT / "analysis"

    generated_whisperx_json = output_dir / f"{video.stem}.json"
    whisperx_json = output_dir / "VIDEO.json"

    whisperx_python = (
        ROOT
        / ".venv-whisperx"
        / "Scripts"
        / "python.exe"
    )

    if not video.exists():
        raise FileNotFoundError(f"Source video not found: {video}")

    if not whisperx_python.exists():
        raise FileNotFoundError(
            "WhisperX environment not found: "
            f"{whisperx_python}. Run setup_windows.ps1 first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print("Source video:", video)
    print("WhisperX Python:", whisperx_python)

    run(
        [
            str(whisperx_python),
            "-c",
            "import whisperx; print('WhisperX import: PASS')",
        ]
    )

    print("\n[1/2] Extracting analysis WAV...")

    run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(AUDIO),
        ]
    )

    if not AUDIO.exists():
        raise FileNotFoundError(
            f"FFmpeg did not create analysis WAV: {AUDIO}"
        )

    print("Analysis WAV:", AUDIO)
    print("\n[2/2] Running WhisperX large-v3...")

    for stale_json in {
        generated_whisperx_json,
        whisperx_json,
    }:
        if stale_json.exists():
            stale_json.unlink()

    run(
        [
            str(whisperx_python),
            "-m",
            "whisperx",
            str(video),
            "--model",
            "large-v3",
            "--device",
            "cpu",
            "--compute_type",
            "int8",
            "--output_format",
            "json",
            "--output_dir",
            str(output_dir),
        ]
    )

    if not generated_whisperx_json.exists():
        raise FileNotFoundError(
            "WhisperX did not create the expected source JSON: "
            f"{generated_whisperx_json}"
        )

    whisperx_json.write_bytes(
        generated_whisperx_json.read_bytes()
    )

    with whisperx_json.open("r", encoding="utf-8") as f:
        transcript = json.load(f)

    segments = transcript.get("segments", [])

    if not segments:
        raise RuntimeError(
            "WhisperX JSON contains no speech segments."
        )

    aligned_words = sum(
        len(segment.get("words", []))
        for segment in segments
    )

    print()
    print("WhisperX JSON:", whisperx_json)
    print("Segments:", len(segments))
    print("Aligned words:", aligned_words)

    if aligned_words == 0:
        raise RuntimeError(
            "WhisperX produced no word-level alignment."
        )

    print("SOURCE PREPROCESSING: PASS")


def run_analyzer():
    print("\n=== PHASE 1: AUDIO EVIDENCE ===\n")

    run([sys.executable, str(ANALYZER)])

    if not EVIDENCE.exists():
        raise FileNotFoundError(
            f"Evidence file missing after analyzer run: {EVIDENCE}"
        )


def clamp(value, low, high):
    return max(low, min(high, value))


def build_review_queue():
    print("\n=== PHASE 2: TARGETED REVIEW QUEUE ===\n")

    with EVIDENCE.open("r", encoding="utf-8") as f:
        evidence = json.load(f)

    duration = float(evidence["media"]["duration_sec"])
    windows = []

    for segment in evidence.get("whisperx_segments", []):
        for word in segment.get("low_confidence_words", []):
            start = clamp(float(word["start"]) - 0.4, 0.0, duration)
            end = clamp(float(word["end"]) + 0.4, 0.0, duration)

            windows.append({
                "priority": "high",
                "type": "transcript_word_check",
                "start": round(start, 3),
                "end": round(end, 3),
                "description": (
                    f"Verify low-confidence word '{word['word']}' "
                    f"(ASR score {word['score']})"
                ),
            })

    for boundary in evidence.get("continuity_boundaries", []):
        if not boundary.get("manual_review_required"):
            continue

        right_index = int(boundary["right_segment"])
        right_segment = evidence["whisperx_segments"][right_index]
        center = float(right_segment["start"])

        start = clamp(center - 1.0, 0.0, duration)
        end = clamp(center + 1.0, 0.0, duration)

        windows.append({
            "priority": "high",
            "type": "speaker_source_boundary_check",
            "start": round(start, 3),
            "end": round(end, 3),
            "description": (
                "Possible speaker/source change at "
                f"{boundary['boundary']}; "
                f"pitch ratio {boundary['pitch_ratio']}x"
            ),
        })

    for segment in evidence.get("review_synthesis", []):
        if "emotion_model_ambiguous" not in segment.get(
            "manual_review_reasons",
            [],
        ):
            continue

        windows.append({
            "priority": "medium",
            "type": "tone_delivery_check",
            "start": round(float(segment["start"]), 3),
            "end": round(float(segment["end"]), 3),
            "description": (
                "Emotion model result is ambiguous; "
                "listen for actual tone and delivery."
            ),
        })

    priority_rank = {"high": 0, "medium": 1, "normal": 2}
    windows.sort(
        key=lambda item: (
            float(item["start"]),
            priority_rank.get(item["priority"], 9),
            item["type"],
        )
    )

    QUEUE.parent.mkdir(parents=True, exist_ok=True)

    with QUEUE.open("w", encoding="utf-8") as f:
        json.dump(windows, f, indent=2, ensure_ascii=False)

    print("Review windows:", len(windows))

    for i, item in enumerate(windows, 1):
        print(
            f"{i}. {item['start']:.3f} --> {item['end']:.3f} | "
            f"{item['priority'].upper()} | {item['type']}"
        )

    return windows


def create_review_clips(windows):
    print("\n=== PHASE 3: TARGETED WAV CLIPS ===\n")

    CLIP_DIR.mkdir(parents=True, exist_ok=True)

    for old in CLIP_DIR.glob("*.wav"):
        old.unlink()

    records = []

    for i, item in enumerate(windows, 1):
        start = float(item["start"])
        end = float(item["end"])
        duration = end - start

        filename = (
            f"{i:02d}_{item['type']}_"
            f"{start:.3f}-{end:.3f}.wav"
        )
        output = CLIP_DIR / filename

        run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{duration:.3f}",
                "-i",
                str(AUDIO),
                "-acodec",
                "pcm_s16le",
                str(output),
            ]
        )

        if not output.exists():
            raise FileNotFoundError(
                f"Review clip was not created: {output}"
            )

        print("CREATED:", filename)

        records.append({
            **item,
            "filename": filename,
            "expected_duration_sec": round(duration, 3),
        })

    with MANIFEST.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    return records


def probe_duration(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    return float(result.stdout.strip())


def validate_review_clips(records):
    print("\n=== PHASE 4: CLIP VALIDATION ===\n")

    failures = []

    for i, record in enumerate(records, 1):
        path = CLIP_DIR / record["filename"]
        expected = float(record["expected_duration_sec"])
        actual = probe_duration(path)
        diff = abs(actual - expected)

        passed = diff <= 0.03

        print(
            f"{i}. {'PASS' if passed else 'FAIL'} | "
            f"actual={actual:.3f}s | "
            f"expected={expected:.3f}s | "
            f"diff={diff:.3f}s | "
            f"{record['filename']}"
        )

        if not passed:
            failures.append(record["filename"])

    if failures:
        raise RuntimeError(
            "Review clip validation failed: "
            + ", ".join(failures)
        )


def main():
    print("===================================")
    print(" MANUSCRIPT II AUDIO REVIEW PIPELINE")
    print("===================================")

    preprocess_source_video()
    run_analyzer()
    windows = build_review_queue()
    records = create_review_clips(windows)
    validate_review_clips(records)

    print()
    print("===================================")
    print(" PIPELINE STATUS: PASS")
    print("===================================")
    print("Evidence:", EVIDENCE)
    print("Review queue:", QUEUE)
    print("Review clips:", CLIP_DIR)
    print("Manifest:", MANIFEST)


if __name__ == "__main__":
    main()
