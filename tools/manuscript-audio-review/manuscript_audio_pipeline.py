from pathlib import Path
import json
import subprocess
import sys

from manuscript_audio_shots import enrich_evidence_with_shots
from manuscript_audio_queue import merge_transcript_review_windows
from manuscript_audio_ui import enrich_evidence_with_ui_candidates
from manuscript_audio_voice import enrich_evidence_with_voice_profiles
from manuscript_audio_optional import run_optional_evidence
from manuscript_audio_task_identity import write_video_identity
from manuscript_audio_asr_consensus import write_asr_consensus_evidence
from manuscript_audio_face_worker import write_face_track_evidence
from manuscript_audio_speaker_mapping import write_speaker_mapping_evidence
from manuscript_audio_sound_events import write_sound_fusion_evidence
from manuscript_audio_report import main as generate_review_report
from manuscript_audio_validator import main as run_audio_validator
from manuscript_audio_master import main as generate_master_packet
ROOT = Path(__file__).resolve().parent

ANALYZER = ROOT / "manuscript_audio_review.py"
EVIDENCE = ROOT / "analysis" / "manuscript_audio_evidence.json"
QUEUE = ROOT / "analysis" / "audio_review_queue.json"
DIARIZATION_QC = ROOT / "analysis" / "diarization_cluster_review.json"
DEFECT_EVIDENCE = ROOT / "analysis" / "recording_defect_evidence.json"
MASKING_EVIDENCE = ROOT / "analysis" / "masking_overlap_evidence.json"
AUDIO = ROOT / "analysis" / "audio.wav"
CLIP_DIR = ROOT / "analysis" / "review_clips"
MANIFEST = CLIP_DIR / "review_clips_manifest.json"
CONTEXT = ROOT / "task_context.json"
SPEAKER_MAP = ROOT / "speaker_map.json"
WHISPERX_JSON = ROOT / "output" / "VIDEO.json"
VIDEO_IDENTITY = ROOT / "analysis" / "video_identity.json"
ASR_CONSENSUS = ROOT / "analysis" / "asr_consensus_evidence.json"
DIARIZATION_EVIDENCE = ROOT / "analysis" / "diarization_evidence.json"
VAD_EVIDENCE = ROOT / "analysis" / "vad_speech_regions.json"
WHISPERX_PYTHON = ROOT / ".venv-whisperx" / "Scripts" / "python.exe"
VISION_PYTHON = ROOT / ".venv-vision" / "Scripts" / "python.exe"
FACE_WORKER = ROOT / "manuscript_audio_face_worker.py"
FACE_TRACK_EVIDENCE = ROOT / "analysis" / "face_track_evidence.json"
SPEAKER_MAPPING_EVIDENCE = ROOT / "analysis" / "speaker_mapping_evidence.json"
AUDIO_EVENTS_PYTHON = ROOT / ".venv-audio-events" / "Scripts" / "python.exe"
SOUND_FUSION_EVIDENCE = ROOT / "analysis" / "sound_fusion_evidence.json"
VIDEO_PATH = None
def run(cmd):
    subprocess.run(cmd, cwd=ROOT, check=True)


def preprocess_source_video():
    print("\n=== PHASE 0: SOURCE PREPROCESSING ===\n")

    if len(sys.argv) >= 2:
        video = Path(sys.argv[1]).expanduser().resolve()
    else:
        video = ROOT / "VIDEO.mp4"

    global VIDEO_PATH
    VIDEO_PATH = video

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

    identity = write_video_identity(
        video,
        VIDEO_IDENTITY,
    )

    print("Source video:", video)
    print(
        "Video fingerprint:",
        identity["video_sha256"][:12] + "...",
    )
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


def run_asr_consensus():
    print("\n=== PHASE 1.5: ASR CONSENSUS (SECONDARY MODEL) ===\n")

    with EVIDENCE.open("r", encoding="utf-8") as f:
        evidence = json.load(f)

    duration = evidence.get("media", {}).get("duration_sec")

    independent_speech_regions = []

    if DIARIZATION_EVIDENCE.exists():
        with DIARIZATION_EVIDENCE.open("r", encoding="utf-8-sig") as f:
            diarization = json.load(f)
        if diarization.get("status") == "complete":
            independent_speech_regions = [
                (t["start"], t["end"]) for t in diarization.get("turns", [])
            ]

    if not independent_speech_regions and VAD_EVIDENCE.exists():
        with VAD_EVIDENCE.open("r", encoding="utf-8-sig") as f:
            vad = json.load(f)
        if vad.get("status") == "complete":
            independent_speech_regions = [
                (r["start"], r["end"]) for r in vad.get("regions", [])
            ]

    try:
        write_asr_consensus_evidence(
            WHISPERX_JSON,
            AUDIO,
            WHISPERX_PYTHON,
            ASR_CONSENSUS,
            duration_sec=duration,
            independent_speech_regions=independent_speech_regions,
        )
    except Exception as exc:  # noqa: BLE001 -- fail soft (design rule 4)
        print(f"ASR CONSENSUS: SKIPPED | {type(exc).__name__}: {exc}")
        ASR_CONSENSUS.parent.mkdir(parents=True, exist_ok=True)
        with ASR_CONSENSUS.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "coverage": None,
                    "word_consensus": [],
                    "conflicts": [],
                    "rerun_windows": [],
                },
                f,
                indent=2,
            )


def run_face_and_speaker_mapping():
    print("\n=== PHASE 1.6: FACE TRACKING / ACTIVE-SPEAKER MAPPING ===\n")

    if VISION_PYTHON.exists() and FACE_WORKER.exists():
        try:
            run([
                str(VISION_PYTHON),
                str(FACE_WORKER),
                str(VIDEO_PATH),
                str(FACE_TRACK_EVIDENCE),
                "--fps",
                "5",
            ])
        except subprocess.CalledProcessError as exc:
            print(f"FACE TRACKING: SKIPPED | subprocess failed: {exc}")
            FACE_TRACK_EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
            with FACE_TRACK_EVIDENCE.open("w", encoding="utf-8") as f:
                json.dump(
                    {"status": "failed", "error": str(exc), "face_tracks": []},
                    f, indent=2,
                )
    else:
        print("FACE TRACKING: SKIPPED | .venv-vision not found (run setup)")
        FACE_TRACK_EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
        with FACE_TRACK_EVIDENCE.open("w", encoding="utf-8") as f:
            json.dump(
                {"status": "unavailable", "error": "vision environment missing",
                 "face_tracks": []},
                f, indent=2,
            )

    try:
        write_speaker_mapping_evidence(
            diarization_path=DIARIZATION_EVIDENCE,
            vad_path=VAD_EVIDENCE,
            face_tracks_path=FACE_TRACK_EVIDENCE,
            output_path=SPEAKER_MAPPING_EVIDENCE,
        )
    except Exception as exc:  # noqa: BLE001 -- fail soft (design rule 4)
        print(f"SPEAKER/FACE MAPPING: SKIPPED | {type(exc).__name__}: {exc}")
        with SPEAKER_MAPPING_EVIDENCE.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "face_tracks": [],
                    "active_speaker_windows": [],
                    "cluster_to_face_candidates": [],
                    "face_to_character_candidates": [],
                },
                f, indent=2,
            )


def run_sound_events():
    print("\n=== PHASE 1.7: SOUND / MUSIC / AMBIENCE EVIDENCE ===\n")

    # 3C is optional evidence. A missing .venv-audio-events, missing worker,
    # model-load failure, inference failure, or malformed output must never
    # break the rest of the pipeline. The orchestrator already fails soft
    # internally; this outer guard catches any unexpected exception so the
    # base packet still generates.
    try:
        write_sound_fusion_evidence()
    except Exception as exc:  # noqa: BLE001 -- fail soft (design rule 4)
        print(f"SOUND FUSION: SKIPPED | {type(exc).__name__}: {exc}")
        SOUND_FUSION_EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
        with SOUND_FUSION_EVIDENCE.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema_version": 1,
                    "status": "unavailable",
                    "error": f"{type(exc).__name__}: {exc}",
                    "worker": {"status": "unavailable", "panns": {}, "clap": {}, "runtime": {}},
                    "sound_events": {"candidates": []},
                    "music": {"regions": [], "overall_confidence": "UNKNOWN", "findings": []},
                    "ambience": {"candidates": [], "findings": []},
                    "source_attribution": {"character_candidates": [], "object_candidates": []},
                    "masking_evidence": {"candidates": []},
                    "review_windows": [],
                    "findings": [],
                },
                f,
                indent=2,
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

    # Flag speech that crosses a Manuscript shot boundary.
    # Internal timestamps are evidence only and never enter Final Audio Text.
    seen_boundaries = set()

    for shot in evidence.get("shot_audio_evidence", []):
        candidates = []

        if shot.get("speech_crosses_into_shot"):
            candidates.append(
                (
                    float(shot["start"]),
                    f"Speech crosses into Shot {shot['shot']} "
                    "from the previous shot."
                )
            )

        if shot.get("speech_crosses_out_of_shot"):
            candidates.append(
                (
                    float(shot["end"]),
                    f"Speech continues out of Shot {shot['shot']} "
                    "into the next shot."
                )
            )

        for center, description in candidates:
            key = round(center, 3)

            if key in seen_boundaries:
                continue

            seen_boundaries.add(key)

            windows.append({
                "priority": "high",
                "type": "shot_boundary_speech_check",
                "start": round(
                    clamp(center - 0.75, 0.0, duration),
                    3,
                ),
                "end": round(
                    clamp(center + 0.75, 0.0, duration),
                    3,
                ),
                "description": description,
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

    # Add suspicious diarization-cluster regions.
    # These are listening cues only, never automatic C# assignments.
    if DIARIZATION_QC.exists():
        with DIARIZATION_QC.open(
            "r",
            encoding="utf-8-sig",
        ) as f:
            diarization_qc = json.load(f)

        for item in diarization_qc.get(
            "review_windows",
            [],
        ):
            start = clamp(
                float(item["start"]),
                0.0,
                duration,
            )

            end = clamp(
                float(item["end"]),
                0.0,
                duration,
            )

            if end <= start:
                continue

            windows.append({
                "priority": item.get(
                    "priority",
                    "high",
                ),
                "type":
                    "diarization_cluster_check",
                "start": round(start, 3),
                "end": round(end, 3),
                "description": item.get(
                    "description",
                    "Verify suspicious diarization cluster.",
                ),
                "speaker_cluster":
                    item.get(
                        "speaker_cluster"
                    ),
            })
    # Recording-defect review cues.
    if DEFECT_EVIDENCE.exists():
        with DEFECT_EVIDENCE.open(
            "r",
            encoding="utf-8-sig",
        ) as f:
            recording_defects = json.load(f)

        for item in recording_defects.get(
            "review_windows",
            [],
        ):
            windows.append({
                "priority": item.get(
                    "priority",
                    "medium",
                ),
                "type":
                    "recording_defect_check",
                "start": round(
                    clamp(
                        float(item["start"]),
                        0.0,
                        duration,
                    ),
                    3,
                ),
                "end": round(
                    clamp(
                        float(item["end"]),
                        0.0,
                        duration,
                    ),
                    3,
                ),
                "description":
                    item.get(
                        "description",
                        "Verify possible recording defect.",
                    ),
                "defect":
                    item.get("defect"),
            })

    # Overlap/intelligibility review cues.
    if MASKING_EVIDENCE.exists():
        with MASKING_EVIDENCE.open(
            "r",
            encoding="utf-8-sig",
        ) as f:
            masking = json.load(f)

        for item in masking.get(
            "review_windows",
            [],
        ):
            windows.append({
                "priority": item.get(
                    "priority",
                    "high",
                ),
                "type":
                    "overlap_intelligibility_check",
                "start": round(
                    clamp(
                        float(item["start"]),
                        0.0,
                        duration,
                    ),
                    3,
                ),
                "end": round(
                    clamp(
                        float(item["end"]),
                        0.0,
                        duration,
                    ),
                    3,
                ),
                "description":
                    item.get(
                        "description",
                        "Verify whether overlap reduces intelligibility.",
                    ),
                "word":
                    item.get("word"),
                "segment":
                    item.get("segment"),
            })
    windows = merge_transcript_review_windows(
        windows,
        max_gap_sec=0.25,
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

    run_optional_evidence()
    run_asr_consensus()
    run_face_and_speaker_mapping()
    run_sound_events()

    enrich_evidence_with_shots(
        EVIDENCE,
        WHISPERX_JSON,
        CONTEXT,
    )
    enrich_evidence_with_ui_candidates(
        EVIDENCE,
    )
    enrich_evidence_with_voice_profiles(
        EVIDENCE,
        SPEAKER_MAP,
    )
    windows = build_review_queue()
    records = create_review_clips(windows)
    validate_review_clips(records)

    print(
        "\n=== PHASE 5: REVIEW REPORT ===\n"
    )

    generate_review_report()

    print(
        "\n=== PHASE 6: VALIDATOR PREFLIGHT ===\n"
    )

    validator_result = run_audio_validator()

    print(
        "\n=== PHASE 7: CONSOLIDATED REVIEW PACKET ===\n"
    )

    generate_master_packet()

    print()
    print("===================================")
    print(" PIPELINE EXECUTION: PASS")
    print("===================================")
    print("Evidence:", EVIDENCE)
    print("Review queue:", QUEUE)
    print("Review clips:", CLIP_DIR)
    print("Manifest:", MANIFEST)
    print("Master packet:", ROOT / "analysis" / "manuscript_audio_review_packet.json")
    print("Human summary:", ROOT / "analysis" / "REVIEW_ME.md")
    print("UI suggestions:", ROOT / "analysis" / "manuscript_audio_ui_suggestions.json")


if __name__ == "__main__":
    main()
















