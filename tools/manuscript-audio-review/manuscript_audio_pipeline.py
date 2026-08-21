from pathlib import Path
import json
import os
import subprocess
import sys


def load_env_file(path=".env"):
    """Best-effort .env loader (local secrets only). Sets HF_TOKEN etc. in
    os.environ BEFORE any subprocess spawns, so the WhisperX diarization
    child (which reads os.environ["HF_TOKEN"]) inherits it. Missing file,
    malformed lines, and empty values are silently ignored -- never raise.
    """
    env_path = Path(path)
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value:
                os.environ.setdefault(key, value)
    except OSError:
        return

from manuscript_audio_shots import enrich_evidence_with_shots
from manuscript_audio_queue import merge_transcript_review_windows
from manuscript_audio_ui import enrich_evidence_with_ui_candidates
from manuscript_audio_voice import enrich_evidence_with_voice_profiles
from manuscript_audio_optional import run_optional_evidence
from manuscript_audio_task_identity import load_bound_json, write_video_identity
from manuscript_audio_seed import parse_seed_text, write_task_context
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
LOUDNORM_AUDIO = ROOT / "analysis" / "audio_loudnorm.wav"
ASR_WORKER = ROOT / "manuscript_audio_asr_worker.py"
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


def transcribe_boosted_primary(whisperx_python, audio_path):
    """Primary ASR on the loudness-normalized WAV with VAD disabled.

    whisperx's CLI pipeline gates transcription behind a pyannote VAD that
    silently drops very quiet speech -- real clips have measured 20+ dB
    below a normal dialogue mix, taking primary coverage from ~100% to
    single digits. The worker (`manuscript_audio_asr_worker.py`) transcribes
    with faster-whisper `vad_filter=False` over the WHOLE clip in strict
    anti-hallucination mode (no conditioning on previous text + repetition
    penalty), so quiet but real dialogue is captured without hallucinated
    continuations. Returns a whisperx-compatible transcript dict, or None on
    any failure so callers can fall back to the CLI.
    """
    try:
        result = subprocess.run(
            [
                str(whisperx_python),
                str(ASR_WORKER),
                str(audio_path),
                "--model",
                "large-v3",
                "--compute-type",
                "int8",
                "--strict",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        payload = json.loads(result.stdout)
    except Exception:  # noqa: BLE001 -- fail soft, caller falls back
        return None

    if not isinstance(payload, dict) or payload.get("status") != "complete":
        return None

    segments = payload.get("segments") or []
    if not segments:
        return None

    return {
        "segments": segments,
        "language": payload.get("language"),
        "boosted_primary": True,
    }


def parse_source_streams(streams):
    """Normalize ffprobe audio/video streams without inventing metadata."""
    out = {}
    for stream in streams or []:
        stream_type = stream.get("codec_type")
        if stream_type == "audio":
            if stream.get("sample_rate"):
                out["source_sample_rate"] = int(stream["sample_rate"])
            if stream.get("channels") is not None:
                out["audio_channels"] = int(stream["channels"])
            if stream.get("codec_name"):
                out["audio_codec"] = stream["codec_name"]
        elif stream_type == "video":
            width = stream.get("width")
            height = stream.get("height")
            if width is not None and height is not None:
                out["resolution"] = f"{int(width)}x{int(height)}"
            rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
            if rate and rate != "0/0":
                numerator, separator, denominator = str(rate).partition("/")
                try:
                    value = float(numerator)
                    if separator:
                        value /= float(denominator)
                    out["fps"] = round(value, 6)
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
            if stream.get("nb_frames") not in (None, "N/A"):
                try:
                    out["frame_count"] = int(stream["nb_frames"])
                except (TypeError, ValueError):
                    pass
            if stream.get("codec_name"):
                out["video_codec"] = stream["codec_name"]
    return out


def probe_source_audio(video_path):
    """Best-effort ffprobe of the ORIGINAL source media streams.

    The historic function name is retained for compatibility, but it now
    captures video integrity metadata too. It never raises.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,codec_name,sample_rate,channels,width,height,"
                "avg_frame_rate,r_frame_rate,nb_frames",
                "-of",
                "json",
                str(video_path),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return {}
        return parse_source_streams(json.loads(result.stdout).get("streams", []))
    except Exception:  # noqa: BLE001 -- best-effort, never break the pipeline
        return {}


def run_seed_ingestion(seed_path):
    """3.5: auto-ingest the locked Manuscript task seed so the pipeline knows
    the real C#/O# structure and locked shot ranges BEFORE analyzing. Without
    this, every stage invents empty characters/objects/shots.
    """
    print("\n=== PHASE 0.5: TASK SEED INGESTION ===\n")

    seed_path = Path(seed_path).expanduser().resolve()

    if not seed_path.exists():
        raise FileNotFoundError(f"Task seed file not found: {seed_path}")

    text = seed_path.read_text(encoding="utf-8-sig")

    if not text.strip():
        raise ValueError(f"Task seed file is empty: {seed_path}")

    parsed = parse_seed_text(text)

    if not parsed["shots"]:
        print("TASK SEED: WARNING | no shot boundaries parsed; "
              "shot-aware evidence will stay empty until a valid seed is used")

    result = write_task_context(parsed, preserve_sha=True)

    meta = result["seed_meta"]

    print("Seed file:", seed_path)
    print("Characters:", ", ".join(result["characters"]) or "(none)")
    print("Objects:", ", ".join(result["objects"]) or "(none)")
    print("Shots:", len(result["shots"]))

    if meta["parse_issues"]:
        print("Parse issues:")
        for issue in meta["parse_issues"]:
            print("  -", issue)

    print("Task context:", CONTEXT)
    return result


def require_current_task_context(current_video_sha256):
    """Return the current task context or fail closed on stale state.

    This is intentionally called even when task_context.json exists. File
    existence alone is not provenance: the workspace is reused between
    clips, so an unbound or differently-bound seed must never reach analysis.
    """
    bound_context = load_bound_json(
        CONTEXT,
        current_video_sha256,
        "TASK SEED",
    )
    if bound_context is None:
        raise RuntimeError(
            "No task seed is bound to this video. Pass the current "
            "locked task seed as the second argument; a task_context.json "
            "from another clip is never reused."
        )
    return bound_context


def preprocess_source_video(video=None):
    print("\n=== PHASE 0: SOURCE PREPROCESSING ===\n")

    if video is None:
        video = ROOT / "VIDEO.mp4"

    video = Path(video).expanduser().resolve()

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

    # 3.5: record the ORIGINAL source audio properties (the analysis WAV is
    # resampled to 16 kHz mono, so its sample rate must never be presented
    # as the source's). Best-effort: if ffprobe fails, the key stays absent
    # and the master reports it honestly as not extracted.
    source_media = probe_source_audio(video)
    if source_media:
        identity.update(source_media)
        with VIDEO_IDENTITY.open("w", encoding="utf-8") as f:
            json.dump(identity, f, indent=2)

    print("Source video:", video)
    print(
        "Video fingerprint:",
        identity["video_sha256"][:12] + "...",
    )
    if identity.get("source_sample_rate"):
        print(
            "Source audio:",
            f"{identity['source_sample_rate']} Hz, "
            f"{identity.get('audio_channels', '?')} ch, "
            f"{identity.get('audio_codec', '?')}",
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
    print("\n[2/2] Running WhisperX large-v3 (loudness-normalized, no VAD)...")

    # Boost quiet dialogue before ASR: many real clips sit 20+ dB below a
    # normal dialogue mix, and whisperx's VAD-gated pipeline then misses the
    # speech entirely (measured 2% coverage on a real task clip). Normalize
    # loudness, then transcribe with vad_filter=False over the whole clip.
    print("Loudness-normalizing analysis WAV for ASR...")
    run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(AUDIO),
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(LOUDNORM_AUDIO),
        ]
    )

    for stale_json in {
        generated_whisperx_json,
        whisperx_json,
    }:
        if stale_json.exists():
            stale_json.unlink()

    transcript = transcribe_boosted_primary(whisperx_python, LOUDNORM_AUDIO)

    if transcript is None:
        # Fallback 1: original whisperx CLI, but on the normalized WAV so it
        # still benefits from the loudness boost.
        print(
            "Boosted primary unavailable; falling back to whisperx CLI "
            "on the normalized WAV..."
        )
        loudnorm_stem_json = output_dir / f"{LOUDNORM_AUDIO.stem}.json"
        if loudnorm_stem_json.exists():
            loudnorm_stem_json.unlink()
        run(
            [
                str(whisperx_python),
                "-m",
                "whisperx",
                str(LOUDNORM_AUDIO),
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
        if loudnorm_stem_json.exists():
            with loudnorm_stem_json.open("r", encoding="utf-8") as f:
                transcript = json.load(f)

    if transcript is None:
        # Fallback 2: original whisperx CLI on the source video (legacy path).
        print(
            "CLI-on-normalized unavailable; falling back to whisperx CLI "
            "on the source video..."
        )
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
        if generated_whisperx_json.exists():
            with generated_whisperx_json.open("r", encoding="utf-8") as f:
                transcript = json.load(f)

    if transcript is None or not isinstance(transcript, dict):
        raise RuntimeError(
            "WhisperX produced no transcript (boosted worker, CLI on "
            "normalized WAV, and CLI on source video all failed)."
        )

    # Normalize into the stable output/VIDEO.json plus the source-named copy.
    whisperx_json.write_text(
        json.dumps(transcript, ensure_ascii=False),
        encoding="utf-8",
    )
    generated_whisperx_json.write_text(
        json.dumps(transcript, ensure_ascii=False),
        encoding="utf-8",
    )

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
    return identity


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
                    "divergence_regions": [],
                    "rerun_windows": [],
                },
                f,
                indent=2,
            )


def _write_face_failure(status, error, diagnostic_code):
    FACE_TRACK_EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    with FACE_TRACK_EVIDENCE.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "status": status,
                "error": error,
                "error_code": diagnostic_code,
                "face_tracks": [],
            },
            f,
            indent=2,
        )


def _normalize_face_failure_artifact():
    """Validate the worker artifact and preserve an actionable failure code."""
    try:
        with FACE_TRACK_EVIDENCE.open("r", encoding="utf-8-sig") as f:
            result = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        _write_face_failure(
            "failed",
            "face worker did not produce valid JSON output",
            "worker_output_invalid",
        )
        return

    if not isinstance(result, dict) or result.get("status") not in ("complete", "failed", "unavailable"):
        _write_face_failure(
            "failed",
            "face worker output is missing a valid status",
            "worker_output_invalid",
        )
        return

    if result.get("status") != "complete":
        result.setdefault("error_code", "worker_failed")
        result.setdefault("face_tracks", [])
        with FACE_TRACK_EVIDENCE.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


def run_face_and_speaker_mapping():
    print("\n=== PHASE 1.6: FACE TRACKING / ACTIVE-SPEAKER MAPPING ===\n")

    if not VISION_PYTHON.exists():
        print("FACE TRACKING: SKIPPED | .venv-vision not found (run setup)")
        _write_face_failure(
            "unavailable",
            "vision environment missing",
            "vision_environment_missing",
        )
    elif not FACE_WORKER.exists():
        print("FACE TRACKING: SKIPPED | face worker script not found")
        _write_face_failure(
            "failed",
            "face worker script not found",
            "worker_launch_failed",
        )
    else:
        try:
            run([
                str(VISION_PYTHON),
                str(FACE_WORKER),
                str(VIDEO_PATH),
                str(FACE_TRACK_EVIDENCE),
                "--fps",
                "5",
            ])
            _normalize_face_failure_artifact()
        except (subprocess.CalledProcessError, OSError) as exc:
            print(f"FACE TRACKING: SKIPPED | worker launch failed: {exc}")
            _write_face_failure(
                "failed",
                str(exc),
                "worker_launch_failed",
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
                    "error_code": "worker_output_invalid",
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
                    "transients": {"events": [], "findings": [], "status": "unavailable"},
                    "review_windows": [],
                    "findings": [],
                },
                f,
                indent=2,
            )


def clamp(value, low, high):
    return max(low, min(high, value))


def shot_of_window(start, end, shots):
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


def build_shot_boundary_review_windows(evidence, duration):
    """Collapse routine visual-cut continuity cues into one listening item.

    A locked visual cut does not imply an audio cut. Keep separate targeted
    checks only when the evidence points to a word crossing the cut, a source
    change, or an intelligibility risk. Routine continuous speech across
    multiple cuts becomes one continuity instruction instead of one warning
    per shot boundary.
    """
    shot_evidence = evidence.get("shot_audio_evidence", []) or []
    segments = evidence.get("whisperx_segments", []) or []
    low_confidence = [
        word
        for segment in segments
        for word in segment.get("low_confidence_words", [])
    ]

    routine_centers = []
    targeted = []
    seen = set()

    for shot in shot_evidence:
        candidates = []
        if shot.get("speech_crosses_into_shot"):
            candidates.append((
                float(shot["start"]),
                f"Speech crosses into Shot {shot['shot']} from the previous shot.",
            ))
        if shot.get("speech_crosses_out_of_shot"):
            candidates.append((
                float(shot["end"]),
                f"Speech continues out of Shot {shot['shot']} into the next shot.",
            ))

        for center, description in candidates:
            center = float(center)
            key = round(center, 3)
            if key in seen:
                continue
            seen.add(key)

            word_crosses = any(
                float(word.get("start", 0.0)) < center < float(word.get("end", 0.0))
                for segment in segments
                for word in segment.get("words", [])
                if word.get("start") is not None and word.get("end") is not None
            )
            nearby_low_confidence = any(
                abs(float(word.get("start", center)) - center) <= 0.35
                or abs(float(word.get("end", center)) - center) <= 0.35
                for word in low_confidence
            )
            source_change = bool(shot.get("possible_speaker_or_source_change"))
            targeted_reason = (
                "word crosses the visual cut" if word_crosses else
                "source identity may change" if source_change else
                "intelligibility risk near the cut" if nearby_low_confidence else
                None
            )

            start = round(clamp(center - 0.75, 0.0, duration), 3)
            end = round(clamp(center + 0.75, 0.0, duration), 3)
            if targeted_reason:
                targeted.append({
                    "priority": "high",
                    "type": "shot_boundary_speech_check",
                    "start": start,
                    "end": end,
                    "description": description + " " + targeted_reason + ".",
                })
            else:
                routine_centers.append(center)

    if routine_centers:
        start = round(clamp(min(routine_centers) - 0.75, 0.0, duration), 3)
        end = round(clamp(max(routine_centers) + 0.75, 0.0, duration), 3)
        if len(routine_centers) > 1:
            description = (
                "Speech continues across multiple locked visual shot "
                "boundaries. Preserve continuity when assigning the speech "
                "to each shot."
            )
        else:
            description = (
                "Speech continues across a locked visual shot boundary. "
                "Preserve continuity when assigning the speech to each shot."
            )
        targeted.append({
            "priority": "high",
            "type": "shot_boundary_continuity_check",
            "start": start,
            "end": end,
            "description": description,
        })

    return targeted


def build_review_queue():
    print("\n=== PHASE 2: TARGETED REVIEW QUEUE ===\n")

    with EVIDENCE.open("r", encoding="utf-8") as f:
        evidence = json.load(f)

    duration = float(evidence["media"]["duration_sec"])
    windows = []

    # 3.5: every review window carries its locked shot when the seed defines
    # one, so reviewers can queue by shot.
    context_shots = []
    seed_listening_targets = []
    if CONTEXT.exists():
        try:
            with CONTEXT.open("r", encoding="utf-8-sig") as f:
                task_context = json.load(f)
            context_shots = task_context.get("shots", []) or []
            seed_listening_targets = (
                task_context.get("seed_meta", {}).get(
                    "human_listening_targets", []
                ) or []
            )
        except (json.JSONDecodeError, OSError):
            context_shots = []
            seed_listening_targets = []

    def with_shot(item):
        if not context_shots:
            return item
        item["shot"] = shot_of_window(
            float(item["start"]), float(item["end"]), context_shots
        )
        # 3.6: cross-shot windows carry their full shots list instead of one
        # forced shot (e.g. a transient spanning Shots 2-3 must not be
        # labeled only Shot 3).
        crossed = []
        for shot in context_shots:
            s_start = float(shot.get("start", 0.0))
            s_end = float(shot.get("end", 0.0))
            if float(item["end"]) > s_start and float(item["start"]) < s_end:
                crossed.append(shot.get("shot"))
        crossed = sorted(s for s in crossed if s is not None)
        if len(crossed) > 1:
            item["shots"] = crossed
            item["shot"] = None  # cross-shot: never force one shot (3.6)
        return item

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

    # A long tail without independent speech is a listen-only safety check,
    # not an automatic ASR rerun. The ASR stage writes this explicitly so the
    # queue preserves the check without trusting a hallucinated tail transcript.
    if ASR_CONSENSUS.exists():
        try:
            with ASR_CONSENSUS.open("r", encoding="utf-8-sig") as f:
                asr_consensus = json.load(f)
        except (json.JSONDecodeError, OSError):
            asr_consensus = {}
        tail_check = asr_consensus.get("clip_tail_check")
        if tail_check:
            windows.append({
                "priority": "medium",
                "type": "clip_tail_check",
                "start": round(clamp(float(tail_check["start"]), 0.0, duration), 3),
                "end": round(clamp(float(tail_check["end"]), 0.0, duration), 3),
                "description": tail_check.get(
                    "action",
                    "Confirm the clip tail is genuinely non-speech.",
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

    # Flag speech that crosses locked visual boundaries. Routine continuity
    # cues are collapsed; only targeted word/source/intelligibility risks stay
    # separate. Internal timestamps are evidence only and never enter Final
    # Audio Text.
    windows.extend(build_shot_boundary_review_windows(evidence, duration))
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
            # 3.6: masking_check queue items exist ONLY from the final
            # masking evidence (masking_overlap_evidence.json) -- the fusion
            # layer never emits them, so the queue can never contradict the
            # packet's masking section.
            windows.append({
                "priority": item.get(
                    "priority",
                    "high",
                ),
                "type":
                    item.get("type", "masking_check"),
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
    # 3.5: sound/music/ambience/transient review windows (Phase 3C/3.5) also
    # enter the queue so clips get cut for them. Transient_sfx_check windows
    # from the independent detector are high-priority when STRONG.
    if SOUND_FUSION_EVIDENCE.exists():
        try:
            with SOUND_FUSION_EVIDENCE.open(
                "r", encoding="utf-8-sig"
            ) as f:
                sound_fusion = json.load(f)
        except (json.JSONDecodeError, OSError):
            sound_fusion = {}

        for item in sound_fusion.get("review_windows", []):
            tier = item.get("tier")
            priority = (
                "high"
                if tier in ("STRONG", "CONFLICT")
                else "medium"
            )
            windows.append({
                "priority": priority,
                "type": item.get("type", "sound_check"),
                "start": round(float(item["start"]), 3),
                "end": round(float(item["end"]), 3),
                "description": item.get("description", "Verify sound by listening."),
                "tier": tier,
                "shots": item.get("shots") or [],
            })

    # Preserve seed-named non-speech claims as ONE consolidated listen/reject
    # task. They are not machine confirmations and must not become automatic
    # Sound events, but silently dropping them makes a completeness audit
    # impossible (for example chewing, wind, and bottle/table contact).
    target_labels = [
        str(item.get("label") or item.get("class") or "").strip()
        for item in seed_listening_targets
        if isinstance(item, dict)
    ]
    target_labels = [label for label in target_labels if label]
    if target_labels:
        windows.append({
            "priority": "high",
            "type": "seed_nonspeech_check",
            "start": 0.0,
            "end": round(duration, 3),
            "description": (
                "Seed names these non-speech sounds; listen through once and "
                "confirm or reject each without assuming it is present: "
                + ", ".join(target_labels)
                + "."
            ),
            "candidate_classes": [
                item.get("class") for item in seed_listening_targets
                if isinstance(item, dict) and item.get("class")
            ],
        })

    windows = [with_shot(w) for w in windows]

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

    # Load local secrets (.env) before any stage runs, so the WhisperX
    # diarization subprocess inherits HF_TOKEN instead of skipping.
    load_env_file()

    # Usage: python manuscript_audio_pipeline.py VIDEO.mp4 [SEED.txt]
    video_arg = sys.argv[1] if len(sys.argv) >= 2 else None
    seed_arg = sys.argv[2] if len(sys.argv) >= 3 else None

    identity = preprocess_source_video(video_arg)

    if seed_arg:
        run_seed_ingestion(seed_arg)
    else:
        require_current_task_context(identity["video_sha256"])
        print(
            "\nTASK SEED: using task_context.json verified against "
            "the current video fingerprint.\n"
        )

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
    print("Evidence ledger:", ROOT / "analysis" / "manuscript_audio_evidence_ledger.json")
    print("Cast audit:", ROOT / "analysis" / "manuscript_audio_cast_audit.json")


if __name__ == "__main__":
    main()











