from pathlib import Path
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parent

WHISPERX_PYTHON = (
    ROOT
    / ".venv-whisperx"
    / "Scripts"
    / "python.exe"
)

DIARIZE_SCRIPT = (
    ROOT
    / "manuscript_audio_diarize.py"
)

DIARIZATION_QC_SCRIPT = (
    ROOT
    / "manuscript_audio_diarization_qc.py"
)

SPEAKER_MAPPING_SCRIPT = (
    ROOT
    / "manuscript_audio_speaker_mapping.py"
)

DEFECT_SCRIPT = ROOT / "manuscript_audio_defects.py"
MASKING_SCRIPT = ROOT / "manuscript_audio_masking.py"

DIARIZATION_EVIDENCE = (
    ROOT
    / "analysis"
    / "diarization_evidence.json"
)

DIARIZATION_QC = (
    ROOT
    / "analysis"
    / "diarization_cluster_review.json"
)

SPEAKER_MAPPING_REVIEW = (
    ROOT
    / "analysis"
    / "speaker_mapping_review.json"
)

DIARIZED_TRANSCRIPT = (
    ROOT
    / "output"
    / "VIDEO.diarized.json"
)

GENERATED_SPEAKER_MAP = (
    ROOT
    / "speaker_map.generated.json"
)

SOUND_EVIDENCE = (
    ROOT
    / "analysis"
    / "sound_event_evidence.json"
)

VAD_REGIONS = (
    ROOT
    / "analysis"
    / "vad_speech_regions.json"
)

AUDIO = (
    ROOT
    / "analysis"
    / "audio.wav"
)

CONTEXT = (
    ROOT
    / "task_context.json"
)


def remove_stale_optional_outputs():
    paths = [
        DIARIZATION_EVIDENCE,
        DIARIZATION_QC,
        SPEAKER_MAPPING_REVIEW,
        DIARIZED_TRANSCRIPT,
        GENERATED_SPEAKER_MAP,
        SOUND_EVIDENCE,
        VAD_REGIONS,
    ]

    for path in paths:
        if path.exists():
            path.unlink()


def run_script(label, python, script):
    try:
        result = subprocess.run(
            [
                str(python),
                str(script),
            ],
            cwd=ROOT,
            check=False,
        )

        if result.returncode != 0:
            print(
                f"{label}: WARNING | "
                f"exit code {result.returncode}"
            )
            return False

        return True

    except Exception as exc:
        print(
            f"{label}: WARNING | {exc}"
        )
        return False


def diarization_complete():
    if not DIARIZATION_EVIDENCE.exists():
        return False

    try:
        with DIARIZATION_EVIDENCE.open(
            "r",
            encoding="utf-8-sig",
        ) as f:
            data = json.load(f)

        return (
            data.get("status")
            == "complete"
        )

    except Exception:
        return False


def run_diarization_layer():
    print(
        "\n=== OPTIONAL DIARIZATION EVIDENCE ===\n"
    )

    if not WHISPERX_PYTHON.exists():
        print(
            "DIARIZATION: SKIPPED | "
            "WhisperX environment missing"
        )
        return

    if not DIARIZE_SCRIPT.exists():
        print(
            "DIARIZATION: SKIPPED | "
            "diarization script missing"
        )
        return

    run_script(
        "DIARIZATION",
        WHISPERX_PYTHON,
        DIARIZE_SCRIPT,
    )

    if not diarization_complete():
        print(
            "DIARIZATION FOLLOW-UP: SKIPPED | "
            "no completed diarization"
        )
        return

    run_script(
        "DIARIZATION CLUSTER QC",
        Path(sys.executable),
        DIARIZATION_QC_SCRIPT,
    )

    run_script(
        "SPEAKER MAPPING REPORT",
        Path(sys.executable),
        SPEAKER_MAPPING_SCRIPT,
    )


def run_sound_layer():
    print(
        "\n=== NON-SPEECH SOUND EVIDENCE ===\n"
    )

    if not AUDIO.exists():
        print(
            "SOUND EVIDENCE: SKIPPED | "
            "analysis WAV missing"
        )
        return

    try:
        from manuscript_audio_sounds import (
            write_sound_evidence,
            enrich_sound_evidence_with_shots,
        )

        write_sound_evidence(
            AUDIO,
            SOUND_EVIDENCE,
        )

        if CONTEXT.exists():
            enrich_sound_evidence_with_shots(
                SOUND_EVIDENCE,
                CONTEXT,
            )
        else:
            print(
                "SOUND SHOT ASSIGNMENT: SKIPPED | "
                "task_context.json missing"
            )

    except Exception as exc:
        print(
            "SOUND EVIDENCE: WARNING |",
            exc,
        )




def run_vad_layer():
    print(
        "\n=== VAD SPEECH-PRESENCE FALLBACK ===\n"
    )

    if not AUDIO.exists():
        print(
            "VAD SPEECH REGIONS: SKIPPED | analysis WAV missing"
        )
        return

    try:
        from manuscript_audio_vad import write_vad_regions

        write_vad_regions(AUDIO)
    except Exception as exc:
        print("VAD SPEECH REGIONS: WARNING |", exc)


def run_quality_layers():
    print(
        "\n=== RECORDING / MASKING EVIDENCE ===\n"
    )

    if DEFECT_SCRIPT.exists():
        run_script(
            "RECORDING DEFECT EVIDENCE",
            Path(sys.executable),
            DEFECT_SCRIPT,
        )

    if MASKING_SCRIPT.exists():
        run_script(
            "MASKING / OVERLAP EVIDENCE",
            Path(sys.executable),
            MASKING_SCRIPT,
        )

def run_optional_evidence():
    print(
        "\n==================================="
    )
    print(
        " OPTIONAL MANUSCRIPT AUDIO EVIDENCE"
    )
    print(
        "==================================="
    )

    remove_stale_optional_outputs()

    run_diarization_layer()
    run_vad_layer()
    run_sound_layer()
    run_quality_layers()
    print()
    print(
        "OPTIONAL AUDIO EVIDENCE: COMPLETE"
    )

