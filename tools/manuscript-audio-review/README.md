# Manuscript II Audio Review Pipeline

Windows setup for the Manuscript II audio-review helper.

The tool takes a source video and creates audio evidence for human review. Automatic transcript, pitch, emotion, and speaker-change signals are leads only. The reviewer must still listen to the actual audio.

## What the pipeline does

1. Extracts a 16 kHz mono WAV with FFmpeg.
2. Runs WhisperX 3.8.6 with `large-v3`, CPU, and INT8 compute.
3. Reads word-level aligned transcript evidence.
4. Flags low-confidence transcript words.
5. Extracts acoustic features with openSMILE.
6. Runs Wav2Vec2 emotion evidence.
7. Checks large pitch changes between adjacent WhisperX segments.
8. Builds a human-review queue.
9. Creates short WAV clips around flagged regions.
10. Validates generated review-clip durations.

The system never automatically chooses final tone or speaker identity.

## Requirements

Install these before running setup:

- Git for Windows
- Python 3.11, 64-bit, including the `py` launcher
- FFmpeg and FFprobe on Windows PATH
- Internet access for Python packages and first-run model downloads

Check them in PowerShell:

```powershell
git --version
py -3.11 --version
ffmpeg -version
ffprobe -version
```

## Fresh laptop setup

Clone the `autoscribe` branch:

```powershell
cd "$env:USERPROFILE\Desktop"
git clone --branch autoscribe https://github.com/jash65571/annotation.git
cd ".\annotation\tools\manuscript-audio-review"
```

Run the installer:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\setup_windows.ps1"
```

A successful setup ends with:

```text
SETUP COMPLETE
```

The installer creates two isolated environments:

```text
.venv-review\
.venv-whisperx\
```

`.venv-review` contains the acoustic and emotion analysis stack.

`.venv-whisperx` contains WhisperX and its dependency stack.

You do not need to activate either environment manually.

## Run a video

From the `tools\manuscript-audio-review` directory:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\review_manuscript_audio.ps1" "C:\path\to\your\video.mp4"
```

If local scripts are already allowed, the shorter form also works:

```powershell
.\review_manuscript_audio.ps1 "C:\path\to\your\video.mp4"
```

Video filenames may contain spaces.

A successful run ends with:

```text
SOURCE PREPROCESSING: PASS
PIPELINE STATUS: PASS
AUDIO REVIEW PIPELINE COMPLETE
```

## Outputs

Generated files stay inside the tool directory and are ignored by Git.

```text
output\VIDEO.json
analysis\audio.wav
analysis\manuscript_audio_evidence.json
analysis\audio_review_queue.json
analysis\review_clips\
analysis\review_clips\review_clips_manifest.json
```

`output\VIDEO.json` is the stable internal WhisperX transcript path. A source-named WhisperX JSON is copied into that stable path after each run.

`manuscript_audio_evidence.json` contains transcript, acoustic, emotion, continuity, and review-synthesis evidence.

`audio_review_queue.json` identifies regions that deserve closer listening.

`review_clips` contains short WAV files around those flagged regions.

## Reviewer environment

Pinned direct packages:

```text
numpy==2.4.6
opensmile==2.6.0
soundfile==0.14.0
torch==2.13.0
transformers==5.15.1
```

They are installed from `requirements-review.txt`.

## WhisperX environment

Pinned package:

```text
whisperx==3.8.6
```

It is installed from `requirements-whisperx.txt`.

## Troubleshooting

### Python 3.11 not found

Run:

```powershell
py -0p
```

Install Python 3.11 if it is missing, then reopen PowerShell.

### FFmpeg not found

Run:

```powershell
Get-Command ffmpeg
Get-Command ffprobe
```

If either command fails, add the FFmpeg `bin` directory to Windows PATH and reopen PowerShell.

### PowerShell blocks scripts

Run the script through a one-process bypass:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\setup_windows.ps1"
```

or:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\review_manuscript_audio.ps1" "C:\path\to\video.mp4"
```

### Verify the reviewer environment

```powershell
.\.venv-review\Scripts\python.exe -c "import numpy, opensmile, soundfile, torch, transformers; print('REVIEW PASS')"
```

### Verify WhisperX

```powershell
.\.venv-whisperx\Scripts\python.exe -c "import whisperx; print('WHISPERX PASS')"
.\.venv-whisperx\Scripts\python.exe -m whisperx --version
```

### First run is slow

The first WhisperX or Hugging Face run may download model files. Later runs reuse the local cache.

## Updating on another laptop

From the cloned repository:

```powershell
git checkout autoscribe
git pull origin autoscribe
cd ".\tools\manuscript-audio-review"
```

If dependency files changed, rerun:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\setup_windows.ps1"
```

## Git safety

The tool-level `.gitignore` excludes:

- `.venv-review`
- `.venv-whisperx`
- `analysis`
- `output`
- video files
- generated audio files
- model/cache files
- Python cache files

Do not force-add task videos, transcripts, or generated evidence.

## Manuscript II review rule

This pipeline is an evidence helper, not annotation authority.

- ASR text must be checked by listening.
- Low confidence means review the word, not automatically mark it wrong.
- Emotion output does not decide final tone.
- Pitch changes do not identify speakers.
- Actual clip audio remains the factual source of truth.
