$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$AudioVenv = Join-Path $Root ".venv-audio-events"
$AudioRequirements = Join-Path $Root "requirements-audio-events.txt"

Write-Host ""
Write-Host "================================================"
Write-Host " MANUSCRIPT PHASE 3C AUDIO-EVENTS SETUP (PANNs + CLAP)"
Write-Host " (optional evidence; failure does not block the core review stack)"
Write-Host "================================================"
Write-Host ""

$PyLauncher = Get-Command py -ErrorAction SilentlyContinue
if (-not $PyLauncher) {
    throw "Python Launcher ('py') was not found. Install Python 3.11 for Windows, then reopen PowerShell."
}

Write-Host "[1/6] Checking Python 3.11..."
& py -3.11 -c "import sys; print('Python:', sys.version); assert sys.version_info[:2] == (3, 11)"
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.11 is required. Run: py -0p to see installed Python versions."
}

Write-Host ""
Write-Host "[2/6] Creating audio-events environment..."
if (-not (Test-Path $AudioVenv)) {
    & py -3.11 -m venv $AudioVenv
    if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv-audio-events." }
} else {
    Write-Host ".venv-audio-events already exists. Reusing it."
}

$AudioPython = Join-Path $AudioVenv "Scripts\python.exe"
if (-not (Test-Path $AudioPython)) {
    throw "Audio-events Python was not created: $AudioPython"
}

& $AudioPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip in .venv-audio-events." }

Write-Host ""
Write-Host "[3/6] Installing torch (CPU wheel, avoids the CUDA bundle and the"
Write-Host "       torch 2.13.x long-path install failure)..."
& $AudioPython -m pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw "Failed to install CPU torch." }

Write-Host ""
Write-Host "[4/6] Installing remaining audio-events requirements..."
& $AudioPython -m pip install -r $AudioRequirements
if ($LASTEXITCODE -ne 0) { throw "Failed to install audio-events requirements." }

Write-Host ""
Write-Host "[5/6] Downloading PANNs runtime files (~/panns_data)..."
# panns_inference shells out to `wget` (absent on Windows) to fetch its
# labels CSV and checkpoint at import/use time. Pre-download them here with
# the venv's own urllib so the worker never depends on wget.
& $AudioPython -c @"
import os, urllib.request
from pathlib import Path
home = Path.home()
os.makedirs(home / 'panns_data', exist_ok=True)
labels = home / 'panns_data' / 'class_labels_indices.csv'
ckpt = home / 'panns_data' / 'Cnn14_mAP=0.431.pth'
if not labels.exists():
    print('  downloading class_labels_indices.csv ...')
    urllib.request.urlretrieve(
        'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv',
        str(labels))
if not (ckpt.exists() and ckpt.stat().st_size > 3e8):
    print('  downloading Cnn14_mAP=0.431.pth (~327 MB) ...')
    urllib.request.urlretrieve(
        'https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1',
        str(ckpt))
print('  labels:', labels.exists(), '| checkpoint:', ckpt.exists(), ckpt.stat().st_size if ckpt.exists() else 0)
"@
if ($LASTEXITCODE -ne 0) { throw "Failed to download PANNs runtime files." }

Write-Host ""
Write-Host "[6/6] Verifying imports + compiling the worker..."
& $AudioPython -c "import torch, numpy, soundfile, transformers; from panns_inference import AudioTagging, labels; from transformers import ClapModel, ClapProcessor; print('audio-events import: PASS'); print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available()); print('numpy:', numpy.__version__); print('soundfile:', soundfile.__version__); print('transformers:', transformers.__version__); print('panns labels:', len(labels))"
if ($LASTEXITCODE -ne 0) { throw "Audio-events import verification failed." }

$Worker = Join-Path $Root "manuscript_audio_sound_events_worker.py"
if (-not (Test-Path $Worker)) {
    throw "Sound-events worker missing: $Worker"
}
& $AudioPython -m py_compile $Worker
if ($LASTEXITCODE -ne 0) { throw "Worker syntax check failed." }

Write-Host ""
Write-Host "================================================"
Write-Host " PHASE 3C AUDIO-EVENTS SETUP COMPLETE"
Write-Host "================================================"
Write-Host ""
Write-Host "Audio-events Python:"
Write-Host "  $AudioPython"
Write-Host ""
Write-Host "First CLAP run downloads laion/clap-htsat-unfused (~2 GB) to the"
Write-Host "HuggingFace cache under ~/.cache/huggingface."
