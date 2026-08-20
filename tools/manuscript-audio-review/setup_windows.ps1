$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ReviewVenv = Join-Path $Root ".venv-review"
$WhisperXVenv = Join-Path $Root ".venv-whisperx"

$ReviewRequirements = Join-Path $Root "requirements-review.txt"
$WhisperXRequirements = Join-Path $Root "requirements-whisperx.txt"

Write-Host ""
Write-Host "========================================"
Write-Host " MANUSCRIPT II AUDIO REVIEW SETUP"
Write-Host "========================================"
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
Write-Host "[2/6] Checking FFmpeg..."
$FFmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
$FFprobe = Get-Command ffprobe -ErrorAction SilentlyContinue
if (-not $FFmpeg -or -not $FFprobe) {
    throw "FFmpeg or FFprobe was not found on PATH. Install FFmpeg, reopen PowerShell, and confirm ffmpeg -version and ffprobe -version work."
}
Write-Host "FFmpeg:" $FFmpeg.Source
Write-Host "FFprobe:" $FFprobe.Source

Write-Host ""
Write-Host "[3/6] Creating review environment..."
if (-not (Test-Path $ReviewVenv)) {
    & py -3.11 -m venv $ReviewVenv
    if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv-review." }
} else {
    Write-Host ".venv-review already exists. Reusing it."
}

$ReviewPython = Join-Path $ReviewVenv "Scripts\python.exe"
if (-not (Test-Path $ReviewPython)) {
    throw "Review Python was not created: $ReviewPython"
}

& $ReviewPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip in .venv-review." }
& $ReviewPython -m pip install -r $ReviewRequirements
if ($LASTEXITCODE -ne 0) { throw "Failed to install reviewer requirements." }

Write-Host ""
Write-Host "[4/6] Creating WhisperX environment..."
if (-not (Test-Path $WhisperXVenv)) {
    & py -3.11 -m venv $WhisperXVenv
    if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv-whisperx." }
} else {
    Write-Host ".venv-whisperx already exists. Reusing it."
}

$WhisperXPython = Join-Path $WhisperXVenv "Scripts\python.exe"
if (-not (Test-Path $WhisperXPython)) {
    throw "WhisperX Python was not created: $WhisperXPython"
}

& $WhisperXPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip in .venv-whisperx." }
& $WhisperXPython -m pip install -r $WhisperXRequirements
if ($LASTEXITCODE -ne 0) { throw "Failed to install WhisperX requirements." }

Write-Host ""
Write-Host "[5/6] Verifying environments..."
& $ReviewPython -c "import numpy, opensmile, soundfile, torch, transformers; print('Review environment: PASS'); print('numpy:', numpy.__version__); print('opensmile:', opensmile.__version__); print('soundfile:', soundfile.__version__); print('torch:', torch.__version__); print('transformers:', transformers.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Review environment verification failed." }

& $WhisperXPython -c "import whisperx; print('WhisperX import: PASS')"
if ($LASTEXITCODE -ne 0) { throw "WhisperX import verification failed." }
& $WhisperXPython -m whisperx --version
if ($LASTEXITCODE -ne 0) { throw "WhisperX CLI verification failed." }

Write-Host ""
Write-Host "[6/6] Checking Manuscript scripts..."
& $ReviewPython -m py_compile (Join-Path $Root "manuscript_audio_review.py") (Join-Path $Root "manuscript_audio_pipeline.py")
if ($LASTEXITCODE -ne 0) { throw "Python script syntax validation failed." }

Write-Host ""
Write-Host "========================================"
Write-Host " SETUP COMPLETE"
Write-Host "========================================"
Write-Host ""
Write-Host "Reviewer Python:"
Write-Host "  $ReviewPython"
Write-Host "WhisperX Python:"
Write-Host "  $WhisperXPython"
Write-Host ""
Write-Host "Run a video with:"
Write-Host '  .\review_manuscript_audio.ps1 "C:\path\to\VIDEO.mp4"'
Write-Host ""
Write-Host "The first WhisperX run may download model files."
