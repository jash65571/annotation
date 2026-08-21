$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VisionVenv = Join-Path $Root ".venv-vision"
$VisionRequirements = Join-Path $Root "requirements-vision.txt"

Write-Host ""
Write-Host "================================================"
Write-Host " MANUSCRIPT PHASE 3B VISION SETUP (mediapipe + opencv)"
Write-Host " (optional evidence; failure does not block the core review stack)"
Write-Host "================================================"
Write-Host ""

$PyLauncher = Get-Command py -ErrorAction SilentlyContinue
if (-not $PyLauncher) {
    throw "Python Launcher ('py') was not found. Install Python 3.11 for Windows, then reopen PowerShell."
}

Write-Host "[1/4] Checking Python 3.11..."
& py -3.11 -c "import sys; print('Python:', sys.version); assert sys.version_info[:2] == (3, 11)"
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.11 is required. Run: py -0p to see installed Python versions."
}

Write-Host ""
Write-Host "[2/4] Creating vision environment..."
if (-not (Test-Path $VisionVenv)) {
    & py -3.11 -m venv $VisionVenv
    if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv-vision." }
} else {
    Write-Host ".venv-vision already exists. Reusing it."
}

$VisionPython = Join-Path $VisionVenv "Scripts\python.exe"
if (-not (Test-Path $VisionPython)) {
    throw "Vision Python was not created: $VisionPython"
}

& $VisionPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip in .venv-vision." }

Write-Host ""
Write-Host "[3/4] Installing vision requirements (mediapipe + opencv headless)..."
& $VisionPython -m pip install -r $VisionRequirements
if ($LASTEXITCODE -ne 0) { throw "Failed to install vision requirements." }

Write-Host ""
Write-Host "[4/4] Verifying imports + compiling the face worker..."
& $VisionPython -c "import mediapipe, cv2, numpy; print('vision import: PASS'); print('mediapipe:', mediapipe.__version__); print('opencv:', cv2.__version__); print('numpy:', numpy.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Vision import verification failed." }

$Worker = Join-Path $Root "manuscript_audio_face_worker.py"
if (-not (Test-Path $Worker)) {
    throw "Face worker missing: $Worker"
}
& $VisionPython -m py_compile $Worker
if ($LASTEXITCODE -ne 0) { throw "Face worker syntax check failed." }

Write-Host ""
Write-Host "================================================"
Write-Host " PHASE 3B VISION SETUP COMPLETE"
Write-Host "================================================"
Write-Host ""
Write-Host "Vision Python:"
Write-Host "  $VisionPython"
Write-Host ""
Write-Host "Face tracking / active-speaker mapping will now run in the"
Write-Host "normal pipeline (Phase 1.6)."
