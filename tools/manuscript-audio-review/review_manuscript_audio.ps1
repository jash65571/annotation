param(
    [Parameter(Position=0)]
    [string]$Video = "VIDEO.mp4"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv-review\Scripts\python.exe"
$Pipeline = Join-Path $Root "manuscript_audio_pipeline.py"

if (-not (Test-Path $Python)) {
    throw "Review Python environment not found: $Python. Run setup_windows.ps1 first."
}

if (-not (Test-Path $Pipeline)) {
    throw "Audio pipeline not found: $Pipeline"
}

if (-not [System.IO.Path]::IsPathRooted($Video)) {
    $Video = Join-Path $Root $Video
}

$Video = [System.IO.Path]::GetFullPath($Video)

if (-not (Test-Path $Video)) {
    throw "Video not found: $Video"
}

Write-Host ""
Write-Host "==================================="
Write-Host " MANUSCRIPT II AUDIO REVIEW"
Write-Host "==================================="
Write-Host "Video: $Video"
Write-Host ""

& $Python $Pipeline $Video

if ($LASTEXITCODE -ne 0) {
    throw "Manuscript audio pipeline failed with exit code $LASTEXITCODE."
}

Write-Host ""
Write-Host "AUDIO REVIEW PIPELINE COMPLETE"
