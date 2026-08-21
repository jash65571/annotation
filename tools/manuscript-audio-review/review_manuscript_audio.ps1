param(
    [Parameter(Position=0)]
    [string]$Video = "VIDEO.mp4",

    [Parameter(Position=1)]
    [string]$Seed = ""
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv-review\Scripts\python.exe"
$Pipeline = Join-Path $Root "manuscript_audio_pipeline.py"

# Load local secrets (.env) so HF_TOKEN reaches the WhisperX diarization
# child even when stages are invoked through the launcher.
$EnvFile = Join-Path $Root ".env"
if ((Test-Path $EnvFile) -and -not $env:HF_TOKEN) {
    Get-Content $EnvFile | ForEach-Object {
        $Line = $_.Trim()
        if ($Line -and -not $Line.StartsWith("#") -and $Line.Contains("=")) {
            $Key, $Value = $Line.Split("=", 2)
            $Key = $Key.Trim()
            $Value = $Value.Trim().Trim('"').Trim("'")
            if ($Key -and $Value) {
                Set-Item -Path "Env:$Key" -Value $Value
            }
        }
    }
}

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

if ($Seed -ne "") {
    if (-not [System.IO.Path]::IsPathRooted($Seed)) {
        $Seed = Join-Path $Root $Seed
    }

    $Seed = [System.IO.Path]::GetFullPath($Seed)

    if (-not (Test-Path $Seed)) {
        throw "Task seed not found: $Seed"
    }
}

Write-Host ""
Write-Host "==================================="
Write-Host " MANUSCRIPT II AUDIO REVIEW"
Write-Host "==================================="
Write-Host "Video: $Video"

if ($Seed -ne "") {
    Write-Host "Seed:  $Seed (locked task auto-ingested before analysis)"
}

Write-Host ""

$PipelineArgs = @($Video)

if ($Seed -ne "") {
    $PipelineArgs += $Seed
}

& $Python $Pipeline @PipelineArgs

if ($LASTEXITCODE -ne 0) {
    throw "Manuscript audio pipeline failed with exit code $LASTEXITCODE."
}

Write-Host ""
Write-Host "AUDIO REVIEW PIPELINE COMPLETE"
