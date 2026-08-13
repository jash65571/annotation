# Build the packaged engine sidecar with PyInstaller (onedir: reliable with
# OpenCV/NumPy, no onefile unpack cost) and stage it for Tauri bundling.
#
# Usage:  powershell -File scripts/build_engine_sidecar.ps1
# Output: desktop/src-tauri/binaries/manuscript-engine-worker/

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

uv run pyinstaller --noconfirm --clean --onedir --console `
    --name manuscript-engine-worker `
    --distpath build\sidecar-dist `
    --workpath build\sidecar-work `
    --specpath build `
    --paths engine `
    --collect-submodules manuscript_reviewer `
    --add-data "$repo\engine\manuscript_reviewer\rules\manuscript_v1.yaml;manuscript_reviewer\rules" `
    --add-data "$repo\engine\manuscript_reviewer\rules\golden_behavior_v1.yaml;manuscript_reviewer\rules" `
    --add-data "$repo\engine\manuscript_reviewer\audio\asr\workers\fw_env\pyproject.toml;manuscript_reviewer\audio\asr\workers\fw_env" `
    --add-data "$repo\engine\manuscript_reviewer\audio\asr\workers\fw_env\worker.py;manuscript_reviewer\audio\asr\workers\fw_env" `
    --add-data "$repo\engine\manuscript_reviewer\audio\asr\workers\wx_env\pyproject.toml;manuscript_reviewer\audio\asr\workers\wx_env" `
    --add-data "$repo\engine\manuscript_reviewer\audio\asr\workers\wx_env\worker.py;manuscript_reviewer\audio\asr\workers\wx_env" `
    scripts\sidecar_entry.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

# Include the uv lockfiles when present so worker bootstrap is fully pinned.
foreach ($env in @("fw_env", "wx_env")) {
    $lock = "engine\manuscript_reviewer\audio\asr\workers\$env\uv.lock"
    if (Test-Path $lock) {
        Copy-Item $lock "build\sidecar-dist\manuscript-engine-worker\_internal\manuscript_reviewer\audio\asr\workers\$env\uv.lock" -Force
    }
}

$target = "desktop\src-tauri\binaries\manuscript-engine-worker"
if (Test-Path $target) { Remove-Item -Recurse -Force $target }
New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
Copy-Item -Recurse "build\sidecar-dist\manuscript-engine-worker" $target

Write-Host "Sidecar staged at $target"
& "$target\manuscript-engine-worker.exe" --help 2>$null
Write-Host "Sidecar size: $((Get-ChildItem -Recurse $target | Measure-Object Length -Sum).Sum / 1MB) MB"
