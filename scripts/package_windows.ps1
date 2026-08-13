# Full Windows packaging: engine sidecar + bundled FFmpeg + bundled uv +
# Tauri NSIS installer.
#
# Usage:  powershell -File scripts/package_windows.ps1
# Prereq: scripts/build_engine_sidecar.ps1 output staged (run automatically),
#         a tested FFmpeg build at $env:MANUSCRIPT_FFMPEG_DIR (or ~/tools/ffmpeg/bin),
#         uv.exe available (bundled copy taken from the current uv install).

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# 1. Engine sidecar
powershell -File scripts\build_engine_sidecar.ps1
if ($LASTEXITCODE -ne 0) { throw "sidecar build failed" }

# 2. FFmpeg: bundle the exact tested build (record its version).
$ffmpegSrc = $env:MANUSCRIPT_FFMPEG_DIR
if (-not $ffmpegSrc) { $ffmpegSrc = "$env:USERPROFILE\tools\ffmpeg\bin" }
if (-not (Test-Path "$ffmpegSrc\ffmpeg.exe")) { throw "ffmpeg.exe not found at $ffmpegSrc" }
$ffmpegDst = "desktop\src-tauri\ffmpeg\bin"
if (Test-Path "desktop\src-tauri\ffmpeg") { Remove-Item -Recurse -Force "desktop\src-tauri\ffmpeg" }
New-Item -ItemType Directory -Force $ffmpegDst | Out-Null
Copy-Item "$ffmpegSrc\ffmpeg.exe", "$ffmpegSrc\ffprobe.exe" $ffmpegDst
& "$ffmpegDst\ffmpeg.exe" -version | Select-Object -First 1 | Tee-Object "desktop\src-tauri\ffmpeg\VERSION.txt"

# 3. uv: bundle the pinned uv executable for ASR worker bootstrap.
$uvExe = (Get-Command uv).Source
$uvDst = "desktop\src-tauri\uv"
if (Test-Path $uvDst) { Remove-Item -Recurse -Force $uvDst }
New-Item -ItemType Directory -Force $uvDst | Out-Null
Copy-Item $uvExe "$uvDst\uv.exe"
& "$uvDst\uv.exe" --version | Tee-Object "$uvDst\VERSION.txt"

# 4. Tauri NSIS bundle
Set-Location desktop
npm run tauri build
if ($LASTEXITCODE -ne 0) { throw "tauri build failed" }
Get-ChildItem src-tauri\target\release\bundle\nsis\*.exe | ForEach-Object {
    Write-Host "Installer: $($_.FullName) ($([math]::Round($_.Length / 1MB, 1)) MB)"
}
