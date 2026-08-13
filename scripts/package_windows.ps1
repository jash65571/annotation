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
# Rust toolchain (rustup default install location) for cargo/tauri build.
if (Test-Path "$env:USERPROFILE\.cargo\bin") { $env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path" }

# 1. Engine sidecar
powershell -File scripts\build_engine_sidecar.ps1
if ($LASTEXITCODE -ne 0) { throw "sidecar build failed" }

# 2. FFmpeg + uv: bundle ONLY the pinned tested runtimes. Every staged
#    executable must hash-match desktop/runtime_versions.json or packaging
#    FAILS (§Phase 6.1-14) — a different local build is never silently bundled.
$pins = Get-Content desktop\runtime_versions.json -Raw | ConvertFrom-Json

function Assert-Hash($file, $expected, $label) {
    $actual = (Get-FileHash $file -Algorithm SHA256).Hash
    if ($actual -ne $expected) {
        throw "$label does not match the pinned tested runtime (expected $expected, got $actual). Run scripts/fetch_runtimes.ps1 to stage the pinned builds."
    }
    Write-Host "$label pin OK"
}

$ffmpegSrc = $env:MANUSCRIPT_FFMPEG_DIR
if (-not $ffmpegSrc) { $ffmpegSrc = "$env:USERPROFILE\tools\ffmpeg\bin" }
if (-not (Test-Path "$ffmpegSrc\ffmpeg.exe")) { throw "ffmpeg.exe not found at $ffmpegSrc" }
Assert-Hash "$ffmpegSrc\ffmpeg.exe" $pins.ffmpeg.ffmpeg_exe_sha256 "ffmpeg.exe"
Assert-Hash "$ffmpegSrc\ffprobe.exe" $pins.ffmpeg.ffprobe_exe_sha256 "ffprobe.exe"
$ffmpegDst = "desktop\src-tauri\ffmpeg\bin"
if (Test-Path "desktop\src-tauri\ffmpeg") { Remove-Item -Recurse -Force "desktop\src-tauri\ffmpeg" }
New-Item -ItemType Directory -Force $ffmpegDst | Out-Null
Copy-Item "$ffmpegSrc\ffmpeg.exe", "$ffmpegSrc\ffprobe.exe" $ffmpegDst
$versionLine = (& "$ffmpegDst\ffmpeg.exe" -version | Select-Object -First 1)
if (-not $versionLine.StartsWith($pins.ffmpeg.version_line_prefix)) {
    throw "ffmpeg version line '$versionLine' does not match pinned '$($pins.ffmpeg.version_line_prefix)'"
}
$versionLine | Tee-Object "desktop\src-tauri\ffmpeg\VERSION.txt"

$uvExe = (Get-Command uv).Source
Assert-Hash $uvExe $pins.uv.uv_exe_sha256 "uv.exe"
$uvDst = "desktop\src-tauri\uv"
if (Test-Path $uvDst) { Remove-Item -Recurse -Force $uvDst }
New-Item -ItemType Directory -Force $uvDst | Out-Null
Copy-Item $uvExe "$uvDst\uv.exe"
& "$uvDst\uv.exe" --version | Tee-Object "$uvDst\VERSION.txt"

# 4. Tauri NSIS bundle (resources overlay only exists for packaged builds so
#    development cargo check/clippy never requires the staged runtimes)
Set-Location desktop
npm run tauri build -- --config src-tauri/tauri.packaged.conf.json
if ($LASTEXITCODE -ne 0) { throw "tauri build failed" }
Get-ChildItem src-tauri\target\release\bundle\nsis\*.exe | ForEach-Object {
    Write-Host "Installer: $($_.FullName) ($([math]::Round($_.Length / 1MB, 1)) MB)"
}
