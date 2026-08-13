# Fetch the PINNED packaged runtimes (FFmpeg + uv) from the URLs in
# desktop/runtime_versions.json and stage them for packaging, verifying every
# executable hash. Used by CI; local packaging may instead verify an already
# present install (scripts/package_windows.ps1) against the same pins.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$pins = Get-Content desktop\runtime_versions.json -Raw | ConvertFrom-Json

function Assert-Hash($file, $expected, $label) {
    $actual = (Get-FileHash $file -Algorithm SHA256).Hash
    if ($actual -ne $expected) {
        throw "$label hash mismatch: expected $expected got $actual"
    }
    Write-Host "$label hash OK"
}

# FFmpeg
$ffZip = "$env:TEMP\mr-pinned-ffmpeg.zip"
Invoke-WebRequest -Uri $pins.ffmpeg.download_url -OutFile $ffZip -UseBasicParsing
$ffTmp = "$env:TEMP\mr-pinned-ffmpeg"
if (Test-Path $ffTmp) { Remove-Item -Recurse -Force $ffTmp }
Expand-Archive $ffZip $ffTmp
$ffmpegExe = Get-ChildItem $ffTmp -Recurse -Filter ffmpeg.exe | Select-Object -First 1
$ffprobeExe = Get-ChildItem $ffTmp -Recurse -Filter ffprobe.exe | Select-Object -First 1
Assert-Hash $ffmpegExe.FullName $pins.ffmpeg.ffmpeg_exe_sha256 "ffmpeg.exe"
Assert-Hash $ffprobeExe.FullName $pins.ffmpeg.ffprobe_exe_sha256 "ffprobe.exe"
$ffDst = "desktop\src-tauri\ffmpeg\bin"
if (Test-Path "desktop\src-tauri\ffmpeg") { Remove-Item -Recurse -Force "desktop\src-tauri\ffmpeg" }
New-Item -ItemType Directory -Force $ffDst | Out-Null
Copy-Item $ffmpegExe.FullName, $ffprobeExe.FullName $ffDst
& "$ffDst\ffmpeg.exe" -version | Select-Object -First 1 | Tee-Object "desktop\src-tauri\ffmpeg\VERSION.txt"

# uv
$uvZip = "$env:TEMP\mr-pinned-uv.zip"
Invoke-WebRequest -Uri $pins.uv.download_url -OutFile $uvZip -UseBasicParsing
$uvTmp = "$env:TEMP\mr-pinned-uv"
if (Test-Path $uvTmp) { Remove-Item -Recurse -Force $uvTmp }
Expand-Archive $uvZip $uvTmp
$uvExe = Get-ChildItem $uvTmp -Recurse -Filter uv.exe | Select-Object -First 1
Assert-Hash $uvExe.FullName $pins.uv.uv_exe_sha256 "uv.exe"
$uvDst = "desktop\src-tauri\uv"
if (Test-Path $uvDst) { Remove-Item -Recurse -Force $uvDst }
New-Item -ItemType Directory -Force $uvDst | Out-Null
Copy-Item $uvExe.FullName "$uvDst\uv.exe"
& "$uvDst\uv.exe" --version | Tee-Object "$uvDst\VERSION.txt"

Write-Host "Pinned runtimes staged."
