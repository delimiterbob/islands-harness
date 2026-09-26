# Install the pinned llama.cpp build for the local models (D9).
#
# Downloads the official Windows CUDA release zips for one build tag from GitHub, refuses
# any zip whose SHA-256 differs from the pin below, and extracts them side by side into
#   %LOCALAPPDATA%\islands\llama.cpp\<build>\
# It writes build.json there, which doctor and the snapshot manifest read.
#
# Usage (Windows PowerShell 5.1 or PowerShell 7):
#   powershell -ExecutionPolicy Bypass -File server\llamacpp\install.ps1
#   powershell -ExecutionPolicy Bypass -File server\llamacpp\install.ps1 -Cuda 12.4
#
# CUDA 13.4 is the default because it carries native code for the RTX 5080 (Blackwell).
# It relies on CUDA minor-version compatibility with the installed driver; if llama-server
# fails to start with a CUDA error, update the NVIDIA driver first, and use -Cuda 12.4 only
# as a fallback. Record which one ran; it is part of the stack identity.
#
# The winget package ggml.llamacpp is not used: it lags the release stream and does not pin
# which backend variant is installed.

[CmdletBinding()]
param(
    [ValidateSet('13.4', '12.4')]
    [string]$Cuda = '13.4',
    [string]$Dest = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# ---- The pin. Change it only by editing this block, and record the change in the changelog.
$Build = 'b11191'
$Commit = '4b1a27fa0eb875bbca4f6cfe936e3d65adc685c0'
$Pins = @{
    '13.4' = @{
        "llama-$Build-bin-win-cuda-13.4-x64.zip"     = '1bcec814e2e916eecc5df165d62f90e1cf4df35cce489101e5c761a0cba972a6'
        'cudart-llama-bin-win-cuda-13.4-x64.zip'      = '738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668'
    }
    '12.4' = @{
        "llama-$Build-bin-win-cuda-12.4-x64.zip"     = '6df7f06043aecfddeac6487432db2c94b88f7974ee7952e399f26dfe189f6010'
        'cudart-llama-bin-win-cuda-12.4-x64.zip'      = '8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6'
    }
}
# --------------------------------------------------------------------------------------------

if (-not $Dest) { $Dest = Join-Path $env:LOCALAPPDATA "islands\llama.cpp\$Build" }
$BaseUrl = "https://github.com/ggml-org/llama.cpp/releases/download/$Build"
$Tmp = Join-Path $env:TEMP "islands-llamacpp-$Build"
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null
New-Item -ItemType Directory -Force -Path $Dest | Out-Null

$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest is very slow with the progress bar on 5.1
# curl.exe ships with Windows 10 and later and is far faster than Invoke-WebRequest on 5.1.
# Either way the SHA-256 check below is what gates the install, not the downloader.
$curl = Get-Command curl.exe -ErrorAction SilentlyContinue
$installed = @()
foreach ($name in $Pins[$Cuda].Keys) {
    $expected = $Pins[$Cuda][$name]
    $zip = Join-Path $Tmp $name
    $partial = "$zip.partial"
    if (-not (Test-Path $zip)) {
        Write-Host "Downloading $name"
        if (Test-Path $partial) { Remove-Item $partial -Force }
        if ($curl) {
            & $curl.Source --fail --location --silent --show-error --retry 3 --output $partial "$BaseUrl/$name"
            if ($LASTEXITCODE -ne 0) { throw "curl.exe failed with exit code $LASTEXITCODE for $name" }
        } else {
            Invoke-WebRequest -Uri "$BaseUrl/$name" -OutFile $partial -UseBasicParsing
        }
        Move-Item $partial $zip   # a zip under its final name is always a complete download
    }
    $actual = (Get-FileHash -Algorithm SHA256 -Path $zip).Hash.ToLowerInvariant()
    if ($actual -ne $expected) {
        Remove-Item $zip -Force
        throw "SHA-256 mismatch for $name. Expected $expected, got $actual. Nothing was installed from it."
    }
    Write-Host "Verified $name"
    Expand-Archive -Path $zip -DestinationPath $Dest -Force
    $installed += [ordered]@{ file = $name; sha256 = $expected }
}

$server = Get-ChildItem -Path $Dest -Recurse -Filter 'llama-server.exe' | Select-Object -First 1
if (-not $server) { throw "llama-server.exe not found under $Dest after extraction." }

$info = [ordered]@{
    build      = $Build
    commit     = $Commit
    cuda       = $Cuda
    server_exe = $server.FullName
    zips       = $installed
    installed  = (Get-Date).ToString('s')
}
# UTF-8 without a byte-order mark (Out-File -Encoding utf8 on 5.1 adds one, which JSON readers reject).
[IO.File]::WriteAllText((Join-Path $Dest 'build.json'), ($info | ConvertTo-Json -Depth 4), (New-Object System.Text.UTF8Encoding($false)))

Write-Host ''
& $server.FullName --version
Write-Host ''
Write-Host "Installed llama.cpp $Build (CUDA $Cuda) at $Dest"
Write-Host "Server: $($server.FullName)"
