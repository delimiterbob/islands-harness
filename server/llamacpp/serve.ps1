# Start llama-server for one local model with the exact flags the snapshot config expects.
#
# Usage, from the harness folder:
#   powershell -ExecutionPolicy Bypass -File server\llamacpp\serve.ps1 -Model gpt-oss-20b
#   powershell -ExecutionPolicy Bypass -File server\llamacpp\serve.ps1 -Model qwen3-14b
#   add -FlashAttn off only for the determinism gate's fallback variant (flash_attn_off)
#
# Or through the harness, which also waits for /health and logs to results\logs:
#   uv run islands serve start gpt-oss-20b     and     uv run islands serve stop
#
# One model at a time: the 16 GB GPU holds one. Stop the server with Ctrl+C before starting
# the other. Every flag below mirrors a field in configs/snapshot-1.yaml `expect`; doctor
# reads GET /props from the running server and blocks a run on any difference.
#
# Why these flags (ARCHITECTURE.md D9 and D12):
#   -np 1               one slot, so batch composition can never vary between requests
#   --no-cache-prompt   llama.cpp documents prompt caching as a source of nondeterminism
#   -b / -ub            fixed prompt-processing chunk sizes
#   -ctk / -ctv f16     unquantized KV cache (quantized KV degrades tool calling)
#   -fa on              flash attention fixed rather than 'auto'
#   --host 127.0.0.1    never reachable from the network
#   --no-webui          no browser UI
# Sampling is not set here: each request carries its own sampler chain and values.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('gpt-oss-20b', 'qwen3-14b', 'qwen3-4b')]
    [string]$Model,
    [string]$Build = 'b11191',
    [ValidateSet('on', 'off')]
    [string]$FlashAttn = 'on',
    [switch]$SkipHashCheck
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Harness = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Models = @{
    'gpt-oss-20b' = @{ file = 'gpt-oss-20b-MXFP4.gguf'; lock = 'models\gpt-oss-20b.lock.json'; port = 8081 }
    'qwen3-14b'   = @{ file = 'Qwen3-14B-Q5_K_M.gguf'; lock = 'models\qwen3-14b.lock.json'; port = 8082 }
    'qwen3-4b'    = @{ file = 'Qwen3-4B-Q5_K_M.gguf'; lock = 'models\qwen3-4b.lock.json'; port = 8083 }
}
$m = $Models[$Model]

$buildDir = Join-Path $env:LOCALAPPDATA "islands\llama.cpp\$Build"
$buildJson = Join-Path $buildDir 'build.json'
if (-not (Test-Path $buildJson)) { throw "llama.cpp $Build is not installed. Run server\llamacpp\install.ps1 first." }
$serverExe = (Get-Content $buildJson -Raw | ConvertFrom-Json).server_exe

$weights = Join-Path $Harness "models\weights\$($m.file)"
if (-not (Test-Path $weights)) { throw "Weights not found: $weights. See README, step 4." }

if (-not $SkipHashCheck) {
    $lock = Get-Content (Join-Path $Harness $m.lock) -Raw | ConvertFrom-Json
    $expected = $lock.files[0].sha256
    Write-Host "Hashing $($m.file) (about half a minute)..."
    $actual = (Get-FileHash -Algorithm SHA256 -Path $weights).Hash.ToLowerInvariant()
    if ($actual -ne $expected) { throw "Weights SHA-256 mismatch for $($m.file). Expected $expected, got $actual." }
    Write-Host 'Weights verified.'
}

$env:CUDA_VISIBLE_DEVICES = '0'
$flags = @(
    '-m', $weights,
    '--alias', $Model,
    '--host', '127.0.0.1',
    '--port', "$($m.port)",
    '-c', '16384',
    '-np', '1',
    '--no-cache-prompt',
    '-ngl', 'all',
    '-fa', $FlashAttn,
    '-ctk', 'f16',
    '-ctv', 'f16',
    '-b', '2048',
    '-ub', '512',
    '--jinja',
    '--reasoning-format', 'auto',
    '--no-webui'
)
Write-Host "Starting llama-server $Build for $Model on http://127.0.0.1:$($m.port)"
Write-Host ($serverExe + ' ' + ($flags -join ' '))
& $serverExe @flags
