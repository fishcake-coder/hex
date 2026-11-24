[CmdletBinding()]
param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA 'Hex\snapdragon'),
    [string]$SourceDir,
    [string]$ModelDir,
    [ValidateRange(8, 2400)]
    [int]$MelFrames = 1280,
    [ValidateRange(1, 4)]
    [int]$LayersPerStage = 2
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Program failed with exit code $LASTEXITCODE."
    }
}

function Assert-Sha256 {
    param([string]$Path, [string]$Sha256)
    if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -ne $Sha256) {
        throw "SHA-256 verification failed for $Path"
    }
}

$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
if ($cpu.Architecture -ne 12) {
    throw 'This setup requires Windows ARM64 on a Snapdragon PC.'
}
if ($MelFrames % 8 -ne 0) {
    throw 'MelFrames must be a multiple of eight.'
}
$root = (New-Item -Path $InstallDir -ItemType Directory -Force).FullName
if (-not $SourceDir) { $SourceDir = Join-Path $root 'models-unified-source' }
if (-not $ModelDir) { $ModelDir = Join-Path $root 'models-unified-qnn' }
$ModelDir = [System.IO.Path]::GetFullPath($ModelDir)
$verified = Join-Path $ModelDir 'sha256.json'
if (Test-Path -LiteralPath $verified -PathType Leaf) {
    $hashes = Get-Content -LiteralPath $verified -Raw | ConvertFrom-Json
    $encoder = Get-Content -LiteralPath (Join-Path $ModelDir 'encoder-manifest.json') -Raw | ConvertFrom-Json
    foreach ($name in @('encoder-manifest.json', 'decoder_joint.onnx', 'decoder-joint-static.onnx', 'vocab.txt') + $encoder.stages) {
        if (-not $hashes.PSObject.Properties[$name]) {
            throw "The installed digest manifest is missing $name"
        }
    }
    foreach ($entry in $hashes.PSObject.Properties) {
        if ([System.IO.Path]::GetFileName($entry.Name) -ne $entry.Name) {
            throw "Invalid model asset name: $($entry.Name)"
        }
        Assert-Sha256 (Join-Path $ModelDir $entry.Name) $entry.Value
    }
    if (
        -not $encoder.PSObject.Properties['compiled_contexts'] -or -not $encoder.compiled_contexts -or
        $encoder.mel_frames -ne $MelFrames -or $encoder.stages.Count -ne [Math]::Ceiling(24 / $LayersPerStage)
    ) {
        throw "The installed encoder uses a different build. Pass a new -ModelDir: $ModelDir"
    }
    Write-Output $ModelDir
    return
}
if (Test-Path -LiteralPath $ModelDir) {
    throw "The target directory already exists. Pass a new -ModelDir: $ModelDir"
}

$uv = (Get-Command uv -ErrorAction Stop).Source
$venv = Join-Path $root 'venv'
$python = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    Invoke-Checked $uv @('python', 'install', 'cpython-3.13.14-windows-aarch64')
    Invoke-Checked $uv @('venv', '--python', 'cpython-3.13.14-windows-aarch64', $venv)
}
Invoke-Checked $uv @(
    'pip', 'install', '--python', $python, '--no-cache',
    'numpy==2.5.5', 'onnxruntime==1.32.0', 'onnxruntime-qnn==2.8.0'
)

$revision = '09e9060322d99c5f070010724786e6ee090fd51d'
$base = "https://huggingface.co/bobNight/parakeet-unified-en-0.6b-onnx/resolve/$revision"
$assets = [ordered]@{
    'offline_fp32_encoder.onnx' = @('encoder.onnx', 41901475, '50083DD0B2AF503B87EA8FCF9FD7D7DC15DDA3130E4113017C8B37ED00418BA4')
    'offline_int8_encoder.onnx' = @('encoder.int8.onnx', 42606669, 'C81ADFAB77634E00C1668A221A14F244C5FB3409E7C14EEEBAF6AC963425910F')
    'encoder.int8.onnx.data' = @('encoder.int8.onnx.data', 611491584, '3D54DD04646C15677BD2844A84DF3770B12CC1CE183481F7B6E0DEF31C92114A')
    'decoder_joint.onnx' = @('decoder_joint.onnx', 35779240, '64648C91935EA4819E9C31EA8A20F011D3B1C960BE3861CB8BC74DF0259CD998')
    'vocab.txt' = @('vocab.txt', 4929, 'AE61C9B743CB47C04CE6FB130444210646ED0F40C953C77EE1855185960AF94F')
}
$source = (New-Item -Path $SourceDir -ItemType Directory -Force).FullName
$downloadLock = [System.IO.File]::Open(
    (Join-Path $source '.download.lock'), [System.IO.FileMode]::OpenOrCreate,
    [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None
)
try {
    foreach ($asset in $assets.GetEnumerator()) {
        $remoteName, $size, $sha256 = $asset.Value
        $destination = Join-Path $source $asset.Key
        if (-not (Test-Path -LiteralPath $destination -PathType Leaf)) {
            $pending = "$destination.download"
            if (-not (Test-Path -LiteralPath $pending -PathType Leaf) -or (Get-Item -LiteralPath $pending).Length -ne $size) {
                Invoke-Checked 'curl.exe' @(
                    '--fail', '--location', '--retry', '3', '--continue-at', '-',
                    '--connect-timeout', '30', '--speed-time', '60', '--speed-limit', '1024',
                    '--output', $pending, "$base/$remoteName"
                )
            }
            Assert-Sha256 $pending $sha256
            Move-Item -LiteralPath $pending -Destination $destination
        }
        Assert-Sha256 $destination $sha256
    }
} finally {
    $downloadLock.Dispose()
}

$parent = (New-Item -Path (Split-Path -Parent $ModelDir) -ItemType Directory -Force).FullName
$stage = Join-Path $parent ('unified-build-' + [guid]::NewGuid().ToString('N'))
New-Item -Path $stage -ItemType Directory | Out-Null
$toolDir = Join-Path $root 'unified-converter-python'
$previousPythonPath = $env:PYTHONPATH
try {
    Invoke-Checked $uv @(
        'pip', 'install', '--python', $python, '--target', $toolDir, '--no-cache', 'onnx==1.23.2'
    )
    $env:PYTHONPATH = $toolDir
    Invoke-Checked $python @(
        '-B', (Join-Path $PSScriptRoot 'prepare_unified_encoder.py'),
        '--source-dir', $source, '--output-dir', $stage,
        '--mel-frames', "$MelFrames", '--layers-per-stage', "$LayersPerStage", '--compile-contexts'
    )
    foreach ($filename in @('decoder_joint.onnx', 'vocab.txt')) {
        Copy-Item -LiteralPath (Join-Path $source $filename) -Destination (Join-Path $stage $filename)
    }
    Invoke-Checked $python @(
        '-B', (Join-Path $PSScriptRoot 'prepare_unified_decoder.py'),
        '--model-dir', $stage, '--compile-context'
    )
    $hashes = [ordered]@{}
    foreach ($file in Get-ChildItem -LiteralPath $stage -File) {
        $hashes[$file.Name] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
    }
    $hashes | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $stage 'sha256.json') -Encoding UTF8
    [System.IO.Directory]::Move($stage, $ModelDir)
    Write-Output $ModelDir
} finally {
    $env:PYTHONPATH = $previousPythonPath
    if (Test-Path -LiteralPath $stage) {
        $resolvedStage = (Resolve-Path -LiteralPath $stage).Path
        if ((Split-Path -Parent $resolvedStage) -eq $parent) {
            Remove-Item -LiteralPath $resolvedStage -Recurse -Force
        }
    }
}
