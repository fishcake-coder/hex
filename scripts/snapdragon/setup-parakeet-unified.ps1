[CmdletBinding()]
param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA 'Hex\snapdragon'),
    [string]$SourceDir,
    [string]$ModelDir,
    [string]$RuntimePython,
    [ValidateRange(8, 2400)]
    [int]$MelFrames = 1280,
    [ValidateRange(1, 4)]
    [int]$LayersPerStage = 2,
    [switch]$Optimize,
    [switch]$Streaming
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

function Install-Acceleration {
    param([string]$ModelPath, [string]$RuntimeRoot)

    $destination = Join-Path $ModelPath 'acceleration'
    $frontendWindows = @(
        @(256, 512, $MelFrames) | Where-Object { $_ -ge 160 -and $_ -le $MelFrames } | Sort-Object -Unique
    )
    $assetNames = @('decoder-joint-static.onnx', 'decoder-joint-static_qnn.bin')
    foreach ($frames in $frontendWindows) {
        $assetNames += "frontend-$frames.onnx", "frontend-${frames}_qnn.bin"
    }
    if (Test-Path -LiteralPath $destination) {
        $hashes = Get-Content -LiteralPath (Join-Path $destination 'sha256.json') -Raw | ConvertFrom-Json
        foreach ($name in $assetNames) {
            Assert-Sha256 (Join-Path $destination $name) $hashes.$name
        }
        return
    }

    $python = $RuntimePython
    $toolDir = $null
    if (-not $python) {
        $python = Join-Path $RuntimeRoot 'venv\Scripts\python.exe'
        $toolDir = Join-Path $RuntimeRoot 'unified-converter-python'
    }
    $build = Join-Path $ModelPath ('acceleration-build-' + [guid]::NewGuid().ToString('N'))
    New-Item -Path $build -ItemType Directory | Out-Null
    $previousPythonPath = $env:PYTHONPATH
    $sourceNames = @('decoder_joint.onnx', 'vocab.txt')
    try {
        if ($toolDir) {
            Invoke-Checked (Get-Command uv -ErrorAction Stop).Source @(
                'pip', 'install', '--python', $python, '--target', $toolDir, '--no-cache', 'onnx==1.23.2'
            )
        }
        $env:PYTHONPATH = $toolDir
        foreach ($name in $sourceNames) {
            Copy-Item -LiteralPath (Join-Path $ModelPath $name) -Destination $build
        }
        Invoke-Checked $python @(
            '-B', (Join-Path $PSScriptRoot 'prepare_unified_decoder.py'),
            '--model-dir', $build, '--compile-context', '--token-output', '--frame-count', '4'
        )
        if ($frontendWindows.Count -gt 0) {
            Invoke-Checked $python (@(
                '-B', (Join-Path $PSScriptRoot 'prepare_unified_frontend.py'),
                '--model-dir', $build, '--mel-frames'
            ) + $frontendWindows)
        }
        $hashes = [ordered]@{}
        foreach ($name in $assetNames) {
            $hashes[$name] = (Get-FileHash -LiteralPath (Join-Path $build $name) -Algorithm SHA256).Hash
        }
        $hashes | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $build 'sha256.json') -Encoding UTF8
        foreach ($name in $sourceNames) {
            Remove-Item -LiteralPath (Join-Path $build $name)
        }
        [System.IO.Directory]::Move($build, $destination)
    } finally {
        $env:PYTHONPATH = $previousPythonPath
        if (Test-Path -LiteralPath $build -PathType Container) {
            foreach ($name in $assetNames + $sourceNames + @('sha256.json')) {
                $path = Join-Path $build $name
                if (Test-Path -LiteralPath $path -PathType Leaf) {
                    Remove-Item -LiteralPath $path
                }
            }
            if (-not (Get-ChildItem -LiteralPath $build -Force)) {
                Remove-Item -LiteralPath $build
            }
        }
    }
}

function Install-Streaming {
    param([string]$ModelPath, [string]$SourcePath, [string]$Python, [string]$ToolPath)

    $destination = Join-Path $ModelPath 'streaming'
    if (Test-Path -LiteralPath $destination) {
        $hashes = Get-Content -LiteralPath (Join-Path $destination 'sha256.json') -Raw | ConvertFrom-Json
        $encoder = Get-Content -LiteralPath (Join-Path $destination 'encoder-manifest.json') -Raw | ConvertFrom-Json
        $required = @('encoder-manifest.json')
        foreach ($stage in $encoder.stages) {
            if ($stage -notmatch '^encoder-\d{2}\.onnx$') {
                throw "Invalid streaming encoder stage: $stage"
            }
            $required += $stage, ([IO.Path]::GetFileNameWithoutExtension($stage) + '_qnn.bin')
        }
        foreach ($name in $required) {
            if (-not $hashes.PSObject.Properties[$name]) {
                throw "The streaming digest manifest is missing $name"
            }
        }
        foreach ($entry in $hashes.PSObject.Properties) {
            if ([System.IO.Path]::GetFileName($entry.Name) -ne $entry.Name) {
                throw "Invalid streaming asset name: $($entry.Name)"
            }
            Assert-Sha256 (Join-Path $destination $entry.Name) $entry.Value
        }
        return
    }

    $build = Join-Path $ModelPath ('streaming-build-' + [guid]::NewGuid().ToString('N'))
    $previousPythonPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = $ToolPath
        Invoke-Checked $Python @(
            '-B', (Join-Path $PSScriptRoot 'prepare_unified_streaming.py'),
            '--source-dir', $SourcePath, '--output-dir', $build
        )
        $hashes = [ordered]@{}
        foreach ($file in Get-ChildItem -LiteralPath $build -File) {
            $hashes[$file.Name] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
        }
        $hashes | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $build 'sha256.json') -Encoding UTF8
        [System.IO.Directory]::Move($build, $destination)
    } finally {
        $env:PYTHONPATH = $previousPythonPath
        if (Test-Path -LiteralPath $build -PathType Container) {
            $resolvedBuild = (Resolve-Path -LiteralPath $build).Path
            if ((Split-Path -Parent $resolvedBuild) -eq $ModelPath) {
                Remove-Item -LiteralPath $resolvedBuild -Recurse -Force
            }
        }
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
$hasModel = Test-Path -LiteralPath $verified -PathType Leaf
if ($hasModel) {
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
    if ($Optimize -or $Streaming) { Install-Acceleration $ModelDir $root }
    if (-not $Streaming) {
        Write-Output $ModelDir
        return
    }
}
if (-not $hasModel -and (Test-Path -LiteralPath $ModelDir)) {
    throw "The target directory already exists. Pass a new -ModelDir: $ModelDir"
}

$python = $RuntimePython
$toolDir = $null
if ($python) {
    $python = (Resolve-Path -LiteralPath $python).Path
} else {
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
    $toolDir = Join-Path $root 'unified-converter-python'
}

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
                    '--fail', '--location', '--retry', '3', '--continue-at', '-', '--proto', '=https',
                    '--connect-timeout', '30', '--speed-time', '60', '--speed-limit', '1024',
                    '--output', $pending, "$base/$remoteName"
                )
            }
            try {
                Assert-Sha256 $pending $sha256
            } catch {
                Remove-Item -LiteralPath $pending
                throw
            }
            Move-Item -LiteralPath $pending -Destination $destination
        }
        Assert-Sha256 $destination $sha256
    }
} finally {
    $downloadLock.Dispose()
}

if ($toolDir) {
    Invoke-Checked $uv @(
        'pip', 'install', '--python', $python, '--target', $toolDir, '--no-cache', 'onnx==1.23.2'
    )
}
if ($hasModel) {
    Install-Streaming $ModelDir $source $python $toolDir
    Write-Output $ModelDir
    return
}

$parent = (New-Item -Path (Split-Path -Parent $ModelDir) -ItemType Directory -Force).FullName
$stage = Join-Path $parent ('unified-build-' + [guid]::NewGuid().ToString('N'))
New-Item -Path $stage -ItemType Directory | Out-Null
$previousPythonPath = $env:PYTHONPATH
try {
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
    if ($Optimize -or $Streaming) { Install-Acceleration $stage $root }
    if ($Streaming) { Install-Streaming $stage $source $python $toolDir }
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
