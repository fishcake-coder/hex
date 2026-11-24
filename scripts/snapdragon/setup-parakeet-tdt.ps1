[CmdletBinding()]
param(
    [string]$AssetDir,
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA 'Hex\snapdragon')
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

function Assert-Hash {
    param([string]$Path, [string]$Expected)
    if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -ne $Expected) {
        throw "SHA-256 verification failed for $Path"
    }
}

$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
if ($cpu.Architecture -ne 12) {
    throw 'This setup requires Windows ARM64 on a Snapdragon PC.'
}
Get-Command uv -ErrorAction Stop | Out-Null

$root = (New-Item -Path $InstallDir -ItemType Directory -Force).FullName
$modelDir = Join-Path $root 'models-tdt'
$venv = Join-Path $root 'venv'
$python = Join-Path $venv 'Scripts\python.exe'

# NVIDIA Parakeet TDT v3 (CC-BY-4.0), exported to ONNX by istupakov.
# OpenWritr (trsdn) converted and quantized the encoder for Hexagon V73.
$encoder = 'https://huggingface.co/trsdn/parakeet-tdt-0.6b-v3-htp-int8-8s/resolve/761c5e095a082df49048215c4e5dc0fbb8b92a7e'
$onnx = 'https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce'
$assets = [ordered]@{
    'encoder-model.onnx' = @("$encoder/encoder-model.onnx", '3459FC342D8546C03FFA9CCDE631902DA62E0A7649C4456A3E0EEC93951EC759')
    'encoder-model.bin' = @("$encoder/encoder-model.bin", '46F287222A477915040D63EF2F51B8F8887B11808CF02224DBF6A38C34509637')
    'decoder_joint-model.int8.onnx' = @("$onnx/decoder_joint-model.int8.onnx", 'EEA7483EE3D1A30375DAEDC8ED83E3960C91B098812127A0D99D1C8977667A70')
    'nemo128.onnx' = @("$onnx/nemo128.onnx", 'A9FDE1486EBFCC08F328D75AD4610C67835FEA58C73BA57E3209A6F6CF019E9F')
    'vocab.txt' = @("$onnx/vocab.txt", 'D58544679EA4BC6AC563D1F545EB7D474BD6CFA467F0A6E2C1DC1C7D37E3C35D')
}

if (Test-Path -LiteralPath $modelDir) {
    foreach ($asset in $assets.GetEnumerator()) {
        Assert-Hash (Join-Path $modelDir $asset.Key) $asset.Value[1]
    }
} else {
    if ($AssetDir) {
        $AssetDir = (Resolve-Path -LiteralPath $AssetDir).Path
    }

    $stage = Join-Path $root ('models-tdt-' + [guid]::NewGuid().ToString('N'))
    New-Item -Path $stage -ItemType Directory | Out-Null
    try {
        foreach ($asset in $assets.GetEnumerator()) {
            $target = Join-Path $stage $asset.Key
            if ($AssetDir) {
                Copy-Item -LiteralPath (Join-Path $AssetDir $asset.Key) -Destination $target
            } else {
                Invoke-Checked 'curl.exe' @(
                    '--fail', '--location', '--retry', '3',
                    '--speed-limit', '1000', '--speed-time', '90',
                    '--output', $target, $asset.Value[0]
                )
            }
            Assert-Hash $target $asset.Value[1]
        }
        [System.IO.Directory]::Move($stage, $modelDir)
    } finally {
        if (Test-Path -LiteralPath $stage) {
            $resolvedStage = (Resolve-Path -LiteralPath $stage).Path
            if ((Split-Path -Parent $resolvedStage) -eq $root) {
                Remove-Item -LiteralPath $resolvedStage -Recurse -Force
            }
        }
    }
}

if (-not (Test-Path -LiteralPath $python)) {
    Invoke-Checked 'uv' @('python', 'install', 'cpython-3.13.14-windows-aarch64')
    Invoke-Checked 'uv' @('venv', '--python', 'cpython-3.13.14-windows-aarch64', $venv)
}

Invoke-Checked 'uv' @(
    'pip', 'install', '--python', $python, '--no-cache',
    'numpy==2.5.5', 'onnxruntime==1.32.0', 'onnxruntime-qnn==2.8.0'
)

$worker = Join-Path $PSScriptRoot 'parakeet_tdt.py'
$response = '{"id":1,"method":"initialize"}' | & $python -u $worker --model-dir $modelDir | ConvertFrom-Json
if (
    $LASTEXITCODE -ne 0 -or -not $response -or -not $response.PSObject.Properties['result'] -or
    $response.result.device -ne 'htp' -or $response.result.graph_devices.encoder -ne 'htp' -or
    $response.result.cpu_fallback
) {
    throw "The Parakeet QNN worker failed to initialize: $($response | ConvertTo-Json -Compress)"
}
Write-Output "Parakeet TDT is ready. Worker: $worker"
Write-Output "Python: $python"
Write-Output "Models: $modelDir"
