[CmdletBinding()]
param(
    [string]$Archive,
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA 'Hex\snapdragon')
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Assert-Hash {
    param([string]$Path, [string]$Expected)
    if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -ne $Expected) {
        throw "SHA-256 verification failed for $Path"
    }
}

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Program failed with exit code $LASTEXITCODE."
    }
}

$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
if ($cpu.Architecture -ne 12) {
    throw 'This setup requires Windows ARM64 on a Snapdragon PC.'
}
Get-Command uv -ErrorAction Stop | Out-Null

$root = (New-Item -Path $InstallDir -ItemType Directory -Force).FullName
$modelDir = Join-Path $root 'models'
$venv = Join-Path $root 'venv'
$python = Join-Path $venv 'Scripts\python.exe'
$release = 'v0.66.0'
$asset = 'whisper_large_v3_turbo_quantized-precompiled_qnn_onnx-w8a16-qualcomm_snapdragon_x_elite'
$base = 'https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/whisper_large_v3_turbo_quantized/releases'
$archiveHash = 'C437E3265BF195C2527A8DF331B4643C1C589B500DD4AFE08ABAEF2FADB4FBAC'
$revision = '41f01f3fe87f28c78e2fbf8b568835947dd65ed9'
$openAi = "https://huggingface.co/openai/whisper-large-v3-turbo/resolve/$revision"
$support = @{
    'tokenizer.json' = @("$openAi/tokenizer.json", '297B13372AC43916285644FB9687ADD3CC62EE2A1ADB60DA3DC25CC94C1871FD')
    'generation_config.json' = @("$openAi/generation_config.json", 'CCE11BFE3AAA6AE9E072EA2637CAAEC8795E68D9B67E655A5AF16EE509681A4C')
    'mel_filters.npz' = @(
        'https://raw.githubusercontent.com/openai/whisper/c5d42560760a05584c1c79546a098287e5a771eb/whisper/assets/mel_filters.npz',
        '7450AE70723A5EF9D341E3CEE628C7CB0177F36CE42C44B7ED2BF3325F0F6D4C'
    )
}
$manifest = Join-Path $modelDir 'sha256.json'

if (Test-Path -LiteralPath $manifest) {
    $hashes = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
    foreach ($file in @('encoder.onnx', 'encoder.bin', 'decoder.onnx', 'decoder.bin', 'metadata.json') + @($support.Keys)) {
        Assert-Hash (Join-Path $modelDir $file) $hashes.$file
    }
} elseif (Test-Path -LiteralPath $modelDir) {
    throw "An existing model directory is missing its verification manifest. Move it to a backup location and rerun setup: $modelDir"
} else {
    $downloaded = -not $Archive
    if ($downloaded) {
        $Archive = Join-Path $root "$asset-$release.zip"
        if (-not (Test-Path -LiteralPath $Archive) -or (Get-Item -LiteralPath $Archive).Length -ne 823697025) {
            Invoke-Checked 'curl.exe' @(
                '--fail', '--location', '--retry', '3', '--continue-at', '-',
                '--output', $Archive, "$base/$release/$asset.zip"
            )
        }
    } else {
        $Archive = (Resolve-Path -LiteralPath $Archive).Path
    }
    Assert-Hash $Archive $archiveHash

    $stage = Join-Path $root ('models-' + [guid]::NewGuid().ToString('N'))
    New-Item -Path $stage -ItemType Directory | Out-Null
    try {
        Invoke-Checked 'tar.exe' @('-xf', $Archive, '-C', $stage, '--strip-components', '1')
        foreach ($entry in $support.GetEnumerator()) {
            $path = Join-Path $stage $entry.Key
            Invoke-Checked 'curl.exe' @('--fail', '--location', '--retry', '3', '--output', $path, $entry.Value[0])
            Assert-Hash $path $entry.Value[1]
        }
        $hashes = [ordered]@{}
        foreach ($file in @('encoder.onnx', 'encoder.bin', 'decoder.onnx', 'decoder.bin', 'metadata.json') + @($support.Keys)) {
            $hashes[$file] = (Get-FileHash -LiteralPath (Join-Path $stage $file) -Algorithm SHA256).Hash
        }
        $hashes | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $stage 'sha256.json') -Encoding UTF8
        [System.IO.Directory]::Move($stage, $modelDir)
        if ($downloaded) {
            Remove-Item -LiteralPath $Archive
        }
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
    'numpy==2.5.5', 'onnxruntime==1.32.0', 'onnxruntime-qnn==2.8.0', 'tokenizers==0.23.3'
)

$worker = Join-Path $PSScriptRoot 'whisper_qnn.py'
$response = '{"id":1,"method":"initialize"}' | & $python -u $worker --model-dir $modelDir | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or -not $response -or -not $response.PSObject.Properties['result'] -or $response.result.device -ne 'htp') {
    throw "The Whisper QNN worker failed to initialize: $($response | ConvertTo-Json -Compress)"
}
Write-Output "Whisper QNN is ready. Worker: $worker"
Write-Output "Python: $python"
Write-Output "Models: $modelDir"
