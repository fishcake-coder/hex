[CmdletBinding()]
param([switch]$VerifyOnly, [switch]$EnableLiveTyping)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ($EnableLiveTyping) {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    if ($principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { exit 1 }

    $settingsPath = Join-Path $env:LOCALAPPDATA 'HexSnapdragon\profile\settings.json'
    if (Test-Path -LiteralPath $settingsPath -PathType Leaf) {
        $settings = Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json
        if ($settings.PSObject.Properties['labs'] -and $settings.labs -and
            $settings.labs.PSObject.Properties['live_typing'] -and $settings.labs.live_typing -eq $false) {
            exit 0
        }
    }
    $helper = Join-Path ([Environment]::GetFolderPath('ProgramFiles')) 'Hex Live Typing\HexLiveTypingProfile.exe'
    & $helper enable
    exit $LASTEXITCODE
}

$runtime = Join-Path $env:LOCALAPPDATA 'Hex\snapdragon'
$package = Join-Path $env:LOCALAPPDATA 'HexSnapdragon\runtime-v1'
$models = Join-Path $runtime 'models-unified-qnn'
$existingPython = Join-Path $runtime 'venv\Scripts\python.exe'
$packagedPython = Join-Path $package 'venv\Scripts\python.exe'
$worker = Join-Path $PSScriptRoot 'unified_rnnt.py'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONPATH = $null
$env:HEX_SNAPDRAGON_PROFILE_DIR = $null
$env:HEX_STREAMING = '1'

function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$(Split-Path -Leaf $Program) exited with code $LASTEXITCODE." }
}

function Get-VerifiedArchive([string]$Uri, [string]$Path, [string]$Hash) {
    $curl = Join-Path $env:SystemRoot 'System32\curl.exe'
    Invoke-Checked $curl @('--fail', '--location', '--retry', '3', '--connect-timeout', '30',
        '--speed-time', '60', '--speed-limit', '1024', '--proto', '=https', '--output', $Path, $Uri)
    if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -ne $Hash) {
        throw "SHA-256 verification failed for $(Split-Path -Leaf $Path)."
    }
}

function Test-StreamingModel([string]$Python) {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf) -or
        -not (Test-Path -LiteralPath (Join-Path $models 'streaming\encoder-manifest.json') -PathType Leaf)) {
        return $false
    }
    $start = [Diagnostics.ProcessStartInfo]::new($Python, ('-I -B -u "{0}"' -f $worker))
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardInput = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $start.EnvironmentVariables['HEX_SNAPDRAGON_MODEL_DIR'] = $models
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $start
    try {
        if (-not $process.Start()) { return $false }
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        $process.StandardInput.WriteLine('{"id":1,"method":"initialize","streaming":true}')
        $process.StandardInput.Close()
        if (-not $process.WaitForExit(120000)) {
            $process.Kill()
            return $false
        }
        if ($process.ExitCode -ne 0) {
            Write-Host (($stderr.Result -split '\r?\n') | Where-Object { $_ } | Select-Object -Last 1)
            return $false
        }
        $reply = $stdout.Result | ConvertFrom-Json
        if ($reply.id -ne 1 -or -not $reply.PSObject.Properties['result']) { return $false }
        $model = $reply.result
        return $model.backend -eq 'qnn' -and $model.device -eq 'htp' -and
            $model.provider -eq 'QNNExecutionProvider' -and $model.cpu_fallback -eq $false -and
            $model.streaming -eq $true -and $model.encoder_precompiled -eq $true -and
            $model.graph_devices.encoder -eq 'htp' -and $model.graph_devices.decoder_joint -eq 'htp'
    } catch {
        Write-Host "QNN verification failed. $($_.Exception.Message)"
        return $false
    } finally {
        $process.Dispose()
    }
}

function Install-PythonRuntime([string]$Stage, [string]$Requirements) {
    $uvArchive = Join-Path $Stage 'uv.zip'
    Get-VerifiedArchive 'https://github.com/astral-sh/uv/releases/download/0.13.10/uv-aarch64-pc-windows-msvc.zip' `
        $uvArchive 'F223BCD3FA010174FFEEB885CC1D5334BB81388640C65BAA5B8B42B47017337E'
    $uvDirectory = Join-Path $Stage 'uv'
    Expand-Archive -LiteralPath $uvArchive -DestinationPath $uvDirectory
    $uv = Get-ChildItem -LiteralPath $uvDirectory -File -Filter 'uv.exe' -Recurse | Select-Object -First 1
    if (-not $uv) { throw 'The verified archive is missing uv.exe.' }

    $basePython = Join-Path $package 'python\python.exe'
    if (-not (Test-Path -LiteralPath $basePython -PathType Leaf)) {
        $pythonArchive = Join-Path $Stage 'python.tar.gz'
        Get-VerifiedArchive 'https://github.com/astral-sh/python-build-standalone/releases/download/20261124/cpython-3.13.18%2B20261124-aarch64-pc-windows-msvc-install_only_stripped.tar.gz' `
            $pythonArchive 'CACAD3E38A40D320A9C0EF604EFBA46B3126239EEAA7CD4AD8765AF9382D7E21'
        Invoke-Checked (Join-Path $env:SystemRoot 'System32\tar.exe') @('-xf', $pythonArchive, '-C', $Stage)
        $stagedPython = Join-Path $Stage 'python'
        if (-not (Test-Path -LiteralPath (Join-Path $stagedPython 'python.exe') -PathType Leaf)) {
            throw 'The verified archive is missing python.exe.'
        }
        [System.IO.Directory]::Move($stagedPython, (Join-Path $package 'python'))
    }
    Invoke-Checked $uv.FullName @('--no-config', '--no-python-downloads', 'venv', '--allow-existing',
        '--python', $basePython, (Join-Path $package 'venv'))
    Invoke-Checked $uv.FullName @('--no-config', 'pip', 'install', '--no-cache', '--reinstall', '--only-binary', ':all:',
        '--default-index', 'https://pypi.org/simple', '--require-hashes', '--python', $packagedPython,
        '-r', $Requirements)
}

try {
    $machine = Get-CimInstance Win32_Processor | Select-Object -First 1
    if ($machine.Architecture -ne 12 -or $machine.Manufacturer -notlike 'Qualcomm*') {
        throw 'Hex Snapdragon needs a Qualcomm Snapdragon processor running Windows 11 ARM64.'
    }
    if ($VerifyOnly) {
        if (Test-StreamingModel $packagedPython) { Write-Output 'Packaged QNN streaming is ready.'; exit 0 }
        if (Test-StreamingModel $existingPython) { Write-Output 'Existing QNN streaming is ready.'; exit 0 }
        throw 'No working Snapdragon QNN streaming runtime was found.'
    }

    $null = New-Item -ItemType Directory -Path $package -Force
    $lock = [System.IO.File]::Open((Join-Path $package 'setup.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    try {
        Write-Host 'Checking for a working English speech model...'
        $receipt = 'ready-package'
        if (Test-StreamingModel $packagedPython) {
            Write-Host 'The packaged model is ready.'
        } elseif (Test-StreamingModel $existingPython) {
            Write-Host 'Reusing the existing Snapdragon model and Python runtime.'
            $receipt = 'ready-shared'
        } else {
            $drive = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($package))
            $required = if (Test-Path -LiteralPath (Join-Path $models 'sha256.json')) { 3GB } else { 4GB }
            if ($drive.AvailableFreeSpace -lt $required) {
                throw 'Free up 5 GB on the drive that holds your Windows user profile, then open Hex Snapdragon again.'
            }
            $stage = Join-Path $package ('download-' + [guid]::NewGuid().ToString('N'))
            $null = New-Item -ItemType Directory -Path $stage
            try {
                Write-Host 'Downloading and verifying the ARM64 Python runtime...'
                Install-PythonRuntime $stage (Join-Path $PSScriptRoot 'runtime-requirements.txt')
                Write-Host 'Downloading and compiling the English speech model. This can take several minutes.'
                & (Join-Path $PSScriptRoot 'setup-parakeet-unified.ps1') -InstallDir $runtime `
                    -ModelDir $models -RuntimePython $packagedPython -Streaming | Out-Host
            } finally {
                $resolvedStage = (Resolve-Path -LiteralPath $stage).Path
                if ((Split-Path -Parent $resolvedStage) -eq $package) {
                    Remove-Item -LiteralPath $resolvedStage -Recurse -Force
                }
            }
            if (-not (Test-StreamingModel $packagedPython)) {
                throw 'The configured model did not initialize with Snapdragon NPU streaming.'
            }
        }
        Set-Content -LiteralPath (Join-Path $package 'ready') -Value $receipt -Encoding Ascii -NoNewline
        Write-Host 'The English speech model is ready.'
    } finally {
        $lock.Dispose()
    }
} catch {
    Write-Host "`nHex Snapdragon setup could not finish. $($_.Exception.Message)" -ForegroundColor Red
    if (-not $VerifyOnly) { $null = Read-Host 'Press Enter to close this window' }
    exit 1
}
