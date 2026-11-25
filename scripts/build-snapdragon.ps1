[CmdletBinding()]
param(
    [string]$LlvmBin,
    [ValidateRange(1, 32)][int]$Jobs = 5,
    [switch]$Clippy,
    [switch]$BuildOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$target = 'aarch64-pc-windows-msvc'
$buildRoot = Join-Path $env:LOCALAPPDATA 'HexSnapdragonBuild'
$targetDir = Join-Path $buildRoot 'target'
$bin = Join-Path $buildRoot 'bin'

if ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -ne 'Arm64') {
    throw 'Run this build on Windows ARM64.'
}

if (-not $env:VSCMD_VER) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
        throw 'Visual Studio Build Tools with the ARM64 C++ toolset and Windows SDK are required.'
    }
    $vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 `
        -property installationPath | Select-Object -First 1
    if (-not $vs) {
        throw 'Visual Studio Build Tools with the ARM64 C++ toolset and Windows SDK are required.'
    }
    Import-Module (Join-Path $vs 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
    Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=arm64 -host_arch=arm64' | Out-Null
}

if ($env:VSCMD_ARG_TGT_ARCH -ne 'arm64') {
    throw 'Start from an ARM64 Developer PowerShell or a regular PowerShell.'
}

if (-not $LlvmBin) {
    $portableLlvm = Join-Path $buildRoot 'llvm\bin'
    if (Test-Path -LiteralPath (Join-Path $portableLlvm 'clang-cl.exe') -PathType Leaf) {
        $LlvmBin = $portableLlvm
    }
}
if ($LlvmBin) {
    $env:Path = "$(Resolve-Path -LiteralPath $LlvmBin);$env:Path"
}
foreach ($tool in 'cargo.exe', 'rustc.exe', 'clang-cl.exe', 'cmake.exe', 'ninja.exe', 'rc.exe') {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue -CommandType Application)) {
        throw "$tool is required. For a portable LLVM installation, pass the path to its bin directory with -LlvmBin."
    }
}
if (-not (& rustc -vV | Select-String -SimpleMatch "host: $target")) {
    throw 'Select the native aarch64-pc-windows-msvc Rust toolchain.'
}

$env:CC = 'clang-cl'
$env:CXX = 'clang-cl'
$env:CMAKE_C_COMPILER = 'clang-cl'
$env:CMAKE_CXX_COMPILER = 'clang-cl'
$env:CMAKE_C_COMPILER_TARGET = $target
$env:CMAKE_CXX_COMPILER_TARGET = $target
$env:CMAKE_GENERATOR = 'Ninja'
$env:CL = '/EHsc'
$env:TRANSCRIBE_CMAKE_ARGS = '-DGGML_NATIVE=OFF -DGGML_OPENMP=OFF'
$env:HEX_LOCAL_BUILD = '1'

$commonArgs = @('--locked', '--release', '--target', $target, '--target-dir', $targetDir,
    '--jobs', $Jobs, '--workspace', '--bin', 'hex', '--bin', 'hex-console')
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    if ($Clippy) {
        & cargo clippy @commonArgs -- -D warnings
        if ($LASTEXITCODE -ne 0) { throw "Clippy exited with code $LASTEXITCODE." }
        return
    }

    & cargo build @commonArgs
    if ($LASTEXITCODE -ne 0) { throw "Cargo exited with code $LASTEXITCODE." }

    $release = Join-Path $targetDir "$target\release"
    if ($BuildOnly) {
        Write-Host "Built $release\hex.exe and $release\hex-console.exe."
        return
    }
    New-Item -ItemType Directory -Path $bin -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $release 'hex.exe') -Destination (Join-Path $bin 'hex.exe')
    Copy-Item -LiteralPath (Join-Path $release 'hex-console.exe') -Destination (Join-Path $bin 'hex.com')
    Write-Host "Built $bin\hex.exe (desktop) and $bin\hex.com (CLI)."
} finally {
    Pop-Location
}
