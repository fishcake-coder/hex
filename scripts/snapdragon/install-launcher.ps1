[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$root = Join-Path $env:LOCALAPPDATA 'HexSnapdragonBuild'
$bin = Join-Path $root 'bin'
$app = Join-Path $bin 'hex.exe'
$launcher = Join-Path $bin 'Hex Snapdragon NPU.exe'
if (-not (Test-Path -LiteralPath $app -PathType Leaf)) {
    throw 'Build Hex first with scripts\build-snapdragon.ps1.'
}

if (-not $env:VSCMD_VER) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 `
        -property installationPath | Select-Object -First 1
    if (-not $vs) { throw 'Install Visual Studio Build Tools with the ARM64 C++ toolset.' }
    Import-Module (Join-Path $vs 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
    Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=arm64 -host_arch=arm64' | Out-Null
}
if ($env:VSCMD_ARG_TGT_ARCH -ne 'arm64') {
    throw 'Run this script in an ARM64 Developer PowerShell or a regular PowerShell.'
}

$source = Join-Path $PSScriptRoot 'launch-windows.cpp'
$object = Join-Path $root 'snapdragon-launcher.obj'
& cl.exe /nologo /std:c++17 /EHsc /O2 /MT /W4 /WX $source "/Fo:$object" "/Fe:$launcher" user32.lib /link /subsystem:windows
if ($LASTEXITCODE -ne 0) { throw "Could not build the ARM64 launcher (exit $LASTEXITCODE)." }
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'whisper_qnn.py') -Destination $bin

$programs = [Environment]::GetFolderPath('Programs')
$shortcut = Join-Path $programs 'Hex Snapdragon NPU.lnk'
$link = (New-Object -ComObject WScript.Shell).CreateShortcut($shortcut)
$link.TargetPath = $launcher
$link.Arguments = ''
$link.WorkingDirectory = $bin
$link.IconLocation = "$app,0"
$link.Description = 'Open Hex dictation with Qualcomm Hexagon NPU acceleration'
$link.WindowStyle = 1
$link.Save()
Write-Output "Installed $shortcut"
