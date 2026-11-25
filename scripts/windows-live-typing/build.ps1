[CmdletBinding()]
param(
    [string]$OutputDir = (Join-Path $env:LOCALAPPDATA 'HexSnapdragonBuild\live-typing')
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$arch = switch ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture) {
    'Arm64' { 'arm64' }
    'X64' { 'amd64' }
    default { throw 'Build the Hex text service on Windows ARM64 or Windows x64.' }
}
if (-not $env:VSCMD_VER) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $component = if ($arch -eq 'arm64') {
        'Microsoft.VisualStudio.Component.VC.Tools.ARM64'
    } else {
        'Microsoft.VisualStudio.Component.VC.Tools.x86.x64'
    }
    $vs = & $vswhere -latest -products '*' -requires $component -property installationPath | Select-Object -First 1
    if (-not $vs) { throw 'Install Visual Studio Build Tools with the native C++ toolset and Windows SDK.' }
    Import-Module (Join-Path $vs 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
    Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments "-arch=$arch -host_arch=$arch" | Out-Null
}
if ($env:VSCMD_ARG_TGT_ARCH -notin @($arch, ($arch -replace '^amd64$', 'x64'))) {
    throw "Use a native $arch developer PowerShell to build the Hex text service."
}

$output = (New-Item -Path $OutputDir -ItemType Directory -Force).FullName
$source = Join-Path $PSScriptRoot 'live-typing-tsf.cpp'
$object = Join-Path $output 'live-typing-tsf.obj'
$dll = Join-Path $output 'HexLiveTypingTsf.dll'
$importLibrary = Join-Path $output 'HexLiveTypingTsf.lib'
$exports = Join-Path $PSScriptRoot 'live-typing-tsf.def'
$flags = @(
    '/nologo', '/LD', '/std:c++17', '/EHsc', '/O2', '/MT', '/W4', '/WX',
    $source, "/Fo:$object", "/Fe:$dll",
    'user32.lib', 'ole32.lib', 'oleaut32.lib', 'uuid.lib', '/link',
    "/IMPLIB:$importLibrary", "/DEF:$exports"
)
& cl.exe @flags | Out-Host
if ($LASTEXITCODE -ne 0) {
    throw "Native compiler exited with code $LASTEXITCODE."
}
$profileSource = Join-Path $PSScriptRoot 'live-typing-profile.cpp'
$profileObject = Join-Path $output 'live-typing-profile.obj'
$profile = Join-Path $output 'HexLiveTypingProfile.exe'
& cl.exe /nologo /std:c++17 /EHsc /O2 /MT /W4 /WX $profileSource "/Fo:$profileObject" "/Fe:$profile" ole32.lib uuid.lib | Out-Host
if ($LASTEXITCODE -ne 0) {
    throw "Windows text profile compiler exited with code $LASTEXITCODE."
}
Write-Output $dll
