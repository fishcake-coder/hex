[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidatePattern('^\d+\.\d+\.\d+$')][string]$Version,
    [string]$Executable,
    [string]$Iscc,
    [switch]$Development,
    [string]$OutDir = (Join-Path $env:LOCALAPPDATA 'HexSnapdragonBuild\release')
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repo = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$buildRoot = Join-Path $env:LOCALAPPDATA 'HexSnapdragonBuild'
if (-not $Iscc) {
    $candidates = @(
        (Join-Path $buildRoot 'packaging-tools\Inno Setup 6\ISCC.exe'),
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
        (Get-Command ISCC.exe -ErrorAction SilentlyContinue | ForEach-Object Source)
    )
    $Iscc = $candidates | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } | Select-Object -First 1
}
if (-not $Iscc) { throw 'Install Inno Setup 6.7.4 or pass the path to its ISCC.exe with -Iscc.' }
$Iscc = (Resolve-Path -LiteralPath $Iscc).Path
if (-not $Executable) {
    & (Join-Path $PSScriptRoot 'build-snapdragon.ps1') -BuildOnly -PublicBuild:(-not $Development)
    $Executable = Join-Path $buildRoot 'target\aarch64-pc-windows-msvc\release\hex.exe'
}
$Executable = (Resolve-Path -LiteralPath $Executable).Path

function Assert-Arm64([string]$Path) {
    $reader = [IO.BinaryReader]::new([IO.File]::OpenRead($Path))
    try {
        if ($reader.ReadUInt16() -ne 0x5A4D) { throw "Invalid Windows binary: $Path" }
        $reader.BaseStream.Position = 0x3C
        $reader.BaseStream.Position = $reader.ReadUInt32()
        if ($reader.ReadUInt32() -ne 0x4550 -or $reader.ReadUInt16() -ne 0xAA64) {
            throw "Expected a Windows ARM64 binary: $Path"
        }
    } finally {
        $reader.Dispose()
    }
}

function Assert-NoBuildPaths([string]$Path) {
    if ($Development) { return }
    $paths = foreach ($privatePath in $env:USERPROFILE, $repo, $buildRoot) {
        $privatePath
        $privatePath.Replace('\', '/')
        $privatePath.Replace('\', '\\')
    }
    $bytes = [IO.File]::ReadAllBytes($Path)
    foreach ($encoding in [Text.Encoding]::UTF8, [Text.Encoding]::Unicode) {
        $offsets = if ($encoding -eq [Text.Encoding]::UTF8) { 0 } else { 0, 1 }
        foreach ($offset in $offsets) {
            $text = $encoding.GetString($bytes, $offset, $bytes.Length - $offset)
            foreach ($privatePath in $paths) {
                if ($text.IndexOf($privatePath, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
                    throw "$(Split-Path -Leaf $Path) contains a private build path. Rebuild with -PublicBuild."
                }
            }
        }
    }
}

Assert-Arm64 $Executable
Assert-NoBuildPaths $Executable
$out = (New-Item -ItemType Directory -Force -Path $OutDir).FullName
$native = Join-Path $buildRoot 'package-native'
$null = New-Item -ItemType Directory -Path $native -Force
& (Join-Path $PSScriptRoot 'windows-live-typing\build.ps1') -OutputDir $native | Out-Host
$launcher = Join-Path $native 'Hex Snapdragon NPU.exe'
$launcherSource = Join-Path $PSScriptRoot 'snapdragon\launch-windows.cpp'
$launcherObject = Join-Path $native 'snapdragon-launcher.obj'
& cl.exe /nologo /std:c++17 /EHsc /O2 /MT /W4 /WX /DHEX_PACKAGED_SNAPDRAGON `
    $launcherSource "/Fo:$launcherObject" "/Fe:$launcher" user32.lib advapi32.lib /link /subsystem:windows | Out-Host
if ($LASTEXITCODE -ne 0) { throw "Could not build the ARM64 launcher (exit $LASTEXITCODE)." }
foreach ($binary in $launcher, (Join-Path $native 'HexLiveTypingTsf.dll'), (Join-Path $native 'HexLiveTypingProfile.exe')) {
    Assert-Arm64 $binary
    Assert-NoBuildPaths $binary
}

$recipe = Join-Path $repo 'packaging\windows\hex-snapdragon.iss'
& $Iscc '/Qp' "/DVersion=$Version" "/DRepository=$repo" "/DOutputDir=$out" "/DBuiltApp=$Executable" `
    "/DBuiltLauncher=$launcher" "/DNativeDirectory=$native" $recipe | Out-Host
if ($LASTEXITCODE -ne 0) { throw "Inno Setup exited with code $LASTEXITCODE." }
$installer = Join-Path $out 'Hex-Snapdragon-Setup-arm64.exe'
Assert-NoBuildPaths $installer
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $installer).Hash.ToLowerInvariant()
Set-Content -LiteralPath "$installer.sha256" -Encoding ascii -Value "$hash  $([IO.Path]::GetFileName($installer))"
Write-Output $installer
