[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet('Register', 'Update', 'Unregister', 'Enable', 'Disable', 'Status')]
    [string]$Action,
    [string]$BuildDir = (Join-Path $env:LOCALAPPDATA 'HexSnapdragonBuild\live-typing')
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$directory = Join-Path ([Environment]::GetFolderPath('ProgramFiles')) 'Hex Live Typing'
$binary = Join-Path $directory 'HexLiveTypingTsf.dll'
$tool = Join-Path $directory 'HexLiveTypingProfile.exe'
$registration = 'Registry::HKEY_LOCAL_MACHINE\SOFTWARE\Classes\CLSID\{BE2D82D0-E30D-4BAF-A7E8-446262604B1F}'
$server = Join-Path $registration 'InprocServer32'

if ($Action -in @('Register', 'Update', 'Unregister')) {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'Windows requires an elevated PowerShell for text service registration.'
    }
}

if (Test-Path -LiteralPath $registration) {
    if (-not (Test-Path -LiteralPath $server) -or (Get-Item -LiteralPath $server).GetValue('') -ne $binary) {
        throw 'The Hex text service identifier is already registered to a different package.'
    }
}

if ($Action -eq 'Register') {
    if (Test-Path -LiteralPath $registration) {
        Write-Output 'Hex Live Typing is already registered.'
        return
    }
    $files = @('HexLiveTypingTsf.dll', 'HexLiveTypingProfile.exe')
    foreach ($name in $files) {
        if (-not (Test-Path -LiteralPath (Join-Path $BuildDir $name) -PathType Leaf)) {
            throw 'Build the native text service first with build.ps1.'
        }
    }
    $null = New-Item -ItemType Directory -Path $directory -Force
    foreach ($name in $files) {
        Copy-Item -LiteralPath (Join-Path $BuildDir $name) -Destination $directory
    }
    try {
        $null = New-Item -Path $registration -Value 'Hex Live Typing'
        $null = New-Item -Path $server -Value $binary
        $null = New-ItemProperty -LiteralPath $server -Name 'ThreadingModel' -Value 'Apartment'
        & $tool register
        if ($LASTEXITCODE -ne 0) { throw 'Windows did not register the Hex speech profile.' }
    } catch {
        if (Test-Path -LiteralPath $server) { Remove-Item -LiteralPath $server }
        if (Test-Path -LiteralPath $registration) { Remove-Item -LiteralPath $registration }
        throw
    }
    Write-Output 'Hex Live Typing is registered. Use Enable in a regular PowerShell.'
    return
}

if ($Action -eq 'Update') {
    if (-not (Test-Path -LiteralPath $registration)) {
        throw 'Register the Hex Live Typing text service first.'
    }
    $source = Join-Path $BuildDir 'HexLiveTypingTsf.dll'
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw 'Build the native text service first with build.ps1.'
    }
    if ((Get-FileHash -LiteralPath $source).Hash -eq (Get-FileHash -LiteralPath $binary).Hash) {
        Write-Output 'Hex Live Typing already has this text service.'
        return
    }
    $update = [Guid]::NewGuid().ToString('N')
    $incoming = Join-Path $directory "HexLiveTypingTsf.$update.new.dll"
    $backup = Join-Path $directory "HexLiveTypingTsf.$update.previous.dll"
    try {
        Copy-Item -LiteralPath $source -Destination $incoming
        [IO.File]::Replace($incoming, $binary, $backup)
    } finally {
        if (Test-Path -LiteralPath $incoming) { Remove-Item -LiteralPath $incoming }
    }
    Write-Output "Hex Live Typing was updated. The previous DLL is at $backup"
    Write-Output 'Reopen apps that already loaded the text service.'
    return
}

if ($Action -eq 'Unregister' -and -not (Test-Path -LiteralPath $registration)) {
    Write-Output 'Hex Live Typing is unregistered.'
    return
}
if (-not (Test-Path -LiteralPath $tool -PathType Leaf)) {
    throw 'Register the Hex Live Typing text service first.'
}
& $tool $Action.ToLowerInvariant()
if ($LASTEXITCODE -ne 0) { throw "Hex Live Typing $Action did not complete." }
if ($Action -eq 'Unregister') {
    Remove-Item -LiteralPath $server
    Remove-Item -LiteralPath $registration
}
