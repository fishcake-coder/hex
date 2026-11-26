#define ProductName "Hex Snapdragon"
#define ProjectUrl "https://github.com/fishcake-coder/hex"
#define ServiceKey "Software\Classes\CLSID\{{BE2D82D0-E30D-4BAF-A7E8-446262604B1F}"
#define ServiceDirectory "{commonpf64}\Hex Live Typing"
#define ServiceLibrary "HexLiveTypingTsf.dll"
#define ServiceTool "HexLiveTypingProfile.exe"
#define Launcher "Hex Snapdragon NPU.exe"
#define VCRuntimeKey "Software\Microsoft\VisualStudio\14.0\VC\Runtimes\arm64"
#define VCRuntimeUrl "https://download.visualstudio.microsoft.com/download/pr/78d061c5-9ef3-42ab-98ed-3309d1095dc4/3C81BCF5B06156858362400D7066F65ECE38E8ACCB5E687FC65BEF5735F31FFC/VC_redist.arm64.exe"
#define VCRuntimeHash "3C81BCF5B06156858362400D7066F65ECE38E8ACCB5E687FC65BEF5735F31FFC"

[Setup]
AppId={{FA317D97-7952-4CCA-9FDD-839ED75C4168}
AppName={#ProductName}
AppPublisher=fishcake-coder
AppPublisherURL={#ProjectUrl}
AppSupportURL={#ProjectUrl}/issues
AppUpdatesURL={#ProjectUrl}/releases
AppVersion={#Version}
ArchitecturesAllowed=arm64
ArchitecturesInstallIn64BitMode=arm64
MinVersion=10.0.26100
PrivilegesRequired=admin
DefaultDirName={commonpf64}\Hex Snapdragon
UsePreviousAppDir=no
DisableDirPage=yes
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\bin\hex.exe
OutputDir={#OutputDir}
OutputBaseFilename=Hex-Snapdragon-Setup-arm64
SetupIconFile={#Repository}\resources\windows\hex.ico
WizardStyle=modern
Compression=lzma2
SolidCompression=yes
CloseApplications=yes
RestartApplications=no
DisableWelcomePage=no

[Files]
Source: "{#BuiltApp}"; DestDir: "{app}\bin"; DestName: "hex.exe"; Flags: ignoreversion
Source: "{#BuiltLauncher}"; DestDir: "{app}\bin"; DestName: "{#Launcher}"; Flags: ignoreversion
Source: "{#Repository}\scripts\snapdragon\install-runtime.ps1"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "{#Repository}\scripts\snapdragon\setup-parakeet-unified.ps1"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "{#Repository}\scripts\snapdragon\prepare_unified_decoder.py"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "{#Repository}\scripts\snapdragon\prepare_unified_encoder.py"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "{#Repository}\scripts\snapdragon\prepare_unified_frontend.py"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "{#Repository}\scripts\snapdragon\prepare_unified_streaming.py"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "{#Repository}\scripts\snapdragon\select_model.py"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "{#Repository}\scripts\snapdragon\unified_rnnt.py"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "{#Repository}\scripts\snapdragon\whisper_qnn.py"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "{#Repository}\scripts\snapdragon\parakeet_tdt.py"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "{#Repository}\packaging\windows\snapdragon-runtime-requirements.txt"; DestDir: "{app}\bin"; DestName: "runtime-requirements.txt"; Flags: ignoreversion
Source: "{#Repository}\LICENSE"; DestDir: "{app}"; DestName: "Hex-LICENSE.txt"; Flags: ignoreversion
Source: "{#Repository}\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Repository}\packaging\windows\snapdragon-model-notices.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#NativeDirectory}\{#ServiceTool}"; DestDir: "{#ServiceDirectory}"; Flags: ignoreversion
Source: "{#NativeDirectory}\{#ServiceLibrary}"; DestDir: "{#ServiceDirectory}"; Flags: ignoreversion restartreplace uninsrestartdelete

[Registry]
Root: HKLM64; Subkey: "{#ServiceKey}"; ValueType: string; ValueName: ""; ValueData: Hex Live Typing; Flags: uninsdeletekeyifempty uninsdeletevalue
Root: HKLM64; Subkey: "{#ServiceKey}\InprocServer32"; ValueType: string; ValueName: ""; ValueData: "{#ServiceDirectory}\{#ServiceLibrary}"; Flags: uninsdeletekey
Root: HKLM64; Subkey: "{#ServiceKey}\InprocServer32"; ValueType: string; ValueName: "ThreadingModel"; ValueData: Apartment

[Icons]
Name: "{commonprograms}\Hex Snapdragon"; Filename: "{app}\bin\{#Launcher}"; WorkingDir: "{app}\bin"; IconFilename: "{app}\bin\hex.exe"
Name: "{commonprograms}\Hex Snapdragon Speech Setup"; Filename: "{app}\bin\{#Launcher}"; Parameters: "--configure"; WorkingDir: "{app}\bin"; IconFilename: "{app}\bin\hex.exe"

[Run]
Filename: "{app}\bin\{#Launcher}"; Description: "Open Hex Snapdragon and set up English dictation"; Flags: postinstall nowait skipifsilent runasoriginaluser

[Code]
function ExpectedServicePath(): String;
begin
  Result := ExpandConstant('{#ServiceDirectory}\{#ServiceLibrary}');
end;

function ServiceRegistrationIsCompatible(): Boolean;
var
  RegisteredPath: String;
begin
  Result := not RegKeyExists(HKLM64, ExpandConstant('{#ServiceKey}'));
  if not Result then
    Result := RegQueryStringValue(HKLM64, ExpandConstant('{#ServiceKey}\InprocServer32'), '', RegisteredPath) and
              (CompareText(RegisteredPath, ExpectedServicePath()) = 0);
end;

function HasVisualCRuntime(): Boolean;
var
  Installed, Major, Minor: Cardinal;
begin
  Result := RegQueryDWordValue(HKLM64, '{#VCRuntimeKey}', 'Installed', Installed) and (Installed = 1) and
            RegQueryDWordValue(HKLM64, '{#VCRuntimeKey}', 'Major', Major) and
            RegQueryDWordValue(HKLM64, '{#VCRuntimeKey}', 'Minor', Minor) and
            ((Major > 14) or ((Major = 14) and (Minor >= 50))) and
            FileExists(ExpandConstant('{sys}\MSVCP140_1.dll'));
end;

function PrepareVisualCRuntime(var NeedsRestart: Boolean): String;
var
  ProcessCode: Integer;
begin
  Result := '';
  if HasVisualCRuntime() then exit;

  try
    DownloadTemporaryFile('{#VCRuntimeUrl}', 'vc_redist.arm64.exe', '{#VCRuntimeHash}', nil);
  except
    Log(GetExceptionMessage());
    Result := 'Setup could not download the Microsoft Visual C++ Runtime. Check your internet connection and try again.';
    exit;
  end;

  ProcessCode := -1;
  Exec(ExpandConstant('{tmp}\vc_redist.arm64.exe'), '/quiet /norestart',
       '', SW_HIDE, ewWaitUntilTerminated, ProcessCode);
  Log('Microsoft Visual C++ Runtime setup returned ' + IntToStr(ProcessCode));
  if (ProcessCode = 3010) or (ProcessCode = 1641) then begin
    NeedsRestart := True;
    Result := 'Restart Windows to finish installing the Microsoft Visual C++ Runtime, then run Hex Snapdragon Setup again.';
  end else if not HasVisualCRuntime() then
    Result := 'Setup could not install the Microsoft Visual C++ Runtime. Run Hex Snapdragon Setup again.';
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  if CompareText(ExpandConstant('{app}'), ExpandConstant('{commonpf64}\Hex Snapdragon')) <> 0 then
    Result := 'Hex Snapdragon must be installed in its Program Files folder.'
  else if not ServiceRegistrationIsCompatible() then
    Result := 'The Windows text-service identifier is in use by another application.'
  else
    Result := PrepareVisualCRuntime(NeedsRestart);
end;

function RunServiceTool(Command: String): Boolean;
var
  ProcessCode: Integer;
begin
  ProcessCode := -1;
  Result := Exec(ExpandConstant('{#ServiceDirectory}\{#ServiceTool}'), Command,
                 ExpandConstant('{#ServiceDirectory}'), SW_HIDE, ewWaitUntilTerminated, ProcessCode) and
            (ProcessCode = 0);
  Log('Hex Live Typing ' + Command + ' returned ' + IntToStr(ProcessCode));
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ProcessCode: Integer;
begin
  if CurStep = ssPostInstall then begin
    if not RunServiceTool('register') then
      RaiseException('Windows could not register Hex Live Typing. Close Hex and run Setup again.');
    ProcessCode := -1;
    ExecAsOriginalUser(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
      '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' +
      ExpandConstant('{app}\bin\install-runtime.ps1') + '" -EnableLiveTyping',
      ExpandConstant('{app}\bin'), SW_HIDE, ewWaitUntilTerminated, ProcessCode);
    Log('Hex Live Typing user activation returned ' + IntToStr(ProcessCode));
  end;
end;

function InitializeUninstall(): Boolean;
begin
  Result := ServiceRegistrationIsCompatible();
  if not Result then
    SuppressibleMsgBox('The Windows text-service identifier is in use by another application.', mbError, MB_OK, IDOK);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if (CurUninstallStep = usUninstall) and RegKeyExists(HKLM64, ExpandConstant('{#ServiceKey}')) then begin
    RunServiceTool('disable');
    if not RunServiceTool('unregister') then
      RaiseException('Windows could not unregister Hex Live Typing. Close Hex and try again.');
  end;
end;
