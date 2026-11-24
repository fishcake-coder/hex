@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "hexBin=%LOCALAPPDATA%\HexSnapdragonBuild\bin"
set "hexRuntime=%LOCALAPPDATA%\Hex\snapdragon"
set "hexSelector=%~dp0scripts\snapdragon\select_model.py"
if not defined HEX_APPLICATION_SUPPORT_DIR set "HEX_APPLICATION_SUPPORT_DIR=%LOCALAPPDATA%\HexSnapdragonBuild\profile"
if not defined HEX_SNAPDRAGON_PYTHON set "HEX_SNAPDRAGON_PYTHON=%hexRuntime%\venv\Scripts\python.exe"

if not exist "%HEX_SNAPDRAGON_PYTHON%" (
    echo The Snapdragon Python environment is missing at "%HEX_SNAPDRAGON_PYTHON%". 1>&2
    goto failed
)
if /I "%~1"=="snapdragon-model" goto snapdragon-model
if not exist "%hexBin%\hex.exe" (
    echo Build Hex first with scripts\build-snapdragon.ps1. 1>&2
    goto failed
)
if "%~1"=="" if exist "%hexBin%\Hex Snapdragon NPU.exe" goto shortcut

if defined HEX_SNAPDRAGON_WORKER if not defined HEX_SNAPDRAGON_MODEL_DIR set "HEX_SNAPDRAGON_MODEL_DIR=%hexRuntime%\models"
if not defined HEX_SNAPDRAGON_WORKER set "HEX_SNAPDRAGON_WORKER=%hexSelector%"
if not exist "%HEX_SNAPDRAGON_WORKER%" (
    echo The Snapdragon transcription worker is missing at "%HEX_SNAPDRAGON_WORKER%". 1>&2
    goto failed
)
if not exist "%HEX_APPLICATION_SUPPORT_DIR%\." mkdir "%HEX_APPLICATION_SUPPORT_DIR%"
if not exist "%HEX_APPLICATION_SUPPORT_DIR%\." goto failed

if "%~1"=="" goto desktop

"%hexBin%\hex.com" %*
exit /b %errorlevel%

:desktop
start "Hex Snapdragon" "%hexBin%\hex.exe"
exit /b %errorlevel%

:shortcut
start "" "%hexBin%\Hex Snapdragon NPU.exe"
exit /b %errorlevel%

:snapdragon-model
"%HEX_SNAPDRAGON_PYTHON%" -u "%hexSelector%" %*
exit /b %errorlevel%

:failed
if "%~1"=="" pause
exit /b 1
