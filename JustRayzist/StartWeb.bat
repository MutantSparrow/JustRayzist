@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
chcp 65001 >nul
cls
set "HOST=127.0.0.1"
set "PORT=37717"

set "PYTHON_EXE="
call :try_python "%CD%\runtime\python\python.exe"
if not defined PYTHON_EXE call :try_python "%CD%\.venv\Scripts\python.exe"
if not defined PYTHON_EXE call :try_python "python"

if not defined PYTHON_EXE (
  echo.
  echo No usable Python interpreter with project dependencies was found.
  echo Tried: runtime\python\python.exe, .venv\Scripts\python.exe, and PATH python.
  echo Repair local env with:
  echo   powershell -ExecutionPolicy Bypass -File scripts\bootstrap_env.ps1
  echo Or install into a selected interpreter:
  echo   python -m pip install -e .
  set "EXIT_CODE=1"
  goto :after_run
)

if exist "launch\ascii_logo_blockier.ans" (
  type "launch\ascii_logo_blockier.ans"
  echo.
)

echo.
echo JustRayzist Web Launcher
echo ========================
echo Select runtime profile:
echo   [1] constrained  (12GB VRAM target)
echo   [2] balanced     (16GB VRAM target, default)
echo   [3] high         (24GB VRAM target)
echo.

choice /c 123 /n /m "Choose profile [1-3]: "
set "PROFILE_CHOICE=%ERRORLEVEL%"
if "%PROFILE_CHOICE%"=="3" set "PROFILE=high"
if "%PROFILE_CHOICE%"=="2" set "PROFILE=balanced"
if "%PROFILE_CHOICE%"=="1" set "PROFILE=constrained"

echo.
echo Select model pack:
set /a PACK_COUNT=0
for /d %%D in ("%CD%\models\packs\*") do (
  if exist "%%~fD\modelpack.yaml" (
    set /a PACK_COUNT+=1
    set "PACK_!PACK_COUNT!=%%~nxD"
  )
)

if !PACK_COUNT! EQU 0 (
  echo No model packs found under models\packs.
  echo Add at least one pack directory with modelpack.yaml and retry.
  set "EXIT_CODE=1"
  goto :after_run
)

if !PACK_COUNT! GTR 9 (
  echo Too many model packs: !PACK_COUNT!. Max supported by launcher is 9.
  set "EXIT_CODE=1"
  goto :after_run
)

set "PACK_CHOICES="
for /l %%I in (1,1,!PACK_COUNT!) do (
  set "PACK_CHOICES=!PACK_CHOICES!%%I"
  echo   [%%I] !PACK_%%I!
)
echo.

choice /c !PACK_CHOICES! /n /m "Choose pack [!PACK_CHOICES!]: "
set "PACK_CHOICE_INDEX=%ERRORLEVEL%"

set "PACK=!PACK_%PACK_CHOICE_INDEX%!"
set "JUSTRAYZIST_PACK=!PACK!"
if /I "!PACK!"=="Rayzist_bf16" (
  call :ensure_rayzist_pack_assets
  if errorlevel 1 (
    set "EXIT_CODE=1"
    goto :after_run
  )
)

call :find_listening_pid !PORT! PORT_PID
if defined PORT_PID (
  echo.
  echo Port !PORT! is already in use by PID !PORT_PID!.
  for /f "tokens=1,* delims=," %%A in ('tasklist /FI "PID eq !PORT_PID!" /FO CSV /NH 2^>nul') do (
    set "PID_IMAGE=%%~A"
  )
  if defined PID_IMAGE if /I not "!PID_IMAGE!"=="INFO: No tasks are running which match the specified criteria." (
    echo Process: !PID_IMAGE!
  )
  choice /c YN /n /m "Kill this process and continue? [Y/N]: "
  if errorlevel 2 (
    echo Launch cancelled.
    set "EXIT_CODE=1"
    goto :after_run
  )
  taskkill /PID !PORT_PID! /F >nul 2>&1
  if errorlevel 1 (
    echo Failed to terminate PID !PORT_PID!.
    set "EXIT_CODE=1"
    goto :after_run
  )
  call :wait_for_port_free !PORT! 30
  if errorlevel 1 (
    echo Port !PORT! is still busy after terminating PID !PORT_PID!.
    set "EXIT_CODE=1"
    goto :after_run
  )
)

echo.
echo Starting web server with profile: !PROFILE!
echo Using model pack: !PACK!
echo URL: http://!HOST!:!PORT!/
echo.

"!PYTHON_EXE!" -m app.cli.main serve --host !HOST! --port !PORT! --profile !PROFILE!
set "EXIT_CODE=%ERRORLEVEL%"

:after_run
if not "%EXIT_CODE%"=="0" (
  echo.
  echo Server exited with code %EXIT_CODE%.
  pause
)

endlocal
goto :eof

:try_python
set "CANDIDATE=%~1"
if /I "%CANDIDATE%"=="python" goto :check_candidate
if not exist "%CANDIDATE%" goto :eof

:check_candidate
"%CANDIDATE%" -c "import typer,fastapi,uvicorn,PIL,torch" >nul 2>&1
if errorlevel 1 goto :eof
set "PYTHON_EXE=%CANDIDATE%"
goto :eof

:ensure_rayzist_pack_assets
set "PACK_ROOT=%CD%\models\packs\Rayzist_bf16"
set "NEEDED_TRANSFORMER=%PACK_ROOT%\weights\Rayzist.v1.0.safetensors"
set "NEEDED_VAE=%PACK_ROOT%\weights\ultrafluxVAEImproved_v10.safetensors"
set "NEEDED_ENCODER=%PACK_ROOT%\config\text_encoder\model.safetensors"
set "MISSING_ASSETS=0"

if not exist "!NEEDED_TRANSFORMER!" set "MISSING_ASSETS=1"
if not exist "!NEEDED_VAE!" set "MISSING_ASSETS=1"
if not exist "!NEEDED_ENCODER!" set "MISSING_ASSETS=1"

if !MISSING_ASSETS! EQU 0 exit /b 0

echo.
echo Missing default model assets for pack Rayzist_bf16.
echo Attempting download from Hugging Face...
call :download_asset "https://huggingface.co/MutantSparrow/Ray/resolve/main/Z-IMAGE-TURBO/Rayzist.v1.0.safetensors" "!NEEDED_TRANSFORMER!"
if errorlevel 1 exit /b 1
call :download_asset "https://huggingface.co/Owen777/UltraFlux-v1/resolve/main/vae/diffusion_pytorch_model.safetensors" "!NEEDED_VAE!"
if errorlevel 1 exit /b 1
call :download_asset "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" "!NEEDED_ENCODER!"
if errorlevel 1 exit /b 1

if not exist "!NEEDED_TRANSFORMER!" (
  echo Missing file after download: !NEEDED_TRANSFORMER!
  exit /b 1
)
if not exist "!NEEDED_VAE!" (
  echo Missing file after download: !NEEDED_VAE!
  exit /b 1
)
if not exist "!NEEDED_ENCODER!" (
  echo Missing file after download: !NEEDED_ENCODER!
  exit /b 1
)

echo Model assets ready.
exit /b 0

:download_asset
set "ASSET_URL=%~1"
set "ASSET_DEST=%~2"
for %%I in ("%ASSET_DEST%") do set "ASSET_DIR=%%~dpI"
if not exist "%ASSET_DIR%" mkdir "%ASSET_DIR%" >nul 2>&1
set "ASSET_TMP=%ASSET_DEST%.part"
if exist "%ASSET_TMP%" del /f /q "%ASSET_TMP%" >nul 2>&1

echo Downloading:
echo   %ASSET_URL%
echo To:
echo   %ASSET_DEST%

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; $wc=New-Object Net.WebClient; $wc.DownloadProgressChanged += { param($s,$e) if($e.ProgressPercentage -ge 0){ Write-Progress -Activity 'Downloading model asset' -Status ($e.ProgressPercentage.ToString() + '%%') -PercentComplete $e.ProgressPercentage } }; $wc.DownloadFile('%ASSET_URL%','%ASSET_TMP%'); Write-Progress -Activity 'Downloading model asset' -Completed"
if errorlevel 1 (
  echo Download failed for %ASSET_URL%
  if exist "%ASSET_TMP%" del /f /q "%ASSET_TMP%" >nul 2>&1
  exit /b 1
)
move /y "%ASSET_TMP%" "%ASSET_DEST%" >nul
if errorlevel 1 (
  echo Failed to finalize downloaded file: %ASSET_DEST%
  if exist "%ASSET_TMP%" del /f /q "%ASSET_TMP%" >nul 2>&1
  exit /b 1
)
exit /b 0

:find_listening_pid
set "%~2="
for /f "tokens=5" %%P in ('netstat -ano -p TCP ^| findstr /R /C:":%~1 .*LISTENING"') do (
  set "%~2=%%P"
  goto :eof
)
goto :eof

:wait_for_port_free
set "WAIT_PORT=%~1"
set "WAIT_TRIES=%~2"
for /l %%T in (1,1,!WAIT_TRIES!) do (
  call :find_listening_pid !WAIT_PORT! WAIT_PID
  if not defined WAIT_PID goto :eof
  >nul ping 127.0.0.1 -n 2
)
exit /b 1
