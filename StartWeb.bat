@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
chcp 65001 >nul
cls

set "HOST=127.0.0.1"
set "DISPLAY_HOST=127.0.0.1"
set "NETWORK_MODE=local"
set "PORT=37717"
set "JUSTRAYZIST_ROOT=%CD%"
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"

echo.
echo JustRayzist Launcher Bootstrap
echo ==============================
echo Detecting runtime...
echo.

set "PYTHON_EXE="
echo Checking local Python environment. This can take a while on slower hardware...
echo.
if defined JUSTRAYZIST_PYTHON call :try_source_python "%JUSTRAYZIST_PYTHON%" "JUSTRAYZIST_PYTHON"
if not defined PYTHON_EXE call :try_source_python "%CD%\.venv\Scripts\python.exe" ".venv\\Scripts\\python.exe"
if not defined PYTHON_EXE call :try_source_python "%CD%\.venv\python.exe" ".venv\\python.exe"
if not defined PYTHON_EXE call :try_source_python "python" "PATH python"
if not defined PYTHON_EXE call :try_python_launcher_paths
if not defined PYTHON_EXE (
  echo.
  echo No usable Python runtime found.
  echo Checked source interpreter candidates - requires launcher/server dependencies and core model packages:
  if defined JUSTRAYZIST_PYTHON echo   JUSTRAYZIST_PYTHON=!JUSTRAYZIST_PYTHON!
  echo   %CD%\.venv\Scripts\python.exe
  echo   %CD%\.venv\python.exe
  echo   PATH python
  echo.
  echo Setup or repair the environment first:
  echo   .\RunMeFirst.bat
  echo.
  echo Manual fallback:
  echo   powershell -ExecutionPolicy Bypass -File scripts\bootstrap_env.ps1
  echo.
  echo The delay before this message came from validating Python startup dependencies.
  set "EXIT_CODE=1"
  goto :after_run
)
echo.
echo Python runtime ready:
echo   !PYTHON_EXE!

if exist "launch\ascii_logo_blockier.ans" (
  type "launch\ascii_logo_blockier.ans"
  echo.
)

echo.
echo JustRayzist Web Launcher
echo ========================
echo.
echo Model pack discovery:
echo Only public, enabled packs are shown here.
set /a PACK_COUNT=0
for /d %%D in ("%CD%\models\packs\*") do (
  if exist "%%~fD\modelpack.yaml" (
    findstr /R /C:"^[ ]*user_visible:[ ]*false" "%%~fD\modelpack.yaml" >nul
    if errorlevel 1 (
      findstr /R /C:"^[ ]*enabled:[ ]*false" "%%~fD\modelpack.yaml" >nul
      if errorlevel 1 (
        set /a PACK_COUNT+=1
        set "PACK_!PACK_COUNT!=%%~nxD"
      )
    )
  )
)

if !PACK_COUNT! EQU 0 (
  echo No public enabled model packs found under models\packs.
  echo Enable at least one public pack and retry.
  set "EXIT_CODE=1"
  goto :after_run
)

if !PACK_COUNT! GTR 9 (
  echo Too many model packs: !PACK_COUNT!. Max supported by launcher is 9.
  set "EXIT_CODE=1"
  goto :after_run
)

if !PACK_COUNT! EQU 1 (
  set "PACK=!PACK_1!"
  echo Auto-selected only available pack: !PACK!
) else (
  echo Select model pack:
  set "PACK_CHOICES="
  for /l %%I in (1,1,!PACK_COUNT!) do (
    set "PACK_CHOICES=!PACK_CHOICES!%%I"
    echo   [%%I] !PACK_%%I!
  )
  echo.
  choice /c !PACK_CHOICES! /n /m "Choose pack [!PACK_CHOICES!]: "
  set "PACK_CHOICE_INDEX=!ERRORLEVEL!"
  call set "PACK=%%PACK_!PACK_CHOICE_INDEX!%%"
)
set "JUSTRAYZIST_PACK=!PACK!"
if /I "!PACK!"=="Rayzist_bf16" (
  call :ensure_default_pack_assets
  if errorlevel 1 (
    set "EXIT_CODE=1"
    goto :after_run
  )
)
if /I "!PACK!"=="Krea2_Turbo" (
  call :ensure_krea2_pack_assets
  if errorlevel 1 (
    set "EXIT_CODE=1"
    goto :after_run
  )
)

echo.
if /I "%JUSTRAYZIST_LISTEN%"=="1" (
  set "HOST=0.0.0.0"
  set "DISPLAY_HOST=[your-LAN-IP]"
  set "NETWORK_MODE=lan"
  echo Network mode forced by JUSTRAYZIST_LISTEN=1: LAN listen ^(0.0.0.0^)
) else (
  echo Select network mode:
  echo   [1] local only  ^(127.0.0.1^)
  echo   [2] LAN listen  ^(0.0.0.0^)
  echo.
  choice /c 12 /n /m "Choose network mode [1-2]: "
  set "NETWORK_CHOICE=!ERRORLEVEL!"
  if "!NETWORK_CHOICE!"=="2" (
    set "HOST=0.0.0.0"
    set "DISPLAY_HOST=[your-LAN-IP]"
    set "NETWORK_MODE=lan"
  ) else (
    set "HOST=127.0.0.1"
    set "DISPLAY_HOST=127.0.0.1"
    set "NETWORK_MODE=local"
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
echo Starting web server with automatic resource-tier detection.
echo Using model pack: !PACK!
echo Bind address: !HOST!:!PORT!
if /I "!NETWORK_MODE!"=="lan" (
  echo Local URL: http://127.0.0.1:!PORT!/
  echo LAN URL:   http://!DISPLAY_HOST!:!PORT!/ ^(replace [your-LAN-IP] with this machine's LAN IP^)
) else (
  echo URL: http://!DISPLAY_HOST!:!PORT!/
)
echo.

"!PYTHON_EXE!" -m app.cli.main serve --host !HOST! --port !PORT!
set "EXIT_CODE=%ERRORLEVEL%"

:after_run
if not "%EXIT_CODE%"=="0" (
  echo.
  echo Server exited with code %EXIT_CODE%.
  pause
)

endlocal
goto :eof

:try_source_python
set "CANDIDATE=%~1"
set "CANDIDATE_LABEL=%~2"
if not defined CANDIDATE_LABEL set "CANDIDATE_LABEL=%~1"
echo [runtime] Checking !CANDIDATE_LABEL!
if /I "%CANDIDATE%"=="python" goto :check_source_candidate
if not exist "%CANDIDATE%" (
  echo [runtime]   not found
  goto :eof
)

:check_source_candidate
"%CANDIDATE%" -c "import importlib.util,sys,typer,fastapi,uvicorn,PIL,safetensors,app.api.main; required=('torch','diffusers','transformers','accelerate'); missing=[name for name in required if importlib.util.find_spec(name) is None]; sys.exit(','.join(missing) if missing else 0)" >nul 2>&1
if errorlevel 1 (
  echo [runtime]   not usable
  goto :eof
)
set "PYTHON_EXE=%CANDIDATE%"
echo [runtime]   usable
goto :eof

:try_python_launcher_paths
for /f "tokens=* delims=" %%L in ('py -0p 2^>nul') do (
  set "PY_LINE=%%L"
  set "PY_LAST="
  for %%P in (!PY_LINE!) do set "PY_LAST=%%P"
  if defined PY_LAST (
    call :try_source_python "!PY_LAST!" "py launcher: !PY_LAST!"
    if defined PYTHON_EXE goto :eof
  )
)
goto :eof

:ensure_default_pack_assets
set "PACK_ROOT=%CD%\models\packs\Rayzist_bf16"
set "NEEDED_TRANSFORMER=%PACK_ROOT%\weights\Rayzist.v1.0.safetensors"
set "NEEDED_VAE=%PACK_ROOT%\weights\diffusion_pytorch_model.safetensors"
set "NEEDED_ENCODER=%PACK_ROOT%\config\text_encoder\model.safetensors"
set "NEEDED_SEEDVR2_DIT=%CD%\models\seedvr2\seedvr2_ema_3b_fp8_e4m3fn.safetensors"
set "NEEDED_SEEDVR2_VAE=%CD%\models\seedvr2\ema_vae_fp16.safetensors"
set "NEEDED_SEEDVR2_RUNTIME=%CD%\models\seedvr2\runtime\ComfyUI-SeedVR2_VideoUpscaler\inference_cli.py"
set "FETCH_SCRIPT=%CD%\scripts\fetch_model_assets.ps1"
set "SEEDVR2_RUNTIME_FETCH_SCRIPT=%CD%\scripts\fetch_seedvr2_runtime.ps1"
set "MISSING_ASSETS=0"

if not exist "!NEEDED_TRANSFORMER!" set "MISSING_ASSETS=1"
if not exist "!NEEDED_VAE!" set "MISSING_ASSETS=1"
if not exist "!NEEDED_ENCODER!" set "MISSING_ASSETS=1"
if not exist "!NEEDED_SEEDVR2_DIT!" set "MISSING_ASSETS=1"
if not exist "!NEEDED_SEEDVR2_VAE!" set "MISSING_ASSETS=1"

if !MISSING_ASSETS! EQU 0 exit /b 0

echo.
echo Missing bundled model assets for Rayzist_bf16.
if not exist "!FETCH_SCRIPT!" (
  echo Missing fetch script: !FETCH_SCRIPT!
  exit /b 1
)
echo Running fetch script:
echo   !FETCH_SCRIPT!
powershell -NoProfile -ExecutionPolicy Bypass -File "!FETCH_SCRIPT!"
if errorlevel 1 (
  echo Failed to fetch default model assets.
  echo Ensure Hugging Face CLI with XET is installed via:
  echo   .\RunMeFirst.bat
  exit /b 1
)

if not exist "!NEEDED_SEEDVR2_RUNTIME!" (
  echo.
  echo Missing SeedVR2 runtime scripts.
  if not exist "!SEEDVR2_RUNTIME_FETCH_SCRIPT!" (
    echo Missing SeedVR2 runtime fetch script: !SEEDVR2_RUNTIME_FETCH_SCRIPT!
    exit /b 1
  )
  echo Running SeedVR2 runtime fetch script:
  echo   !SEEDVR2_RUNTIME_FETCH_SCRIPT!
  powershell -NoProfile -ExecutionPolicy Bypass -File "!SEEDVR2_RUNTIME_FETCH_SCRIPT!"
  if errorlevel 1 (
    echo Failed to fetch SeedVR2 runtime scripts.
    echo Run .\RunMeFirst.bat to repair setup.
    exit /b 1
  )
)

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
if not exist "!NEEDED_SEEDVR2_DIT!" (
  echo Missing file after download: !NEEDED_SEEDVR2_DIT!
  exit /b 1
)
if not exist "!NEEDED_SEEDVR2_VAE!" (
  echo Missing file after download: !NEEDED_SEEDVR2_VAE!
  exit /b 1
)
if not exist "!NEEDED_SEEDVR2_RUNTIME!" (
  echo Missing file after download: !NEEDED_SEEDVR2_RUNTIME!
  exit /b 1
)

echo Model assets ready.
exit /b 0

:ensure_krea2_pack_assets
set "KREA_ROOT=%CD%\models\packs\Krea2_Turbo"
set "KREA_TRANSFORMER=%KREA_ROOT%\weights\krea2_turbo_fp8.safetensors"
set "KREA_ENCODER=%KREA_ROOT%\weights\qwen3vl_4b_fp8_scaled.safetensors"
set "KREA_VAE=%KREA_ROOT%\weights\qwen_image_vae.safetensors"
set "FETCH_SCRIPT=%CD%\scripts\fetch_model_assets.ps1"
set "KREA_MISSING=0"

if not exist "!KREA_TRANSFORMER!" set "KREA_MISSING=1"
if not exist "!KREA_ENCODER!" set "KREA_MISSING=1"
if not exist "!KREA_VAE!" set "KREA_MISSING=1"

if !KREA_MISSING! EQU 0 (
  echo Krea2_Turbo model assets ready.
  exit /b 0
)

echo.
echo Missing Krea2_Turbo model weights (~18 GB).
echo Krea2-Turbo weights are governed by the Krea 2 Community License, which is
echo distinct from the Z-Image assets and is downloaded for local use only.
choice /c YN /n /m "Accept the Krea 2 Community License and download now? [Y/N]: "
if errorlevel 2 (
  echo Krea2_Turbo launch cancelled. You can fetch later with:
  echo   powershell -ExecutionPolicy Bypass -File scripts\fetch_model_assets.ps1 -IncludeKrea2 -AcceptKrea2License
  exit /b 1
)
if not exist "!FETCH_SCRIPT!" (
  echo Missing fetch script: !FETCH_SCRIPT!
  exit /b 1
)
echo Running fetch script:
echo   !FETCH_SCRIPT! -IncludeKrea2 -AcceptKrea2License
powershell -NoProfile -ExecutionPolicy Bypass -File "!FETCH_SCRIPT!" -IncludeKrea2 -AcceptKrea2License
if errorlevel 1 (
  echo Failed to fetch Krea2_Turbo model assets.
  echo Ensure Hugging Face CLI with XET is installed via:
  echo   .\RunMeFirst.bat
  exit /b 1
)
if not exist "!KREA_TRANSFORMER!" (
  echo Missing file after download: !KREA_TRANSFORMER!
  exit /b 1
)
if not exist "!KREA_ENCODER!" (
  echo Missing file after download: !KREA_ENCODER!
  exit /b 1
)
if not exist "!KREA_VAE!" (
  echo Missing file after download: !KREA_VAE!
  exit /b 1
)
echo Krea2_Turbo model assets ready.
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
