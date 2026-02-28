@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
chcp 65001 >nul
cls

set "HOST=127.0.0.1"
set "PORT=37717"
set "JUSTRAYZIST_ROOT=%CD%"
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"

set "WEB_EXE=%CD%\bin\web\justrayzist-web.exe"
set "RUN_MODE="
set "PYTHON_EXE="
if exist "!WEB_EXE!" (
  set "RUN_MODE=exe"
) else (
  if defined JUSTRAYZIST_PYTHON call :try_source_python "%JUSTRAYZIST_PYTHON%"
  if not defined PYTHON_EXE call :try_source_python "%CD%\.venv\Scripts\python.exe"
  if not defined PYTHON_EXE call :try_source_python "%CD%\.venv\python.exe"
  if not defined PYTHON_EXE call :try_source_python "python"
  if not defined PYTHON_EXE call :try_python_launcher_paths
  if not defined PYTHON_EXE (
    echo.
    echo No packaged web executable found with a usable source Python runtime.
    echo Checked packaged path:
    echo   !WEB_EXE!
    echo Checked source interpreter candidates - requires dependencies and ZImage diffusers symbols:
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
    set "EXIT_CODE=1"
    goto :after_run
  )
  set "RUN_MODE=python"
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
echo   [2] balanced     (16GB VRAM target)
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

set "RELEASE_LANE=cu128"
if exist "%CD%\release_lane.txt" (
  set /p RELEASE_LANE=<"%CD%\release_lane.txt"
  if not defined RELEASE_LANE set "RELEASE_LANE=cu128"
)

if /I "!RUN_MODE!"=="exe" (
  if /I "%JUSTRAYZIST_SKIP_GPU_PREFLIGHT%"=="1" (
    echo GPU preflight skipped due to JUSTRAYZIST_SKIP_GPU_PREFLIGHT=1.
  ) else (
    call :validate_gpu_lane
    if errorlevel 1 (
      echo.
      echo GPU preflight reported a lane/driver mismatch.
      choice /c YN /n /m "Continue launch anyway? [Y/N]: "
      if errorlevel 2 (
        set "EXIT_CODE=1"
        goto :after_run
      )
    )
  )
) else (
  echo GPU preflight skipped in source mode.
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
echo Runtime lane: !RELEASE_LANE!
echo URL: http://!HOST!:!PORT!/
echo.

if /I "!RUN_MODE!"=="exe" (
  "!WEB_EXE!" --host !HOST! --port !PORT! --profile !PROFILE!
) else (
  "!PYTHON_EXE!" -m app.cli.main serve --host !HOST! --port !PORT! --profile !PROFILE!
)
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
if /I "%CANDIDATE%"=="python" goto :check_source_candidate
if not exist "%CANDIDATE%" goto :eof

:check_source_candidate
"%CANDIDATE%" -c "import typer,fastapi,uvicorn,PIL,torch,diffusers,transformers,accelerate,safetensors; from diffusers import ZImagePipeline, ZImageTransformer2DModel, ZImageImg2ImgPipeline" >nul 2>&1
if errorlevel 1 goto :eof
set "PYTHON_EXE=%CANDIDATE%"
goto :eof

:try_python_launcher_paths
for /f "tokens=* delims=" %%L in ('py -0p 2^>nul') do (
  set "PY_LINE=%%L"
  set "PY_LAST="
  for %%P in (!PY_LINE!) do set "PY_LAST=%%P"
  if defined PY_LAST (
    call :try_source_python "!PY_LAST!"
    if defined PYTHON_EXE goto :eof
  )
)
goto :eof

:validate_gpu_lane
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$lane='%RELEASE_LANE%'.Trim().ToLowerInvariant();" ^
  "$floors=@{cu126=[version]'561.17';cu128=[version]'572.61'};" ^
  "$cmd=Get-Command nvidia-smi -ErrorAction SilentlyContinue;" ^
  "if(-not $cmd){ Write-Host 'GPU preflight: nvidia-smi not found; skipping lane gate.'; exit 0 };" ^
  "$rows=& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null;" ^
  "if(-not $rows){ Write-Host 'GPU preflight: no NVIDIA GPU detected; continuing.'; exit 0 };" ^
  "$first=@($rows)[0].ToString();" ^
  "$parts=$first.Split(',',2);" ^
  "if($parts.Count -lt 2){ Write-Host ('GPU preflight: unexpected nvidia-smi row: ' + $first); exit 2 };" ^
  "$gpu=$parts[0].Trim();" ^
  "$driverText=$parts[1].Trim();" ^
  "try { $driver=[version]$driverText } catch { Write-Host ('GPU preflight: unable to parse driver version: ' + $driverText); exit 2 };" ^
  "$is50=$gpu -match 'RTX\s*50';" ^
  "if($lane -eq 'cu126' -and $is50){ Write-Host ('GPU preflight failed: ' + $gpu + ' requires cu128 lane and driver >= 572.61.'); exit 2 };" ^
  "$required=$floors[$lane];" ^
  "if(-not $required){ Write-Host ('GPU preflight: unknown lane ' + $lane + '; skipping lane gate.'); exit 0 };" ^
  "if($driver -lt $required){ Write-Host ('GPU preflight failed: GPU=' + $gpu + ', driver=' + $driver + ', lane=' + $lane + ', required>=' + $required); exit 2 };" ^
  "Write-Host ('GPU preflight OK: GPU=' + $gpu + ', driver=' + $driver + ', lane=' + $lane + ', required>=' + $required); exit 0"
if errorlevel 2 exit /b 1
if errorlevel 1 exit /b 1
exit /b 0

:ensure_rayzist_pack_assets
set "PACK_ROOT=%CD%\models\packs\Rayzist_bf16"
set "NEEDED_TRANSFORMER=%PACK_ROOT%\weights\Rayzist.v1.0.safetensors"
set "NEEDED_VAE=%PACK_ROOT%\weights\ultrafluxVAEImproved_v10.safetensors"
set "NEEDED_ENCODER=%PACK_ROOT%\config\text_encoder\model.safetensors"
set "NEEDED_UPSCALER=%CD%\models\upscaler\2x_RealESRGAN_x2plus.pth"
set "FETCH_SCRIPT=%CD%\scripts\fetch_model_assets.ps1"
set "MISSING_ASSETS=0"

if not exist "!NEEDED_TRANSFORMER!" set "MISSING_ASSETS=1"
if not exist "!NEEDED_VAE!" set "MISSING_ASSETS=1"
if not exist "!NEEDED_ENCODER!" set "MISSING_ASSETS=1"
if not exist "!NEEDED_UPSCALER!" set "MISSING_ASSETS=1"

if !MISSING_ASSETS! EQU 0 exit /b 0

echo.
echo Missing default model assets for pack Rayzist_bf16.
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
if not exist "!NEEDED_UPSCALER!" (
  echo Missing file after download: !NEEDED_UPSCALER!
  exit /b 1
)

echo Model assets ready.
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
