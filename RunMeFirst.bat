@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
chcp 65001 >nul
cls

if exist "launch\ascii_logo_blockier.ans" (
  type "launch\ascii_logo_blockier.ans"
  echo.
)

echo.
echo JustRayzist Setup and Repair
echo ============================
echo This script will install or repair your local JustRayzist environment.
echo.

set "SETUP_SCRIPT=%CD%\scripts\setup\runmefirst.ps1"
if not exist "!SETUP_SCRIPT!" (
  echo Setup script not found:
  echo   !SETUP_SCRIPT!
  echo.
  set "EXIT_CODE=1"
  goto :after_run
)

powershell -NoProfile -ExecutionPolicy Bypass -File "!SETUP_SCRIPT!"
set "EXIT_CODE=%ERRORLEVEL%"

:after_run
if not "%EXIT_CODE%"=="0" (
  echo.
  echo Setup failed with code %EXIT_CODE%.
  echo You can rerun RunMeFirst.bat to attempt repair.
  pause
) else (
  echo.
  echo Setup completed successfully.
  echo Launch the app with StartWeb.bat or the desktop shortcut.
  pause
)

endlocal
exit /b %EXIT_CODE%
