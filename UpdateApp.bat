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
echo JustRayzist In-Place Updater
echo ============================
echo This updates a packaged JustRayzist install from the latest matching GitHub release.
echo Local runtime folders such as models, outputs, data, and .venv are preserved.
echo.

set "UPDATE_SCRIPT=%CD%\scripts\update_release.ps1"
if not exist "!UPDATE_SCRIPT!" (
  echo Updater script not found:
  echo   !UPDATE_SCRIPT!
  echo.
  set "EXIT_CODE=1"
  goto :after_run
)

powershell -NoProfile -ExecutionPolicy Bypass -File "!UPDATE_SCRIPT!"
set "EXIT_CODE=%ERRORLEVEL%"

:after_run
if not "%EXIT_CODE%"=="0" (
  echo.
  echo Update failed with code %EXIT_CODE%.
  pause
) else (
  echo.
  echo Update completed successfully.
  echo Launch the app with StartWeb.bat.
  pause
)

endlocal
exit /b %EXIT_CODE%
