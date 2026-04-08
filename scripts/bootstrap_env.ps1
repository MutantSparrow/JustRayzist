param(
  [string]$PythonExe = "python",
  [ValidateSet("cu126", "cu128", "default", "auto")]
  [string]$Lane = "cu128"
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$helperScript = Join-Path $projectRoot "scripts\portable\bootstrap_env.py"

if (-not (Test-Path $helperScript)) {
  throw "Portable bootstrap helper not found: $helperScript"
}

& $PythonExe $helperScript --project-root $projectRoot --python-exe $PythonExe --lane $Lane --platform windows
if ($LASTEXITCODE -ne 0) {
  throw "Portable bootstrap helper failed with exit code $LASTEXITCODE."
}
