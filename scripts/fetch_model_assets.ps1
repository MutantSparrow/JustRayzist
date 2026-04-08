param(
  [switch]$Force
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$helperScript = Join-Path $projectRoot "scripts\portable\fetch_model_assets.py"

if (-not (Test-Path $helperScript)) {
  throw "Portable asset fetch helper not found: $helperScript"
}

$pythonCandidates = @(
  (Join-Path $projectRoot ".venv\Scripts\python.exe"),
  "python"
)

$pythonExe = $null
foreach ($candidate in $pythonCandidates) {
  if ($candidate -ne "python" -and -not (Test-Path $candidate)) {
    continue
  }
  $previousPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    & $candidate -c "import sys; sys.exit(0)" *> $null
    if ($LASTEXITCODE -eq 0) {
      $pythonExe = $candidate
      break
    }
  } finally {
    $ErrorActionPreference = $previousPreference
  }
}

if (-not $pythonExe) {
  throw "Python executable not found. Run .\RunMeFirst.bat to install or repair the environment."
}

$arguments = @($helperScript, "--project-root", $projectRoot, "--platform", "windows")
if ($Force) {
  $arguments += "--force"
}

& $pythonExe @arguments
if ($LASTEXITCODE -ne 0) {
  throw "Portable asset fetch helper failed with exit code $LASTEXITCODE."
}
