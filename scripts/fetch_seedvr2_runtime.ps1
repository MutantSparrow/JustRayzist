param(
  [switch]$Force,
  [string]$Revision = "main"
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$helperScript = Join-Path $projectRoot "scripts\portable\fetch_seedvr2_runtime.py"

if (-not (Test-Path $helperScript)) {
  throw "Portable SeedVR2 runtime helper not found: $helperScript"
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

$arguments = @(
  $helperScript,
  "--project-root",
  $projectRoot,
  "--revision",
  $Revision
)
if ($Force) {
  $arguments += "--force"
}

& $pythonExe @arguments
if ($LASTEXITCODE -ne 0) {
  throw "Portable SeedVR2 runtime helper failed with exit code $LASTEXITCODE."
}
