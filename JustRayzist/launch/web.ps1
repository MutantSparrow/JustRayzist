param(
  [string]$BindHost = "127.0.0.1",
  [int]$Port = 37717,
  [string]$PythonExe = ""
)

$rootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$portablePython = Join-Path $rootDir "runtime\python\python.exe"
$venvPython = Join-Path $rootDir ".venv\Scripts\python.exe"

function Resolve-ReadyPython {
  param(
    [string]$PreferredPythonExe,
    [string]$PortableCandidate,
    [string]$VenvCandidate
  )

  $candidates = @()
  if (-not [string]::IsNullOrWhiteSpace($PreferredPythonExe)) {
    $candidates += $PreferredPythonExe
  } else {
    $candidates += @($PortableCandidate, $VenvCandidate, "python")
  }

  foreach ($candidate in $candidates) {
    $isPathCommand = $candidate -eq "python"
    if (-not $isPathCommand -and -not (Test-Path $candidate)) {
      continue
    }
    & $candidate -c "import typer,fastapi,uvicorn,PIL,torch" *> $null
    if ($LASTEXITCODE -eq 0) {
      return $candidate
    }
  }
  return $null
}

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
  $PythonExe = Resolve-ReadyPython -PreferredPythonExe "" -PortableCandidate $portablePython -VenvCandidate $venvPython
} else {
  $PythonExe = Resolve-ReadyPython -PreferredPythonExe $PythonExe -PortableCandidate $portablePython -VenvCandidate $venvPython
}

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
  Write-Host "No usable Python interpreter with project dependencies was found." -ForegroundColor Red
  Write-Host "Tried: runtime\\python\\python.exe, .venv\\Scripts\\python.exe, and PATH python."
  Write-Host "Repair local env with:"
  Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\\bootstrap_env.ps1"
  Write-Host "Or install dependencies into your selected interpreter:"
  Write-Host "  python -m pip install -e ."
  exit 1
}

& $PythonExe -m app.cli.main serve --host $BindHost --port $Port
