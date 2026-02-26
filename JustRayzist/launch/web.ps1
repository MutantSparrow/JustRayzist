param(
  [string]$BindHost = "127.0.0.1",
  [int]$Port = 37717,
  [string]$PythonExe = ""
)

$rootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$portablePython = Join-Path $rootDir "runtime\python\python.exe"
$venvPython = Join-Path $rootDir ".venv\python.exe"
$allowPathPython = Test-Path (Join-Path $rootDir "scripts\bootstrap_env.ps1")

function Test-PythonRuntimeFiles {
  param([string]$PythonPath)

  if (-not (Test-Path $PythonPath)) {
    return $false
  }
  $pythonDir = Split-Path -Parent $PythonPath
  if (Test-Path (Join-Path $pythonDir "python3.dll")) {
    return $true
  }
  $anyVersionedDll = Get-ChildItem -Path $pythonDir -Filter "python*.dll" -File -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $anyVersionedDll)
}

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
    $candidates += @($PortableCandidate, $VenvCandidate)
    if ($allowPathPython) {
      $candidates += "python"
    }
  }

  foreach ($candidate in $candidates) {
    $isPathCommand = $candidate -eq "python"
    if (-not $isPathCommand) {
      if (-not (Test-Path $candidate)) {
        continue
      }
      if (-not (Test-PythonRuntimeFiles -PythonPath $candidate)) {
        continue
      }
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
  $bootstrapScript = Join-Path $rootDir "scripts\bootstrap_env.ps1"
  Write-Host "No usable Python interpreter with project dependencies was found." -ForegroundColor Red
  if ($allowPathPython) {
    Write-Host "Tried: runtime\\python\\python.exe, .venv\\python.exe, and PATH python."
  } else {
    Write-Host "Tried: runtime\\python\\python.exe and .venv\\python.exe."
  }
  if (Test-Path $bootstrapScript) {
    Write-Host "Repair local env with:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\\bootstrap_env.ps1"
  }
  Write-Host "Or install dependencies into your selected interpreter:"
  Write-Host "  python -m pip install -e ."
  exit 1
}

& $PythonExe -m app.cli.main serve --host $BindHost --port $Port
