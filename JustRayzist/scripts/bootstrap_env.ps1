param(
  [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvRoot = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvRoot "Scripts\\python.exe"

function Test-ModuleImport {
  param(
    [string]$PythonPath,
    [string]$ModuleName
  )

  $previousPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    & $PythonPath -c "import $ModuleName" *> $null
    return ($LASTEXITCODE -eq 0)
  } catch {
    return $false
  } finally {
    $ErrorActionPreference = $previousPreference
  }
}

function Invoke-BestEffort {
  param(
    [string]$PythonPath,
    [string[]]$Arguments
  )

  $previousPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    & $PythonPath @Arguments *> $null
    return ($LASTEXITCODE -eq 0)
  } catch {
    return $false
  } finally {
    $ErrorActionPreference = $previousPreference
  }
}

if (-not (Test-Path $venvPython)) {
  & $PythonExe -m venv $venvRoot
}

if (-not (Test-ModuleImport -PythonPath $venvPython -ModuleName "pip")) {
  [void](Invoke-BestEffort -PythonPath $venvPython -Arguments @("-m", "ensurepip", "--upgrade"))
}

if (-not (Test-ModuleImport -PythonPath $venvPython -ModuleName "pip")) {
  Write-Host ".venv is incomplete. Rebuilding virtual environment..."
  & $PythonExe -m venv --clear $venvRoot
  [void](Invoke-BestEffort -PythonPath $venvPython -Arguments @("-m", "ensurepip", "--upgrade"))
}

if (-not (Test-ModuleImport -PythonPath $venvPython -ModuleName "pip")) {
  throw "Failed to bootstrap pip inside .venv."
}

Push-Location $projectRoot
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -e .[dev]
Pop-Location

Write-Host "Environment ready. Use $venvPython for commands."
