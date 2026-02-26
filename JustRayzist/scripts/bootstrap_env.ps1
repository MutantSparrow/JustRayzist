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

  & $PythonPath -c "import $ModuleName" *> $null
  return ($LASTEXITCODE -eq 0)
}

if (-not (Test-Path $venvPython)) {
  & $PythonExe -m venv $venvRoot
}

if (-not (Test-ModuleImport -PythonPath $venvPython -ModuleName "pip")) {
  & $venvPython -m ensurepip --upgrade
}

if (-not (Test-ModuleImport -PythonPath $venvPython -ModuleName "pip")) {
  Write-Host ".venv is incomplete. Rebuilding virtual environment..."
  & $PythonExe -m venv --clear $venvRoot
  & $venvPython -m ensurepip --upgrade
}

if (-not (Test-ModuleImport -PythonPath $venvPython -ModuleName "pip")) {
  throw "Failed to bootstrap pip inside .venv."
}

Push-Location $projectRoot
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -e .[dev]
Pop-Location

Write-Host "Environment ready. Use $venvPython for commands."
