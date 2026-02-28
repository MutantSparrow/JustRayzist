param(
  [string]$PythonExe = "python",
  [ValidateSet("cu126", "cu128")]
  [string]$Lane = "cu128"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvRoot = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvRoot "Scripts\\python.exe"
$tmpRoot = Join-Path $projectRoot ".build\\tmp"
$torchRequirements = Join-Path $projectRoot ("requirements\\torch-" + $Lane + ".txt")

New-Item -ItemType Directory -Path $tmpRoot -Force | Out-Null
$env:TEMP = $tmpRoot
$env:TMP = $tmpRoot

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

function Invoke-Checked {
  param(
    [Parameter(Mandatory = $true)][string]$Executable,
    [Parameter(Mandatory = $true)][string[]]$Arguments,
    [string]$WorkingDirectory = ""
  )

  if ($WorkingDirectory) {
    Push-Location $WorkingDirectory
  }
  try {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
      throw "Command failed: $Executable $($Arguments -join ' ') (exit code $LASTEXITCODE)"
    }
  } finally {
    if ($WorkingDirectory) {
      Pop-Location
    }
  }
}

if (-not (Test-Path $venvPython)) {
  Invoke-Checked -Executable $PythonExe -Arguments @("-m", "venv", $venvRoot)
}

if (-not (Test-ModuleImport -PythonPath $venvPython -ModuleName "pip")) {
  [void](Invoke-BestEffort -PythonPath $venvPython -Arguments @("-m", "ensurepip", "--upgrade"))
}

if (-not (Test-ModuleImport -PythonPath $venvPython -ModuleName "pip")) {
  Write-Host ".venv is incomplete. Rebuilding virtual environment..."
  Invoke-Checked -Executable $PythonExe -Arguments @("-m", "venv", "--clear", $venvRoot)
  [void](Invoke-BestEffort -PythonPath $venvPython -Arguments @("-m", "ensurepip", "--upgrade"))
}

if (-not (Test-ModuleImport -PythonPath $venvPython -ModuleName "pip")) {
  throw "Failed to bootstrap pip inside .venv."
}

if (-not (Test-Path $torchRequirements)) {
  throw "Missing torch requirements file for lane ${Lane}: $torchRequirements"
}

Invoke-Checked -Executable $venvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked -Executable $venvPython -Arguments @("-m", "pip", "install", "--upgrade", "setuptools", "wheel")
Invoke-Checked -Executable $venvPython -Arguments @("-m", "pip", "install", "-r", $torchRequirements)
Invoke-Checked -Executable $venvPython -Arguments @("-m", "pip", "install", "--no-build-isolation", "-e", ".[dev]") -WorkingDirectory $projectRoot

Write-Host "Environment ready. Use $venvPython for commands. Lane=$Lane."
