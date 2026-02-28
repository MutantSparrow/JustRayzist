param(
  [ValidateSet("cu126", "cu128")]
  [string]$Lane = "cu128",
  [string]$PythonExe = "python",
  [string]$BuildVenvDir = ".build\\pyinstaller\\venv",
  [string]$DistRoot = "dist\\pyinstaller",
  [switch]$UseActivePython,
  [switch]$Clean,
  [switch]$SkipDependencyInstall
)

$ErrorActionPreference = "Stop"

function Invoke-Robust {
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

$rootDir = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
$venvDir = [System.IO.Path]::GetFullPath((Join-Path $rootDir $BuildVenvDir))
$distBaseDir = [System.IO.Path]::GetFullPath((Join-Path $rootDir $DistRoot))
$laneDistDir = Join-Path $distBaseDir $Lane
$workDir = Join-Path $rootDir ".build\\pyinstaller\\work\\$Lane"
$specDir = Join-Path $rootDir ".build\\pyinstaller\\spec\\$Lane"
$tempRoot = Join-Path $rootDir (".build\\tmp_" + [System.Guid]::NewGuid().ToString("N"))

New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
$env:TEMP = $tempRoot
$env:TMP = $tempRoot

if ($Clean) {
  foreach ($path in @($laneDistDir, $workDir, $specDir)) {
    if (Test-Path $path) {
      Remove-Item $path -Recurse -Force
    }
  }
}

if ($UseActivePython) {
  $venvPython = $PythonExe
  if (-not (Test-ModuleImport -PythonPath $venvPython -ModuleName "pip")) {
    [void](Invoke-BestEffort -PythonPath $venvPython -Arguments @("-m", "ensurepip", "--upgrade"))
  }
  if (-not (Test-ModuleImport -PythonPath $venvPython -ModuleName "pip")) {
    throw "Active Python is missing pip and could not be repaired with ensurepip: $venvPython"
  }
} else {
  if (-not (Test-Path $venvDir)) {
    Invoke-Robust -Executable $PythonExe -Arguments @("-m", "venv", $venvDir)
  }

  $venvPython = Join-Path $venvDir "Scripts\\python.exe"
  if (-not (Test-Path $venvPython)) {
    throw "Build virtual environment python executable not found: $venvPython"
  }

  if (-not (Test-ModuleImport -PythonPath $venvPython -ModuleName "pip")) {
    [void](Invoke-BestEffort -PythonPath $venvPython -Arguments @("-m", "ensurepip", "--upgrade"))
  }
  if (-not (Test-ModuleImport -PythonPath $venvPython -ModuleName "pip")) {
    Invoke-Robust -Executable $PythonExe -Arguments @("-m", "venv", "--clear", $venvDir)
    [void](Invoke-BestEffort -PythonPath $venvPython -Arguments @("-m", "ensurepip", "--upgrade"))
  }
  if (-not (Test-ModuleImport -PythonPath $venvPython -ModuleName "pip")) {
    throw "Build virtual environment is incomplete: pip is unavailable in $venvPython"
  }
}

if (-not $SkipDependencyInstall) {
  $torchRequirements = Join-Path $rootDir ("requirements\\torch-" + $Lane + ".txt")
  if (-not (Test-Path $torchRequirements)) {
    throw "Missing torch requirements file for lane ${Lane}: $torchRequirements"
  }

  Invoke-Robust -Executable $venvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools>=68", "wheel")
  Invoke-Robust -Executable $venvPython -Arguments @("-m", "pip", "install", "--upgrade", "pyinstaller")
  Invoke-Robust -Executable $venvPython -Arguments @("-m", "pip", "install", "-r", $torchRequirements)
  Invoke-Robust -Executable $venvPython -Arguments @("-m", "pip", "install", "-e", ".") -WorkingDirectory $rootDir
}

New-Item -ItemType Directory -Path $laneDistDir -Force | Out-Null
New-Item -ItemType Directory -Path $workDir -Force | Out-Null
New-Item -ItemType Directory -Path $specDir -Force | Out-Null

$excludeArgs = @(
  "--exclude-module", "tensorflow",
  "--exclude-module", "keras",
  "--exclude-module", "jax",
  "--exclude-module", "flax",
  "--exclude-module", "gradio",
  "--exclude-module", "matplotlib",
  "--exclude-module", "sklearn",
  "--exclude-module", "scipy",
  "--exclude-module", "pandas",
  "--exclude-module", "IPython"
)

$commonArgs = @(
  "-m", "PyInstaller",
  "--noconfirm",
  "--clean",
  "--onedir",
  "--distpath", $laneDistDir,
  "--workpath", $workDir,
  "--specpath", $specDir,
  "--paths", $rootDir,
  "--hidden-import", "app.cli.main",
  "--hidden-import", "app.api.main"
) + $excludeArgs

$env:USE_TF = "0"
$env:TRANSFORMERS_NO_TF = "1"
$env:TRANSFORMERS_NO_FLAX = "1"

Invoke-Robust -Executable $venvPython -Arguments ($commonArgs + @(
    "--name", "justrayzist-web",
    "app\\entrypoints\\web_entry.py"
  )) -WorkingDirectory $rootDir

Invoke-Robust -Executable $venvPython -Arguments ($commonArgs + @(
    "--name", "justrayzist-cli",
    "app\\entrypoints\\cli_entry.py"
  )) -WorkingDirectory $rootDir

Write-Host ""
Write-Host "PyInstaller one-dir build complete."
Write-Host "Lane:      $Lane"
Write-Host "Output:    $laneDistDir"
Write-Host "Web binary: $laneDistDir\\justrayzist-web\\justrayzist-web.exe"
Write-Host "CLI binary: $laneDistDir\\justrayzist-cli\\justrayzist-cli.exe"
