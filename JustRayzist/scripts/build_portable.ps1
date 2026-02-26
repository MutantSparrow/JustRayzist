param(
  [string]$OutputDir = "..\\dist\\JustRayzistPortable",
  [string]$PythonExe = "",
  [switch]$Clean,
  [switch]$SkipModels
)

$ErrorActionPreference = "Stop"

function Invoke-Robocopy {
  param(
    [Parameter(Mandatory = $true)][string]$Source,
    [Parameter(Mandatory = $true)][string]$Destination,
    [string[]]$ExtraArgs = @()
  )

  $args = @($Source, $Destination, "/E", "/R:1", "/W:1", "/NFL", "/NDL", "/NJH", "/NJS", "/NP") + $ExtraArgs
  & robocopy @args | Out-Null
  if ($LASTEXITCODE -ge 8) {
    throw "Robocopy failed for '$Source' -> '$Destination' (exit code $LASTEXITCODE)."
  }
}

function Resolve-PythonHome {
  param([string]$PythonExecutable)

  if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    $PythonExecutable = (Get-Command python -ErrorAction Stop).Source
  }

  $resolvedExe = (Resolve-Path $PythonExecutable).Path
  $exeDir = Split-Path -Parent $resolvedExe
  $pythonHome = $exeDir
  $venvCfgCandidates = @(
    (Join-Path $exeDir "pyvenv.cfg"),
    (Join-Path (Split-Path -Parent $exeDir) "pyvenv.cfg")
  )
  $venvCfg = $null
  foreach ($candidateCfg in $venvCfgCandidates) {
    if (Test-Path $candidateCfg) {
      $venvCfg = $candidateCfg
      break
    }
  }
  $isVirtualEnv = $false
  $sitePackagesSource = Join-Path $pythonHome "Lib\\site-packages"

  if ($venvCfg) {
    $isVirtualEnv = $true
    $venvRoot = Split-Path -Parent $venvCfg
    $venvSitePackages = Join-Path $venvRoot "Lib\\site-packages"
    if (Test-Path $venvSitePackages) {
      $sitePackagesSource = (Resolve-Path $venvSitePackages).Path
    }
    $homeLine = Get-Content $venvCfg | Where-Object { $_ -match "^\s*home\s*=" } | Select-Object -First 1
    if ($homeLine) {
      $candidate = $homeLine.Split("=", 2)[1].Trim()
      if ($candidate -and (Test-Path (Join-Path $candidate "python.exe"))) {
        $pythonHome = (Resolve-Path $candidate).Path
      }
    }
  }

  if (-not (Test-Path (Join-Path $pythonHome "python.exe"))) {
    throw "Could not resolve a valid Python home from '$PythonExecutable'."
  }

  return @{
    Executable = $resolvedExe
    Home = $pythonHome
    IsVirtualEnv = $isVirtualEnv
    SitePackagesSource = $sitePackagesSource
  }
}

$rootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$portableRoot = Join-Path $rootDir $OutputDir
$portableRoot = [System.IO.Path]::GetFullPath($portableRoot)

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
  $localVenvPython = Join-Path $rootDir ".venv\\Scripts\\python.exe"
  if (Test-Path $localVenvPython) {
    $PythonExe = $localVenvPython
  }
}

if ($Clean -and (Test-Path $portableRoot)) {
  Remove-Item $portableRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $portableRoot -Force | Out-Null

$pythonInfo = Resolve-PythonHome -PythonExecutable $PythonExe
$runtimeDir = Join-Path $portableRoot "runtime\\python"
New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null

Write-Host "Bundling Python runtime from: $($pythonInfo.Home)"
Invoke-Robocopy -Source $pythonInfo.Home -Destination $runtimeDir -ExtraArgs @(
  "/XD", "__pycache__", ".pytest_cache", ".ruff_cache", "Scripts\\__pycache__", "Lib\\__pycache__"
)
if ($pythonInfo.IsVirtualEnv -and (Test-Path $pythonInfo.SitePackagesSource)) {
  Write-Host "Overlaying site-packages from virtual environment: $($pythonInfo.SitePackagesSource)"
  $runtimeSitePackages = Join-Path $runtimeDir "Lib\\site-packages"
  New-Item -ItemType Directory -Path $runtimeSitePackages -Force | Out-Null
  Invoke-Robocopy -Source $pythonInfo.SitePackagesSource -Destination $runtimeSitePackages -ExtraArgs @(
    "/XD", "__pycache__", ".pytest_cache", ".ruff_cache"
  )
}

Write-Host "Copying application files..."
$projectDirs = @("app", "img", "launch", "docs")
foreach ($dir in $projectDirs) {
  Invoke-Robocopy -Source (Join-Path $rootDir $dir) -Destination (Join-Path $portableRoot $dir) -ExtraArgs @(
    "/XD", "__pycache__", ".pytest_cache", ".ruff_cache"
  )
}

$upscalerSourceDir = Join-Path $rootDir "models\\upscaler"
$upscalerDestDir = Join-Path $portableRoot "models\\upscaler"
if (-not (Test-Path $upscalerSourceDir)) {
  throw "Required upscaler directory not found: $upscalerSourceDir"
}
Invoke-Robocopy -Source $upscalerSourceDir -Destination $upscalerDestDir -ExtraArgs @(
  "/XD", "__pycache__", ".cache"
)

if (-not $SkipModels) {
  Invoke-Robocopy -Source (Join-Path $rootDir "models\\packs") -Destination (Join-Path $portableRoot "models\\packs") -ExtraArgs @(
    "/XD", "__pycache__", ".cache"
  )
} else {
  New-Item -ItemType Directory -Path (Join-Path $portableRoot "models\\packs") -Force | Out-Null
}

$filesToCopy = @(
  "StartWeb.bat",
  "pyproject.toml"
)
foreach ($file in $filesToCopy) {
  Copy-Item (Join-Path $rootDir $file) -Destination (Join-Path $portableRoot $file) -Force
}

New-Item -ItemType Directory -Path (Join-Path $portableRoot "outputs") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $portableRoot "data") -Force | Out-Null
if (-not (Test-Path (Join-Path $portableRoot "data\\.gitkeep"))) {
  New-Item -ItemType File -Path (Join-Path $portableRoot "data\\.gitkeep") -Force | Out-Null
}

$portablePython = Join-Path $runtimeDir "python.exe"
if (-not (Test-Path $portablePython)) {
  throw "Portable build is missing runtime\\python\\python.exe."
}

$defaultUpscalerCheckpoint = Join-Path $upscalerDestDir "2x_RealESRGAN_x2plus.pth"
if (-not (Test-Path $defaultUpscalerCheckpoint)) {
  throw "Portable build is missing default upscaler checkpoint: $defaultUpscalerCheckpoint"
}

$dependencyCheckScript = "import importlib.util,sys;required=('typer','fastapi','uvicorn','PIL','torch');missing=[n for n in required if importlib.util.find_spec(n) is None];print(', '.join(missing));sys.exit(1 if missing else 0)"
$missingModuleOutput = & $portablePython -c $dependencyCheckScript
if ($LASTEXITCODE -ne 0) {
  $missingModules = ($missingModuleOutput -join ", ").Trim()
  if (-not $missingModules) {
    $missingModules = "(unknown)"
  }
  throw "Portable runtime is missing required dependencies: $missingModules. Build using a Python environment with project dependencies installed."
}

Write-Host ""
Write-Host "Portable bundle created at:"
Write-Host "  $portableRoot"
Write-Host ""
Write-Host "Launch with:"
Write-Host "  $portableRoot\\StartWeb.bat"
