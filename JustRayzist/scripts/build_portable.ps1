param(
  [string]$OutputDir = "..\\dist\\JustRayzistPortable",
  [string]$PythonExe = "",
  [switch]$Clean,
  [switch]$SkipModels,
  [switch]$SkipVenv
)

$ErrorActionPreference = "Stop"

function Test-PythonDependencySet {
  param([string]$PythonPath)

  $previousPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    & $PythonPath -c "import typer,fastapi,uvicorn,PIL,torch" *> $null
    return ($LASTEXITCODE -eq 0)
  } catch {
    return $false
  } finally {
    $ErrorActionPreference = $previousPreference
  }
}

function Resolve-ReadyPython {
  param(
    [string]$PreferredPythonExe,
    [string[]]$Candidates
  )

  $candidateList = @()
  if (-not [string]::IsNullOrWhiteSpace($PreferredPythonExe)) {
    $candidateList += $PreferredPythonExe
  } else {
    $candidateList += $Candidates
  }

  foreach ($candidate in $candidateList) {
    $isPathCommand = $candidate -eq "python"
    if (-not $isPathCommand -and -not (Test-Path $candidate)) {
      continue
    }
    if (Test-PythonDependencySet -PythonPath $candidate) {
      return $candidate
    }
  }

  return $null
}

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

function Ensure-PortableVenvLayout {
  param([string]$VenvDir)

  $scriptsDir = Join-Path $VenvDir "Scripts"
  New-Item -ItemType Directory -Path $scriptsDir -Force | Out-Null

  $rootPython = Join-Path $VenvDir "python.exe"
  $scriptsPython = Join-Path $scriptsDir "python.exe"
  if ((Test-Path $scriptsPython) -and -not (Test-Path $rootPython)) {
    Copy-Item $scriptsPython -Destination $rootPython -Force
  } elseif ((Test-Path $rootPython) -and -not (Test-Path $scriptsPython)) {
    Copy-Item $rootPython -Destination $scriptsPython -Force
  }

  $rootPythonw = Join-Path $VenvDir "pythonw.exe"
  $scriptsPythonw = Join-Path $scriptsDir "pythonw.exe"
  if ((Test-Path $scriptsPythonw) -and -not (Test-Path $rootPythonw)) {
    Copy-Item $scriptsPythonw -Destination $rootPythonw -Force
  } elseif ((Test-Path $rootPythonw) -and -not (Test-Path $scriptsPythonw)) {
    Copy-Item $rootPythonw -Destination $scriptsPythonw -Force
  }

  $runtimeDlls = Get-ChildItem -Path $VenvDir -Filter "python*.dll" -File -ErrorAction SilentlyContinue
  foreach ($dll in $runtimeDlls) {
    $scriptsDllPath = Join-Path $scriptsDir $dll.Name
    if (-not (Test-Path $scriptsDllPath)) {
      Copy-Item $dll.FullName -Destination $scriptsDllPath -Force
    }
  }
}

function Resolve-PythonHome {
  param([string]$PythonExecutable)

  if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    $PythonExecutable = (Get-Command python -ErrorAction Stop).Source
  } elseif (-not (Test-Path $PythonExecutable)) {
    $command = Get-Command $PythonExecutable -ErrorAction Stop
    $PythonExecutable = $command.Source
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

$localVenvRootPython = Join-Path $rootDir ".venv\\python.exe"
$localVenvScriptsPython = Join-Path $rootDir ".venv\\Scripts\\python.exe"
$PythonExe = Resolve-ReadyPython -PreferredPythonExe $PythonExe -Candidates @($localVenvRootPython, $localVenvScriptsPython, "python")
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
  throw "No usable Python interpreter with required dependencies found. Run scripts\\bootstrap_env.ps1 or pass -PythonExe."
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

$sourceVenvDir = Join-Path $rootDir ".venv"
if (-not $SkipVenv) {
  $portableVenvDir = Join-Path $portableRoot ".venv"
  $sourceVenvRootPython = Join-Path $sourceVenvDir "python.exe"
  $sourceVenvScriptsPython = Join-Path $sourceVenvDir "Scripts\\python.exe"
  $sourceVenvPython = if (Test-Path $sourceVenvRootPython) { $sourceVenvRootPython } else { $sourceVenvScriptsPython }
  if ((Test-Path $sourceVenvPython) -and (Test-PythonDependencySet -PythonPath $sourceVenvPython)) {
    Write-Host "Copying local virtual environment to portable bundle: $sourceVenvDir"
    Invoke-Robocopy -Source $sourceVenvDir -Destination $portableVenvDir -ExtraArgs @(
      "/XD", "__pycache__", ".pytest_cache", ".ruff_cache", "Lib\\site-packages\\__pycache__", "Scripts\\__pycache__"
    )
  } else {
    Write-Host "Local .venv unavailable or missing dependencies. Creating portable .venv from bundled runtime."
    Invoke-Robocopy -Source $runtimeDir -Destination $portableVenvDir -ExtraArgs @(
      "/XD", "__pycache__", ".pytest_cache", ".ruff_cache", "Lib\\__pycache__", "Lib\\site-packages\\__pycache__"
    )
  }
  Ensure-PortableVenvLayout -VenvDir $portableVenvDir
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
  Invoke-Robocopy -Source (Join-Path $rootDir "models\\packs") -Destination (Join-Path $portableRoot "models\\packs") -ExtraArgs @(
    "/XD", "__pycache__", ".cache",
    "/XF", "*.safetensors", "*.gguf"
  )
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
$portablePythonDll = Join-Path $runtimeDir "python3.dll"
$portableVersionedDll = Get-ChildItem -Path $runtimeDir -Filter "python*.dll" -File -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not (Test-Path $portablePythonDll) -and -not $portableVersionedDll) {
  throw "Portable build is missing Python runtime DLLs under runtime\\python."
}
if (-not $SkipVenv) {
  $portableVenvPython = Join-Path $portableRoot ".venv\\python.exe"
  if (-not (Test-Path $portableVenvPython)) {
    throw "Portable build is missing .venv\\python.exe."
  }
  $portableVenvDll = Join-Path $portableRoot ".venv\\python3.dll"
  $portableVenvVersionedDll = Get-ChildItem -Path (Join-Path $portableRoot ".venv") -Filter "python*.dll" -File -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not (Test-Path $portableVenvDll) -and -not $portableVenvVersionedDll) {
    throw "Portable build is missing Python runtime DLLs under .venv."
  }
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
if (-not $SkipVenv) {
  $portableVenvPython = Join-Path $portableRoot ".venv\\python.exe"
  $missingVenvModuleOutput = & $portableVenvPython -c $dependencyCheckScript
  if ($LASTEXITCODE -ne 0) {
    $missingVenvModules = ($missingVenvModuleOutput -join ", ").Trim()
    if (-not $missingVenvModules) {
      $missingVenvModules = "(unknown)"
    }
    throw "Portable .venv is missing required dependencies: $missingVenvModules."
  }
}

Write-Host ""
Write-Host "Portable bundle created at:"
Write-Host "  $portableRoot"
Write-Host ""
Write-Host "Launch with:"
Write-Host "  $portableRoot\\StartWeb.bat"
