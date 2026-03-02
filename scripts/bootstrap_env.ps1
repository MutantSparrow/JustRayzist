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
$runtimeRequirements = Join-Path $projectRoot "requirements\\runtime-lock.txt"
$seedVr2Requirements = Join-Path $projectRoot "requirements\\seedvr2-lock.txt"
$devRequirements = Join-Path $projectRoot "requirements\\dev-lock.txt"

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

function Test-ZImageDiffusersSymbols {
  param(
    [string]$PythonPath
  )

  return (Invoke-BestEffort -PythonPath $PythonPath -Arguments @(
      "-c",
      "from diffusers import ZImagePipeline, ZImageTransformer2DModel, ZImageImg2ImgPipeline"
    ))
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
if (-not (Test-Path $runtimeRequirements)) {
  throw "Missing runtime lock file: $runtimeRequirements"
}
if (-not (Test-Path $seedVr2Requirements)) {
  throw "Missing SeedVR2 lock file: $seedVr2Requirements"
}
if (-not (Test-Path $devRequirements)) {
  throw "Missing dev lock file: $devRequirements"
}

function Install-DiffusersWithFallback {
  param(
    [Parameter(Mandatory = $true)][string]$PythonPath
  )

  $attempts = @(
    [PSCustomObject]@{
      Label = "diffusers==0.36.0"
      Args = @("-m", "pip", "install", "--upgrade", "diffusers==0.36.0")
    },
    [PSCustomObject]@{
      Label = "diffusers==0.36.0.dev0"
      Args = @("-m", "pip", "install", "--upgrade", "diffusers==0.36.0.dev0")
    },
    [PSCustomObject]@{
      Label = "pre-release diffusers>=0.36.0"
      Args = @("-m", "pip", "install", "--upgrade", "--pre", "diffusers>=0.36.0")
    },
    [PSCustomObject]@{
      Label = "diffusers main branch zip"
      Args = @(
        "-m",
        "pip",
        "install",
        "--upgrade",
        "https://github.com/huggingface/diffusers/archive/refs/heads/main.zip"
      )
    }
  )

  foreach ($attempt in $attempts) {
    Write-Host ("Installing {0}..." -f $attempt.Label)
    $installed = $false
    try {
      Invoke-Checked -Executable $PythonPath -Arguments $attempt.Args
      $installed = $true
    } catch {
      Write-Host ("Install attempt failed for {0}: {1}" -f $attempt.Label, $_.Exception.Message) -ForegroundColor Yellow
    }
    if (-not $installed) {
      continue
    }
    if (Test-ZImageDiffusersSymbols -PythonPath $PythonPath) {
      Write-Host ("Using {0} (ZImage symbols verified)." -f $attempt.Label)
      return
    }
    Write-Host ("Installed {0} but ZImage symbols are still missing." -f $attempt.Label) -ForegroundColor Yellow
  }

  throw (
    "Unable to install a diffusers build exposing ZImagePipeline/ZImageTransformer2DModel/ZImageImg2ImgPipeline. " +
    "Check internet access and rerun RunMeFirst.bat."
  )
}

Invoke-Checked -Executable $venvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked -Executable $venvPython -Arguments @("-m", "pip", "install", "--upgrade", "setuptools", "wheel")
Invoke-Checked -Executable $venvPython -Arguments @(
  "-m", "pip", "install", "huggingface_hub[hf_xet]==0.35.0"
)
Invoke-Checked -Executable $venvPython -Arguments @("-m", "pip", "install", "-r", $torchRequirements)
$runtimeNoDiffusers = Join-Path $tmpRoot "runtime-lock.no-diffusers.txt"
$runtimeLines = Get-Content -Path $runtimeRequirements
$filteredRuntime = @()
foreach ($line in $runtimeLines) {
  $trimmed = $line.Trim()
  if (-not $trimmed) {
    continue
  }
  if ($trimmed.StartsWith("#")) {
    continue
  }
  if ($trimmed -match '^diffusers(\s|==|>=|<=|~=|@|$)') {
    continue
  }
  $filteredRuntime += $line
}
if ($filteredRuntime.Count -gt 0) {
  Set-Content -Path $runtimeNoDiffusers -Value $filteredRuntime -Encoding ascii
  Invoke-Checked -Executable $venvPython -Arguments @("-m", "pip", "install", "-r", $runtimeNoDiffusers)
}
Invoke-Checked -Executable $venvPython -Arguments @("-m", "pip", "install", "-r", $seedVr2Requirements)
Install-DiffusersWithFallback -PythonPath $venvPython
Invoke-Checked -Executable $venvPython -Arguments @("-m", "pip", "install", "-r", $devRequirements)
Invoke-Checked -Executable $venvPython -Arguments @("-m", "pip", "install", "--no-build-isolation", "--no-deps", "-e", ".") -WorkingDirectory $projectRoot

Write-Host "Environment ready. Use $venvPython for commands. Lane=$Lane."
