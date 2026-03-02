param(
  [switch]$ForceRepair
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
$bootstrapScript = Join-Path $projectRoot "scripts\\bootstrap_env.ps1"
$fetchScript = Join-Path $projectRoot "scripts\\fetch_model_assets.ps1"
$fetchSeedVr2RuntimeScript = Join-Path $projectRoot "scripts\\fetch_seedvr2_runtime.ps1"
$manifestPath = Join-Path $PSScriptRoot "python_manifest.json"
$startWebPath = Join-Path $projectRoot "StartWeb.bat"
$venvPython = Join-Path $projectRoot ".venv\\Scripts\\python.exe"
$releaseLanePath = Join-Path $projectRoot "release_lane.txt"
$seedVr2RuntimeScript = Join-Path $projectRoot "models\\seedvr2\\runtime\\ComfyUI-SeedVR2_VideoUpscaler\\inference_cli.py"
$seedVr2DitPath = Join-Path $projectRoot "models\\seedvr2\\seedvr2_ema_3b_fp8_e4m3fn.safetensors"
$seedVr2VaePath = Join-Path $projectRoot "models\\seedvr2\\ema_vae_fp16.safetensors"

$script:StepCurrent = 0
$script:StepTotal = 0

function Write-Banner {
  Clear-Host
  $asciiPath = Join-Path $projectRoot "launch\\ascii_logo_blockier.ans"
  if (Test-Path $asciiPath) {
    Get-Content $asciiPath
    Write-Host ""
  }
  Write-Host "JustRayzist RunMeFirst" -ForegroundColor Cyan
  Write-Host "======================" -ForegroundColor Cyan
  Write-Host "Setup, sanity-check, and auto-repair workflow." -ForegroundColor Gray
  Write-Host ""
}

function Set-StepTotal {
  param([int]$Total)
  $script:StepCurrent = 0
  $script:StepTotal = $Total
}

function Invoke-Step {
  param(
    [Parameter(Mandatory = $true)][string]$Title,
    [Parameter(Mandatory = $true)][scriptblock]$Action
  )

  $script:StepCurrent += 1
  Write-Host ("[Step {0}/{1}] {2}" -f $script:StepCurrent, $script:StepTotal, $Title) -ForegroundColor Yellow
  $stepTimer = [System.Diagnostics.Stopwatch]::StartNew()
  & $Action
  $stepTimer.Stop()
  Write-Host ("[OK] {0} ({1:n1}s)" -f $Title, $stepTimer.Elapsed.TotalSeconds) -ForegroundColor Green
  Write-Host ""
}

function Invoke-External {
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

function Try-RunPython {
  param(
    [Parameter(Mandatory = $true)][string]$PythonExe,
    [Parameter(Mandatory = $true)][string]$Code
  )

  $previousPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    $result = & $PythonExe -c $Code 2>$null
    return [PSCustomObject]@{
      Success = ($LASTEXITCODE -eq 0)
      Output = ($result -join "`n").Trim()
    }
  } catch {
    return [PSCustomObject]@{
      Success = $false
      Output = ""
    }
  } finally {
    $ErrorActionPreference = $previousPreference
  }
}

function Try-RunCommand {
  param(
    [Parameter(Mandatory = $true)][string]$Executable,
    [Parameter(Mandatory = $true)][string[]]$Arguments
  )

  $previousPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    $result = & $Executable @Arguments 2>$null
    return [PSCustomObject]@{
      Success = ($LASTEXITCODE -eq 0)
      Output = ($result -join "`n").Trim()
    }
  } catch {
    return [PSCustomObject]@{
      Success = $false
      Output = ""
    }
  } finally {
    $ErrorActionPreference = $previousPreference
  }
}

function Test-Python311 {
  param([string]$PythonExe)
  if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    return $false
  }
  if ($PythonExe -ne "python" -and -not (Test-Path $PythonExe)) {
    return $false
  }
  $probe = Try-RunPython -PythonExe $PythonExe -Code "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
  if (-not $probe.Success) {
    return $false
  }
  return ($probe.Output -eq "3.11")
}

function Resolve-Python311 {
  $candidateList = New-Object System.Collections.Generic.List[string]

  $localVenvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
  if (Test-Path $localVenvPython) {
    $candidateList.Add($localVenvPython)
  }

  if ($env:JUSTRAYZIST_PYTHON) {
    $candidateList.Add($env:JUSTRAYZIST_PYTHON)
  }
  $candidateList.Add("python")

  $pyPreferred = Try-RunCommand -Executable "py" -Arguments @("-c", "import sys; print(sys.executable)")
  if ($pyPreferred.Success -and -not [string]::IsNullOrWhiteSpace($pyPreferred.Output)) {
    $candidateList.Add($pyPreferred.Output)
  }

  $py311Preferred = Try-RunCommand -Executable "py" -Arguments @("-3.11", "-c", "import sys; print(sys.executable)")
  if ($py311Preferred.Success -and -not [string]::IsNullOrWhiteSpace($py311Preferred.Output)) {
    $candidateList.Add($py311Preferred.Output)
  }

  foreach ($candidate in $candidateList) {
    if (Test-Python311 -PythonExe $candidate) {
      if ($candidate -eq "python") {
        $resolved = Try-RunPython -PythonExe "python" -Code "import sys; print(sys.executable)"
        if ($resolved.Success -and -not [string]::IsNullOrWhiteSpace($resolved.Output)) {
          return $resolved.Output
        }
      }
      return $candidate
    }
  }
  return $null
}

function Get-LaneSelection {
  $floors = @{
    cu126 = [Version]"561.17"
    cu128 = [Version]"572.61"
  }

  $result = [PSCustomObject]@{
    Lane = "cu128"
    GpuName = "Unknown"
    DriverVersion = "Unknown"
    Message = "No GPU-specific override detected. Using cu128."
  }

  $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
  if (-not $nvidiaSmi) {
    $result.Message = "nvidia-smi not found. Using default lane cu128."
    return $result
  }

  $rows = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null
  if (-not $rows) {
    $result.Message = "No NVIDIA GPU detected. Using default lane cu128."
    return $result
  }

  $firstRow = @($rows)[0].ToString()
  $parts = $firstRow.Split(",", 2)
  if ($parts.Count -lt 2) {
    $result.Message = "Unexpected nvidia-smi output row '$firstRow'. Using cu128."
    return $result
  }
  $gpu = $parts[0].Trim()
  $driverText = $parts[1].Trim()
  $result.GpuName = $gpu
  $result.DriverVersion = $driverText

  try {
    $driver = [Version]$driverText
  } catch {
    $result.Message = "Could not parse driver version '$driverText'. Using cu128."
    return $result
  }

  if ($gpu -match "RTX\s*50") {
    $result.Lane = "cu128"
    if ($driver -lt $floors.cu128) {
      $result.Message = "Detected $gpu with driver $driverText. cu128 selected; update driver to >= $($floors.cu128)."
    } else {
      $result.Message = "Detected $gpu with driver $driverText. cu128 selected."
    }
    return $result
  }

  if ($driver -ge $floors.cu128) {
    $result.Lane = "cu128"
    $result.Message = "Detected $gpu with driver $driverText. cu128 selected."
    return $result
  }

  if ($driver -ge $floors.cu126) {
    $result.Lane = "cu126"
    $result.Message = "Detected $gpu with driver $driverText. cu126 selected for compatibility."
    return $result
  }

  $result.Lane = "cu126"
  $result.Message = "Driver $driverText is below lane floors. cu126 selected; driver update recommended."
  return $result
}

function Install-PythonFromManifest {
  if (-not (Test-Path $manifestPath)) {
    throw "Python manifest not found: $manifestPath"
  }
  $manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json

  $arch = "amd64"
  if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") {
    $arch = "arm64"
  } elseif (-not [Environment]::Is64BitOperatingSystem) {
    $arch = "x86"
  }

  $installerSpec = $manifest.installers.$arch
  if (-not $installerSpec) {
    throw "No installer manifest entry found for architecture '$arch'."
  }

  $tmpDir = Join-Path $projectRoot ".build\\bootstrap"
  New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null
  $installerPath = Join-Path $tmpDir ("python-" + $manifest.python_version + "-" + $arch + ".exe")

  Write-Host ("Downloading Python {0} ({1})..." -f $manifest.python_version, $arch) -ForegroundColor Gray
  Invoke-WebRequest -Uri $installerSpec.url -OutFile $installerPath -MaximumRedirection 10

  if ($installerSpec.md5) {
    $actualMd5 = (Get-FileHash -Path $installerPath -Algorithm MD5).Hash.ToLowerInvariant()
    $expectedMd5 = $installerSpec.md5.ToLowerInvariant()
    if ($actualMd5 -ne $expectedMd5) {
      Remove-Item -Path $installerPath -Force -ErrorAction SilentlyContinue
      throw "Python installer checksum mismatch. Expected MD5 $expectedMd5, got $actualMd5."
    }
    Write-Host ("Installer checksum verified (md5: {0})." -f $actualMd5) -ForegroundColor DarkGray
  }

  Write-Host "Installing Python silently..." -ForegroundColor Gray
  $process = Start-Process -FilePath $installerPath -ArgumentList $manifest.silent_args -PassThru -Wait
  if ($process.ExitCode -ne 0) {
    throw "Python installer exited with code $($process.ExitCode)."
  }

  $pythonExe = Resolve-Python311
  if (-not $pythonExe) {
    $commonPath = Join-Path $env:LOCALAPPDATA "Programs\\Python\\Python311\\python.exe"
    if (Test-Python311 -PythonExe $commonPath) {
      $pythonExe = $commonPath
    }
  }
  if (-not $pythonExe) {
    throw "Python 3.11 installation finished but interpreter could not be located."
  }

  return $pythonExe
}

function Test-EnvironmentHealth {
  $issues = New-Object System.Collections.Generic.List[string]

  if (-not (Test-Path $venvPython)) {
    $issues.Add(".venv python executable not found.")
    return [PSCustomObject]@{
      Healthy = $false
      Issues = $issues
    }
  }

  $importProbe = Try-RunPython -PythonExe $venvPython -Code "import typer,fastapi,uvicorn,PIL,torch,diffusers,transformers,accelerate,safetensors; from diffusers import ZImagePipeline, ZImageTransformer2DModel, ZImageImg2ImgPipeline"
  if (-not $importProbe.Success) {
    $issues.Add("Core runtime imports failed in .venv (including required ZImage diffusers symbols).")
  }
  $seedImportProbe = Try-RunPython -PythonExe $venvPython -Code "import cv2,omegaconf,peft,einops,rotary_embedding_torch"
  if (-not $seedImportProbe.Success) {
    $issues.Add("SeedVR2 runtime dependencies are missing in .venv.")
  }

  if (-not (Test-Path $seedVr2RuntimeScript)) {
    $issues.Add("SeedVR2 runtime script not found.")
  }
  if (-not (Test-Path $seedVr2DitPath)) {
    $issues.Add("SeedVR2 DiT checkpoint not found.")
  }
  if (-not (Test-Path $seedVr2VaePath)) {
    $issues.Add("SeedVR2 VAE checkpoint not found.")
  }

  $previousPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    & $venvPython -m app.cli.main doctor *> $null
    if ($LASTEXITCODE -ne 0) {
      $issues.Add("doctor command failed.")
    }
    & $venvPython -m app.cli.main validate-models *> $null
    if ($LASTEXITCODE -ne 0) {
      $issues.Add("validate-models failed.")
    }
  } finally {
    $ErrorActionPreference = $previousPreference
  }

  return [PSCustomObject]@{
    Healthy = ($issues.Count -eq 0)
    Issues = $issues
  }
}

function Ensure-Shortcut {
  if (-not (Test-Path $startWebPath)) {
    throw "StartWeb.bat not found at project root: $startWebPath"
  }

  $desktop = [Environment]::GetFolderPath("Desktop")
  if ([string]::IsNullOrWhiteSpace($desktop)) {
    Write-Host "Desktop path unavailable; skipping shortcut creation." -ForegroundColor Yellow
    return
  }

  $shortcutPath = Join-Path $desktop "JustRayzist.lnk"
  $wshShell = New-Object -ComObject WScript.Shell
  $shortcut = $wshShell.CreateShortcut($shortcutPath)
  $shortcut.TargetPath = $startWebPath
  $shortcut.WorkingDirectory = $projectRoot
  $shortcut.Description = "Launch JustRayzist"
  $iconPath = Join-Path $projectRoot "img\\favicon.ico"
  if (Test-Path $iconPath) {
    $shortcut.IconLocation = $iconPath
  }
  $shortcut.Save()

  Write-Host ("Desktop shortcut ready: {0}" -f $shortcutPath) -ForegroundColor DarkGray
}

function Ensure-ReleaseLaneFile {
  param([string]$Lane)
  Set-Content -Path $releaseLanePath -Value $Lane -Encoding ascii
}

Set-Location $projectRoot
Write-Banner

$timer = [System.Diagnostics.Stopwatch]::StartNew()

try {
  if (-not (Test-Path $bootstrapScript)) {
    throw "Missing bootstrap script: $bootstrapScript"
  }
  if (-not (Test-Path $fetchScript)) {
    throw "Missing model fetch script: $fetchScript"
  }
  if (-not (Test-Path $fetchSeedVr2RuntimeScript)) {
    throw "Missing SeedVR2 runtime fetch script: $fetchSeedVr2RuntimeScript"
  }

  $laneSelection = Get-LaneSelection
  Write-Host $laneSelection.Message -ForegroundColor Cyan
  Write-Host ""

  $existingHealth = Test-EnvironmentHealth
  if ($existingHealth.Healthy -and -not $ForceRepair) {
    Set-StepTotal -Total 3

    Invoke-Step -Title "Sanity check existing environment" -Action {
      Write-Host ".venv, dependencies, and model validation checks are healthy." -ForegroundColor Gray
    }

    Invoke-Step -Title "Refresh runtime lane marker" -Action {
      Ensure-ReleaseLaneFile -Lane $laneSelection.Lane
      Write-Host ("release_lane.txt updated to {0}" -f $laneSelection.Lane) -ForegroundColor Gray
    }

    Invoke-Step -Title "Ensure StartWeb shortcut" -Action {
      Ensure-Shortcut
    }

    $timer.Stop()
    Write-Host ("Environment is healthy. No repair was needed. ({0:n1}s)" -f $timer.Elapsed.TotalSeconds) -ForegroundColor Green
    exit 0
  }

  if ($existingHealth.Issues.Count -gt 0) {
    Write-Host "Environment check found issues; repair will run:" -ForegroundColor Yellow
    foreach ($issue in $existingHealth.Issues) {
      Write-Host ("  - {0}" -f $issue) -ForegroundColor Yellow
    }
    Write-Host ""
  }

  Set-StepTotal -Total 8
  $script:ResolvedPythonExe = Resolve-Python311

  Invoke-Step -Title "Record target runtime lane" -Action {
    Ensure-ReleaseLaneFile -Lane $laneSelection.Lane
    Write-Host ("Using lane: {0}" -f $laneSelection.Lane) -ForegroundColor Gray
  }

  Invoke-Step -Title "Ensure Python 3.11 is available" -Action {
    if ([string]::IsNullOrWhiteSpace($script:ResolvedPythonExe)) {
      $script:ResolvedPythonExe = Install-PythonFromManifest
      Write-Host ("Installed Python at: {0}" -f $script:ResolvedPythonExe) -ForegroundColor Gray
    } else {
      Write-Host ("Python 3.11 detected at: {0}" -f $script:ResolvedPythonExe) -ForegroundColor Gray
    }
    if ([string]::IsNullOrWhiteSpace($script:ResolvedPythonExe)) {
      throw "Python 3.11 interpreter resolution failed."
    }
  }

  Invoke-Step -Title "Create or repair virtual environment" -Action {
    Invoke-External -Executable "powershell" -Arguments @(
      "-NoProfile",
      "-ExecutionPolicy", "Bypass",
      "-File", $bootstrapScript,
      "-PythonExe", $script:ResolvedPythonExe,
      "-Lane", $laneSelection.Lane
    )
  }

  Invoke-Step -Title "Download default model assets" -Action {
    Invoke-External -Executable "powershell" -Arguments @(
      "-NoProfile",
      "-ExecutionPolicy", "Bypass",
      "-File", $fetchScript
    )
  }

  Invoke-Step -Title "Fetch SeedVR2 runtime scripts" -Action {
    Invoke-External -Executable "powershell" -Arguments @(
      "-NoProfile",
      "-ExecutionPolicy", "Bypass",
      "-File", $fetchSeedVr2RuntimeScript
    )
  }

  Invoke-Step -Title "Run doctor check" -Action {
    Invoke-External -Executable $venvPython -Arguments @("-m", "app.cli.main", "doctor")
  }

  Invoke-Step -Title "Run model validation" -Action {
    Invoke-External -Executable $venvPython -Arguments @("-m", "app.cli.main", "validate-models")
  }

  Invoke-Step -Title "Ensure StartWeb shortcut" -Action {
    Ensure-Shortcut
  }

  $timer.Stop()
  Write-Host ("Setup and repair complete. ({0:n1}s)" -f $timer.Elapsed.TotalSeconds) -ForegroundColor Green
  Write-Host "Next step: launch StartWeb.bat or use the desktop shortcut." -ForegroundColor Cyan
  exit 0
} catch {
  $timer.Stop()
  Write-Host ""
  Write-Host ("[FAIL] {0}" -f $_.Exception.Message) -ForegroundColor Red
  Write-Host "RunMeFirst could not complete setup. Re-run this script to retry repair." -ForegroundColor Yellow
  Write-Host ("Elapsed: {0:n1}s" -f $timer.Elapsed.TotalSeconds) -ForegroundColor DarkGray
  exit 1
}
