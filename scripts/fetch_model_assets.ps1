param(
  [switch]$Force
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Resolve-HfCliExecutable {
  $candidates = @(
    (Join-Path $projectRoot ".venv\Scripts\hf.exe"),
    "hf"
  )

  foreach ($candidate in $candidates) {
    if ($candidate -ne "hf" -and -not (Test-Path $candidate)) {
      continue
    }
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
      & $candidate "version" *> $null
      if ($LASTEXITCODE -eq 0) {
        return $candidate
      }
    } catch {
      continue
    } finally {
      $ErrorActionPreference = $previousPreference
    }
  }
  return $null
}

function Invoke-HfCli {
  param(
    [Parameter(Mandatory = $true)][string]$HfExe,
    [Parameter(Mandatory = $true)][string[]]$Arguments
  )

  & $HfExe @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "HF CLI command failed: $HfExe $($Arguments -join ' ') (exit code $LASTEXITCODE)"
  }
}

function Ensure-HfCliPrerequisites {
  param([Parameter(Mandatory = $true)][string]$HfExe)

  Invoke-HfCli -HfExe $HfExe -Arguments @("version")

  & $HfExe "download" "--help" *> $null
  if ($LASTEXITCODE -ne 0) {
    throw "HF CLI command failed: $HfExe download --help (exit code $LASTEXITCODE)"
  }
}

function Download-Asset {
  param(
    [Parameter(Mandatory = $true)][string]$HfExe,
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$RepoId,
    [Parameter(Mandatory = $true)][string]$RepoFile,
    [string]$Revision = "main",
    [Parameter(Mandatory = $true)][string]$RelativeOutputPath,
    [string]$Sha256 = "",
    [switch]$Overwrite
  )

  $outputPath = Join-Path $projectRoot $RelativeOutputPath
  $outputDir = Split-Path -Parent $outputPath
  $stageRoot = Join-Path $projectRoot ".build\hf_downloads"
  New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
  New-Item -ItemType Directory -Path $stageRoot -Force | Out-Null

  $expectedHash = ""
  if ($null -ne $Sha256) {
    $expectedHash = [string]$Sha256
  }
  $expectedHash = $expectedHash.ToLowerInvariant()
  $hasExpectedHash = -not [string]::IsNullOrWhiteSpace($expectedHash)

  if ((Test-Path $outputPath) -and (-not $Overwrite)) {
    if ($hasExpectedHash) {
      $actualHash = (Get-FileHash -Path $outputPath -Algorithm SHA256).Hash.ToLowerInvariant()
      if ($actualHash -eq $expectedHash) {
        $sizeMb = [math]::Round(((Get-Item $outputPath).Length / 1MB), 2)
        Write-Host "[skip] $Name already exists and passed SHA256 check ($sizeMb MB): $outputPath"
        return
      }
      Write-Warning "$Name exists but SHA256 does not match expected value. Re-downloading..."
      Write-Host "  expected: $expectedHash"
      Write-Host "  actual:   $actualHash"
    } else {
      $sizeMb = [math]::Round(((Get-Item $outputPath).Length / 1MB), 2)
      Write-Host "[skip] $Name already exists ($sizeMb MB): $outputPath"
      return
    }
  }

  $stageDir = Join-Path $stageRoot ([Guid]::NewGuid().ToString("N"))
  $tmpPath = "$outputPath.part"
  if (Test-Path $tmpPath) {
    Remove-Item $tmpPath -Force
  }

  Write-Host "[download] $Name"
  Write-Host "  repo: $RepoId"
  Write-Host "  file: $RepoFile"
  Write-Host "  to:   $outputPath"

  New-Item -ItemType Directory -Path $stageDir -Force | Out-Null
  try {
    Invoke-HfCli -HfExe $HfExe -Arguments @(
      "download",
      $RepoId,
      $RepoFile,
      "--repo-type", "model",
      "--revision", $Revision,
      "--local-dir", $stageDir,
      "--max-workers", "8"
    )

    $repoFileRelative = $RepoFile -replace "/", "\"
    $downloadedPath = Join-Path $stageDir $repoFileRelative
    if (-not (Test-Path $downloadedPath)) {
      throw "HF CLI download completed but file not found at expected path: $downloadedPath"
    }

    Move-Item -Path $downloadedPath -Destination $tmpPath -Force
  } finally {
    if (Test-Path $stageDir) {
      Remove-Item -Path $stageDir -Recurse -Force -ErrorAction SilentlyContinue
    }
  }

  if ($hasExpectedHash) {
    $actualHash = (Get-FileHash -Path $tmpPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $expectedHash) {
      Remove-Item $tmpPath -Force -ErrorAction SilentlyContinue
      throw "SHA256 mismatch for $Name. Expected $expectedHash, got $actualHash."
    }
    Write-Host "  sha256: $actualHash"
  }

  Move-Item -Path $tmpPath -Destination $outputPath -Force
  $sizeMb = [math]::Round(((Get-Item $outputPath).Length / 1MB), 2)
  Write-Host "[ok] $Name saved ($sizeMb MB)"
}

$assets = @(
  @{
    Name = "Transformer checkpoint"
    RepoId = "MutantSparrow/Ray"
    RepoFile = "Z-IMAGE-TURBO/Rayzist.v1.0.safetensors"
    RelativeOutputPath = "models/packs/Rayzist_bf16/weights/Rayzist.v1.0.safetensors"
    Sha256 = "e1d396329a3d5ebde6d81df5d4753367a61fa9f0cb45ed6fa78336f69bd975a1"
  },
  @{
    Name = "VAE checkpoint"
    RepoId = "Tongyi-MAI/Z-Image-Turbo"
    RepoFile = "vae/diffusion_pytorch_model.safetensors"
    RelativeOutputPath = "models/packs/Rayzist_bf16/weights/diffusion_pytorch_model.safetensors"
    Sha256 = "f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3"
  },
  @{
    Name = "Text encoder checkpoint"
    RepoId = "Comfy-Org/z_image_turbo"
    RepoFile = "split_files/text_encoders/qwen_3_4b.safetensors"
    RelativeOutputPath = "models/packs/Rayzist_bf16/config/text_encoder/model.safetensors"
    Sha256 = "6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a"
  },
  @{
    Name = "Upscaler checkpoint (RealESRGAN x2plus)"
    RepoId = "imagepipeline/superresolution"
    RepoFile = "RealESRGAN_x2plus.pth"
    RelativeOutputPath = "models/upscaler/2x_RealESRGAN_x2plus.pth"
    Sha256 = "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb"
  }
)

$hfExe = Resolve-HfCliExecutable
if ([string]::IsNullOrWhiteSpace($hfExe)) {
  throw (
    "Hugging Face CLI executable (hf) not found. " +
    "Run .\RunMeFirst.bat to install/repair the environment."
  )
}
if ($hfExe -eq "hf") {
  Write-Warning (
    "Using 'hf' from PATH. For strict reproducibility, prefer .venv\Scripts\hf.exe " +
    "by running .\RunMeFirst.bat first."
  )
}

$env:HF_XET_HIGH_PERFORMANCE = "1"
Remove-Item Env:HF_HUB_DISABLE_XET -ErrorAction SilentlyContinue

Ensure-HfCliPrerequisites -HfExe $hfExe

foreach ($asset in $assets) {
  Download-Asset -HfExe $hfExe -Name $asset.Name -RepoId $asset.RepoId -RepoFile $asset.RepoFile -RelativeOutputPath $asset.RelativeOutputPath -Sha256 $asset.Sha256 -Overwrite:$Force
}

$deprecatedVaePath = Join-Path $projectRoot "models/packs/Rayzist_bf16/weights/ultrafluxVAEImproved_v10.safetensors"
if (Test-Path $deprecatedVaePath) {
  Remove-Item -Path $deprecatedVaePath -Force
  Write-Host "[cleanup] Removed deprecated VAE file: $deprecatedVaePath"
}

Write-Host ""
Write-Host "Model asset fetch complete (HF CLI + XET)."
Write-Host "Next step:"
Write-Host "  python -m app.cli.main validate-models"
