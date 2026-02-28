param(
  [switch]$Force
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Resolve-HfUrl {
  param([string]$Url)
  return ($Url -replace "/blob/", "/resolve/")
}

function Download-Asset {
  param(
    [string]$Name,
    [string]$Url,
    [string]$RelativeOutputPath,
    [string]$Sha256,
    [switch]$Overwrite
  )

  $outputPath = Join-Path $projectRoot $RelativeOutputPath
  $outputDir = Split-Path -Parent $outputPath
  if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
  }

  $expectedHash = $Sha256.ToLowerInvariant()
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

  $resolvedUrl = Resolve-HfUrl -Url $Url
  $tmpPath = "$outputPath.part"
  if (Test-Path $tmpPath) {
    Remove-Item $tmpPath -Force
  }

  Write-Host "[download] $Name"
  Write-Host "  from: $resolvedUrl"
  Write-Host "  to:   $outputPath"
  Invoke-WebRequest -Uri $resolvedUrl -OutFile $tmpPath -MaximumRedirection 10
  Move-Item -Path $tmpPath -Destination $outputPath -Force

  if ($hasExpectedHash) {
    $actualHash = (Get-FileHash -Path $outputPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $expectedHash) {
      Remove-Item $outputPath -Force -ErrorAction SilentlyContinue
      throw "SHA256 mismatch for $Name. Expected $expectedHash, got $actualHash."
    }
    Write-Host "  sha256: $actualHash"
  }

  $sizeMb = [math]::Round(((Get-Item $outputPath).Length / 1MB), 2)
  Write-Host "[ok] $Name saved ($sizeMb MB)"
}

$assets = @(
  @{
    Name = "Transformer checkpoint"
    Url = "https://huggingface.co/MutantSparrow/Ray/blob/main/Z-IMAGE-TURBO/Rayzist.v1.0.safetensors"
    RelativeOutputPath = "models/packs/Rayzist_bf16/weights/Rayzist.v1.0.safetensors"
    Sha256 = "e1d396329a3d5ebde6d81df5d4753367a61fa9f0cb45ed6fa78336f69bd975a1"
  },
  @{
    Name = "VAE checkpoint"
    Url = "https://huggingface.co/Owen777/UltraFlux-v1/blob/main/vae/diffusion_pytorch_model.safetensors"
    RelativeOutputPath = "models/packs/Rayzist_bf16/weights/ultrafluxVAEImproved_v10.safetensors"
    Sha256 = "2bf9ad685686b480b03651a8d8595951e4a5578016b8ead4af5e22d3dc9b3409"
  },
  @{
    Name = "Text encoder checkpoint"
    Url = "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors"
    RelativeOutputPath = "models/packs/Rayzist_bf16/config/text_encoder/model.safetensors"
    Sha256 = "6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a"
  },
  @{
    Name = "Upscaler checkpoint (RealESRGAN x2plus)"
    Url = "https://huggingface.co/imagepipeline/superresolution/resolve/main/RealESRGAN_x2plus.pth"
    RelativeOutputPath = "models/upscaler/2x_RealESRGAN_x2plus.pth"
    Sha256 = "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb"
  }
)

foreach ($asset in $assets) {
  Download-Asset -Name $asset.Name -Url $asset.Url -RelativeOutputPath $asset.RelativeOutputPath -Sha256 $asset.Sha256 -Overwrite:$Force
}

Write-Host ""
Write-Host "Model asset fetch complete."
Write-Host "Next step:"
Write-Host "  python -m app.cli.main validate-models"
