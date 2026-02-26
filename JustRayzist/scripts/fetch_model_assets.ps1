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
    [switch]$Overwrite
  )

  $outputPath = Join-Path $projectRoot $RelativeOutputPath
  $outputDir = Split-Path -Parent $outputPath
  if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
  }

  if ((Test-Path $outputPath) -and (-not $Overwrite)) {
    $sizeMb = [math]::Round(((Get-Item $outputPath).Length / 1MB), 2)
    Write-Host "[skip] $Name already exists ($sizeMb MB): $outputPath"
    return
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

  $sizeMb = [math]::Round(((Get-Item $outputPath).Length / 1MB), 2)
  Write-Host "[ok] $Name saved ($sizeMb MB)"
}

$assets = @(
  @{
    Name = "Transformer checkpoint"
    Url = "https://huggingface.co/MutantSparrow/Ray/blob/main/Z-IMAGE-TURBO/Rayzist.v1.0.safetensors"
    RelativeOutputPath = "models/packs/Rayzist_bf16/weights/Rayzist.v1.0.safetensors"
  },
  @{
    Name = "VAE checkpoint"
    Url = "https://huggingface.co/Owen777/UltraFlux-v1/blob/main/vae/diffusion_pytorch_model.safetensors"
    RelativeOutputPath = "models/packs/Rayzist_bf16/weights/ultrafluxVAEImproved_v10.safetensors"
  },
  @{
    Name = "Text encoder checkpoint"
    Url = "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors"
    RelativeOutputPath = "models/packs/Rayzist_bf16/config/text_encoder/model.safetensors"
  }
)

foreach ($asset in $assets) {
  Download-Asset -Name $asset.Name -Url $asset.Url -RelativeOutputPath $asset.RelativeOutputPath -Overwrite:$Force
}

Write-Host ""
Write-Host "Model asset fetch complete."
Write-Host "Next step:"
Write-Host "  python -m app.cli.main validate-models"
