param(
  [switch]$Force,
  [switch]$AcceptCommunityLicense
)

# Fetches Krea2-Turbo weights + config into models/packs/Krea2_Turbo.
#
# Krea2-Turbo is opt-in and governed by the Krea 2 Community License (distinct from the Z-Image
# assets). This script downloads for LOCAL use only and requires -AcceptCommunityLicense to proceed.
# See JustRayzist-Krea.md §12 and models/packs/Krea2_Turbo/weights/README.md.
#
# NOTE (WP-0): the exact repo filenames below are placeholders to be confirmed against
# https://huggingface.co/krea/Krea-2-Turbo when the coexistence spike is run on real hardware.

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$packDir = Join-Path $projectRoot "models\packs\Krea2_Turbo"
$weightsDir = Join-Path $packDir "weights"
$configDir = Join-Path $packDir "config"
$repoId = "krea/Krea-2-Turbo"

Write-Host "==============================================================="
Write-Host " Krea2-Turbo assets - Krea 2 Community License"
Write-Host " Weights are downloaded for LOCAL use only. Redistribution or"
Write-Host " bundling requires review of the Krea 2 Community License."
Write-Host "==============================================================="

if (-not $AcceptCommunityLicense) {
  throw "Re-run with -AcceptCommunityLicense once you have reviewed and accepted the Krea 2 Community License."
}

foreach ($dir in @($weightsDir, $configDir)) {
  if (-not (Test-Path $dir)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
  }
}

# Prefer the project venv's hf CLI, fall back to PATH.
$hfCandidates = @(
  (Join-Path $projectRoot ".venv\Scripts\hf.exe"),
  "hf"
)
$hfExe = $null
foreach ($candidate in $hfCandidates) {
  if ($candidate -ne "hf" -and -not (Test-Path $candidate)) { continue }
  $hfExe = $candidate
  break
}
if (-not $hfExe) {
  throw "Hugging Face CLI ('hf') not found. Run .\RunMeFirst.bat to set up the environment."
}

$downloadArgs = @("download", $repoId, "--local-dir", $packDir)
if ($Force) { $downloadArgs += "--force-download" }

Write-Host "Downloading $repoId into $packDir ..."
& $hfExe @downloadArgs
if ($LASTEXITCODE -ne 0) {
  throw "hf download failed with exit code $LASTEXITCODE."
}

Write-Host "Done. Copy modelpack.yaml.template -> modelpack.yaml and adjust paths if needed."
