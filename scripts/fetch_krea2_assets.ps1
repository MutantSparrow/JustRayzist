param(
  [switch]$Force,
  [switch]$AcceptCommunityLicense
)

# Thin wrapper: fetch the Krea2_Turbo pack weights via the shared model-asset fetcher.
#
# Krea2-Turbo is opt-in and governed by the Krea 2 Community License (distinct from the Z-Image
# assets). Weights download for LOCAL use only. Re-run with -AcceptCommunityLicense to proceed.
# See models/packs/Krea2_Turbo/weights/README.md and JustRayzist-Krea.md.
#
# The authoritative asset list (repo ids, filenames, SHA256, output paths) lives in
# scripts/portable/fetch_model_assets.py (OPTIONAL_KREA2_ASSETS). This wrapper only forwards flags
# so the two paths never drift.

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$fetchScript = Join-Path $scriptDir "fetch_model_assets.ps1"

if (-not (Test-Path $fetchScript)) {
  throw "Shared fetch script not found: $fetchScript"
}

Write-Host "==============================================================="
Write-Host " Krea2-Turbo assets - Krea 2 Community License"
Write-Host " Weights are downloaded for LOCAL use only. Redistribution or"
Write-Host " bundling requires review of the Krea 2 Community License."
Write-Host "==============================================================="

if (-not $AcceptCommunityLicense) {
  throw "Re-run with -AcceptCommunityLicense once you have reviewed and accepted the Krea 2 Community License."
}

$fetchArgs = @("-IncludeKrea2", "-AcceptKrea2License")
if ($Force) {
  $fetchArgs += "-Force"
}

& $fetchScript @fetchArgs
if ($LASTEXITCODE -ne 0) {
  throw "Krea2 asset fetch failed with exit code $LASTEXITCODE."
}
