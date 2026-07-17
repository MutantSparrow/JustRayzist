param(
  [switch]$Force
)

# Thin wrapper: fetch the Krea2_Turbo pack weights via the shared model-asset fetcher.
#
# The authoritative asset list (repo ids, filenames, SHA256, output paths) lives in
# scripts/portable/fetch_model_assets.py (OPTIONAL_KREA2_ASSETS). This wrapper only forwards flags
# so the two paths never drift. Krea2_Turbo weights are provisioned from the operator's own
# finetuned checkpoint repo — licensing is handled off-repo, not here.

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$fetchScript = Join-Path $scriptDir "fetch_model_assets.ps1"

if (-not (Test-Path $fetchScript)) {
  throw "Shared fetch script not found: $fetchScript"
}

$fetchArgs = @("-IncludeKrea2")
if ($Force) {
  $fetchArgs += "-Force"
}

& $fetchScript @fetchArgs
if ($LASTEXITCODE -ne 0) {
  throw "Krea2 asset fetch failed with exit code $LASTEXITCODE."
}
