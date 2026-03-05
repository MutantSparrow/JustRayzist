param(
  [switch]$Force,
  [string]$Revision = "main"
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$targetRoot = Join-Path $projectRoot "models\seedvr2\runtime"
$targetRepo = Join-Path $targetRoot "ComfyUI-SeedVR2_VideoUpscaler"
$repoUrl = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"

function Apply-AllocatorCompatPatch {
  param(
    [Parameter(Mandatory = $true)][string]$RuntimeScriptPath
  )

  $original = Get-Content -Path $RuntimeScriptPath -Raw -Encoding utf8
  if ($original -notmatch "PYTORCH_CUDA_ALLOC_CONF") {
    Write-Host "[patch] Runtime allocator env var already compatible."
    return
  }

  $updated = $original -replace "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"
  if ($updated -eq $original) {
    Write-Host "[patch] Runtime allocator env var already compatible."
    return
  }

  Set-Content -Path $RuntimeScriptPath -Value $updated -Encoding utf8
  Write-Host "[patch] Applied allocator env compatibility patch (PYTORCH_ALLOC_CONF)."
}

function Invoke-Checked {
  param(
    [Parameter(Mandatory = $true)][string]$Executable,
    [Parameter(Mandatory = $true)][string[]]$Arguments
  )

  & $Executable @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed: $Executable $($Arguments -join ' ') (exit code $LASTEXITCODE)"
  }
}

$gitCmd = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitCmd) {
  throw "Git executable not found in PATH. Install Git and rerun RunMeFirst.bat."
}

New-Item -ItemType Directory -Path $targetRoot -Force | Out-Null

if (-not (Test-Path (Join-Path $targetRepo ".git"))) {
  if (Test-Path $targetRepo) {
    if (-not $Force) {
      throw (
        "SeedVR2 runtime directory exists but is not a git repository: $targetRepo. " +
        "Delete it or rerun with -Force."
      )
    }
    Remove-Item -Path $targetRepo -Recurse -Force
  }
  Write-Host "[download] Cloning SeedVR2 runtime repository..."
  Invoke-Checked -Executable "git" -Arguments @(
    "clone",
    "--depth",
    "1",
    "--branch",
    $Revision,
    $repoUrl,
    $targetRepo
  )
} else {
  Write-Host "[update] Refreshing SeedVR2 runtime repository..."
  Invoke-Checked -Executable "git" -Arguments @("-C", $targetRepo, "fetch", "--depth", "1", "origin", $Revision)
  Invoke-Checked -Executable "git" -Arguments @("-C", $targetRepo, "checkout", "--force", "FETCH_HEAD")
}

$runtimeScript = Join-Path $targetRepo "inference_cli.py"
if (-not (Test-Path $runtimeScript)) {
  throw "SeedVR2 runtime fetch completed but inference_cli.py is missing: $runtimeScript"
}

Apply-AllocatorCompatPatch -RuntimeScriptPath $runtimeScript

Write-Host "[ok] SeedVR2 runtime ready:"
Write-Host "  $runtimeScript"
