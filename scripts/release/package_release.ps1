param(
  [ValidateSet("cu126", "cu128")]
  [string]$Lane = "cu128",
  [string]$Version = "v0.0.0",
  [string]$OutputRoot = "dist",
  [switch]$NoZip,
  [switch]$Clean
)

$ErrorActionPreference = "Stop"

function Invoke-RobocopySafe {
  param(
    [Parameter(Mandatory = $true)][string]$Source,
    [Parameter(Mandatory = $true)][string]$Destination,
    [string[]]$ExtraArgs = @()
  )

  if (-not (Test-Path $Source)) {
    throw "Source path not found: $Source"
  }
  New-Item -ItemType Directory -Path $Destination -Force | Out-Null
  $args = @($Source, $Destination, "/E", "/R:1", "/W:1", "/NFL", "/NDL", "/NJH", "/NJS", "/NP") + $ExtraArgs
  & robocopy @args | Out-Null
  if ($LASTEXITCODE -ge 8) {
    throw "Robocopy failed for '$Source' -> '$Destination' (exit code $LASTEXITCODE)."
  }
}

function Copy-TrackedTreeSubset {
  param(
    [Parameter(Mandatory = $true)][string]$RootDir,
    [Parameter(Mandatory = $true)][string]$RelativeRoot,
    [Parameter(Mandatory = $true)][string]$DestinationRoot
  )

  $tracked = @(& git -C $RootDir ls-files -- $RelativeRoot 2>$null)
  foreach ($relativePath in $tracked) {
    if ([string]::IsNullOrWhiteSpace($relativePath)) {
      continue
    }
    if ($relativePath -match '\.(safetensors|gguf|pth)$') {
      continue
    }

    $sourcePath = Join-Path $RootDir $relativePath
    if (-not (Test-Path $sourcePath -PathType Leaf)) {
      continue
    }

    $destPath = Join-Path $DestinationRoot $relativePath
    $destDir = Split-Path -Parent $destPath
    if ($destDir) {
      New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    }
    Copy-Item -Path $sourcePath -Destination $destPath -Force
  }
}

function Copy-CommonReleaseContent {
  param(
    [Parameter(Mandatory = $true)][string]$RootDir,
    [Parameter(Mandatory = $true)][string]$ReleaseDir
  )

  Invoke-RobocopySafe -Source (Join-Path $RootDir "app") -Destination (Join-Path $ReleaseDir "app")
  Invoke-RobocopySafe -Source (Join-Path $RootDir "docs") -Destination (Join-Path $ReleaseDir "docs") -ExtraArgs @("/XF", "metadata_config.html")
  Invoke-RobocopySafe -Source (Join-Path $RootDir "img") -Destination (Join-Path $ReleaseDir "img")
  Invoke-RobocopySafe -Source (Join-Path $RootDir "launch") -Destination (Join-Path $ReleaseDir "launch")
  Invoke-RobocopySafe -Source (Join-Path $RootDir "readme_images") -Destination (Join-Path $ReleaseDir "readme_images")
  Invoke-RobocopySafe -Source (Join-Path $RootDir "requirements") -Destination (Join-Path $ReleaseDir "requirements")
  Invoke-RobocopySafe -Source (Join-Path $RootDir "scripts") -Destination (Join-Path $ReleaseDir "scripts")
  Copy-TrackedTreeSubset -RootDir $RootDir -RelativeRoot "models/packs" -DestinationRoot $ReleaseDir
  Copy-TrackedTreeSubset -RootDir $RootDir -RelativeRoot "models/upscaler" -DestinationRoot $ReleaseDir

  Copy-Item (Join-Path $RootDir "StartWeb.bat") -Destination (Join-Path $ReleaseDir "StartWeb.bat") -Force
  Copy-Item (Join-Path $RootDir "RunMeFirst.bat") -Destination (Join-Path $ReleaseDir "RunMeFirst.bat") -Force
  Copy-Item (Join-Path $RootDir "UpdateApp.bat") -Destination (Join-Path $ReleaseDir "UpdateApp.bat") -Force
  Copy-Item (Join-Path $RootDir "README.md") -Destination (Join-Path $ReleaseDir "README.md") -Force
  Copy-Item (Join-Path $RootDir "pyproject.toml") -Destination (Join-Path $ReleaseDir "pyproject.toml") -Force
  Copy-Item (Join-Path $RootDir "LICENSE") -Destination (Join-Path $ReleaseDir "LICENSE") -Force
}

$rootDir = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
$outputRootAbs = [System.IO.Path]::GetFullPath((Join-Path $rootDir $OutputRoot))
$mode = "bootstrap"
$releaseName = "JustRayzist_win64_${Lane}_${Version}_${mode}"
$releaseDir = Join-Path $outputRootAbs $releaseName

if ($Clean -and (Test-Path $releaseDir)) {
  Remove-Item $releaseDir -Recurse -Force
}
New-Item -ItemType Directory -Path $releaseDir -Force | Out-Null

Copy-CommonReleaseContent -RootDir $rootDir -ReleaseDir $releaseDir

Set-Content -Path (Join-Path $releaseDir "release_lane.txt") -Value $Lane -Encoding ascii
([ordered]@{
  app_name = "JustRayzist"
  version = $Version
  lane = $Lane
  mode = $mode
  generated_at = (Get-Date).ToString("s")
} | ConvertTo-Json) | Set-Content -Path (Join-Path $releaseDir "release_manifest.json") -Encoding ascii
Set-Content -Path (Join-Path $releaseDir "cuda_baseline.json") -Value @"
{
  "generated_at": "$(Get-Date -Format s)",
  "mode": "$mode",
  "lane": "$Lane",
  "driver_floors": {
    "cu126": "561.17",
    "cu128": "572.61"
  },
  "gpu_guidance": {
    "20xx_30xx_40xx": {
      "preferred_lane": "cu128",
      "fallback_lane": "cu126"
    },
    "50xx": {
      "required_lane": "cu128"
    }
  }
}
"@ -Encoding ascii

New-Item -ItemType Directory -Path (Join-Path $releaseDir "outputs") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $releaseDir "data") -Force | Out-Null
if (-not (Test-Path (Join-Path $releaseDir "data\\.gitkeep"))) {
  New-Item -ItemType File -Path (Join-Path $releaseDir "data\\.gitkeep") -Force | Out-Null
}

if (-not (Test-Path (Join-Path $releaseDir "models\\upscaler\\README.txt"))) {
  Set-Content -Path (Join-Path $releaseDir "models\\upscaler\\README.txt") -Value @"
No upscaler checkpoints are bundled in release artifacts.
Use RunMeFirst.bat or scripts\fetch_model_assets.ps1 to download default assets from Hugging Face using HF CLI + XET (checksum-verified).
You may also place a custom local .pth file in this folder.
"@ -Encoding ascii
}

if (-not $NoZip) {
  $zipPath = Join-Path $outputRootAbs "$releaseName.zip"
  if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
  }
  Compress-Archive -Path (Join-Path $releaseDir "*") -DestinationPath $zipPath -CompressionLevel Optimal
}

Write-Host ""
Write-Host "Release package created:"
Write-Host "  Lane: $Lane"
Write-Host "  $releaseDir"
if (-not $NoZip) {
  Write-Host "  $outputRootAbs\\$releaseName.zip"
}
