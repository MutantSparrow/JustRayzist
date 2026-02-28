param(
  [ValidateSet("cu126", "cu128")]
  [string]$Lane = "cu128",
  [string]$Version = "v0.0.0",
  [string]$PythonExe = "python",
  [string]$OutputRoot = "dist",
  [string]$BuildRoot = "dist\\pyinstaller",
  [switch]$UseActivePython,
  [switch]$SkipDependencyInstall,
  [switch]$SkipBuild,
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

$rootDir = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
$outputRootAbs = [System.IO.Path]::GetFullPath((Join-Path $rootDir $OutputRoot))
$buildRootAbs = [System.IO.Path]::GetFullPath((Join-Path $rootDir $BuildRoot))
$buildLaneDir = Join-Path $buildRootAbs $Lane
$buildScript = Join-Path $rootDir "scripts\\pyinstaller\\build_onedir.ps1"

if (-not $SkipBuild) {
  $buildArgs = @("-Lane", $Lane, "-PythonExe", $PythonExe)
  if ($UseActivePython) {
    $buildArgs += "-UseActivePython"
  }
  if ($SkipDependencyInstall) {
    $buildArgs += "-SkipDependencyInstall"
  }
  if ($Clean) {
    $buildArgs += "-Clean"
  }
  & powershell -NoProfile -ExecutionPolicy Bypass -File $buildScript @buildArgs
  if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed with exit code $LASTEXITCODE."
  }
}

$webSource = Join-Path $buildLaneDir "justrayzist-web"
$cliSource = Join-Path $buildLaneDir "justrayzist-cli"
if (-not (Test-Path (Join-Path $webSource "justrayzist-web.exe"))) {
  throw "Missing web executable at $webSource"
}
if (-not (Test-Path (Join-Path $cliSource "justrayzist-cli.exe"))) {
  throw "Missing CLI executable at $cliSource"
}

$releaseName = "JustRayzist_win64_${Lane}_${Version}"
$releaseDir = Join-Path $outputRootAbs $releaseName
if ($Clean -and (Test-Path $releaseDir)) {
  Remove-Item $releaseDir -Recurse -Force
}

New-Item -ItemType Directory -Path $releaseDir -Force | Out-Null
Invoke-RobocopySafe -Source $webSource -Destination (Join-Path $releaseDir "bin\\web")
Invoke-RobocopySafe -Source $cliSource -Destination (Join-Path $releaseDir "bin\\cli")
Invoke-RobocopySafe -Source (Join-Path $rootDir "app\\ui") -Destination (Join-Path $releaseDir "app\\ui")
Invoke-RobocopySafe -Source (Join-Path $rootDir "img") -Destination (Join-Path $releaseDir "img")
Invoke-RobocopySafe -Source (Join-Path $rootDir "docs") -Destination (Join-Path $releaseDir "docs")
Invoke-RobocopySafe -Source (Join-Path $rootDir "launch") -Destination (Join-Path $releaseDir "launch")
Invoke-RobocopySafe -Source (Join-Path $rootDir "models\\packs") -Destination (Join-Path $releaseDir "models\\packs") -ExtraArgs @(
  "/XF", "*.safetensors", "*.gguf", "*.pth"
)

New-Item -ItemType Directory -Path (Join-Path $releaseDir "scripts") -Force | Out-Null
Copy-Item (Join-Path $rootDir "StartWeb.bat") -Destination (Join-Path $releaseDir "StartWeb.bat") -Force
Copy-Item (Join-Path $rootDir "scripts\\fetch_model_assets.ps1") -Destination (Join-Path $releaseDir "scripts\\fetch_model_assets.ps1") -Force
Copy-Item (Join-Path $rootDir "README.md") -Destination (Join-Path $releaseDir "README.md") -Force

New-Item -ItemType Directory -Path (Join-Path $releaseDir "models\\upscaler") -Force | Out-Null
Set-Content -Path (Join-Path $releaseDir "models\\upscaler\\README.txt") -Value @"
No upscaler checkpoints are bundled in release artifacts.
Use StartWeb.bat or scripts\fetch_model_assets.ps1 to download default assets from Hugging Face.
You may also place a custom local .pth file in this folder.
"@ -Encoding ascii

New-Item -ItemType Directory -Path (Join-Path $releaseDir "outputs") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $releaseDir "data") -Force | Out-Null
if (-not (Test-Path (Join-Path $releaseDir "data\\.gitkeep"))) {
  New-Item -ItemType File -Path (Join-Path $releaseDir "data\\.gitkeep") -Force | Out-Null
}

Set-Content -Path (Join-Path $releaseDir "release_lane.txt") -Value $Lane -Encoding ascii
Set-Content -Path (Join-Path $releaseDir "cuda_baseline.json") -Value @"
{
  "generated_at": "$(Get-Date -Format s)",
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

if (-not $NoZip) {
  $zipPath = Join-Path $outputRootAbs "$releaseName.zip"
  if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
  }
  Compress-Archive -Path (Join-Path $releaseDir "*") -DestinationPath $zipPath -CompressionLevel Optimal
}

Write-Host ""
Write-Host "Release package created:"
Write-Host "  $releaseDir"
if (-not $NoZip) {
  Write-Host "  $outputRootAbs\\$releaseName.zip"
}
