param(
  [ValidateSet("cu126", "cu128")]
  [string]$Lane = "cu128",
  [string]$Version = "v0.0.0",
  [ValidateSet("bootstrap", "bundled")]
  [string]$Mode = "bootstrap",
  [string]$PythonExe = "python",
  [string]$OutputRoot = "dist",
  [string]$BuildRoot = "dist\\pyinstaller",
  [switch]$UseActivePython,
  [switch]$SkipDependencyInstall,
  [switch]$SkipBuild,
  [switch]$IncludeCliBinary,
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

function Invoke-BuildOnedir {
  param(
    [Parameter(Mandatory = $true)][string]$BuildScriptPath
  )

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

  & powershell -NoProfile -ExecutionPolicy Bypass -File $BuildScriptPath @buildArgs
  if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed with exit code $LASTEXITCODE."
  }
}

function Copy-CommonReleaseContent {
  param(
    [Parameter(Mandatory = $true)][string]$RootDir,
    [Parameter(Mandatory = $true)][string]$ReleaseDir
  )

  Invoke-RobocopySafe -Source (Join-Path $RootDir "app") -Destination (Join-Path $ReleaseDir "app")
  Invoke-RobocopySafe -Source (Join-Path $RootDir "docs") -Destination (Join-Path $ReleaseDir "docs")
  Invoke-RobocopySafe -Source (Join-Path $RootDir "img") -Destination (Join-Path $ReleaseDir "img")
  Invoke-RobocopySafe -Source (Join-Path $RootDir "launch") -Destination (Join-Path $ReleaseDir "launch")
  Invoke-RobocopySafe -Source (Join-Path $RootDir "requirements") -Destination (Join-Path $ReleaseDir "requirements")
  Invoke-RobocopySafe -Source (Join-Path $RootDir "scripts") -Destination (Join-Path $ReleaseDir "scripts")
  Invoke-RobocopySafe -Source (Join-Path $RootDir "models\\packs") -Destination (Join-Path $ReleaseDir "models\\packs") -ExtraArgs @(
    "/XF", "*.safetensors", "*.gguf", "*.pth"
  )
  Invoke-RobocopySafe -Source (Join-Path $RootDir "models\\upscaler") -Destination (Join-Path $ReleaseDir "models\\upscaler") -ExtraArgs @(
    "/XF", "*.safetensors", "*.gguf", "*.pth"
  )

  Copy-Item (Join-Path $RootDir "StartWeb.bat") -Destination (Join-Path $ReleaseDir "StartWeb.bat") -Force
  Copy-Item (Join-Path $RootDir "RunMeFirst.bat") -Destination (Join-Path $ReleaseDir "RunMeFirst.bat") -Force
  Copy-Item (Join-Path $RootDir "README.md") -Destination (Join-Path $ReleaseDir "README.md") -Force
  Copy-Item (Join-Path $RootDir "pyproject.toml") -Destination (Join-Path $ReleaseDir "pyproject.toml") -Force
}

$rootDir = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
$outputRootAbs = [System.IO.Path]::GetFullPath((Join-Path $rootDir $OutputRoot))
$buildRootAbs = [System.IO.Path]::GetFullPath((Join-Path $rootDir $BuildRoot))
$buildLaneDir = Join-Path $buildRootAbs $Lane
$buildScript = Join-Path $rootDir "scripts\\pyinstaller\\build_onedir.ps1"

$releaseName = "JustRayzist_win64_${Lane}_${Version}_${Mode}"
$releaseDir = Join-Path $outputRootAbs $releaseName

if ($Clean -and (Test-Path $releaseDir)) {
  Remove-Item $releaseDir -Recurse -Force
}
New-Item -ItemType Directory -Path $releaseDir -Force | Out-Null

if ($Mode -eq "bundled") {
  if (-not $SkipBuild) {
    Invoke-BuildOnedir -BuildScriptPath $buildScript
  }

  $webSource = Join-Path $buildLaneDir "justrayzist-web"
  if (-not (Test-Path (Join-Path $webSource "justrayzist-web.exe"))) {
    throw "Missing web executable at $webSource"
  }

  Invoke-RobocopySafe -Source $webSource -Destination (Join-Path $releaseDir "bin\\web")

  if ($IncludeCliBinary) {
    $cliSource = Join-Path $buildLaneDir "justrayzist-cli"
    if (-not (Test-Path (Join-Path $cliSource "justrayzist-cli.exe"))) {
      throw "Missing CLI executable at $cliSource"
    }
    Invoke-RobocopySafe -Source $cliSource -Destination (Join-Path $releaseDir "bin\\cli")
  }
}

Copy-CommonReleaseContent -RootDir $rootDir -ReleaseDir $releaseDir

Set-Content -Path (Join-Path $releaseDir "release_lane.txt") -Value $Lane -Encoding ascii
Set-Content -Path (Join-Path $releaseDir "cuda_baseline.json") -Value @"
{
  "generated_at": "$(Get-Date -Format s)",
  "mode": "$Mode",
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
Use RunMeFirst.bat or scripts\fetch_model_assets.ps1 to download default assets from Hugging Face (checksum-verified).
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
Write-Host "  Mode: $Mode"
Write-Host "  Lane: $Lane"
if ($Mode -eq "bundled") {
  Write-Host ("  Bundled CLI: {0}" -f $IncludeCliBinary)
}
Write-Host "  $releaseDir"
if (-not $NoZip) {
  Write-Host "  $outputRootAbs\\$releaseName.zip"
}
