param(
  [string]$RepoOwner = "MutantSparrow",
  [string]$RepoName = "JustRayzist",
  [switch]$Force,
  [switch]$RunRepair,
  [switch]$ApplyFromExtracted,
  [switch]$CompatibilityMode,
  [string]$TargetRoot,
  [string]$ExtractedRoot
)

$ErrorActionPreference = "Stop"

function Write-Section {
  param([Parameter(Mandatory = $true)][string]$Message)
  Write-Host ""
  Write-Host $Message -ForegroundColor Cyan
}

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

function Get-NormalizedVersion {
  param([AllowNull()][string]$Value)

  if ([string]::IsNullOrWhiteSpace($Value)) {
    return $null
  }

  $trimmed = $Value.Trim() -replace '^[vV]', ''
  $parts = @($trimmed -split '\.')
  if ($parts.Count -eq 0 -or $parts.Count -gt 4) {
    return $null
  }

  foreach ($part in $parts) {
    if ($part -notmatch '^\d+$') {
      return $null
    }
  }

  while ($parts.Count -lt 4) {
    $parts += "0"
  }

  try {
    return [version]($parts -join '.')
  } catch {
    return $null
  }
}

function Get-ReleaseManifest {
  param([Parameter(Mandatory = $true)][string]$RootDir)

  $manifestPath = Join-Path $RootDir "release_manifest.json"
  if (-not (Test-Path $manifestPath)) {
    return $null
  }

  try {
    return Get-Content -Path $manifestPath -Raw | ConvertFrom-Json
  } catch {
    throw "Failed to read release manifest at $manifestPath. $_"
  }
}

function Get-LocalInstallInfo {
  param([Parameter(Mandatory = $true)][string]$RootDir)

  $manifest = Get-ReleaseManifest -RootDir $RootDir
  $lanePath = Join-Path $RootDir "release_lane.txt"
  if ($manifest -and $manifest.lane) {
    $lane = [string]$manifest.lane
  } elseif (Test-Path $lanePath) {
    $lane = (Get-Content -Path $lanePath -Raw).Trim()
  } else {
    $lane = $null
  }

  if ([string]::IsNullOrWhiteSpace($lane)) {
    throw "Unable to determine the installed release lane. Expected release_manifest.json or release_lane.txt."
  }

  if ($manifest -and $manifest.mode) {
    $mode = [string]$manifest.mode
  } else {
    $mode = "bootstrap"
  }

  $version = $null
  if ($manifest -and $manifest.version) {
    $version = [string]$manifest.version
  }

  return [pscustomobject]@{
    Manifest = $manifest
    Lane = $lane.Trim().ToLowerInvariant()
    Mode = $mode.Trim().ToLowerInvariant()
    Version = $version
    NormalizedVersion = Get-NormalizedVersion -Value $version
  }
}

function Get-LatestReleaseInfo {
  param(
    [Parameter(Mandatory = $true)][string]$RepoOwner,
    [Parameter(Mandatory = $true)][string]$RepoName
  )

  $uri = "https://api.github.com/repos/$RepoOwner/$RepoName/releases/latest"
  $headers = @{
    Accept = "application/vnd.github+json"
    "User-Agent" = "JustRayzist-Updater"
  }

  try {
    return Invoke-RestMethod -Uri $uri -Headers $headers -Method Get
  } catch {
    throw "Failed to query the latest GitHub release from $uri. $_"
  }
}

function Select-ReleaseAsset {
  param(
    [Parameter(Mandatory = $true)]$Release,
    [Parameter(Mandatory = $true)][string]$Lane,
    [Parameter(Mandatory = $true)][string]$Mode
  )

  $tag = [regex]::Escape([string]$Release.tag_name)
  $lanePattern = [regex]::Escape($Lane)
  $modePattern = [regex]::Escape($Mode)
  $exactPattern = "^JustRayzist_win64_${lanePattern}_${tag}_${modePattern}\.zip$"
  $fallbackPattern = "^JustRayzist_win64_${lanePattern}_.+_${modePattern}\.zip$"

  $assets = @($Release.assets)
  $asset = $assets | Where-Object { $_.name -match $exactPattern } | Select-Object -First 1
  if (-not $asset) {
    $asset = $assets | Where-Object { $_.name -match $fallbackPattern } | Select-Object -First 1
  }
  return $asset
}

function Test-ShouldUpdate {
  param(
    [AllowNull()][version]$LocalVersion,
    [AllowNull()][version]$RemoteVersion,
    [string]$LocalVersionText,
    [string]$RemoteVersionText,
    [switch]$Force
  )

  if ($Force) {
    return $true
  }

  if ($LocalVersion -and $RemoteVersion) {
    return $LocalVersion -lt $RemoteVersion
  }

  if ([string]::IsNullOrWhiteSpace($LocalVersionText)) {
    return $true
  }

  return $LocalVersionText.Trim() -ne $RemoteVersionText.Trim()
}

function Replace-ManagedDirectory {
  param(
    [Parameter(Mandatory = $true)][string]$SourceRoot,
    [Parameter(Mandatory = $true)][string]$TargetRoot,
    [Parameter(Mandatory = $true)][string]$RelativePath
  )

  $sourcePath = Join-Path $SourceRoot $RelativePath
  if (-not (Test-Path $sourcePath)) {
    return
  }

  $targetPath = Join-Path $TargetRoot $RelativePath
  if (Test-Path $targetPath) {
    Remove-Item -Path $targetPath -Recurse -Force
  }
  Invoke-RobocopySafe -Source $sourcePath -Destination $targetPath
}

function Merge-ManagedDirectory {
  param(
    [Parameter(Mandatory = $true)][string]$SourceRoot,
    [Parameter(Mandatory = $true)][string]$TargetRoot,
    [Parameter(Mandatory = $true)][string]$RelativePath,
    [string[]]$ExtraArgs = @()
  )

  $sourcePath = Join-Path $SourceRoot $RelativePath
  if (-not (Test-Path $sourcePath)) {
    return
  }

  $targetPath = Join-Path $TargetRoot $RelativePath
  Invoke-RobocopySafe -Source $sourcePath -Destination $targetPath -ExtraArgs $ExtraArgs
}

function Copy-ManagedFile {
  param(
    [Parameter(Mandatory = $true)][string]$SourceRoot,
    [Parameter(Mandatory = $true)][string]$TargetRoot,
    [Parameter(Mandatory = $true)][string]$RelativePath
  )

  $sourcePath = Join-Path $SourceRoot $RelativePath
  if (-not (Test-Path $sourcePath)) {
    return
  }

  $targetPath = Join-Path $TargetRoot $RelativePath
  $targetDir = Split-Path -Parent $targetPath
  if ($targetDir) {
    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
  }
  Copy-Item -Path $sourcePath -Destination $targetPath -Force
}

function Write-ReleaseManifest {
  param(
    [Parameter(Mandatory = $true)][string]$TargetRoot,
    [Parameter(Mandatory = $true)][string]$Version,
    [Parameter(Mandatory = $true)][string]$Lane,
    [Parameter(Mandatory = $true)][string]$Mode
  )

  $manifestPath = Join-Path $TargetRoot "release_manifest.json"
  ([ordered]@{
    app_name = "JustRayzist"
    version = $Version
    lane = $Lane
    mode = $Mode
    generated_at = (Get-Date).ToString("s")
  } | ConvertTo-Json) | Set-Content -Path $manifestPath -Encoding ascii
}

function Invoke-ApplyUpdate {
  param(
    [Parameter(Mandatory = $true)][string]$TargetRoot,
    [Parameter(Mandatory = $true)][string]$ExtractedRoot,
    [switch]$CompatibilityMode,
    [string]$ReleaseVersion,
    [string]$ReleaseLane,
    [string]$ReleaseMode
  )

  $managedReplaceDirs = @(
    "app",
    "docs",
    "img",
    "launch",
    "readme_images",
    "requirements",
    "bin"
  )

  if (-not $CompatibilityMode) {
    $managedReplaceDirs += "scripts"
  }

  $managedMergeDirs = @(
    @{ Path = "models\\packs"; ExtraArgs = @("/XF", "*.safetensors", "*.gguf", "*.pth") },
    @{ Path = "models\\upscaler"; ExtraArgs = @("/XF", "*.safetensors", "*.gguf", "*.pth") }
  )

  if ($CompatibilityMode) {
    $managedMergeDirs += @{ Path = "scripts"; ExtraArgs = @() }
  }

  $managedFiles = @(
    "StartWeb.bat",
    "RunMeFirst.bat",
    "UpdateApp.bat",
    "README.md",
    "pyproject.toml",
    "LICENSE",
    "cuda_baseline.json",
    "release_manifest.json"
  )

  $requiredReleasePaths = @(
    (Join-Path $ExtractedRoot "app"),
    (Join-Path $ExtractedRoot "StartWeb.bat")
  )

  foreach ($requiredPath in $requiredReleasePaths) {
    if (-not (Test-Path $requiredPath)) {
      throw "Extracted release is missing required content: $requiredPath"
    }
  }

  if ($CompatibilityMode) {
    Write-Section "Applying update (compatibility mode)"
    Write-Host "Older release layout detected; preserving the current updater files while overlaying app content."
  } else {
    Write-Section "Applying update"
  }
  foreach ($dir in $managedReplaceDirs) {
    Replace-ManagedDirectory -SourceRoot $ExtractedRoot -TargetRoot $TargetRoot -RelativePath $dir
  }

  foreach ($entry in $managedMergeDirs) {
    Merge-ManagedDirectory -SourceRoot $ExtractedRoot -TargetRoot $TargetRoot -RelativePath $entry.Path -ExtraArgs $entry.ExtraArgs
  }

  foreach ($file in $managedFiles) {
    Copy-ManagedFile -SourceRoot $ExtractedRoot -TargetRoot $TargetRoot -RelativePath $file
  }

  $targetLanePath = Join-Path $TargetRoot "release_lane.txt"
  if (-not (Test-Path $targetLanePath)) {
    Copy-ManagedFile -SourceRoot $ExtractedRoot -TargetRoot $TargetRoot -RelativePath "release_lane.txt"
  }

  if ($CompatibilityMode -and -not [string]::IsNullOrWhiteSpace($ReleaseVersion)) {
    Write-ReleaseManifest -TargetRoot $TargetRoot -Version $ReleaseVersion -Lane $ReleaseLane -Mode $ReleaseMode
  } elseif ((-not (Test-Path (Join-Path $ExtractedRoot "release_manifest.json"))) -and -not [string]::IsNullOrWhiteSpace($ReleaseVersion)) {
    Write-ReleaseManifest -TargetRoot $TargetRoot -Version $ReleaseVersion -Lane $ReleaseLane -Mode $ReleaseMode
  }

  New-Item -ItemType Directory -Path (Join-Path $TargetRoot "outputs") -Force | Out-Null
  New-Item -ItemType Directory -Path (Join-Path $TargetRoot "data") -Force | Out-Null
}

if ($ApplyFromExtracted) {
  if ([string]::IsNullOrWhiteSpace($TargetRoot) -or [string]::IsNullOrWhiteSpace($ExtractedRoot)) {
    throw "Apply mode requires -TargetRoot and -ExtractedRoot."
  }

  $targetRootAbs = [System.IO.Path]::GetFullPath($TargetRoot)
  $extractedRootAbs = [System.IO.Path]::GetFullPath($ExtractedRoot)
  Invoke-ApplyUpdate -TargetRoot $targetRootAbs -ExtractedRoot $extractedRootAbs -CompatibilityMode:$CompatibilityMode
  Write-Host ""
  Write-Host "Update applied to $targetRootAbs" -ForegroundColor Green
  exit 0
}

$rootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (Test-Path (Join-Path $rootDir ".git")) {
  Write-Host "This updater is for packaged installs, not git worktrees." -ForegroundColor Yellow
  Write-Host "Use git pull in the source repository instead." -ForegroundColor Yellow
  exit 1
}

$install = Get-LocalInstallInfo -RootDir $rootDir
$release = Get-LatestReleaseInfo -RepoOwner $RepoOwner -RepoName $RepoName
$remoteVersionText = [string]$release.tag_name
$remoteVersion = Get-NormalizedVersion -Value $remoteVersionText
$asset = Select-ReleaseAsset -Release $release -Lane $install.Lane -Mode $install.Mode
if (-not $asset) {
  throw "No release asset found for lane '$($install.Lane)' and mode '$($install.Mode)' in release '$remoteVersionText'."
}

$shouldUpdate = Test-ShouldUpdate `
  -LocalVersion $install.NormalizedVersion `
  -RemoteVersion $remoteVersion `
  -LocalVersionText $install.Version `
  -RemoteVersionText $remoteVersionText `
  -Force:$Force

if (-not $shouldUpdate) {
  Write-Host "Already up to date." -ForegroundColor Green
  if ([string]::IsNullOrWhiteSpace($install.Version)) {
    Write-Host "Installed version: unknown"
  } else {
    Write-Host "Installed version: $($install.Version)"
  }
  Write-Host "Latest release:    $remoteVersionText"
  exit 0
}

Write-Section "Checking latest release"
if ([string]::IsNullOrWhiteSpace($install.Version)) {
  Write-Host "Installed version: unknown"
} else {
  Write-Host "Installed version: $($install.Version)"
}
Write-Host "Latest release:    $remoteVersionText"
Write-Host "Selected asset:    $($asset.name)"

$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("JustRayzist-update-" + [guid]::NewGuid().ToString("N"))
$downloadPath = Join-Path $tempRoot "release.zip"
$extractRoot = Join-Path $tempRoot "extracted"

try {
  New-Item -ItemType Directory -Path $extractRoot -Force | Out-Null

  Write-Section "Downloading update"
  Write-Host $asset.browser_download_url
  Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $downloadPath -Headers @{ "User-Agent" = "JustRayzist-Updater" }

  Write-Section "Extracting update"
  Expand-Archive -Path $downloadPath -DestinationPath $extractRoot -Force

  $applyScript = Join-Path $extractRoot "scripts\update_release.ps1"
  if (Test-Path $applyScript) {
    $applyArgs = @(
      "-NoProfile",
      "-ExecutionPolicy", "Bypass",
      "-File", $applyScript,
      "-ApplyFromExtracted",
      "-TargetRoot", $rootDir,
      "-ExtractedRoot", $extractRoot
    )

    Write-Section "Applying update"
    $applyProcess = Start-Process -FilePath "powershell" -ArgumentList $applyArgs -Wait -PassThru -NoNewWindow
    if ($applyProcess.ExitCode -ne 0) {
      throw "Update apply step failed with exit code $($applyProcess.ExitCode)."
    }
  } else {
    Write-Section "Applying compatibility update"
    Invoke-ApplyUpdate `
      -TargetRoot $rootDir `
      -ExtractedRoot $extractRoot `
      -CompatibilityMode `
      -ReleaseVersion $remoteVersionText `
      -ReleaseLane $install.Lane `
      -ReleaseMode $install.Mode
  }

  if ($RunRepair) {
    $repairBat = Join-Path $rootDir "RunMeFirst.bat"
    if (Test-Path $repairBat) {
      Write-Section "Running repair"
      $repairProcess = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "`"$repairBat`"" -Wait -PassThru -NoNewWindow
      if ($repairProcess.ExitCode -ne 0) {
        throw "RunMeFirst.bat exited with code $($repairProcess.ExitCode)."
      }
    }
  }
} finally {
  if (Test-Path $tempRoot) {
    Remove-Item -Path $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
  }
}

Write-Host ""
Write-Host "Update complete." -ForegroundColor Green
Write-Host "Your models, outputs, data, and .venv were preserved."
Write-Host "Launch the app with StartWeb.bat."
exit 0
