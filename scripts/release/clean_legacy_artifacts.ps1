param(
  [string]$DistRoot = "dist"
)

$ErrorActionPreference = "Stop"

$rootDir = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
$distDir = [System.IO.Path]::GetFullPath((Join-Path $rootDir $DistRoot))

if (-not (Test-Path $distDir)) {
  Write-Host "Nothing to clean. Dist directory not found: $distDir"
  exit 0
}

$patterns = @(
  "JustRayzistPortable*"
)

foreach ($pattern in $patterns) {
  Get-ChildItem -Path $distDir -Filter $pattern -Force -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_.PSIsContainer) {
      Remove-Item $_.FullName -Recurse -Force
    } else {
      Remove-Item $_.FullName -Force
    }
    Write-Host "Removed: $($_.FullName)"
  }
}

Write-Host "Legacy artifact cleanup complete."
