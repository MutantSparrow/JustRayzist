param(
  [string]$OutputDir = "..\\dist\\JustRayzistPortable",
  [string]$ZipPath = "",
  [string]$PythonExe = "",
  [switch]$SkipModels,
  [switch]$NoClean
)

$ErrorActionPreference = "Stop"

$rootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$buildScript = Join-Path $PSScriptRoot "build_portable.ps1"

$buildArgs = @("-OutputDir", $OutputDir)
if (-not $NoClean) {
  $buildArgs += "-Clean"
}
if ($PythonExe) {
  $buildArgs += @("-PythonExe", $PythonExe)
}
if ($SkipModels) {
  $buildArgs += "-SkipModels"
}

Write-Host "Building portable bundle..."
& powershell -NoProfile -ExecutionPolicy Bypass -File $buildScript @buildArgs

$bundleDir = [System.IO.Path]::GetFullPath((Join-Path $rootDir $OutputDir))
if (-not (Test-Path $bundleDir)) {
  throw "Portable bundle directory was not created: $bundleDir"
}

$roadmapPath = Join-Path $bundleDir "JustRayzist.md"
if (Test-Path $roadmapPath) {
  Remove-Item $roadmapPath -Force
}

if ([string]::IsNullOrWhiteSpace($ZipPath)) {
  $zipName = [System.IO.Path]::GetFileName($bundleDir) + ".zip"
  $ZipPath = Join-Path (Split-Path -Parent $bundleDir) $zipName
}
$zipFullPath = [System.IO.Path]::GetFullPath((Join-Path $rootDir $ZipPath))

if (Test-Path $zipFullPath) {
  Remove-Item $zipFullPath -Force
}

Write-Host "Creating release archive..."
Compress-Archive -Path (Join-Path $bundleDir "*") -DestinationPath $zipFullPath -CompressionLevel Optimal

Write-Host ""
Write-Host "Release archive created:"
Write-Host "  $zipFullPath"
