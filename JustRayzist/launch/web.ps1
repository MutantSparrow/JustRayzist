param(
  [string]$BindHost = "127.0.0.1",
  [int]$Port = 37717,
  [string]$Profile = "balanced",
  [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$rootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$webExe = Join-Path $rootDir "bin\web\justrayzist-web.exe"

Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$env:PYTHONNOUSERSITE = "1"
$env:JUSTRAYZIST_ROOT = $rootDir

if (Test-Path $webExe) {
  & $webExe --host $BindHost --port $Port --profile $Profile
  exit $LASTEXITCODE
}

Push-Location $rootDir
try {
  & $PythonExe -m app.cli.main serve --host $BindHost --port $Port --profile $Profile
  exit $LASTEXITCODE
} finally {
  Pop-Location
}

