param(
  [string]$BindHost = "127.0.0.1",
  [int]$Port = 37717,
  [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$rootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$env:PYTHONNOUSERSITE = "1"
$env:JUSTRAYZIST_ROOT = $rootDir

Push-Location $rootDir
try {
  & $PythonExe -m app.cli.main serve --host $BindHost --port $Port
  exit $LASTEXITCODE
} finally {
  Pop-Location
}
