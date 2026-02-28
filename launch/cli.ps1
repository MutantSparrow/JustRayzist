param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ArgsList
)

$ErrorActionPreference = "Stop"
$rootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$cliExe = Join-Path $rootDir "bin\cli\justrayzist-cli.exe"
$pythonExe = if ($env:JUSTRAYZIST_PYTHON) { $env:JUSTRAYZIST_PYTHON } else { "python" }

Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$env:PYTHONNOUSERSITE = "1"
$env:JUSTRAYZIST_ROOT = $rootDir

if (Test-Path $cliExe) {
  & $cliExe @ArgsList
  exit $LASTEXITCODE
}

Push-Location $rootDir
try {
  & $pythonExe -m app.cli.main @ArgsList
  exit $LASTEXITCODE
} finally {
  Pop-Location
}
