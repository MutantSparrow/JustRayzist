param()

$ErrorActionPreference = "Stop"
$rootDir = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path

$requiredPaths = @(
  "pyproject.toml",
  "README.md",
  "StartWeb.bat",
  "scripts\\bootstrap_env.ps1",
  "scripts\\fetch_model_assets.ps1",
  "scripts\\pyinstaller\\build_onedir.ps1",
  "scripts\\release\\package_release.ps1",
  "requirements\\runtime-lock.txt",
  "requirements\\dev-lock.txt",
  "requirements\\build-lock.txt",
  "requirements\\torch-cu126.txt",
  "requirements\\torch-cu128.txt",
  "models\\packs\\modelpack.yaml.example",
  "models\\upscaler\\README.md"
)

$errors = New-Object System.Collections.Generic.List[string]
$warnings = New-Object System.Collections.Generic.List[string]

foreach ($relativePath in $requiredPaths) {
  $absolutePath = Join-Path $rootDir $relativePath
  if (-not (Test-Path $absolutePath)) {
    $errors.Add("Missing required path: $relativePath")
  }
}

$trackedModelPatterns = @(
  "models/**/*.safetensors",
  "models/**/*.gguf",
  "models/**/*.pth"
)

foreach ($pattern in $trackedModelPatterns) {
  $matches = @(& git -C $rootDir ls-files $pattern 2>$null)
  foreach ($match in $matches) {
    if (-not [string]::IsNullOrWhiteSpace($match)) {
      $errors.Add("Tracked model binary detected (should be local-only): $match")
    }
  }
}

& git -C $rootDir check-ignore -q dist
if ($LASTEXITCODE -ne 0) {
  $warnings.Add("'dist' is not ignored by git; local artifacts may be accidentally committed.")
}

if ($errors.Count -gt 0) {
  Write-Host "Repository readiness check failed:" -ForegroundColor Red
  foreach ($message in $errors) {
    Write-Host "  - $message"
  }
  foreach ($message in $warnings) {
    Write-Host "  - warning: $message" -ForegroundColor Yellow
  }
  exit 1
}

Write-Host "Repository readiness check passed." -ForegroundColor Green
foreach ($message in $warnings) {
  Write-Host "  - warning: $message" -ForegroundColor Yellow
}
exit 0
