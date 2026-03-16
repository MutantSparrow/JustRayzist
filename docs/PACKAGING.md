# Packaging

## Packaging Strategy

The repository supports two Windows packaging flows:

- `bootstrap`: ships the app, scripts, and metadata without bundling Python/CUDA runtime payloads
- `bundled`: adds PyInstaller one-dir binaries for the CLI and web entrypoints

Model weights are not bundled in either mode.

## Main Scripts

- `scripts\pyinstaller\build_onedir.ps1`
- `scripts\release\package_release.ps1`
- `scripts\release\verify_repo_readiness.ps1`
- `scripts\release\clean_legacy_artifacts.ps1`

## Build One-Dir Binaries

```powershell
powershell -ExecutionPolicy Bypass -File scripts\pyinstaller\build_onedir.ps1 -Lane cu128 -Clean
```

Useful flags:

- `-Lane cu126|cu128`
- `-PythonExe C:\Path\To\python.exe`
- `-SkipDependencyInstall`

## Create a Bootstrap Release

```powershell
powershell -ExecutionPolicy Bypass -File scripts\release\package_release.ps1 -Mode bootstrap -Lane cu128 -Version v0.1.0 -Clean
```

## Create a Bundled Release

```powershell
powershell -ExecutionPolicy Bypass -File scripts\release\package_release.ps1 -Mode bundled -Lane cu128 -Version v0.1.0 -Clean
```

Useful flags:

- `-UseActivePython`
- `-SkipDependencyInstall`
- `-SkipBuild`
- `-IncludeCliBinary`
- `-NoZip`

## Repository Readiness Check

```powershell
powershell -ExecutionPolicy Bypass -File scripts\release\verify_repo_readiness.ps1
```

## Cleanup Legacy Release Artifacts

```powershell
powershell -ExecutionPolicy Bypass -File scripts\release\clean_legacy_artifacts.ps1
```

## Runtime Asset Policy

- `RunMeFirst.bat` is the primary setup and repair path
- `StartWeb.bat` can auto-fetch the default `Rayzist_bf16` assets when they are missing
- downloads use Hugging Face CLI with XET acceleration
- fetched assets are SHA256-verified before acceptance

## CUDA / Driver Baseline

- `cu126`: driver `>= 561.17`
- `cu128`: driver `>= 572.61`

`StartWeb.bat` reads `release_lane.txt` when present and applies lane-specific GPU preflight checks.

## Expected Outputs

Typical packaging outputs are written under `dist/`, including:

- release folders such as `dist\JustRayzist_win64_cu128_v0.1.0_bootstrap`
- optional zip archives for release distribution
- PyInstaller one-dir build folders under `dist\pyinstaller\`
