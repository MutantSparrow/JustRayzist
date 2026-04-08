# Packaging

Windows-only release engineering document. Linux and macOS support is source-mode only.

## Packaging Strategy

Release artifacts are bootstrap-only. They ship the app, scripts, and metadata without bundling Python/CUDA runtime payloads.

Model weights are not bundled in release artifacts. Packaged releases also include `UpdateApp.bat`, `scripts\update_release.ps1`, `release_manifest.json`, and the local README image assets so installs can update in place from GitHub releases.

The packaged UX now matches source mode:

- startup prompts only for public model-pack selection and local/LAN listen mode
- memory strategy is auto-detected at runtime through internal `resource_tier` selection
- no normal-user packaged flow asks for `high`, `balanced`, or `constrained`

## Main Scripts

- `scripts\release\package_release.ps1`
- `scripts\release\verify_repo_readiness.ps1`
- `scripts\release\clean_legacy_artifacts.ps1`
- `scripts\update_release.ps1`

## Engineering-Only One-Dir Builds

```powershell
powershell -ExecutionPolicy Bypass -File scripts\pyinstaller\build_onedir.ps1 -Lane cu128 -Clean
```

Useful flags:

- `-Lane cu126|cu128`
- `-PythonExe C:\Path\To\python.exe`
- `-SkipDependencyInstall`

These outputs are for engineering validation only and are not shipped as release artifacts.

## Create a Release

```powershell
powershell -ExecutionPolicy Bypass -File scripts\release\package_release.ps1 -Lane cu128 -Version v0.1.0 -Clean
```

Useful flags:

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
- `StartWeb.bat` can auto-fetch the selected default pack assets when they are missing
- install/setup provisions `Rayzist_bf16` as the bundled default enabled pack
- `StartWeb.bat` now asks users only for pack selection when more than one public enabled pack exists, plus local/LAN listen mode
- `StartWeb.bat` shows only public enabled packs and keeps hidden/experimental/disabled packs out of the normal packaged launcher flow
- release packaging copies tracked pack metadata only, so local custom packs and local weight files are left to the installer/local machine
- downloads use Hugging Face CLI with XET acceleration
- fetched assets are SHA256-verified before acceptance

## Packaged Update Expectations

- `UpdateApp.bat` preserves `models/`, `outputs/`, `data/`, `.venv/`, and `release_lane.txt`
- packaged updates should not overwrite local user outputs or model assets
- after update, `StartWeb.bat` should still present the no-profile startup flow and public-pack-only selection

## CUDA / Driver Baseline

- `cu126`: driver `>= 561.17`
- `cu128`: driver `>= 572.61`

`StartWeb.bat` reads `release_lane.txt` when present and applies lane-specific GPU preflight checks.

## Expected Outputs

Typical packaging outputs are written under `dist/`, including:

- release folders such as `dist\JustRayzist_win64_cu128_v0.1.0_bootstrap`
- optional zip archives for release distribution
- optional engineering-only PyInstaller build folders under `dist\pyinstaller\`
