# Packaging and Compatibility

## Packaging Strategy
- Default packaging format: bootstrap runtime bundle (no embedded Python/CUDA binaries)
- Optional packaging format: PyInstaller `--onedir` (`bundled` mode)
- Models are never bundled in artifacts.
- Runtime entrypoints:
  - `RunMeFirst.bat` (setup/repair)
  - `StartWeb.bat` (launch app)

Runtime model acquisition:
- `RunMeFirst.bat` prefetches default assets from Hugging Face.
- `StartWeb.bat` still auto-downloads missing default assets if needed.
- `scripts/fetch_model_assets.ps1` can prefetch all defaults (including upscaler checkpoint) before first run.
- Download backend is Hugging Face CLI (`hf download`) with XET acceleration enabled.
- Downloaded assets are verified via SHA256.

Dependency lock baseline:
- `requirements/runtime-lock.txt`
- `requirements/dev-lock.txt`
- `requirements/build-lock.txt`
- `requirements/torch-cu126.txt`
- `requirements/torch-cu128.txt`

## Build Commands
Create bootstrap release artifact (recommended):
```powershell
powershell -ExecutionPolicy Bypass -File scripts\release\package_release.ps1 -Mode bootstrap -Lane cu128 -Version v0.10.0-beta.02 -Clean
```

Optional: build lane binaries for bundled release:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\pyinstaller\build_onedir.ps1 -Lane cu128 -Clean
```

Create bundled release artifact (large):
```powershell
powershell -ExecutionPolicy Bypass -File scripts\release\package_release.ps1 -Mode bundled -Lane cu128 -Version v0.10.0-beta.02 -PythonExe .\.venv\Scripts\python.exe -Clean
```

If the host Python cannot create virtual environments, add `-UseActivePython`.
If dependencies are already present in the selected interpreter, add `-SkipDependencyInstall`.

Cleanup legacy dist outputs:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\release\clean_legacy_artifacts.ps1
```

Repository readiness check:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\release\verify_repo_readiness.ps1
```

## CUDA/Driver Baseline
As of `2026-02-27`:

- Lane `cu126`
  - PyTorch wheels: `torch==2.9.1`, `torchvision==0.24.1`, `torchaudio==2.9.1`
  - Driver floor: `561.17`
  - Intended GPUs: 20xx/30xx/40xx fallback lane

- Lane `cu128`
  - PyTorch wheels: `torch==2.9.1`, `torchvision==0.24.1`, `torchaudio==2.9.1`
  - Driver floor: `572.61`
  - Intended GPUs: preferred lane for 20xx/30xx/40xx; required for 50xx

Pinned lane wheel sets are defined in:
- `requirements/torch-cu126.txt`
- `requirements/torch-cu128.txt`

`StartWeb.bat` reads `release_lane.txt` and enforces lane-specific GPU preflight at launch.
