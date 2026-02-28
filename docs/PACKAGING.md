# Packaging and Compatibility

## Packaging Strategy
- Packaging format: PyInstaller `--onedir`
- Models are never bundled in artifacts.
- Runtime entrypoint: `StartWeb.bat`
- Release binaries:
  - `bin/web/justrayzist-web.exe`
  - `bin/cli/justrayzist-cli.exe`

Runtime model acquisition:
- `StartWeb.bat` auto-downloads missing default assets from Hugging Face.
- `scripts/fetch_model_assets.ps1` can prefetch all defaults (including upscaler checkpoint) before first run.
- Downloaded assets are verified via SHA256.

Dependency lock baseline:
- `requirements/runtime-lock.txt`
- `requirements/dev-lock.txt`
- `requirements/build-lock.txt`
- `requirements/torch-cu126.txt`
- `requirements/torch-cu128.txt`

## Build Commands
Build lane binaries:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\pyinstaller\build_onedir.ps1 -Lane cu128 -Clean
```

Create release artifact:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\release\package_release.ps1 -Lane cu128 -Version v0.10.0-beta.01 -PythonExe .\.venv\Scripts\python.exe -Clean
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
