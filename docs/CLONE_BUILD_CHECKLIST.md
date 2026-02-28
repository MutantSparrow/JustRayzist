# Clone and Build Checklist

Use this checklist on a fresh machine to confirm the repo is self-sufficient.

## 1) Clone

```powershell
git clone <repo-url>
cd JustRayzist
```

## 2) Verify Repository Readiness

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\release\verify_repo_readiness.ps1
```

This checks required scripts/files and confirms model binaries are not tracked.

## 3) Run Setup / Repair

```powershell
.\RunMeFirst.bat
```

This is the primary setup path and includes Python install, `.venv` setup, dependency install, model fetch, and sanity checks.
It also installs Hugging Face CLI + XET support used by model downloads.

## 4) Manual Bootstrap (Optional / Advanced)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_env.ps1 -PythonExe E:\APPS\Python_3.11\python.exe -Lane cu128
```

Notes:

- `requirements/runtime-lock.txt` and `requirements/dev-lock.txt` pin app/dev dependency baselines.
- `requirements/torch-cu126.txt` and `requirements/torch-cu128.txt` pin CUDA lane torch wheels.

## 5) Fetch Default Model Assets (Optional Manual Step)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fetch_model_assets.ps1
```

The script downloads required assets via Hugging Face CLI (`hf download`) with XET enabled and validates SHA256 checksums.

## 6) Sanity Checks

```powershell
python -m app.cli.main doctor
python -m app.cli.main validate-models
python -m ruff check app
```

## 7) Launch App from Source

```powershell
.\StartWeb.bat
```

## 8) Build One-Dir Binaries (Optional Bundled Workflow)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\pyinstaller\build_onedir.ps1 -Lane cu128 -Clean
```

## 9) Create Bootstrap Release Artifact

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\release\package_release.ps1 -Mode bootstrap -Lane cu128 -Version v0.0.0 -Clean
```

## 10) Create Bundled Release Artifact (Optional, Large)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\release\package_release.ps1 -Mode bundled -Lane cu128 -Version v0.0.0 -Clean
```
