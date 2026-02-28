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

## 3) Bootstrap Python Environment

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_env.ps1 -PythonExe E:\APPS\Python_3.11\python.exe -Lane cu128
```

Notes:

- `requirements/runtime-lock.txt` and `requirements/dev-lock.txt` pin app/dev dependency baselines.
- `requirements/torch-cu126.txt` and `requirements/torch-cu128.txt` pin CUDA lane torch wheels.

## 4) Fetch Default Model Assets (One-Time Online Step)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fetch_model_assets.ps1
```

The script downloads required assets and validates SHA256 checksums.

## 5) Sanity Checks

```powershell
python -m app.cli.main doctor
python -m app.cli.main validate-models
python -m ruff check app
```

## 6) Launch App from Source

```powershell
.\StartWeb.bat
```

## 7) Build One-Dir Binaries

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\pyinstaller\build_onedir.ps1 -Lane cu128 -Clean
```

## 8) Create Release Artifact

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\release\package_release.ps1 -Lane cu128 -Version v0.0.0 -Clean
```
