# Clone and Build Checklist

Use this checklist on a fresh machine to verify that the source tree is runnable and package-ready.

## 1. Clone

```powershell
git clone <repo-url>
cd JustRayzist
```

## 2. Verify release scripts and tracked files

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\release\verify_repo_readiness.ps1
```

## 3. Run setup or repair

```powershell
.\RunMeFirst.bat
```

```bash
./RunMeFirst.sh
```

## 4. Optional manual bootstrap

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_env.ps1 -PythonExe C:\Path\To\python.exe -Lane cu128
powershell -ExecutionPolicy Bypass -File .\scripts\fetch_model_assets.ps1
```

## 5. Run source-mode sanity checks

```powershell
python -m app.cli.main doctor
python -m app.cli.main validate-models
python -m ruff check app tests
python -m pytest -q tests -p no:cacheprovider
```

## 6. Launch the app

```powershell
.\StartWeb.bat
```

```bash
./StartWeb.sh
```

Normal startup should now ask only for the public model pack and local/LAN listen mode. The app should auto-detect its internal resource tier from available VRAM instead of prompting for `high`, `balanced`, or `constrained`.

Quick acceptance checks after launch:

- only public packs are shown in the launcher
- the app opens on `http://127.0.0.1:37717/`
- `GET /model-packs` returns only public packs
- `/API` loads current examples from the internal manifest-backed tester

## 7. Optional packaged build

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\pyinstaller\build_onedir.ps1 -Lane cu128 -Clean
powershell -ExecutionPolicy Bypass -File .\scripts\release\package_release.ps1 -Lane cu128 -Version v0.1.0 -Clean
```
