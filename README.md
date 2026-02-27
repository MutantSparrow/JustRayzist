# JustRayzist

Offline-first local image generation app with:
- FastAPI web UI/API
- Typer CLI
- local model packs (`.safetensors` / `.gguf`)
- optional upscale + refine flow
- portable Windows packaging

## Features
- Local-only runtime controls (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`).
- Runtime profiles for `constrained`, `balanced`, and `high` VRAM targets.
- Model-pack validation and local component path checks.
- PNG output with metadata and SQLite-backed gallery index.
- Queue-based web generation/upscale workflow.
- PyInstaller `--onedir` Windows packaging.
- `StartWeb.bat` model bootstrap for default Rayzist pack assets from Hugging Face.

## Repository Layout
```text
.
|- JustRayzist/        # app workspace (code, docs, launchers, models, scripts)
|  |- app/             # API, CLI, core runtime, storage, web UI assets
|  |- docs/            # usage and troubleshooting docs
|  |- launch/          # PowerShell launch helpers
|  |- models/          # model pack configs + upscaler checkpoints
|  |- scripts/         # environment/bootstrap/build/release scripts
|  |- StartWeb.bat     # interactive web launcher (profile + pack selection)
|  \- pyproject.toml
|- dist/               # portable build/release output
\- README.md
```

## Tech Stack
- Python 3.11+
- PyTorch, Diffusers, Transformers, Accelerate
- FastAPI + Uvicorn
- Typer
- Pillow
- SQLite
- PowerShell + batch launch/build tooling (Windows-first workflow)

## Installation
From repo root:

```powershell
cd .\JustRayzist
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_env.ps1 -PythonExe E:\APPS\Python_3.11\python.exe -Lane cu128
```

This creates/repairs `.venv`, installs lane-matched CUDA PyTorch wheels, and installs project dependencies.

## Quick Start
From `JustRayzist/`:

```powershell
.\StartWeb.bat
```

Launcher flow:
1. Select runtime profile.
2. Select model pack.
3. If pack is `Rayzist_bf16`, missing default assets are auto-downloaded from Hugging Face.
4. Web server starts at `http://127.0.0.1:37717/`.

## Configuration
Environment variables:
- `JUSTRAYZIST_ROOT`: override workspace root path.
- `JUSTRAYZIST_PROFILE`: runtime profile (`constrained|balanced|high`).
- `JUSTRAYZIST_PACK`: preferred model pack name.
- `JUSTRAYZIST_OFFLINE`: `1` (default) enables offline guard flags.
- `JUSTRAYZIST_ENV`: environment label (default `dev`).
- `JUSTRAYZIST_PYTHON`: optional source-mode launcher override for Python executable path.

## CLI Usage
From `JustRayzist/`:

```powershell
python -m app.cli.main status
python -m app.cli.main doctor
python -m app.cli.main validate-models
python -m app.cli.main serve --host 127.0.0.1 --port 37717
```

Generate:
```powershell
python -m app.cli.main generate --pack Rayzist_bf16 --prompt "cinematic skyline at sunrise" --profile balanced
```

Upscale + refine:
```powershell
python -m app.cli.main upscale-refine --pack Rayzist_bf16 --input-image outputs\sample.png --prompt "portrait photo" --profile balanced
```

## API Endpoints
- `GET /health`
- `GET /config`
- `GET /model-packs`
- `POST /generate`
- `POST /upscale`
- `GET /images`
- `GET /images/{filename}`
- `DELETE /images/{filename}`
- `DELETE /gallery`
- `POST /server/kill`
- `GET /` (web UI)

## Build And Release
Build PyInstaller one-dir binaries:
```powershell
powershell -ExecutionPolicy Bypass -File .\JustRayzist\scripts\pyinstaller\build_onedir.ps1 -Lane cu128 -Clean
```

Create release zip:
```powershell
powershell -ExecutionPolicy Bypass -File .\JustRayzist\scripts\release\package_release.ps1 -Lane cu128 -Version v0.10.0-beta.01 -PythonExe .\JustRayzist\.venv\Scripts\python.exe -Clean
```

Notes:
- Release artifacts never bundle model weights (`.safetensors`, `.gguf`, `.pth`).
- CUDA lanes:
  - `cu126` (driver >= `561.17`) for 20xx/30xx/40xx fallback.
  - `cu128` (driver >= `572.61`) as preferred default, required for 50xx.
- Release users should download uploaded **release assets**, not GitHub auto-generated source zip archives.

## Development Workflow
From `JustRayzist/`:
```powershell
python -m app.cli.main doctor
python -m app.cli.main validate-models
python -m app.cli.main serve
```

Optional checks:
```powershell
python -m ruff check .
```

If a local `tests/` directory is present in your workspace:
```powershell
python -m pytest -q tests
```

## Troubleshooting
See:
- `JustRayzist/docs/USAGE.md`
- `JustRayzist/docs/PACKAGING.md`
- `JustRayzist/docs/TROUBLESHOOTING.md`

## Known Limitations
- Windows-first launcher and packaging workflow.
- Runtime depends on local model availability and compatible GPU/driver stack for CUDA acceleration.
- No authentication is implemented for local API control endpoints (`/server/kill`, deletion endpoints).

## Contributing
1. Keep changes focused and behavior-safe.
2. Validate with lint/tests/smoke checks where available.
3. Update docs when behavior, flags, or release flow changes.

## License
No license file is currently included in this repository.
