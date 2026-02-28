# JustRayzist

JustRayzist is a Windows-first, offline-first local image generation app built around:

- FastAPI web API + browser UI
- Typer CLI
- local model packs (`.safetensors` / `.gguf`)
- profile-based runtime behavior for different VRAM classes
- optional x2 upscaling + img2img refinement flow
- PyInstaller one-dir release packaging

The app is designed to run without runtime internet dependencies once required assets are present locally.

## Features

- Local/offline runtime guards (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`).
- Runtime profiles: `constrained`, `balanced`, `high`.
- Model pack validation and local path enforcement.
- PNG metadata writing and SQLite gallery indexing.
- Web gallery with filtering, fullscreen, compare-hold for upscaled images, and queued jobs.
- CLI workflows for generation, upscaling/refinement, soak runs, and soak reporting.
- Lane-based release packaging (`cu126`, `cu128`) with GPU driver preflight.
- Release artifacts do not bundle model weights.

## Repository Layout

```text
.
|- app/                       # API, CLI, core runtime, storage, UI assets
|- docs/                      # Usage, packaging, troubleshooting docs
|- launch/                    # PowerShell launcher helpers
|- models/                    # Model pack configs and upscaler checkpoint folder
|- requirements/              # Lane-pinned torch wheels
|- scripts/                   # Bootstrap, fetch, pyinstaller, release tooling
|- StartWeb.bat               # Interactive launcher
|- pyproject.toml
|- dist/                      # Build/release output
\- README.md                  # Canonical repo documentation
```

`dist/` is local build output and is intentionally not source-controlled.

## Tech Stack

- Python 3.11+
- PyTorch + CUDA wheels (`cu126`/`cu128`)
- Diffusers + Transformers + Accelerate
- FastAPI + Uvicorn
- Typer
- Pillow
- SQLite
- PowerShell + batch tooling for bootstrap/build/release

## Requirements

- Windows host (primary supported workflow).
- NVIDIA GPU recommended for target performance.
- Python 3.11 interpreter available locally.

## Installation

From repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_env.ps1 -PythonExe E:\APPS\Python_3.11\python.exe -Lane cu128
```

This creates/repairs `.venv`, installs lane-matched torch wheels, and installs project dependencies.

## Model Assets (One-Time Online Setup)

From repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fetch_model_assets.ps1
```

Or use `StartWeb.bat` and let it auto-fetch missing default assets for `Rayzist_bf16`.

## Quick Start

From repository root:

```powershell
.\StartWeb.bat
```

Launcher flow:

1. Select runtime profile.
2. Select model pack.
3. Optional model asset bootstrap for `Rayzist_bf16`.
4. Open `http://127.0.0.1:37717/`.

## Configuration

Environment variables:

- `JUSTRAYZIST_ROOT`: override workspace root.
- `JUSTRAYZIST_PROFILE`: `constrained|balanced|high`.
- `JUSTRAYZIST_PACK`: default model pack name.
- `JUSTRAYZIST_OFFLINE`: `1` (default) enables offline env guards.
- `JUSTRAYZIST_ENV`: environment label (`dev` default).
- `JUSTRAYZIST_PYTHON`: optional interpreter override for source-mode launcher.
- `JUSTRAYZIST_SKIP_GPU_PREFLIGHT`: set `1` to bypass lane/driver preflight in packaged mode.

## CLI Usage

From repository root:

```powershell
python -m app.cli.main status
python -m app.cli.main doctor
python -m app.cli.main validate-models
python -m app.cli.main serve --host 127.0.0.1 --port 37717 --profile balanced
```

Generate:

```powershell
python -m app.cli.main generate --pack Rayzist_bf16 --prompt "cinematic skyline at sunrise" --profile balanced
```

Upscale + refine:

```powershell
python -m app.cli.main upscale-refine --pack Rayzist_bf16 --input-image outputs\sample.png --prompt "portrait photo" --profile balanced
```

Soak test:

```powershell
python -m app.cli.main soak --pack Rayzist_bf16 --prompt "stress prompt" --iterations 20 --profile constrained
```

Soak report:

```powershell
python -m app.cli.main soak-report --list-sessions
python -m app.cli.main soak-report --session-id <session_id>
```

## API Summary

Base URL: `http://127.0.0.1:37717`

- `GET /health`
- `GET /config`
- `GET /model-packs`
- `POST /generate`
- `POST /upscale`
- `GET /images`
- `GET /images/{filename}`
- `DELETE /images/{filename}?confirm=DELETE`
- `DELETE /gallery?confirm=DELETE`
- `POST /server/kill`
- `GET /API` (interactive API documentation + tester)

### API Example: Generate

```http
POST /generate
Content-Type: application/json

{
  "prompt": "A cinematic skyline at sunrise",
  "width": 1024,
  "height": 1024,
  "pack": "Rayzist_bf16",
  "seed": 123456,
  "scheduler_mode": "euler",
  "enhance_prompt": false
}
```

### API Example: Upscale

```http
POST /upscale
Content-Type: application/json

{
  "filename": "justrayzist_20260228_120000_000.png",
  "pack": "Rayzist_bf16",
  "seed": 123456,
  "scheduler_mode": "euler",
  "enhance_prompt": false
}
```

## Build and Release

Build PyInstaller one-dir binaries:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\pyinstaller\build_onedir.ps1 -Lane cu128 -Clean
```

Create release package:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\release\package_release.ps1 -Lane cu128 -Version v0.10.0-beta.01 -Clean
```

Cleanup legacy artifacts:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\release\clean_legacy_artifacts.ps1
```

## CUDA Lane Baseline

- `cu126`: NVIDIA driver `>= 561.17` (20xx/30xx/40xx fallback lane)
- `cu128`: NVIDIA driver `>= 572.61` (preferred lane; required for 50xx)

## Development Workflow

From repository root:

```powershell
python -m app.cli.main doctor
python -m app.cli.main validate-models
python -m ruff check app
```

If tests are present in your local workspace:

```powershell
python -m pytest -q tests
```

## Troubleshooting

See:

- `docs/USAGE.md`
- `docs/PACKAGING.md`
- `docs/TROUBLESHOOTING.md`

## Known Limitations

- Windows-first launcher/build flow.
- No authentication on local destructive endpoints (`/server/kill`, delete routes).
- Runtime quality/performance depend on local model pack quality and GPU/driver compatibility.

## Contributing

1. Keep changes behavior-safe and focused.
2. Validate with lint/tests/smoke checks.
3. Keep docs aligned with real command behavior.
4. Prefer small, reviewable commits.

## License

No license file is currently included in this repository.
