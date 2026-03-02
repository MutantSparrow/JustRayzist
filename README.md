# JustRayzist

JustRayzist is a Windows-first, offline-first local image generation app built around:

- FastAPI web API + browser UI
- Typer CLI
- Z-Image Turbo, and more specifically my very own finetune: [Rayzist](https://huggingface.co/MutantSparrow/Ray)
- local model packs (`.safetensors` / `.gguf`)
- profile-based runtime behavior for different VRAM classes
- custom mixed-model fast upscale flow
- RunMeFirst bootstrap installation and auto-repair

The app is designed to run without runtime internet dependencies once required assets are present locally.

## Features

- Local/offline runtime guards (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`).
- Runtime profiles: `constrained`, `balanced`, `high`.
- Model pack validation and local path enforcement.
- PNG metadata writing and SQLite gallery indexing.
- Web gallery with filtering, fullscreen, compare-hold for upscaled images, and queued jobs.
- CLI workflows for generation, mixed-model upscaling, soak runs, and soak reporting.
- Lane-aware bootstrap packaging (`cu126`, `cu128`) with GPU driver preflight.
- Release artifacts do not bundle model weights.

## Tech Stack

- Python 3.11+
- PyTorch + CUDA wheels (`cu126`/`cu128`)
- Diffusers + Transformers + Accelerate
- FastAPI + Uvicorn
- Typer
- Pillow
- SQLite

## Requirements

- Windows host (primary supported workflow).
- NVIDIA GPU recommended for target performance.
- Internet access for first-time setup (Python/dependencies/model downloads).

## Installation

From repository root:

```powershell
.\RunMeFirst.bat
```

`RunMeFirst.bat` will:
- install Python 3.11 if missing
- create or repair `.venv`
- install lane-matched torch/runtime dependencies
- install Hugging Face CLI + XET support in the environment
- fetch default model assets from Hugging Face (checksum-verified)
- run sanity checks and create a desktop shortcut

## Model Assets (One-Time Online Setup)

From repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fetch_model_assets.ps1
```

`RunMeFirst.bat` prefetches defaults automatically. `StartWeb.bat` can still auto-fetch missing default assets required by the default pack and custom mixed-model fast upscale path.
Downloads are performed through Hugging Face CLI (`hf download`) with XET acceleration enabled (`HF_XET_HIGH_PERFORMANCE=1`), and each file is SHA256-verified before acceptance.

## Quick Start

From repository root:

```powershell
.\RunMeFirst.bat
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

`POST /upscale` uses the app's default custom mixed-model fast upscale path.

## Release Packaging

Release packaging docs were moved to:
- `scripts/release/README.md`

## CUDA Lane Baseline

- `cu126`: NVIDIA driver `>= 561.17` (20xx/30xx/40xx fallback lane)
- `cu128`: NVIDIA driver `>= 572.61` (preferred lane; required for 50xx)

## Troubleshooting

See:

- `docs/USAGE.md`
- `docs/PACKAGING.md`
- `docs/TROUBLESHOOTING.md`
- `docs/CLONE_BUILD_CHECKLIST.md`

## Known Limitations

- Windows-first launcher/build flow.
- No authentication on local destructive endpoints (`/server/kill`, delete routes).
- Runtime quality/performance depend on local model pack quality and GPU/driver compatibility.

## License

This project is licensed under the Apache License 2.0.
See the [LICENSE](LICENSE) file for full terms.

## Acknowledgements

Default model assets are provided by the following model owners and repositories:

- MutantSparrow (Ray): https://huggingface.co/MutantSparrow/Ray
- Tongyi-MAI (Z-Image-Turbo): https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
- Comfy-Org (z_image_turbo): https://huggingface.co/Comfy-Org/z_image_turbo
- ByteDance-Seed (SeedVR2-3B original): https://huggingface.co/ByteDance-Seed/SeedVR2-3B
- themindstudio (SeedVR2-3B FP8 quantized provider): https://huggingface.co/themindstudio/SeedVR2-3B-FP8-e4m3fn
- imagepipeline (superresolution/x2 upscaler): https://huggingface.co/imagepipeline/superresolution

Model weights remain under their respective upstream licenses and terms.
