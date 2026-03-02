# JustRayzist
<img width="1900" height="921" alt="image" src="https://github.com/user-attachments/assets/8ade356b-d74b-43bd-b983-06befb5fc230" />


<br>Not feeling like ComfyUI? Too broke to get a monthly sub to an image platform?<br>
Got 35GB of space on a drive somewhere and an RTX card? <br>

### Enter Just Rayzist!
A lightweight, easy to install and easier to run app that just runs.<br>
Built around my Z-Image-Turbo finetune, it gives you a fast image generation platform, available through a local web page, command line or via local API so your favorite AI agents can use it.<br>
It generates images up to 1536x1536 and can upscale up to double that in less than a minute. <br>
It even has a built in prompt enhancement feature and a cool image browser.<br><br>
<img height="200" alt="image" src="https://github.com/user-attachments/assets/8d4374d7-e2f3-48a7-89d2-514ecf19abc3" />
<img height="200" alt="image" src="https://github.com/user-attachments/assets/8ae8c35c-dfab-4e4f-95d2-cf4780957095" />



## Specs

- FastAPI web API + browser UI
- Typer CLI
- Z-Image Turbo, and more specifically my very own finetune: [Rayzist](https://huggingface.co/MutantSparrow/Ray)
- local model packs (`.safetensors` / `.gguf`)
- Runtime profiles `constrained`, `balanced`, `high` support different VRAM classes
- custom mixed-model fast upscale flow. It's not the best in the world, but it's the best at that speed!
- RunMeFirst bootstrap installation and auto-repair.
- Run it locally or open it to LAN access
- Model pack system to support custom -Image-Turbo models, VAEs or encoder models.
- PNG metadata writing and SQLite gallery indexing.
- Web gallery with filtering, fullscreen, compare-hold for upscaled images, and queued jobs.
- CLI workflows for generation, mixed-model upscaling, soak runs, and soak reporting.
- Lane-aware bootstrap packaging (`cu126`, `cu128`) with GPU driver preflight.

The app is designed to run 100% without runtime internet dependencies once installed locally.

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
- NVIDIA GPU required.
- Internet access for first-time setup (Python/dependencies/model downloads(everything is coming from HuggingFace, so you need access to that)).

### CUDA Lane Baseline

- `cu126`: NVIDIA driver `>= 561.17` (20xx/30xx/40xx fallback lane)
- `cu128`: NVIDIA driver `>= 572.61` (preferred lane; required for 50xx)

12, 16 and 24GB RTX card supported: 20xx, 30xx, 40xx and 50xx sries and up.<br>
Tested on 4090, 4080 3090, 3060ti.<br>
It will work on 8GB cards provided you have enough system RAM, but it will slow down considerably.<br>
It *should* run purely on CPU thanks to smart offload but you *probably* do not want to do this.

## Installation

From repository root:

```powershell
.\RunMeFirst.bat
```

`RunMeFirst.bat` will:
- install Python 3.11 if missing
- create or repair `.venv`
- installmatching torch/runtime dependencies based on your detected GPU
- install Hugging Face CLI + XET support in the environment
- fetch default model assets from Hugging Face
- run sanity checks and create a desktop shortcut

Downloads are performed through Hugging Face CLI (`hf download`) with XET acceleration enabled (`HF_XET_HIGH_PERFORMANCE=1`), and each file is SHA256-verified before acceptance.

## Quick Start

From repository root:

```powershell
.\StartWeb.bat
```
<br>
...or use the desktop shortcut.<br>
<br>
Launcher flow:

1. Select runtime profile.
2. Select model pack.
3. Select if the server will listen to LAN connections.
4. Open `http://127.0.0.1:37717/`.

## CLI Usage

Environment variables (used for CLI)

- `JUSTRAYZIST_ROOT`: override workspace root.
- `JUSTRAYZIST_PROFILE`: `constrained|balanced|high`.
- `JUSTRAYZIST_PACK`: default model pack name.
- `JUSTRAYZIST_OFFLINE`: `1` (default) enables offline env guards.
- `JUSTRAYZIST_ENV`: environment label (`dev` default).
- `JUSTRAYZIST_PYTHON`: optional interpreter override for source-mode launcher.
- `JUSTRAYZIST_SKIP_GPU_PREFLIGHT`: set `1` to bypass lane/driver preflight in packaged mode.
<br><br>

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
- With multiple users on LAN or multiple web pages open, requests made in one place will only be picked up in the page when it next refreshes. (no push)

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
