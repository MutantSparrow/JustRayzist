# JustRayzist Usage

## Prerequisites
- Python 3.11+ environment with dependencies installed.
- Local model pack configured at `models/packs/<pack_name>/modelpack.yaml`.
- For GPU tests: CUDA-enabled PyTorch and NVIDIA driver.

## Bootstrap Local `.venv`
```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_env.ps1 -PythonExe E:\APPS\Python_3.11\python.exe -Lane cu128
```

## Fetch Required Model Assets (One-Time Online Setup)
Run this before `validate-models` on a fresh clone:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\fetch_model_assets.ps1
```

This downloads and places:
- `models/packs/Rayzist_bf16/weights/Rayzist.v1.0.safetensors`
- `models/packs/Rayzist_bf16/weights/ultrafluxVAEImproved_v10.safetensors`
- `models/packs/Rayzist_bf16/config/text_encoder/model.safetensors`

## Validate Environment
```powershell
python -m app.cli.main doctor
python -m app.cli.main validate-models
```

## Single Generation (CLI)
```powershell
python -m app.cli.main generate `
  --pack Rayzist_bf16 `
  --prompt "A cinematic skyline at sunrise" `
  --enhance-prompt `
  --width 1024 `
  --height 1024 `
  --profile balanced
```

Outputs:
- PNG file in `outputs/`
- Generation metric row in `data/generation_metrics.jsonl`

## Upscale Test (Profile Sweep)
Run superscaling on a local PNG using a local upscaler checkpoint (`.pth` or `.safetensors`):
```powershell
python -m app.cli.main upscale-test `
  --input-image outputs/_Upscale_test.png `
  --checkpoint "models/upscaler/2x_RealESRGAN_x2plus.pth" `
  --profiles high,balanced,constrained
```

Optional upscaler tiling overrides:
- `--tile-size`: force upscale tiling (`0` means full-frame upscale).
- `--tile-overlap`: overlap for tiled upscale.

Outputs:
- One upscaled PNG per tested profile in `outputs/`
- One metric row per tested profile in `data/generation_metrics.jsonl` (`mode=upscale_test`)

## Upscale + Refine (2-Step)
Run a two-stage path: `RealESRGAN x2plus` upscaling, then Z-Image Turbo `img2img` refinement on the x2 image.
```powershell
python -m app.cli.main upscale-refine `
  --pack Rayzist_bf16 `
  --input-image outputs/_Upscale_test.png `
  --checkpoint models/upscaler/2x_RealESRGAN_x2plus.pth `
  --prompt "portrait photo, natural skin, realistic detail" `
  --strength 0.20 `
  --refine-steps 6 `
  --profile balanced
```

Useful flags:
- `--enhance-prompt`: rewrite prompt with loaded text encoder before refine pass.
- `--refine-tile-size`: force tiled refine size (`0` means full-frame refine).
- `--refine-tile-overlap`: overlap used when tiled refine is active.
- `--scheduler-mode euler|dpm`: choose scheduler for img2img refine.

Adaptive refine defaults when `--refine-tile-size` is omitted:
- `high`: full-frame for smaller outputs (`max side <= 1024`), otherwise adaptive tiling (2x2 grid, cap 896).
- `balanced`: adaptive tiling (3x3 grid, cap 1024).
- `constrained`: adaptive tiling (4x4 grid, cap 896).
- On CUDA OOM during refine, runtime retries with smaller tiles before failing.

Outputs:
- One final refined PNG in `outputs/`
- One metric row in `data/generation_metrics.jsonl` (`mode=upscale_then_img2img`)

## Soak Testing
Run repeated generations and track memory drift/recycles:
```powershell
python -m app.cli.main soak `
  --pack Rayzist_bf16 `
  --prompt "Stress test prompt" `
  --iterations 20 `
  --width 1024 `
  --height 1024 `
  --steps 9 `
  --profile constrained `
  --warmup
```

Useful flags:
- `--save-images`: save each soak output
- `--no-warmup`: skip warmup
- `--seed-start`: deterministic seed sequence
- `--enhance-prompt`: rewrite prompt with loaded text encoder before generation
- `--drift-threshold-mb`: override profile default drift threshold
- `--recycle-every`: override profile default recycle cadence

Profile soak defaults:
- `high`: drift threshold `256MB`, recycle every `0` (disabled)
- `balanced`: drift threshold `128MB`, recycle every `0` (disabled)
- `constrained`: drift threshold `64MB`, recycle every `24`

## Soak Report
List sessions:
```powershell
python -m app.cli.main soak-report --list-sessions
```

Report latest session:
```powershell
python -m app.cli.main soak-report
```

Report as JSON:
```powershell
python -m app.cli.main soak-report --json
```

Report a specific session:
```powershell
python -m app.cli.main soak-report --session-id <session_id>
```

## Run Web Server
```powershell
python -m app.cli.main serve --host 127.0.0.1 --port 37717
```

Windows profile launcher:
- Run `StartWeb.bat`
- Optional: set `JUSTRAYZIST_PYTHON` to force source-mode interpreter path (example: `set JUSTRAYZIST_PYTHON=E:\APPS\Python\python.exe`).
- Choose profile:
  - `1` = constrained
  - `2` = balanced
  - `3` = high
- Choose model pack from dynamically discovered entries under `models/packs/*/modelpack.yaml`
- If `Rayzist_bf16` assets are missing, launcher auto-downloads them from Hugging Face before startup

Web API:
- `GET /health`
- `GET /config`
- `POST /generate` with JSON `{ "prompt": "...", "width": 1024, "height": 1024, "enhance_prompt": false }`
- `POST /upscale` with JSON `{ "filename": "justrayzist_....png", "seed": 1234, "scheduler_mode": "euler", "enhance_prompt": false }`
- `GET /images?prompt=<keyword>&limit=120&offset=0`
- `GET /images/{filename}`
- `GET /model-packs`
- `DELETE /images/{filename}?confirm=DELETE`
- `DELETE /gallery?confirm=DELETE`
- `POST /server/kill`

Web UI:
- Open `http://127.0.0.1:37717/`
- Top bar includes prompt input, generate icon button, and settings popup.
- Settings popup includes:
  - resolution selector (grouped by ratio)
  - portrait/landscape toggle
  - freeze-seed toggle
  - prompt enhancer toggle (uses already-loaded text encoder)
  - delete gallery action (requires typing `DELETE`)
  - kill server action (confirmation modal + CRT disconnect animation)
- Press `Enter` (or click the generate icon) to run generation.
- During generation, gallery shows a black placeholder tile with `GENERATING...` spinner at the target aspect ratio.
- Gallery displays local PNG outputs in masonry layout with:
  - prompt keyword filter
  - reverse order toggle
  - hover metadata with tile actions: `Download` and magenta `Upscale`
  - generation-style placeholder tile for queued/running upscale jobs
  - fullscreen viewer with wheel zoom (`0.01x` to `10x`) and drag-to-pan
  - fullscreen actions: `Use Prompt`, `Copy Prompt`, `Upscale`, `Download`
  - for upscaled images: metadata line `Upscaled from <source> | HOLD TO SEE ORIGINAL | <resolution>`
  - hold-to-compare link temporarily swaps the preview to the original source image
  - keyboard delete in fullscreen (`Delete`/`Backspace`) opens Yes/No confirmation

## Build PyInstaller One-Dir (Windows)
Build lane-specific one-dir binaries:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\pyinstaller\build_onedir.ps1 -Lane cu128 -Clean
```

Useful flags:
- `-Lane cu126|cu128`
- `-PythonExe C:\Path\To\python.exe`
- `-SkipDependencyInstall` (reuse existing build venv dependencies)

Outputs:
- `dist\pyinstaller\<lane>\justrayzist-web\justrayzist-web.exe`
- `dist\pyinstaller\<lane>\justrayzist-cli\justrayzist-cli.exe`

## Create Release Artifact (Windows)
Build (unless skipped), assemble release folder, and zip it:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\release\package_release.ps1 -Lane cu128 -Version v0.10.0-beta.01 -Clean
```

Useful flags:
- `-Lane cu126|cu128`
- `-Version vX.Y.Z`
- `-PythonExe C:\Path\To\python.exe`
- `-UseActivePython` (skip build-venv creation and build with `-PythonExe` directly)
- `-SkipDependencyInstall` (assume PyInstaller + runtime deps are already installed)
- `-SkipBuild`
- `-NoZip`

Release policy:
- Model weights are never bundled in artifacts (`.safetensors`, `.gguf`, `.pth` are removed).
- `StartWeb.bat` downloads missing `Rayzist_bf16` assets from Hugging Face on first launch.
- Runtime lane marker is written to `release_lane.txt`.

CUDA/driver baseline:
- `cu126`: NVIDIA driver `>= 561.17` (20xx/30xx/40xx fallback lane)
- `cu128`: NVIDIA driver `>= 572.61` (preferred lane, required for 50xx)

Run packaged app:
- Open `<release>\StartWeb.bat`
