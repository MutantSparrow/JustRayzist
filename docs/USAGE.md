# JustRayzist Usage

## Prerequisites
- Windows host with internet access for initial setup.
- Local model pack configured at `models/packs/<pack_name>/modelpack.yaml`.
- For GPU tests: CUDA-enabled PyTorch and NVIDIA driver.

## Run Setup / Repair (Recommended)
```powershell
.\RunMeFirst.bat
```

`RunMeFirst.bat` detects/installs Python 3.11, creates or repairs `.venv`, installs lane-matched dependencies, downloads default model assets, validates the environment, and refreshes a desktop shortcut to `StartWeb.bat`.
Model downloads are performed with Hugging Face CLI and XET acceleration.

## Manual `.venv` Bootstrap (Advanced)
```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_env.ps1 -PythonExe E:\APPS\Python_3.11\python.exe -Lane cu128
```

## Fetch Required Model Assets (One-Time Online Setup)
Run this before `validate-models` on a fresh clone:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\fetch_model_assets.ps1
```
Downloads are SHA256-verified before being accepted.
Transfer backend: Hugging Face CLI (`hf download`) with `HF_XET_HIGH_PERFORMANCE=1`.

This downloads and places:
- `models/packs/Rayzist_bf16/weights/Rayzist.v1.0.safetensors`
- `models/packs/Rayzist_bf16/weights/diffusion_pytorch_model.safetensors`
- `models/packs/Rayzist_bf16/config/text_encoder/model.safetensors`
- `models/seedvr2/seedvr2_ema_3b_fp8_e4m3fn.safetensors`
- `models/seedvr2/ema_vae_fp16.safetensors`
- `models/upscaler/2x_RealESRGAN_x2plus.pth`

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
- Metrics include runtime execution diagnostics such as:
  - `runtime_profile`
  - `execution_mode` (`full_cuda`, `model_offload`, `sequential_offload`)
  - `cuda_total_bytes`
  - `cuda_reserved_after_load_bytes`

## SeedVR2 Benchmark (A/B)
Run paired comparisons between `x2plus` baseline and `SeedVR2`:
```powershell
python -m app.cli.main seedvr2-benchmark `
  --input-image outputs/_Upscale_test.png `
  --profiles high,balanced,constrained `
  --timeout-seconds 240 `
  --max-consecutive-failures 3
```
Outputs:
- Paired benchmark PNGs in `outputs/`
- CSV report in `data/seedvr2_benchmark_<timestamp>.csv`
- JSONL report in `data/seedvr2_benchmark_<timestamp>.jsonl`

## SeedVR2 + X2 Alpha Blend Benchmark
Run `x2` and `SeedVR2` passes, then blend `x2` over `SeedVR2` with alpha values:
```powershell
python -m app.cli.main seedvr2-blend-benchmark `
  --profile high `
  --alphas 25,50,75 `
  --timeout-seconds 240
```

Notes:
- If `--inputs` is omitted, the command auto-selects the latest two `1024x1024` `justrayzist_*.png` images from `outputs/`.
- Blend formula is `x2 over SeedVR2`: `blend = alpha*x2 + (1-alpha)*seedvr2`.

Outputs:
- Per source image:
  - one `x2` PNG
  - one `SeedVR2` PNG
  - three blend PNGs (`a25`, `a50`, `a75`)
- CSV report in `data/seedvr2_blend_benchmark_<timestamp>.csv`
- JSONL report in `data/seedvr2_blend_benchmark_<timestamp>.jsonl`

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
LAN listen mode:
```powershell
python -m app.cli.main serve --host 0.0.0.0 --port 37717
```

Windows profile launcher:
- Run `StartWeb.bat`
- Optional: set `JUSTRAYZIST_PYTHON` to force source-mode interpreter path (example: `set JUSTRAYZIST_PYTHON=E:\APPS\Python\python.exe`).
- Optional: set `JUSTRAYZIST_LISTEN=1` to force LAN listen mode (`0.0.0.0`) in launcher.
- Choose profile:
  - `1` = constrained
  - `2` = balanced
  - `3` = high
- `high` uses adaptive execution mode selection at startup:
  - prefers `full_cuda` when reserved VRAM ratio is healthy
  - automatically uses `model_offload` when full residency is likely to thrash memory
- Choose model pack from dynamically discovered entries under `models/packs/*/modelpack.yaml`
- If `Rayzist_bf16` assets are missing, launcher auto-downloads them from Hugging Face before startup (including default upscaler checkpoint)

Web API:
- `GET /health`
- `GET /config`
- `POST /generate` with JSON `{ "prompt": "...", "width": 1024, "height": 1024, "enhance_prompt": false }`
- `POST /upscale` with JSON `{ "filename": "justrayzist_....png", "seed": 1234, "scheduler_mode": "euler", "enhance_prompt": false }`
  - Uses the default production upscale chain in-app: `x2plus + SeedVR2 + 50% blend` (`upscale_engine=x2_seedvr2_blend`).
- `GET /images?prompt=<keyword>&limit=120&offset=0`
- `GET /images/{filename}`
- `GET /model-packs`
- `DELETE /images/{filename}?confirm=DELETE`
- `DELETE /gallery?confirm=DELETE`
- `POST /server/kill`
- `GET /API` (interactive endpoint list + request tester)

API examples (PowerShell):
```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:37717/health"

$genBody = @{
  prompt = "Cinematic skyline at sunrise"
  width = 1024
  height = 1024
  pack = "Rayzist_bf16"
  seed = 123456
  scheduler_mode = "euler"
  enhance_prompt = $false
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:37717/generate" -ContentType "application/json" -Body $genBody

$upscaleBody = @{
  filename = "justrayzist_20260228_120000_000.png"
  pack = "Rayzist_bf16"
  seed = 123456
  scheduler_mode = "euler"
  enhance_prompt = $false
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:37717/upscale" -ContentType "application/json" -Body $upscaleBody
```

Web UI:
- Open `http://127.0.0.1:37717/`
- API reference/testing page: `http://127.0.0.1:37717/API`
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
Build lane-specific one-dir binaries (optional bundled workflow):
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
Default bootstrap release (small artifact):
```powershell
powershell -ExecutionPolicy Bypass -File scripts\release\package_release.ps1 -Mode bootstrap -Lane cu128 -Version v0.10.0-beta.02 -Clean
```

Optional bundled release (large artifact, offline-style):
```powershell
powershell -ExecutionPolicy Bypass -File scripts\release\package_release.ps1 -Mode bundled -Lane cu128 -Version v0.10.0-beta.02 -Clean
```

Useful flags:
- `-Mode bootstrap|bundled`
- `-Lane cu126|cu128`
- `-Version vX.Y.Z`
- `-PythonExe C:\Path\To\python.exe`
- `-UseActivePython` (skip build-venv creation and build with `-PythonExe` directly)
- `-SkipDependencyInstall` (assume PyInstaller + runtime deps are already installed)
- `-SkipBuild`
- `-IncludeCliBinary` (bundled mode only; adds second large binary payload)
- `-NoZip`

Release policy:
- Model weights are never bundled in artifacts (`.safetensors`, `.gguf`, `.pth` are removed).
- `RunMeFirst.bat` performs setup/repair and prefetches default assets before launch.
- `StartWeb.bat` downloads missing `Rayzist_bf16` assets from Hugging Face on first launch, including `models/upscaler/2x_RealESRGAN_x2plus.pth`.
- Runtime lane marker is written to `release_lane.txt`.

CUDA/driver baseline:
- `cu126`: NVIDIA driver `>= 561.17` (20xx/30xx/40xx fallback lane)
- `cu128`: NVIDIA driver `>= 572.61` (preferred lane, required for 50xx)

Run packaged app:
- Open `<release>\RunMeFirst.bat` first, then launch `<release>\StartWeb.bat`
