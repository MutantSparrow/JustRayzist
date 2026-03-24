# Troubleshooting

## `validate-models` fails

Check:

- `modelpack.yaml` paths exist and are local paths
- file extensions match declared formats (`safetensors` or `gguf`)
- `pipeline_config_dir` and `required_configs` exist
- `user_visible`, if present, is a boolean
- advanced runtime hints (`storage_mode`, `storage_dtype`, `compute_dtype`) are valid for the selected component

Actions:

```powershell
.\RunMeFirst.bat
python -m app.cli.main validate-models
```

## `UpdateApp.bat` fails

Check:

- the folder is a packaged release, not a git checkout
- internet access to GitHub is available
- a matching release asset exists for your current lane and mode

Actions:

```powershell
.\UpdateApp.bat
```

If the updater says no matching asset was found, download the correct release package for your lane manually.
## `Missing client id` from the API

Cause:

- client-scoped routes require `X-JustRayzist-Client`
- direct image links can use `?client_id=<client-id>` instead

Example:

```powershell
$headers = @{ "X-JustRayzist-Client" = "desktop-client" }
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:37717/images" -Headers $headers
```

## CUDA is not detected

Check:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

If CUDA is unavailable, the app can still fall back to CPU, but generation and upscale performance will be very poor.

## Runtime dependency import errors

Examples:

- `ModuleNotFoundError: typer`
- `ModuleNotFoundError: fastapi`
- `ModuleNotFoundError: torch`

Actions:

```powershell
.\RunMeFirst.bat
```

Or repair manually:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_env.ps1 -PythonExe C:\Path\To\python.exe -Lane cu128
```

## `cannot import name 'ZImagePipeline' from 'diffusers'`

Cause:

- the selected interpreter has a diffusers build that does not contain the required ZImage classes

Action:

```powershell
.\RunMeFirst.bat
```

The bootstrap script verifies the required diffusers symbols and repairs the environment if needed.

## Model fetch fails

Actions:

```powershell
.\RunMeFirst.bat
powershell -ExecutionPolicy Bypass -File scripts\fetch_model_assets.ps1
```

If SeedVR2 runtime files are missing:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\fetch_seedvr2_runtime.ps1
```

## Launcher fails GPU preflight

Current driver floors:

- `cu126`: `>= 561.17`
- `cu128`: `>= 572.61`

Actions:

- update the NVIDIA driver
- or use a release built for the other lane
- or set `JUSTRAYZIST_SKIP_GPU_PREFLIGHT=1` only if you accept the risk and know the environment is compatible

## Web gallery is empty but images exist

Check:

- images are under `outputs/`
- `data/gallery.db` exists
- the request is using the same client id that created the images
- `GET /images` and `GET /model-packs` are being checked from the same running instance you launched

Actions:

```powershell
python -m app.cli.main serve --host 127.0.0.1 --port 37717
```

Server startup performs gallery sync.

## A model pack is missing from the launcher or `GET /model-packs`

Check:

- the pack folder contains `modelpack.yaml`
- `python -m app.cli.main validate-models` passes
- the pack is not marked `user_visible: false`
- the pack is not marked `enabled: false`

Important:

- only public enabled packs appear in `StartWeb.bat` and `GET /model-packs`
- when exactly one public enabled pack exists, `StartWeb.bat` auto-selects it and skips the pack prompt
- hidden packs can still be loaded explicitly for engineering workflows
- `Rayzist_fp8_full` and compatible custom real-FP8 packs can be public; if they are missing, validate their pack files and weights
- `fp8_storage` is not a user pack anymore; constrained conditions may derive `<base>__auto_fp8_storage` automatically
- real FP8 packs currently run through BF16 compute with FP8-at-rest preservation; native FP8 inference is not implemented yet

Actions:

```powershell
python -m app.cli.main validate-models
python -m app.cli.main validate-models --all
python -m app.cli.main status
```

## `GET /images/{filename}` returns not found

Check:

- the file still exists on disk under `outputs/`
- the request uses the correct `client_id` or matching client header
- the filename is not being requested from another client scope

## Soak drift is too high

Inspect the report:

```powershell
python -m app.cli.main soak-report --session-id <session_id>
```

Mitigations:

- lower `--drift-threshold-mb`
- set `--recycle-every`
- let auto resource-tiering downgrade naturally on lower-VRAM hardware
- use engineering-only forced profile overrides only when you are benchmarking or diagnosing a specific memory strategy

## High VRAM pressure or unexpected fallback behavior

Check the current baseline and detected memory tier:

```powershell
python -m app.cli.main status
python -m app.cli.main doctor
```

Look for:

- `runtime_profile`: stable baseline defaults for normal behavior
- `resource_tier`: current auto-detected memory strategy
- `resource_tier_override`: should normally be `null`
- `auto_resource_tier`: should normally be `true`

If a request falls back to a heavier offload mode, that usually means the preflight guard found less free CUDA-visible VRAM than was needed for the current run. This can happen even if Windows Task Manager looks relatively empty, because CUDA allocatable VRAM and desktop-reported usage do not always match exactly.

Normal guidance:

- do not create a second user pack just for FP8 storage
- let constrained conditions derive the internal FP8-storage runtime strategy automatically when the selected base pack is compatible
- keep forced `--profile` / `--profiles` flags for engineering diagnostics and benchmark commands only

Useful engineering probes:

```powershell
python -m app.cli.main pack-compare --prompt "diagnostic prompt"
python -m app.cli.main prompt-grid-benchmark --pack Rayzist_bf16 --prompt "PROMPT 1" --prompt "PROMPT 2" --prompt "PROMPT 3"
```

## SeedVR2 or upscale assets are missing

Expected local files include:

- `models\upscaler\2x_RealESRGAN_x2plus.pth`
- `models\seedvr2\seedvr2_ema_3b_fp8_e4m3fn.safetensors`
- `models\seedvr2\ema_vae_fp16.safetensors`
- `models\seedvr2\runtime\ComfyUI-SeedVR2_VideoUpscaler\inference_cli.py`

Repair paths:

```powershell
.\RunMeFirst.bat
powershell -ExecutionPolicy Bypass -File scripts\fetch_model_assets.ps1
powershell -ExecutionPolicy Bypass -File scripts\fetch_seedvr2_runtime.ps1
```
