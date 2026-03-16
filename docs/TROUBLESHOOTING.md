# Troubleshooting

## `validate-models` fails

Check:

- `modelpack.yaml` paths exist and are local paths
- file extensions match declared formats (`safetensors` or `gguf`)
- `pipeline_config_dir` and `required_configs` exist

Actions:

```powershell
.\RunMeFirst.bat
python -m app.cli.main validate-models
```

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

Actions:

```powershell
python -m app.cli.main serve --host 127.0.0.1 --port 37717 --profile balanced
```

Server startup performs gallery sync.

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
- switch to `constrained` profile on lower-VRAM hardware

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
