# Troubleshooting

## `validate-models` fails
Check:
- `modelpack.yaml` paths exist and are local paths.
- File extension matches declared `format` (`safetensors` or `gguf`).
- `pipeline_config_dir` exists and contains required config files.

Action:
- Run the full setup/repair workflow:
  - `.\RunMeFirst.bat`
- Fetch required model weights into expected local paths:
  - `powershell -ExecutionPolicy Bypass -File scripts\fetch_model_assets.ps1`

## Generation fails with model key mismatch
Possible cause:
- Transformer checkpoint format does not match expected architecture mapping.

Current behavior:
- Prefixed/fused Z-Image transformer keys (`model.diffusion_model.*`, fused `attention.qkv.weight`) are converted automatically.

Action:
- Ensure the model is truly Z-Image-compatible.
- Confirm `transformer/config.json` matches model architecture version.

## GGUF component fails to load
Check:
- `modelpack.yaml` component `format` is `gguf` and path ends in `.gguf`.
- `pipeline_config_dir` contains matching local config for the component.
- For GGUF text encoder, `config/text_encoder/config.json` is present and valid.

Action:
- Re-run `python -m app.cli.main validate-models`.
- Start with one GGUF component at a time to isolate incompatibility.
- If needed, fall back to `.safetensors` for the failing component.

## CUDA not detected
Check:
- `python -c "import torch; print(torch.cuda.is_available())"` returns `True`.
- Correct CUDA-enabled PyTorch build is installed.
- NVIDIA driver and CUDA runtime are compatible.

If CUDA is unavailable:
- App still runs on CPU, but performance and memory behavior are not representative of target hardware.

## `ModuleNotFoundError` for runtime dependencies (`typer`, `fastapi`, `uvicorn`, `PIL`, `torch`)
Cause:
- The selected Python interpreter does not have project dependencies installed.

Action:
- Preferred repair path:
  - `.\RunMeFirst.bat`
- Install dependencies into the same interpreter you use to launch:
  - `python -m pip install -r requirements\runtime-lock.txt`
  - `python -m pip install -r requirements\dev-lock.txt`
  - `python -m pip install --no-deps -e .`
- Rebuild/repair local `.venv` automatically:
  - `powershell -ExecutionPolicy Bypass -File scripts\bootstrap_env.ps1 -PythonExe E:\APPS\Python_3.11\python.exe -Lane cu128`
- Or use launcher wrappers:
  - `StartWeb.bat`
  - `launch\web.ps1`
  - `launch\cli.ps1`

## `cannot import name 'ZImagePipeline' from 'diffusers'`
Cause:
- A standard/stable `diffusers` build was installed, but this app requires ZImage classes from newer dev builds.

Action:
- Preferred repair path:
  - `.\RunMeFirst.bat`
- Manual repair path:
  - `powershell -ExecutionPolicy Bypass -File scripts\bootstrap_env.ps1 -PythonExe E:\APPS\Python_3.11\python.exe -Lane cu128`

Notes:
- `scripts/bootstrap_env.ps1` now verifies `ZImagePipeline`, `ZImageTransformer2DModel`, and `ZImageImg2ImgPipeline` imports and auto-falls back to a compatible diffusers source when needed.

If you are testing on another machine:
- Do not use GitHub auto-generated source zip archives.
- Download the uploaded release asset zip from **Releases -> Assets**.
- Run `RunMeFirst.bat` once before launching `StartWeb.bat`.

## Model fetch fails due to missing Hugging Face CLI / XET
Cause:
- `scripts/fetch_model_assets.ps1` uses Hugging Face CLI (`hf download`) with XET acceleration.
- Environment is missing `huggingface_hub` CLI module and/or `hf_xet`.

Action:
- Run full setup/repair:
  - `.\RunMeFirst.bat`
- Or manually repair bootstrap environment:
  - `powershell -ExecutionPolicy Bypass -File scripts\bootstrap_env.ps1 -PythonExe E:\APPS\Python_3.11\python.exe -Lane cu128`
- Re-run:
  - `powershell -ExecutionPolicy Bypass -File scripts\fetch_model_assets.ps1`

## Launcher fails GPU preflight for lane/driver mismatch
Cause:
- Release lane in `release_lane.txt` does not match installed NVIDIA driver.

Current floors:
- `cu126`: driver `>= 561.17`
- `cu128`: driver `>= 572.61` (required for RTX 50xx)

Action:
- Update NVIDIA driver to meet lane floor.
- Or use artifact built for another lane.

## Upscale fails: missing upscaler checkpoint
Cause:
- No upscaler checkpoint is bundled in release artifacts.

Action:
- Fetch defaults from Hugging Face:
  - `powershell -ExecutionPolicy Bypass -File scripts\fetch_model_assets.ps1`
- Or launch with `StartWeb.bat` to auto-download default assets (including `2x_RealESRGAN_x2plus.pth`).
- You can still provide a custom `.pth` or `.safetensors` checkpoint via CLI `--checkpoint`.

## Upscale output quality is poor
Check:
- Source image may already contain heavy artifacts.
- Prompt used for refine may be too strong/off-topic.
- Scheduler/seed combination may be unstable for that image.

Action:
- Keep defaults first (`strength=0.20`, `refine_steps=6`, `scheduler_mode=euler`).
- If outputs look overcooked, lower refine strength:
  - `--strength 0.12`
- If outputs look too soft, increase refine steps modestly:
  - `--refine-steps 8`
- Keep adaptive tiling enabled unless you are intentionally tuning tile behavior.

## Slow first iteration in soak runs
Expected:
- Initial load and first pass are slower due to model materialization and warmup.

Recommendation:
- Keep `--warmup` enabled for soak tests.

## `high` profile refine is too slow or times out
Cause:
- Refine stage can become expensive on large post-upscale images when full-frame img2img is used.

Current behavior:
- Adaptive tiling is used by default for larger outputs.
- `high` uses full-frame only for smaller outputs (`max side <= 1024`), then switches to tiled refine.
- If refine hits CUDA OOM, runtime retries with progressively smaller tiles.

## `high` profile text2img is unexpectedly slower than `balanced`
Cause:
- On some 24GB-class GPUs, effective free VRAM can be lower than nominal due to driver/desktop/runtime overhead.
- If full GPU residency creates heavy memory pressure, throughput can drop instead of improve.

Current behavior:
- `high` now auto-selects execution mode at startup using observed CUDA reserved memory ratio.
- If reserved VRAM is high, runtime switches `high` to model offload for stability/throughput.
- During repeated generation, `high` can fall back from full CUDA to model offload if sustained pressure is detected.

Action:
- Re-launch server after major workload changes to re-evaluate startup mode.
- Compare `execution_mode` in metrics rows (`full_cuda` vs `model_offload`) when diagnosing performance.

Action:
- Keep auto tiling (omit `--refine-tile-size`) unless you are intentionally tuning.
- For strict runtime caps, force tiling explicitly:
  - `--refine-tile-size 1024 --refine-tile-overlap 64`

## Large memory drift during soak
Use:
- `python -m app.cli.main soak-report --session-id <id>`

Inspect:
- `drift_max_mb`
- `drift_slope_mb_per_iteration`
- `recycle_count`

Mitigation:
- Lower `--drift-threshold-mb` below the active profile default
- Set `--recycle-every` to force periodic backend recycle
- Use constrained profile for lower-VRAM systems

## Pytest cache warning on Windows
Warning:
- Cache path creation can fail in restricted directories.

Action:
- This does not invalidate test results.
- Optionally run pytest with cache disabled:
  - `python -m pytest -q -p no:cacheprovider`

## Web gallery is empty but images exist
Check:
- Images are stored in `outputs/*.png`.
- API has indexed files in `data/gallery.db`.

Action:
- Restart web server (startup performs output sync to SQLite index).
- Call `GET /images` directly to verify API list response.

## `GET /images/{filename}` returns not found for an existing DB row
Cause:
- Row path is missing on disk or outside managed `outputs/` directory.

Action:
- Keep generated images under `outputs/`.
- If gallery DB was manually edited or copied from another machine, reindex by restarting the server.
