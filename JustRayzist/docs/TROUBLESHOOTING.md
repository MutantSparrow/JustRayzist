# Troubleshooting

## `validate-models` fails
Check:
- `modelpack.yaml` paths exist and are local paths.
- File extension matches declared `format` (`safetensors` or `gguf`).
- `pipeline_config_dir` exists and contains required config files.

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

## `ModuleNotFoundError: No module named 'typer'`
Cause:
- The selected Python interpreter does not have project dependencies installed.

Action:
- Install dependencies into the same interpreter you use to launch:
  - `python -m pip install -e .`
- Rebuild/repair local `.venv` automatically:
  - `powershell -ExecutionPolicy Bypass -File scripts\bootstrap_env.ps1`
- Or use the bundled/venv launchers that select project-local Python first:
  - `StartWeb.bat`
  - `launch\web.ps1`
  - `launch\cli.ps1`

## Upscale fails: missing `2x_RealESRGAN_x2plus.pth`
Cause:
- Default upscaler checkpoint is not present at `models/upscaler/2x_RealESRGAN_x2plus.pth`.

Action:
- Ensure the file exists at that exact path in the project/bundle.
- Rebuild portable bundle after restoring the checkpoint:
  - `powershell -ExecutionPolicy Bypass -File scripts\build_portable.ps1 -Clean`

Portable build now fails early if this checkpoint is missing, to prevent shipping a broken upscale path.

## Slow first iteration in soak runs
Expected:
- Initial load and first pass are slower due to model materialization and warmup.

Recommendation:
- Keep `--warmup` enabled for soak tests.

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
