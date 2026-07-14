# Local Weights (Not Stored In Git)

The Krea2-Turbo pack uses the ComfyUI-native fp8 weights from
[`AlperKTS/Krea2_FP8`](https://huggingface.co/AlperKTS/Krea2_FP8). Place these three files in this
folder before running generation (the app converts their ComfyUI key layout at load time — see
`app/core/pipeline_factory/krea_comfy_convert.py`):

- `krea2_turbo_fp8.safetensors`       (Krea2Transformer2DModel — ~12B, fp8)
- `qwen3vl_4b_fp8_scaled.safetensors` (Qwen3VLModel text encoder, scaled fp8)
- `qwen_image_vae.safetensors`        (AutoencoderKLQwenImage)

They are intentionally excluded from GitHub because of large size limits. The pack's diffusers
config dirs (`../config/`) ARE committed, so only these weights need fetching.

From project root, the automated fetch commands are:
```powershell
# Windows
powershell -ExecutionPolicy Bypass -File .\scripts\fetch_model_assets.ps1 -IncludeKrea2 -AcceptKrea2License
```
```bash
# Any platform
python scripts/portable/fetch_model_assets.py --include-krea2 --accept-krea2-license
```

`StartWeb.bat` / `StartWeb.sh` also prompt to fetch these automatically the first time you select
the `Krea2_Turbo` pack.

## Licensing (READ BEFORE DISTRIBUTING)

Krea2-Turbo is released under the **Krea 2 Community License**, not the same terms as the Z-Image
assets. The fetch step downloads weights for local use only. Redistribution or bundling of the
weights, and any product positioning, must be reviewed against that license first — this is a
non-technical decision requiring human sign-off (see JustRayzist-Krea.md §12). The fetch script
therefore prints the license notice and requires explicit opt-in.

## VRAM

The bf16 transformer alone is ~24GB. On <=24GB cards use the fp8 backend (`fp8_krea`, listed first
in `backend_preference`). See JustRayzist-Krea.md §6 / WP-6 for tier behavior.
