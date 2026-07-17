# Local Weights (Not Stored In Git)

The Krea2-Turbo pack expects three native/ComfyUI-format fp8 weight files here. The operator
supplies these — licensing / distribution is handled off-repo. Place these files in this folder
before running generation (the app converts their ComfyUI key layout at load time — see
`app/core/pipeline_factory/krea_comfy_convert.py`):

- `krea2_turbo_fp8.safetensors`       (Krea2Transformer2DModel — ~12B, fp8)
- `qwen3vl_4b_fp8_scaled.safetensors` (Qwen3VLModel text encoder, scaled fp8)
- `qwen_image_vae.safetensors`        (AutoencoderKLQwenImage)

They are intentionally excluded from GitHub because of size limits. The pack's diffusers
config dirs (`../config/`) ARE committed, so only these weights need fetching.

From project root, the automated fetch commands are:
```powershell
# Windows
powershell -ExecutionPolicy Bypass -File .\scripts\fetch_model_assets.ps1 -IncludeKrea2
```
```bash
# Any platform
python scripts/portable/fetch_model_assets.py --include-krea2
```

`StartWeb.bat` / `StartWeb.sh` also prompt to fetch these automatically the first time you
select the `Krea2_Turbo` pack. The fetch target repo id lives in
`scripts/portable/fetch_model_assets.py` (`_KREA2_FINETUNE_REPO`); the shipped default is a
placeholder — see the `TODO(krea2 finetune)` marker there for how to swap in the real repo
plus per-file SHA256s.

## VRAM

The bf16 transformer alone is ~24 GB. On <=24 GB cards use the fp8 backend (`fp8_krea`, listed
first in `backend_preference`). See `docs/KREA2_IMPLEMENTATION_STATUS.md` for tier behavior and
optimization notes.
