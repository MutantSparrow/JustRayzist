# Local Weights (Not Stored In Git)

Place the Krea2-Turbo weight files in this folder before running generation:

- `krea2_transformer.safetensors`  (Krea2Transformer2DModel — ~12B params)
- `krea2_vae.safetensors`          (AutoencoderKLQwenImage)

The text-encoder weights (`Qwen3VLModel`) are fetched into `../config/text_encoder/`.

They are intentionally excluded from GitHub because of large size limits.

Source: `krea/Krea-2-Turbo` on Hugging Face (`https://huggingface.co/krea/Krea-2-Turbo`).

From project root, the automated fetch command is:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fetch_krea2_assets.ps1
```

## Licensing (READ BEFORE DISTRIBUTING)

Krea2-Turbo is released under the **Krea 2 Community License**, not the same terms as the Z-Image
assets. The fetch step downloads weights for local use only. Redistribution or bundling of the
weights, and any product positioning, must be reviewed against that license first — this is a
non-technical decision requiring human sign-off (see JustRayzist-Krea.md §12). The fetch script
therefore prints the license notice and requires explicit opt-in.

## VRAM

The bf16 transformer alone is ~24GB. On <=24GB cards use the fp8 backend (`fp8_krea`, listed first
in `backend_preference`). See JustRayzist-Krea.md §6 / WP-6 for tier behavior.
