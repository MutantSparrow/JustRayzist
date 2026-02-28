# Local Weights (Not Stored In Git)

Place these files in this folder before running generation:

- `Rayzist.v1.0.safetensors`
- `ultrafluxVAEImproved_v10.safetensors`

They are intentionally excluded from GitHub because of large size limits.

Source URLs:
- `Rayzist.v1.0.safetensors`: `https://huggingface.co/MutantSparrow/Ray/blob/main/Z-IMAGE-TURBO/Rayzist.v1.0.safetensors`
- `ultrafluxVAEImproved_v10.safetensors`: `https://huggingface.co/Owen777/UltraFlux-v1/blob/main/vae/diffusion_pytorch_model.safetensors`

From project root, the automated fetch command is:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fetch_model_assets.ps1
```

The fetch script uses Hugging Face CLI (`hf download`) with XET acceleration and verifies SHA256 checksums.
