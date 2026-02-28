# Upscaler Checkpoints

This folder stores local upscaler checkpoints used by `upscale-test` and `upscale-refine`.

Default expected file:

- `2x_RealESRGAN_x2plus.pth`

How to fetch defaults:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\fetch_model_assets.ps1
```

`StartWeb.bat` also attempts to auto-download missing default assets (including this upscaler checkpoint).
Downloaded defaults are validated with SHA256 checks.
