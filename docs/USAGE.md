# Usage

This document focuses on source-mode usage and the main operational commands.

## Prerequisites

- Windows host
- Python 3.11+
- Model pack under `models/packs/<pack_name>/modelpack.yaml`
- NVIDIA GPU recommended for practical performance

## Setup

Recommended:

```powershell
.\RunMeFirst.bat
```

Manual alternative:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_env.ps1 -PythonExe C:\Path\To\python.exe -Lane cu128
powershell -ExecutionPolicy Bypass -File scripts\fetch_model_assets.ps1
```

## Update a Packaged Install

If you are running from a packaged release folder, not a git checkout:

```powershell
.\UpdateApp.bat
```

The updater preserves `models/`, `outputs/`, `data/`, `.venv/`, and `release_lane.txt` while replacing the shipped app files from the latest matching GitHub release.
## Sanity Checks

```powershell
python -m app.cli.main doctor
python -m app.cli.main validate-models
python -m app.cli.main --help
```

## Serve the App

Source mode:

```powershell
python -m app.cli.main serve --host 127.0.0.1 --port 37717 --profile balanced
```

Launcher mode:

```powershell
.\StartWeb.bat
```

## CLI Commands

Status and validation:

```powershell
python -m app.cli.main status
python -m app.cli.main doctor
python -m app.cli.main validate-models
```

Generate:

```powershell
python -m app.cli.main generate `
  --pack Rayzist_bf16 `
  --prompt "A cinematic skyline at sunrise" `
  --width 1024 `
  --height 1024 `
  --profile balanced
```

Supported generation cap: `1536x1536`.

Upscale test:

```powershell
python -m app.cli.main upscale-test `
  --input-image outputs\_Upscale_test.png `
  --checkpoint models\upscaler\2x_RealESRGAN_x2plus.pth `
  --profiles high,balanced,constrained
```

Upscale and refine:

```powershell
python -m app.cli.main upscale-refine `
  --pack Rayzist_bf16 `
  --input-image outputs\sample.png `
  --prompt "portrait photo" `
  --profile balanced
```

Soak run and report:

```powershell
python -m app.cli.main soak `
  --pack Rayzist_bf16 `
  --prompt "stress prompt" `
  --iterations 20 `
  --profile constrained

python -m app.cli.main soak-report --list-sessions
python -m app.cli.main soak-report --session-id <session_id>
```

SeedVR2 benchmark commands:

```powershell
python -m app.cli.main seedvr2-benchmark --profiles high,balanced,constrained
python -m app.cli.main seedvr2-blend-benchmark --profile high --alphas 25,50,75
```

## API Usage

Base URL: `http://127.0.0.1:37717`

Supported generation cap: `1536x1536`.
Client-scoped routes require `X-JustRayzist-Client`.

```powershell
$headers = @{ "X-JustRayzist-Client" = "desktop-client" }
```

Generate:

```powershell
$body = @{
  prompt = "Cinematic skyline at sunrise"
  width = 1024
  height = 1024
  pack = "Rayzist_bf16"
  seed = 123456
  scheduler_mode = "euler"
  enhance_prompt = $false
  use_random_latent = $false
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:37717/generate" -Headers $headers -ContentType "application/json" -Body $body
```

List images:

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:37717/images?limit=50&offset=0&newest_first=true" -Headers $headers
```

Download one image:

```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:37717/images/justrayzist_20260316_120000_000.png?client_id=desktop-client" -OutFile sample.png
```

Delete one image:

```powershell
$deleteBody = @{ confirm = "DELETE" } | ConvertTo-Json
Invoke-RestMethod -Method Delete -Uri "http://127.0.0.1:37717/images/justrayzist_20260316_120000_000.png?confirm=DELETE" -Headers $headers -ContentType "application/json" -Body $deleteBody
```

Import from another gallery source:

```powershell
$body = @{ source_id = "__legacy_root__"; dry_run = $false } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:37717/gallery/import" -Headers $headers -ContentType "application/json" -Body $body
```

## Output Locations

- Generated images: `outputs/`
- Metrics: `data/generation_metrics.jsonl`
- Gallery database: `data/gallery.db`
- Benchmark reports: `data/seedvr2*_benchmark_*.csv` and `.jsonl`

## Environment Variables

- `JUSTRAYZIST_ROOT`
- `JUSTRAYZIST_PROFILE`
- `JUSTRAYZIST_PACK`
- `JUSTRAYZIST_OFFLINE`
- `JUSTRAYZIST_ENV`
- `JUSTRAYZIST_PYTHON`
- `JUSTRAYZIST_LISTEN`
- `JUSTRAYZIST_SKIP_GPU_PREFLIGHT`
