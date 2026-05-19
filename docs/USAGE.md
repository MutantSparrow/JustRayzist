# Usage

This document focuses on source-mode usage and the main operational commands.

## Prerequisites

- Windows for packaged installs and the default source workflow
- Linux and macOS source mode support
- Python 3.11+
- Model pack under `models/packs/<pack_name>/modelpack.yaml`
- NVIDIA GPU recommended for practical performance
- macOS setup is best-effort; accelerated generation is not guaranteed

## Setup

Windows:

```powershell
.\RunMeFirst.bat
```

Linux or macOS source mode:

```bash
./RunMeFirst.sh
```

Manual alternatives:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_env.ps1 -PythonExe C:\Path\To\python.exe -Lane cu128
powershell -ExecutionPolicy Bypass -File scripts\fetch_model_assets.ps1
```

```bash
python3 scripts/portable/bootstrap_env.py --python-exe python3 --lane auto
python3 scripts/portable/fetch_model_assets.py
python3 scripts/portable/fetch_seedvr2_runtime.py
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
python -m app.cli.main validate-models --all
python -m app.cli.main --help
```

## Serve the App

Source mode:

```powershell
python -m app.cli.main serve --host 127.0.0.1 --port 37717
```

Launcher mode:

```powershell
.\StartWeb.bat
```

```bash
./StartWeb.sh
```

Launcher flow:

1. Select a public enabled model pack only when more than one is installed.
2. Select local-only or LAN listen mode.
3. Let the app auto-detect a memory strategy from available VRAM.

`StartWeb.sh` accepts `--host`, `--port`, and `--pack`. If more than one public enabled pack exists and no TTY is available, provide `--pack` or `JUSTRAYZIST_PACK`.

## Web Gallery

- Masonry gallery layout is now the default.
- Pending jobs survive refresh for the same client and can be cancelled directly from the gallery.
- Color swatches beside `Newest First` filter the gallery by dominant image color.
- The first launch after a color-classifier update may briefly show `Updating gallery color cache...` while the cached gallery color data is rebuilt in the background.

## R+ Mode

- The web UI can toggle `R+` for normal generate jobs.
- When `R+` is on in the web UI, the request is sent as `inference_process="rplus"` and the UI pins that run to `20` steps.
- `R+` exposes `vibrance` and `bias` controls in the web UI and CLI. Raw API callers can send the same values directly.
- `R+` is generate-only in the web UI today. Reference-image `img2img` keeps the standard path and greys out the `R+` controls while active.

## Auto Resource Tiering

Normal runs no longer ask the user to choose `high`, `balanced`, or `constrained`.

- `runtime_profile` stays on the stable `balanced` baseline for normal quality behavior.
- `resource_tier` is auto-detected from current free VRAM and drives memory strategy only.
- Internal execution modes such as `full_cuda`, `model_offload`, and `sequential_offload` are selected automatically.
- The app can downgrade or re-upgrade the internal resource tier between requests as available VRAM changes.
- Public pack lists (`StartWeb.bat`, `StartWeb.sh`, `GET /model-packs`) show only public enabled packs.
- Setup and asset fetch provision the bundled `Rayzist_bf16` pack.
- Setup can also fetch optional `Rayzist_qwen3_4b_fp8` from [MutantSparrow/qwen3_4b_Rayzist_v1.0_fp8](https://huggingface.co/MutantSparrow/qwen3_4b_Rayzist_v1.0_fp8); it reuses `Rayzist_bf16` model weights and replaces only the text encoder.
- Derived FP8 storage remains an internal constrained-memory strategy; native FP8 inference is not implemented in the current release.

- In constrained conditions, compatible safetensors packs may auto-derive an internal FP8-storage runtime variant such as `Rayzist_bf16__auto_fp8_storage`.

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
  --height 1024

python -m app.cli.main generate `
  --pack Rayzist_bf16 `
  --prompt "A cinematic skyline at sunrise" `
  --inference-process rplus `
  --rplus-vibrance 0.2 `
  --rplus-initial-bias-level 0.1
```

Supported generation cap: UI presets up to `1536x1536`; raw API requests up to `2048x2048`.

SeedVR2 x2 still benchmark:

```powershell
python -m app.cli.main seedvr2-still-benchmark `
  --inputs outputs\sample.png `
  --presets seed_faithful,seed_sharp `
  --runtime-preset current_baseline
```

Soak run and report:

```powershell
python -m app.cli.main soak `
  --pack Rayzist_bf16 `
  --prompt "stress prompt" `
  --iterations 20

python -m app.cli.main soak-report --list-sessions
python -m app.cli.main soak-report --session-id <session_id>
```

Engineering-only benchmark commands:

These commands keep forced `--profile` / `--profiles` flags for diagnostics and reproducible comparisons. They are not part of the normal startup/runtime flow.

```powershell
python -m app.cli.main seedvr2-still-benchmark --inputs outputs\sample.png --presets seed_faithful,seed_sharp --runtime-preset current_baseline
```

Procedural latent preview:

```powershell
python -m app.cli.main procedural-latent-preview --count 16 --seed-start 1 --creativity 2
```

Engineering compare/probe commands:

These commands are discoverable in the CLI, but they are intended for diagnostics, benchmarking, and regression work rather than normal daily generation.

```powershell
python -m app.cli.main pack-compare `
  --prompt "A cinematic skyline at sunrise"

python -m app.cli.main rplus-compare `
  --pack Rayzist_bf16 `
  --prompt "A cinematic skyline at sunrise" `
  --seed 17

python -m app.cli.main pack-compare-suite `
  --iterations 3

python -m app.cli.main prompt-grid-benchmark `
  --pack Rayzist_bf16 `
  --prompt "PROMPT 1" `
  --prompt "PROMPT 2" `
  --prompt "PROMPT 3"

python -m app.cli.main seedvr2-still-benchmark `
  --inputs outputs\sample.png `
  --presets seed_faithful,seed_sharp `
  --runtime-preset current_baseline
```

What they are for:

- `pack-compare`: compare a base pack against a candidate runtime strategy such as derived FP8 storage.
- `rplus-compare`: generate the same prompt and seed once with `standard` and once with `rplus`, then write paired reports and diff metrics.
- `pack-compare-suite`: run the multi-pack benchmark matrix and generate contact sheets/reports.
- `prompt-grid-benchmark`: run forced-tier plus auto-tier prompt grids with generation/upscale artifacts.
- `seedvr2-still-benchmark`: benchmark SeedVR2 x2 direct behavior with still-image presets and explicit runtime preset control.

## API Usage

Base URL: `http://127.0.0.1:37717`

Supported generation cap: UI presets up to `1536x1536`; raw API requests up to `2048x2048`.
Client-scoped routes require `X-JustRayzist-Client`.
Use `procedural_creativity` (`0-3`) to control Creative Mode.
In the main UI, scheduler behavior is derived automatically from Creative Mode. `scheduler_mode` remains optional for raw API and CLI calls.
`GET /health` and `GET /config` report both the stable `runtime_profile` baseline and the currently detected `resource_tier`.
`GET /model-packs` returns public packs only.
The built-in `/API` page gets its route descriptions and sample payloads from the internal `GET /api-manifest` feed so the examples stay aligned with the real handlers.

```powershell
$headers = @{ "X-JustRayzist-Client" = "desktop-client" }
```

<!-- BEGIN GENERATED API ROUTES -->
- `GET /health`
- `GET /config`
- `GET /model-packs`
- `GET /loras`
- `GET /wildcards`
- `POST /wildcards`
- `PATCH /wildcards/{wildcard_id}`
- `DELETE /wildcards/{wildcard_id}`
- `POST /wildcards/suggestions`
- `GET /chat/history`
- `DELETE /chat/history`
- `POST /chat`
- `POST /lora-drafts`
- `POST /lora-drafts/{draft_id}/detect-triggers`
- `POST /loras`
- `PATCH /loras/{lora_id}`
- `GET /loras/{lora_id}/preview`
- `DELETE /loras/{lora_id}`
- `POST /generate`
- `POST /img2img`
- `POST /upscale`
- `POST /clarity`
- `GET /client-jobs`
- `POST /client-jobs/cancel`
- `GET /images?prompt=skyline&color=blue&favorite=true&limit=50&offset=0&newest_first=true`
- `GET /images/{filename}?client_id=<client-id>`
- `POST /images/{filename}/favorite`
- `POST /images/download-zip`
- `DELETE /images/{filename}?confirm=DELETE`
- `DELETE /gallery?confirm=DELETE`
- `POST /gallery/rebuild`
- `GET /gallery/import-sources`
- `POST /gallery/import`
- `POST /server/kill`
- `POST /server/restart`
<!-- END GENERATED API ROUTES -->

<!-- BEGIN GENERATED API EXAMPLES -->
### `GET /health`

Service health plus current baseline/defaults and detected memory strategy.

Sample response:

```json
{
  "status": "ok",
  "app": "JustRayzist",
  "version": "1.8.7",
  "runtime_profile": "balanced",
  "resource_tier": "high",
  "active_pack": "Rayzist_bf16",
  "selected_pack": "Rayzist_bf16",
  "effective_pack": "Rayzist_bf16",
  "active_backend": "diffusers_zimage",
  "fp8_fallback_used": false,
  "fp8_fallback_reason": null,
  "fp8_runtime_mode": null,
  "fp8_storage_preserved_tensor_count": 0,
  "fp8_promoted_tensor_count": 0,
  "lora_capable": true,
  "wildcard_suggestions_capable": true,
  "gallery_color_cache_active": false,
  "gallery_color_cache_version": "dominant_v6",
  "gallery_color_cache_target_version": "dominant_v6",
  "gallery_color_cache_error": null,
  "offline_mode": true
}
```

### `GET /config`

Resolved runtime configuration, paths, and current runtime status.

Sample response:

```json
{
  "app_name": "JustRayzist",
  "app_version": "1.8.7",
  "environment": "dev",
  "offline_mode": true,
  "runtime_profile": {
    "name": "balanced",
    "description": "16GB-class profile with moderate offload and stable throughput."
  },
  "resource_tier": {
    "name": "high",
    "description": "24GB-class profile with minimal offload and highest throughput."
  },
  "resource_tier_override": null,
  "auto_resource_tier": true,
  "paths": {
    "root_dir": "S:\\STABLEDIFFUSION\\JustRayzist",
    "models_dir": "S:\\STABLEDIFFUSION\\JustRayzist\\models",
    "model_packs_dir": "S:\\STABLEDIFFUSION\\JustRayzist\\models\\packs",
    "outputs_dir": "S:\\STABLEDIFFUSION\\JustRayzist\\outputs",
    "data_dir": "S:\\STABLEDIFFUSION\\JustRayzist\\data",
    "ui_dir": "S:\\STABLEDIFFUSION\\JustRayzist\\app\\ui"
  },
  "runtime": {
    "runtime_profile": "balanced",
    "resource_tier": "high",
    "resource_tier_description": "24GB-class profile with minimal offload and highest throughput.",
    "resource_tier_override": null,
    "auto_resource_tier": true,
    "active_pack": "Rayzist_bf16",
    "selected_pack": "Rayzist_bf16",
    "effective_pack": "Rayzist_bf16",
    "active_backend": "diffusers_zimage",
    "execution_mode": "model_offload",
    "fp8_checkpoint": false,
    "fp8_fallback_used": false,
    "fp8_fallback_reason": null,
    "fp8_runtime_mode": null,
    "fp8_normalized_tensor_count": 0,
    "fp8_storage_preserved_tensor_count": 0,
    "fp8_promoted_tensor_count": 0,
    "lora_capable": true,
    "wildcard_suggestions_capable": true,
    "gallery_color_cache_active": false,
    "gallery_color_cache_version": "dominant_v6",
    "gallery_color_cache_target_version": "dominant_v6",
    "gallery_color_cache_error": null
  }
}
```

### `GET /model-packs`

List discovered, valid, public, and enabled model packs.

Sample response:

```json
{
  "count": 1,
  "items": [
    {
      "name": "Rayzist_bf16",
      "path": "S:\\STABLEDIFFUSION\\JustRayzist\\models\\packs\\Rayzist_bf16\\modelpack.yaml",
      "architecture": "z_image_turbo"
    }
  ]
}
```

### `GET /loras`

List installed LoRAs, preview URLs, saved trigger words, detected trigger suggestions, and runtime LoRA capabilities.

Sample response:

```json
{
  "count": 1,
  "items": [
    {
      "id": "cinematic-style",
      "display_name": "cinematic-style",
      "source_filename": "cinematic-style.safetensors",
      "preview_url": "/loras/cinematic-style/preview",
      "trigger_words": [
        "cinematic style"
      ],
      "detected_trigger_words": [
        "cinematic style",
        "moody light"
      ],
      "preview_is_custom": true,
      "metadata_summary": {
        "ss_output_name": "cinematic-style"
      },
      "file_size_bytes": 12345678
    }
  ],
  "capabilities": {
    "supported": true,
    "active_pack": "Rayzist_bf16",
    "max_active": 3,
    "min_weight": -2.0,
    "max_weight": 2.0,
    "default_weight": 1.0
  }
}
```

### `GET /wildcards`

List installed wildcards, their editable prompt tokens, multiline content, and runtime wildcard capabilities.

Sample response:

```json
{
  "count": 1,
  "items": [
    {
      "id": "3c03cc4d8cf5476e831d6603626d7843",
      "display_name": "Picturesque Locations",
      "token": "picturesque-locations",
      "placeholder": "__picturesque-locations__",
      "content_text": "a cabin in the Schwarzwald\na chalet in the French Alps",
      "entry_count": 2,
      "created_at": "2026-04-08T12:00:00+00:00",
      "updated_at": "2026-04-08T12:00:00+00:00"
    }
  ],
  "capabilities": {
    "supported": true,
    "active_pack": "Rayzist_bf16",
    "suggestions_supported": true
  }
}
```

### `POST /wildcards`

Create one wildcard with a display name, editable prompt token, and multiline entries.

Sample request body:

```json
{
  "display_name": "Picturesque Locations",
  "token": "picturesque-locations",
  "content_text": "a cabin in the Schwarzwald\na chalet in the French Alps"
}
```

Sample response:

```json
{
  "status": "ok",
  "item": {
    "id": "3c03cc4d8cf5476e831d6603626d7843",
    "display_name": "Picturesque Locations",
    "token": "picturesque-locations",
    "placeholder": "__picturesque-locations__",
    "content_text": "a cabin in the Schwarzwald\na chalet in the French Alps",
    "entry_count": 2,
    "created_at": "2026-04-08T12:00:00+00:00",
    "updated_at": "2026-04-08T12:00:00+00:00"
  },
  "capabilities": {
    "supported": true,
    "active_pack": "Rayzist_bf16",
    "suggestions_supported": true
  }
}
```

### `PATCH /wildcards/{wildcard_id}`

Update one wildcard's display name, editable prompt token, and multiline entries.

Sample request body:

```json
{
  "display_name": "Picturesque Locations",
  "token": "picturesque-locations",
  "content_text": "a cabin in the Schwarzwald\na chalet in the French Alps\na white sandy beach in Bora-Bora"
}
```

Sample response:

```json
{
  "status": "ok",
  "item": {
    "id": "3c03cc4d8cf5476e831d6603626d7843",
    "display_name": "Picturesque Locations",
    "token": "picturesque-locations",
    "placeholder": "__picturesque-locations__",
    "content_text": "a cabin in the Schwarzwald\na chalet in the French Alps\na white sandy beach in Bora-Bora",
    "entry_count": 3,
    "created_at": "2026-04-08T12:00:00+00:00",
    "updated_at": "2026-04-08T12:10:00+00:00"
  }
}
```

### `DELETE /wildcards/{wildcard_id}`

Delete one wildcard definition from the library.

Sample response:

```json
{
  "status": "ok",
  "id": "3c03cc4d8cf5476e831d6603626d7843",
  "deleted": true
}
```

### `POST /wildcards/suggestions`

Ask the active text encoder for 10 wildcard entry suggestions that match a theme and stay within the example-length constraint.

Sample request body:

```json
{
  "theme": "picturesque locations",
  "format_example": "a cabin in the Schwarzwald",
  "seed": 123456,
  "existing_entries": [
    "a cabin in the Schwarzwald"
  ]
}
```

Sample response:

```json
{
  "status": "ok",
  "suggestions": [
    "a chalet in the French Alps",
    "a white sandy beach in Bora-Bora",
    "a small cafe in a Parisian side street"
  ],
  "accepted_count": 3,
  "target_count": 10,
  "seed": 123456,
  "example_word_count": 5,
  "min_words": 5,
  "max_words": 5,
  "partial": true,
  "message": "Returned a partial set because the example-length filter was restrictive."
}
```

### `GET /chat/history`

Load the per-client Rayzist Chat history and active encoder label.

Requires `X-JustRayzist-Client`.

Sample response:

```json
{
  "status": "ok",
  "history": {
    "owner_id": "example-client",
    "next_number": 3,
    "exchange_count": 1,
    "exchanges": [
      {
        "user": {
          "number": 1,
          "role": "user",
          "content": "Help me make this prompt moodier.",
          "created_at": "2026-04-08T12:00:00+00:00",
          "error": false
        },
        "assistant": {
          "number": 2,
          "role": "assistant",
          "content": "Add stronger lighting contrast, specific weather, and camera framing.",
          "created_at": "2026-04-08T12:00:05+00:00",
          "error": false,
          "actions": [
            {
              "type": "set_prompt",
              "label": "Use Prompt",
              "prompt": "rainy neon city street, reflected signs, low camera angle"
            }
          ]
        }
      }
    ]
  },
  "capabilities": {
    "supported": true,
    "active_pack": "Rayzist_bf16",
    "encoder": "text_encoder.gguf"
  }
}
```

### `DELETE /chat/history`

Clear the per-client Rayzist Chat JSON history.

Requires `X-JustRayzist-Client`.

Sample response:

```json
{
  "status": "ok",
  "history": {
    "owner_id": "example-client",
    "next_number": 1,
    "exchange_count": 0,
    "exchanges": []
  },
  "capabilities": {
    "supported": true,
    "active_pack": "Rayzist_bf16",
    "encoder": "text_encoder.gguf"
  }
}
```

### `POST /chat`

Send one Rayzist Chat message through the active text encoder and append the numbered exchange to local JSON history.

Requires `X-JustRayzist-Client`.

Sample request body:

```json
{
  "message": "Give me three ways to improve a rainy city prompt.",
  "app_state": {
    "current_prompt": "rainy city street",
    "resolution": "1024x1024",
    "prompt_enhance": true,
    "queue_status": "0/5"
  },
  "max_new_tokens": 256,
  "temperature": 0.75
}
```

Sample response:

```json
{
  "status": "ok",
  "exchange": {
    "user": {
      "number": 1,
      "role": "user",
      "content": "Give me three ways to improve a rainy city prompt.",
      "created_at": "2026-04-08T12:00:00+00:00",
      "error": false
    },
    "assistant": {
      "number": 2,
      "role": "assistant",
      "content": "Specify rain intensity, reflected neon, street materials, and camera height.",
      "created_at": "2026-04-08T12:00:08+00:00",
      "error": false,
      "actions": [
        {
          "type": "set_prompt",
          "label": "Use Prompt",
          "prompt": "rainy neon city street, wet asphalt, reflected signs, low camera angle"
        },
        {
          "type": "open_route",
          "label": "Open API",
          "href": "/API"
        }
      ]
    }
  },
  "history": {
    "owner_id": "example-client",
    "next_number": 3,
    "exchange_count": 1
  },
  "capabilities": {
    "supported": true,
    "active_pack": "Rayzist_bf16",
    "encoder": "text_encoder.gguf"
  },
  "seed": 123456,
  "encoder": "text_encoder.gguf",
  "actions": [
    {
      "type": "set_prompt",
      "label": "Use Prompt",
      "prompt": "rainy neon city street, wet asphalt, reflected signs, low camera angle"
    },
    {
      "type": "open_route",
      "label": "Open API",
      "href": "/API"
    }
  ]
}
```

### `POST /lora-drafts`

Upload one `.safetensors` LoRA into draft storage for metadata inspection before saving it into the live library. LoRA uploads are capped at 10 GiB.

Sample request body:

```text
multipart/form-data with one file field named `file`
```

Sample response:

```json
{
  "status": "ok",
  "draft": {
    "draft_id": "cinematic-style",
    "display_name": "cinematic-style",
    "source_filename": "cinematic-style.safetensors",
    "detected_trigger_words": [
      "cinematic style",
      "moody light"
    ],
    "metadata_summary": {
      "ss_output_name": "cinematic-style"
    },
    "file_size_bytes": 12345678
  }
}
```

### `POST /lora-drafts/{draft_id}/detect-triggers`

Re-scan a staged LoRA draft for trigger words and metadata suggestions.

Sample response:

```json
{
  "status": "ok",
  "draft": {
    "draft_id": "cinematic-style",
    "display_name": "cinematic-style",
    "source_filename": "cinematic-style.safetensors",
    "detected_trigger_words": [
      "cinematic style",
      "moody light"
    ],
    "metadata_summary": {
      "ss_output_name": "cinematic-style"
    },
    "file_size_bytes": 12345678
  }
}
```

### `POST /loras`

Finalize a staged LoRA draft into the live library with a chosen name, saved trigger words, and an optional thumbnail image. Thumbnail uploads are capped at 10 MiB.

Sample request body:

```text
multipart/form-data with `draft_id`, `display_name`, `trigger_words` (JSON string), and optional `thumbnail` image
```

Sample response:

```json
{
  "status": "ok",
  "item": {
    "id": "cinematic-style",
    "display_name": "Cinematic Style",
    "source_filename": "cinematic-style.safetensors",
    "preview_url": "/loras/cinematic-style/preview",
    "preview_is_custom": true,
    "trigger_words": [
      "cinematic style",
      "moody light"
    ],
    "detected_trigger_words": [
      "cinematic style",
      "moody light"
    ],
    "metadata_summary": {
      "ss_output_name": "cinematic-style"
    },
    "file_size_bytes": 12345678
  },
  "capabilities": {
    "supported": true,
    "active_pack": "Rayzist_bf16",
    "max_active": 3,
    "min_weight": -2.0,
    "max_weight": 2.0,
    "default_weight": 1.0
  }
}
```

### `PATCH /loras/{lora_id}`

Update the display name, saved trigger words, and optional thumbnail image for one installed LoRA without replacing the weights file. Thumbnail uploads are capped at 10 MiB.

Sample request body:

```text
multipart/form-data with `display_name`, `trigger_words` (JSON string), and optional `thumbnail` image
```

Sample response:

```json
{
  "status": "ok",
  "item": {
    "id": "cinematic-style",
    "display_name": "Cinematic Style",
    "source_filename": "cinematic-style.safetensors",
    "preview_url": "/loras/cinematic-style/preview",
    "preview_is_custom": true,
    "trigger_words": [
      "cinematic style",
      "moody light"
    ],
    "detected_trigger_words": [
      "cinematic style",
      "moody light"
    ],
    "metadata_summary": {
      "ss_output_name": "cinematic-style"
    },
    "file_size_bytes": 12345678
  }
}
```

### `GET /loras/{lora_id}/preview`

Download the current preview image for one installed LoRA.

Sample response:

```text
PNG binary response
```

### `DELETE /loras/{lora_id}`

Delete one installed LoRA plus its sidecar JSON and preview image.

Sample response:

```json
{
  "status": "ok",
  "id": "cinematic-style",
  "deleted_files": 3
}
```

### `POST /generate`

Generate one image from prompt and dimensions in the current client scope.

Requires `X-JustRayzist-Client`.

Sample request body:

```json
{
  "job_id": "pending_1712345678901_abcd1234",
  "prompt": "A cinematic skyline at sunrise",
  "width": 1024,
  "height": 1024,
  "pack": "Rayzist_bf16",
  "seed": 123456,
  "scheduler_mode": "euler",
  "enhance_prompt": false,
  "procedural_creativity": 0,
  "loras": [
    {
      "id": "cinematic-style",
      "weight": 1.0
    }
  ]
}
```

Sample response:

```json
{
  "filename": "justrayzist_YYYYMMDD_hhmmss_000.png",
  "output_path": "S:\\STABLEDIFFUSION\\JustRayzist\\outputs\\example-client\\justrayzist_YYYYMMDD_hhmmss_000.png",
  "prompt": "A cinematic skyline at sunrise",
  "prompt_original": "A cinematic skyline at sunrise with __picturesque-locations__",
  "prompt_wildcard_resolved": "A cinematic skyline at sunrise with a chalet in the French Alps",
  "width": 1024,
  "height": 1024,
  "duration_ms": 12345,
  "url": "/images/justrayzist_YYYYMMDD_hhmmss_000.png",
  "prompt_enhanced": false,
  "prompt_effective_base": "A cinematic skyline at sunrise with a chalet in the French Alps",
  "prompt_effective": "A cinematic skyline at sunrise with a chalet in the French Alps, cinematic style",
  "scheduler_mode": "euler",
  "procedural_creativity": 0,
  "wildcard_count": 1,
  "wildcards": [
    {
      "id": "3c03cc4d8cf5476e831d6603626d7843",
      "display_name": "Picturesque Locations",
      "token": "picturesque-locations",
      "placeholder": "__picturesque-locations__",
      "selected_entry": "a chalet in the French Alps",
      "occurrence_index": 0,
      "prompt_offset": 31
    }
  ],
  "lora_count": 1,
  "loras": [
    {
      "id": "cinematic-style",
      "name": "cinematic-style",
      "weight": 1.0
    }
  ]
}
```

### `POST /img2img`

Generate one variation from a reference image upload plus prompt and similarity.

Requires `X-JustRayzist-Client`.

Sample request fields (`multipart/form-data`):

```json
{
  "image": "<binary image upload>",
  "prompt": "A cinematic skyline at sunrise",
  "pack": "Rayzist_bf16",
  "job_id": "pending_img2img_1712345678901_abcd1234",
  "seed": 123456,
  "scheduler_mode": "euler",
  "enhance_prompt": false,
  "similarity": 0.8,
  "loras": [
    {
      "id": "cinematic-style",
      "weight": 1.0
    }
  ]
}
```

Sample response:

```json
{
  "filename": "justrayzist_YYYYMMDD_hhmmss_001.png",
  "mode": "img2img",
  "source_filename": "reference.png",
  "source_width": 1024,
  "source_height": 768,
  "similarity": 0.8,
  "duration_ms": 12345,
  "url": "/images/justrayzist_YYYYMMDD_hhmmss_001.png",
  "prompt_enhanced": false,
  "prompt_effective_base": "A cinematic skyline at sunrise with a chalet in the French Alps",
  "prompt_effective": "A cinematic skyline at sunrise with a chalet in the French Alps, cinematic style",
  "scheduler_mode": "euler",
  "wildcard_count": 1,
  "lora_count": 1
}
```

### `POST /upscale`

Upscale one gallery image with the content-aware x2 path for photos and illustration.

Requires `X-JustRayzist-Client`.

Sample request body:

```json
{
  "job_id": "pending_upscale_1712345678901_abcd1234",
  "filename": "justrayzist_YYYYMMDD_hhmmss_000.png",
  "pack": "Rayzist_bf16",
  "seed": 123456,
  "scheduler_mode": "euler",
  "enhance_prompt": false
}
```

Sample response:

```json
{
  "filename": "justrayzist_YYYYMMDD_hhmmss_001.png",
  "mode": "api_upscale",
  "source_filename": "justrayzist_YYYYMMDD_hhmmss_000.png",
  "upscale_engine": "content_aware_ai_x2",
  "execution_mode": "content_aware_ai_x2",
  "duration_ms": 23456,
  "url": "/images/justrayzist_YYYYMMDD_hhmmss_001.png"
}
```

### `POST /clarity`

Run the fast FS clarity pipeline on one gallery image and return it at the original size.

Requires `X-JustRayzist-Client`.

Sample request body:

```json
{
  "job_id": "pending_clarity_1712345678901_abcd1234",
  "filename": "justrayzist_YYYYMMDD_hhmmss_000.png",
  "pack": "Rayzist_bf16",
  "seed": 123456,
  "scheduler_mode": "euler",
  "enhance_prompt": false
}
```

Sample response:

```json
{
  "filename": "justrayzist_YYYYMMDD_hhmmss_002.png",
  "mode": "api_clarity",
  "source_filename": "justrayzist_YYYYMMDD_hhmmss_000.png",
  "clarity_engine": "fs_unsharp_downscale",
  "working_width": 2048,
  "working_height": 2048,
  "duration_ms": 16789,
  "url": "/images/justrayzist_YYYYMMDD_hhmmss_002.png"
}
```

### `GET /client-jobs`

Return the current active generation, img2img, clarity, or upscale job for the requesting client.

Requires `X-JustRayzist-Client`.

Sample response:

```json
{
  "active_job": {
    "job_id": "pending_1712345678901_abcd1234",
    "kind": "generate",
    "status": "generating",
    "prompt": "A cinematic skyline at sunrise",
    "width": 1024,
    "height": 1024,
    "pack": "Rayzist_bf16",
    "seed": 123456,
    "enhance_prompt": false,
    "procedural_creativity": 0,
    "started_at": "2026-03-25T12:34:56+00:00"
  }
}
```

### `POST /client-jobs/cancel`

Cancel the current active client-scoped job or a specific active job id.

Requires `X-JustRayzist-Client`.

Sample request body:

```json
{
  "job_id": "pending_1712345678901_abcd1234"
}
```

Sample response:

```json
{
  "status": "ok",
  "cancel_requested": true,
  "job_id": "pending_1712345678901_abcd1234",
  "message": "Cancellation requested."
}
```

### `GET /images?prompt=skyline&color=blue&favorite=true&limit=50&offset=0&newest_first=true`

List images for the current client scope, with optional prompt, color, and favorite filtering.

Requires `X-JustRayzist-Client`.

Sample response:

```json
{
  "count": 1,
  "limit": 50,
  "offset": 0,
  "items": [
    {
      "filename": "justrayzist_YYYYMMDD_hhmmss_000.png",
      "favorite": 1
    }
  ],
  "color_cache": {
    "active": false,
    "version": "dominant_v6",
    "target_version": "dominant_v6",
    "needs_rebuild": false,
    "last_error": null
  }
}
```

### `GET /images/{filename}?client_id=<client-id>`

Download one image by filename. Use the query parameter for direct links and image tags.

Requires `X-JustRayzist-Client`.

Sample response:

```text
PNG binary response
```

### `POST /images/{filename}/favorite`

Set or clear the favorite flag for one client-scoped image.

Requires `X-JustRayzist-Client`.

Sample request body:

```json
{
  "favorite": true
}
```

Sample response:

```json
{
  "status": "ok",
  "filename": "justrayzist_YYYYMMDD_hhmmss_000.png",
  "favorite": true,
  "item": {
    "filename": "justrayzist_YYYYMMDD_hhmmss_000.png",
    "favorite": 1
  }
}
```

### `POST /images/download-zip`

Download a ZIP archive containing the selected client-scoped images.

Requires `X-JustRayzist-Client`.

Sample request body:

```json
{
  "filenames": [
    "justrayzist_YYYYMMDD_hhmmss_000.png",
    "justrayzist_YYYYMMDD_hhmmss_001.png"
  ]
}
```

Sample response:

```text
ZIP binary response (attachment filename: <client>_selection.zip)
```

### `DELETE /images/{filename}?confirm=DELETE`

Delete one image and its index entry in the current client scope.

Requires `X-JustRayzist-Client`.

Sample request body:

```json
{
  "confirm": "DELETE"
}
```

Sample response:

```json
{
  "status": "ok",
  "deleted_files": 1,
  "deleted_rows": 1,
  "remaining_rows": 0,
  "filename": "..."
}
```

### `DELETE /gallery?confirm=DELETE`

Delete all gallery images for the current client scope.

Requires `X-JustRayzist-Client`.

Sample request body:

```json
{
  "confirm": "DELETE"
}
```

Sample response:

```json
{
  "status": "ok",
  "deleted_files": 42,
  "deleted_rows": 42,
  "remaining_rows": 0
}
```

### `POST /gallery/rebuild`

Rebuild the current client-scoped gallery index after manual PNG copies, replacements, or deletions in the gallery folder.

Requires `X-JustRayzist-Client`.

Sample response:

```json
{
  "status": "ok",
  "owner_id": "example-client",
  "scanned_files": 12,
  "indexed": 2,
  "updated": 10,
  "removed_missing": 1,
  "total_items": 12
}
```

### `GET /gallery/import-sources`

List gallery import candidates from the legacy root or other userspaces.

Requires `X-JustRayzist-Client`.

Sample response:

```json
{
  "count": 2,
  "items": [
    {
      "source_id": "__legacy_root__",
      "image_count": 10
    }
  ]
}
```

### `POST /gallery/import`

Copy PNGs from another gallery source into the current client userspace.

Requires `X-JustRayzist-Client`.

Sample request body:

```json
{
  "source_id": "__legacy_root__",
  "dry_run": false
}
```

Sample response:

```json
{
  "status": "ok",
  "source_id": "__legacy_root__",
  "target_owner_id": "example-client",
  "imported": 12,
  "skipped": 0,
  "failed": 0
}
```

### `POST /server/kill`

Request local server shutdown from the hosting machine and local app origin.

Sample request body:

```json
{}
```

Sample response:

```json
{
  "status": "ok",
  "message": "Server shutdown initiated."
}
```

### `POST /server/restart`

Request local server restart from the hosting machine and local app origin.

Sample request body:

```json
{}
```

Sample response:

```json
{
  "status": "ok",
  "message": "Server restart initiated."
}
```
<!-- END GENERATED API EXAMPLES -->

## Output Locations

- Generated images: `outputs/`
- Metrics: `data/generation_metrics.jsonl`
- Gallery database: `data/gallery.db`
- Benchmark reports: `data/seedvr2*_benchmark_*.csv` and `.jsonl`

## Environment Variables

- `JUSTRAYZIST_ROOT`
- `JUSTRAYZIST_PROFILE` (engineering-only runtime tier override)
- `JUSTRAYZIST_PACK`
- `JUSTRAYZIST_OFFLINE`
- `JUSTRAYZIST_ENV`
- `JUSTRAYZIST_PYTHON`
- `JUSTRAYZIST_LISTEN`
- `JUSTRAYZIST_SKIP_GPU_PREFLIGHT`
