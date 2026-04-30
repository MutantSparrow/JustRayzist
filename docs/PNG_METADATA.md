# PNG Output Metadata Reference

All generated images are saved as PNG with metadata text chunks written by `app/storage/png_output.py`.

By default, release outputs keep compact metadata. Set `JUSTRAYZIST_METADEBUG=1` to keep the full diagnostic metadata payload for troubleshooting.

## Fixed Fields

Always present regardless of generation mode.

| Key | Value |
|---|---|
| `timestamp` | UTC ISO timestamp |
| `prompt` | Final prompt used for the saved image |
| `application_name` | App name from settings |
| `application_version` | App version from settings |
| `generated_with` | `"Just Rayzist!"` |
| `model_page` | HuggingFace model URL |

## Per-Mode Fields

`G` = Generate · `I` = Img2img · `U` = Upscale · `C` = Clarity

| Key | G | I | U | C | Notes |
|---|:---:|:---:|:---:|:---:|---|
| `owner_id` | ✓ | ✓ | ✓ | ✓ | |
| `mode` | — | `img2img` | `api_upscale` | `api_clarity` | Generate omits this key |
| **Prompt** | | | | | |
| `prompt_original` | ✓ | ✓ | ✓ | ✓ | User's raw input |
| `prompt_wildcard_resolved` | ✓ | ✓ | — | — | After wildcard token substitution |
| `prompt_effective_base` | ✓ | ✓ | — | — | Base prompt before LoRA trigger injection |
| `prompt_effective` | ✓ | ✓ | ✓ | ✓ | Final prompt fed to pipeline |
| `prompt_enhanced` | ✓ | ✓ | ✓ | ✓ | Bool — whether enhancement ran |
| **Image dimensions** | | | | | |
| `width` | ✓ | ✓ | ✓ | ✓ | Output image width |
| `height` | ✓ | ✓ | ✓ | ✓ | Output image height |
| `working_width` | — | — | ✓ | ✓ | Internal processing resolution |
| `working_height` | — | — | ✓ | ✓ | Internal processing resolution |
| **Generation params** | | | | | |
| `steps` | ✓ | ✓ | `0` | `0` | |
| `guidance_scale` | ✓ | ✓ | `0.0` | `0.0` | |
| `seed` | ✓ | ✓ | ✓ | ✓ | |
| `scheduler_mode` | ✓ | ✓ | ✓ | ✓ | |
| `inference_process` | ✓ | — | — | — | |
| `procedural_creativity` | ✓ | — | — | — | `0`–`3` |
| `duration_ms` | ✓ | ✓ | ✓ | ✓ | Total wall time ms |
| **Model / pack** | | | | | |
| `backend` | ✓ | ✓ | ✓ | ✓ | |
| `device` | ✓ | ✓ | ✓ | ✓ | |
| `model_pack` | ✓ | ✓ | ✓ | ✓ | |
| `selected_pack` | ✓ | ✓ | — | — | |
| `effective_pack` | ✓ | ✓ | — | — | After donor resolution |
| `derived_strategy` | ✓ | ✓ | — | — | e.g. `auto_fp8_storage` |
| `runtime_profile` | ✓ | ✓ | ✓ | ✓ | e.g. `balanced` |
| `resource_tier` | ✓ | ✓ | ✓ | ✓ | `high` / `balanced` / `constrained` |
| `execution_mode` | ✓ | ✓ | ✓ | ✓ | |
| **FP8** | | | | | |
| `fp8_checkpoint` | ✓ | ✓ | — | — | |
| `fp8_fallback_used` | ✓ | ✓ | — | — | |
| `fp8_fallback_reason` | ✓ | ✓ | — | — | |
| `fp8_runtime_mode` | ✓ | ✓ | — | — | |
| `fp8_normalized_tensor_count` | ✓ | ✓ | — | — | |
| `fp8_storage_preserved_tensor_count` | ✓ | ✓ | — | — | |
| `fp8_promoted_tensor_count` | ✓ | ✓ | — | — | |
| **LoRA** | | | | | |
| `loras_json` | ✓ | ✓ | — | — | JSON array of applied LoRAs |
| `lora_count` | ✓ | ✓ | — | — | |
| **Wildcards** | | | | | |
| `wildcards_json` | ✓ | ✓ | — | — | JSON array of expanded wildcards |
| `wildcard_count` | ✓ | ✓ | — | — | |
| **Img2img-only** | | | | | |
| `refine_strength` | — | ✓ | — | — | |
| `refine_pass_count` | — | ✓ | — | — | |
| `refine_pass1_steps` | — | ✓ | — | — | |
| `refine_pass2_steps` | — | ✓ | — | — | |
| `refine_pass2_strength` | — | ✓ | — | — | |
| `similarity` | — | ✓ | — | — | Normalized from user input |
| `source_filename` | — | ✓ | — | — | |
| `source_width` | — | ✓ | — | — | Normalized input size |
| `source_height` | — | ✓ | — | — | Normalized input size |
| `source_original_width` | — | ✓ | — | — | Pre-normalization |
| `source_original_height` | — | ✓ | — | — | Pre-normalization |
| **Upscale-only** | | | | | |
| `upscale_engine` | — | — | ✓ | — | |
| `upscale_duration_ms` | — | — | ✓ | — | |
| `step_timings_ms.*` | — | — | ✓ | — | Dynamic per-step timing keys |
| **Upscale + Clarity shared** | | | | | |
| `source_image` | — | — | ✓ | ✓ | Full path to source file |
| `source_generation_seed` | — | — | ✓ | ✓ | Seed read from source PNG metadata |
| `request_enhance_prompt` | — | — | ✓ | ✓ | |
| **Clarity-only** | | | | | |
| `clarity_engine` | — | — | — | ✓ | |
| `clarity_variant` | — | — | — | ✓ | |
| `clarity_duration_ms` | — | — | — | ✓ | |
| `clarity_fs_method` | — | — | — | ✓ | e.g. `linear` |
| `clarity_fs_type` | — | — | — | ✓ | e.g. `gaussian` |
| `clarity_fs_intensity` | — | — | — | ✓ | |
| `clarity_unsharp_stage` | — | — | — | ✓ | |
| `clarity_unsharp_radius` | — | — | — | ✓ | |
| `clarity_unsharp_percent` | — | — | — | ✓ | |
| `clarity_unsharp_threshold` | — | — | — | ✓ | |
| `step_timings_ms.*` | — | — | — | ✓ | Dynamic per-step timing keys |

## Source Locations

| Concern | File | Lines |
|---|---|---|
| Writer | `app/storage/png_output.py` | compact filter and PNG write |
| Generate metadata | `app/api/inference_service.py` | API generation save payload |
| Img2img metadata | `app/api/inference_service.py` | API img2img save payload |
| Upscale metadata | `app/api/inference_service.py` | API upscale save payload |
| Clarity metadata | `app/api/inference_service.py` | API clarity save payload |
| Generate result telemetry | `app/core/backends/diffusers_zimage.py` | 149–246 |
| Upscale result telemetry | `app/core/deblur.py` | 232–240 |
| Clarity result telemetry | `app/core/deblur.py` | 257–265 |
