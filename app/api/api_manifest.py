from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from app.version import APP_VERSION


@dataclass(frozen=True)
class ApiExample:
    method: str
    path: str
    description: str
    requires_client: bool
    request: Any
    response: Any
    include_in_readme: bool = True
    include_in_usage: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


API_EXAMPLES: tuple[ApiExample, ...] = (
    ApiExample(
        method="GET",
        path="/health",
        description="Service health plus current baseline/defaults and detected memory strategy.",
        requires_client=False,
        request=None,
        response={
            "status": "ok",
            "app": "JustRayzist",
            "version": APP_VERSION,
            "runtime_profile": "balanced",
            "resource_tier": "high",
            "active_pack": "Rayzist_bf16",
            "selected_pack": "Rayzist_bf16",
            "effective_pack": "Rayzist_bf16",
            "active_backend": "diffusers_zimage",
            "fp8_fallback_used": False,
            "fp8_fallback_reason": None,
            "fp8_runtime_mode": None,
            "fp8_storage_preserved_tensor_count": 0,
            "fp8_promoted_tensor_count": 0,
            "lora_capable": True,
            "wildcard_suggestions_capable": True,
            "gallery_color_cache_active": False,
            "gallery_color_cache_version": "dominant_v6",
            "gallery_color_cache_target_version": "dominant_v6",
            "gallery_color_cache_error": None,
            "offline_mode": True,
        },
    ),
    ApiExample(
        method="GET",
        path="/config",
        description="Resolved runtime configuration, paths, and current runtime status.",
        requires_client=False,
        request=None,
        response={
            "app_name": "JustRayzist",
            "app_version": APP_VERSION,
            "environment": "dev",
            "offline_mode": True,
            "runtime_profile": {
                "name": "balanced",
                "description": "16GB-class profile with moderate offload and stable throughput.",
            },
            "resource_tier": {
                "name": "high",
                "description": "24GB-class profile with minimal offload and highest throughput.",
            },
            "resource_tier_override": None,
            "auto_resource_tier": True,
            "paths": {
                "root_dir": "S:\\STABLEDIFFUSION\\JustRayzist",
                "models_dir": "S:\\STABLEDIFFUSION\\JustRayzist\\models",
                "model_packs_dir": "S:\\STABLEDIFFUSION\\JustRayzist\\models\\packs",
                "outputs_dir": "S:\\STABLEDIFFUSION\\JustRayzist\\outputs",
                "data_dir": "S:\\STABLEDIFFUSION\\JustRayzist\\data",
                "ui_dir": "S:\\STABLEDIFFUSION\\JustRayzist\\app\\ui",
            },
            "runtime": {
                "runtime_profile": "balanced",
                "resource_tier": "high",
                "resource_tier_description": "24GB-class profile with minimal offload and highest throughput.",
                "resource_tier_override": None,
                "auto_resource_tier": True,
                "active_pack": "Rayzist_bf16",
                "selected_pack": "Rayzist_bf16",
                "effective_pack": "Rayzist_bf16",
                "active_backend": "diffusers_zimage",
                "execution_mode": "model_offload",
                "fp8_checkpoint": False,
                "fp8_fallback_used": False,
                "fp8_fallback_reason": None,
                "fp8_runtime_mode": None,
                "fp8_normalized_tensor_count": 0,
                "fp8_storage_preserved_tensor_count": 0,
                "fp8_promoted_tensor_count": 0,
                "lora_capable": True,
                "wildcard_suggestions_capable": True,
                "gallery_color_cache_active": False,
                "gallery_color_cache_version": "dominant_v6",
                "gallery_color_cache_target_version": "dominant_v6",
                "gallery_color_cache_error": None,
            },
        },
    ),
    ApiExample(
        method="GET",
        path="/model-packs",
        description="List discovered, valid, public, and enabled model packs.",
        requires_client=False,
        request=None,
        response={
            "count": 1,
            "items": [
                {
                    "name": "Rayzist_bf16",
                    "path": "S:\\STABLEDIFFUSION\\JustRayzist\\models\\packs\\Rayzist_bf16\\modelpack.yaml",
                    "architecture": "z_image_turbo",
                },
            ],
        },
    ),
    ApiExample(
        method="GET",
        path="/loras",
        description="List installed LoRAs, preview URLs, saved trigger words, detected trigger suggestions, and runtime LoRA capabilities.",
        requires_client=False,
        request=None,
        response={
            "count": 1,
            "items": [
                {
                    "id": "cinematic-style",
                    "display_name": "cinematic-style",
                    "source_filename": "cinematic-style.safetensors",
                    "preview_url": "/loras/cinematic-style/preview",
                    "trigger_words": ["cinematic style"],
                    "detected_trigger_words": ["cinematic style", "moody light"],
                    "preview_is_custom": True,
                    "metadata_summary": {"ss_output_name": "cinematic-style"},
                    "file_size_bytes": 12345678,
                },
            ],
            "capabilities": {
                "supported": True,
                "active_pack": "Rayzist_bf16",
                "max_active": 3,
                "min_weight": -2.0,
                "max_weight": 2.0,
                "default_weight": 1.0,
            },
        },
    ),
    ApiExample(
        method="GET",
        path="/loras/events",
        description="Server-sent event stream that pushes LoRA library revision changes to connected browser clients.",
        requires_client=False,
        request=None,
        response="text/event-stream with revision numbers and periodic keep-alives",
        include_in_readme=False,
        include_in_usage=False,
    ),
    ApiExample(
        method="GET",
        path="/wildcards",
        description="List installed wildcards, their editable prompt tokens, multiline content, and runtime wildcard capabilities.",
        requires_client=False,
        request=None,
        response={
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
                    "updated_at": "2026-04-08T12:00:00+00:00",
                },
            ],
            "capabilities": {
                "supported": True,
                "active_pack": "Rayzist_bf16",
                "suggestions_supported": True,
            },
        },
    ),
    ApiExample(
        method="GET",
        path="/wildcards/events",
        description="Server-sent event stream that pushes wildcard library revision changes to connected browser clients.",
        requires_client=False,
        request=None,
        response="text/event-stream with revision numbers and periodic keep-alives",
        include_in_readme=False,
        include_in_usage=False,
    ),
    ApiExample(
        method="POST",
        path="/wildcards",
        description="Create one wildcard with a display name, editable prompt token, and multiline entries.",
        requires_client=False,
        request={
            "display_name": "Picturesque Locations",
            "token": "picturesque-locations",
            "content_text": "a cabin in the Schwarzwald\na chalet in the French Alps",
        },
        response={
            "status": "ok",
            "item": {
                "id": "3c03cc4d8cf5476e831d6603626d7843",
                "display_name": "Picturesque Locations",
                "token": "picturesque-locations",
                "placeholder": "__picturesque-locations__",
                "content_text": "a cabin in the Schwarzwald\na chalet in the French Alps",
                "entry_count": 2,
                "created_at": "2026-04-08T12:00:00+00:00",
                "updated_at": "2026-04-08T12:00:00+00:00",
            },
            "capabilities": {
                "supported": True,
                "active_pack": "Rayzist_bf16",
                "suggestions_supported": True,
            },
        },
    ),
    ApiExample(
        method="PATCH",
        path="/wildcards/{wildcard_id}",
        description="Update one wildcard's display name, editable prompt token, and multiline entries.",
        requires_client=False,
        request={
            "display_name": "Picturesque Locations",
            "token": "picturesque-locations",
            "content_text": "a cabin in the Schwarzwald\na chalet in the French Alps\na white sandy beach in Bora-Bora",
        },
        response={
            "status": "ok",
            "item": {
                "id": "3c03cc4d8cf5476e831d6603626d7843",
                "display_name": "Picturesque Locations",
                "token": "picturesque-locations",
                "placeholder": "__picturesque-locations__",
                "content_text": "a cabin in the Schwarzwald\na chalet in the French Alps\na white sandy beach in Bora-Bora",
                "entry_count": 3,
                "created_at": "2026-04-08T12:00:00+00:00",
                "updated_at": "2026-04-08T12:10:00+00:00",
            },
        },
    ),
    ApiExample(
        method="DELETE",
        path="/wildcards/{wildcard_id}",
        description="Delete one wildcard definition from the library.",
        requires_client=False,
        request=None,
        response={"status": "ok", "id": "3c03cc4d8cf5476e831d6603626d7843", "deleted": True},
    ),
    ApiExample(
        method="POST",
        path="/wildcards/suggestions",
        description="Ask the active text encoder for 10 wildcard entry suggestions that match a theme and stay within the example-length constraint.",
        requires_client=False,
        request={
            "theme": "picturesque locations",
            "format_example": "a cabin in the Schwarzwald",
            "seed": 123456,
            "existing_entries": ["a cabin in the Schwarzwald"],
        },
        response={
            "status": "ok",
            "suggestions": [
                "a chalet in the French Alps",
                "a white sandy beach in Bora-Bora",
                "a small cafe in a Parisian side street",
            ],
            "accepted_count": 3,
            "target_count": 10,
            "seed": 123456,
            "example_word_count": 5,
            "min_words": 5,
            "max_words": 5,
            "partial": True,
            "message": "Returned a partial set because the example-length filter was restrictive.",
        },
    ),
    ApiExample(
        method="POST",
        path="/lora-drafts",
        description="Upload one `.safetensors` LoRA into draft storage for metadata inspection before saving it into the live library. LoRA uploads are capped at 10 GiB.",
        requires_client=False,
        request="multipart/form-data with one file field named `file`",
        response={
            "status": "ok",
            "draft": {
                "draft_id": "cinematic-style",
                "display_name": "cinematic-style",
                "source_filename": "cinematic-style.safetensors",
                "detected_trigger_words": ["cinematic style", "moody light"],
                "metadata_summary": {"ss_output_name": "cinematic-style"},
                "file_size_bytes": 12345678,
            },
        },
    ),
    ApiExample(
        method="POST",
        path="/lora-drafts/{draft_id}/detect-triggers",
        description="Re-scan a staged LoRA draft for trigger words and metadata suggestions.",
        requires_client=False,
        request=None,
        response={
            "status": "ok",
            "draft": {
                "draft_id": "cinematic-style",
                "display_name": "cinematic-style",
                "source_filename": "cinematic-style.safetensors",
                "detected_trigger_words": ["cinematic style", "moody light"],
                "metadata_summary": {"ss_output_name": "cinematic-style"},
                "file_size_bytes": 12345678,
            },
        },
    ),
    ApiExample(
        method="POST",
        path="/loras",
        description="Finalize a staged LoRA draft into the live library with a chosen name, saved trigger words, and an optional thumbnail image. Thumbnail uploads are capped at 10 MiB.",
        requires_client=False,
        request="multipart/form-data with `draft_id`, `display_name`, `trigger_words` (JSON string), and optional `thumbnail` image",
        response={
            "status": "ok",
            "item": {
                "id": "cinematic-style",
                "display_name": "Cinematic Style",
                "source_filename": "cinematic-style.safetensors",
                "preview_url": "/loras/cinematic-style/preview",
                "preview_is_custom": True,
                "trigger_words": ["cinematic style", "moody light"],
                "detected_trigger_words": ["cinematic style", "moody light"],
                "metadata_summary": {"ss_output_name": "cinematic-style"},
                "file_size_bytes": 12345678,
            },
            "capabilities": {
                "supported": True,
                "active_pack": "Rayzist_bf16",
                "max_active": 3,
                "min_weight": -2.0,
                "max_weight": 2.0,
                "default_weight": 1.0,
            },
        },
    ),
    ApiExample(
        method="PATCH",
        path="/loras/{lora_id}",
        description="Update the display name, saved trigger words, and optional thumbnail image for one installed LoRA without replacing the weights file. Thumbnail uploads are capped at 10 MiB.",
        requires_client=False,
        request="multipart/form-data with `display_name`, `trigger_words` (JSON string), and optional `thumbnail` image",
        response={
            "status": "ok",
            "item": {
                "id": "cinematic-style",
                "display_name": "Cinematic Style",
                "source_filename": "cinematic-style.safetensors",
                "preview_url": "/loras/cinematic-style/preview",
                "preview_is_custom": True,
                "trigger_words": ["cinematic style", "moody light"],
                "detected_trigger_words": ["cinematic style", "moody light"],
                "metadata_summary": {"ss_output_name": "cinematic-style"},
                "file_size_bytes": 12345678,
            },
        },
    ),
    ApiExample(
        method="GET",
        path="/loras/{lora_id}/preview",
        description="Download the current preview image for one installed LoRA.",
        requires_client=False,
        request=None,
        response="PNG binary response",
    ),
    ApiExample(
        method="DELETE",
        path="/loras/{lora_id}",
        description="Delete one installed LoRA plus its sidecar JSON and preview image.",
        requires_client=False,
        request=None,
        response={"status": "ok", "id": "cinematic-style", "deleted_files": 3},
    ),
    ApiExample(
        method="POST",
        path="/generate",
        description="Generate one image from prompt and dimensions in the current client scope.",
        requires_client=True,
        request={
            "job_id": "pending_1712345678901_abcd1234",
            "prompt": "A cinematic skyline at sunrise",
            "width": 1024,
            "height": 1024,
            "pack": "Rayzist_bf16",
            "seed": 123456,
            "scheduler_mode": "euler",
            "enhance_prompt": False,
            "procedural_creativity": 0,
            "loras": [{"id": "cinematic-style", "weight": 1.0}],
        },
        response={
            "filename": "justrayzist_YYYYMMDD_hhmmss_000.png",
            "output_path": "S:\\STABLEDIFFUSION\\JustRayzist\\outputs\\example-client\\justrayzist_YYYYMMDD_hhmmss_000.png",
            "prompt": "A cinematic skyline at sunrise",
            "prompt_original": "A cinematic skyline at sunrise with __picturesque-locations__",
            "prompt_wildcard_resolved": "A cinematic skyline at sunrise with a chalet in the French Alps",
            "width": 1024,
            "height": 1024,
            "duration_ms": 12345,
            "url": "/images/justrayzist_YYYYMMDD_hhmmss_000.png",
            "prompt_enhanced": False,
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
                    "prompt_offset": 31,
                }
            ],
            "lora_count": 1,
            "loras": [{"id": "cinematic-style", "name": "cinematic-style", "weight": 1.0}],
        },
    ),
    ApiExample(
        method="POST",
        path="/upscale",
        description="Upscale one gallery image with the fixed SeedVR2 direct x2 faithful path.",
        requires_client=True,
        request={
            "job_id": "pending_upscale_1712345678901_abcd1234",
            "filename": "justrayzist_YYYYMMDD_hhmmss_000.png",
            "pack": "Rayzist_bf16",
            "seed": 123456,
            "scheduler_mode": "euler",
            "enhance_prompt": False,
        },
        response={
            "filename": "justrayzist_YYYYMMDD_hhmmss_001.png",
            "mode": "api_upscale",
            "source_filename": "justrayzist_YYYYMMDD_hhmmss_000.png",
            "upscale_engine": "seedvr2_direct_x2_faithful",
            "execution_mode": "seedvr2_direct_x2_faithful",
            "duration_ms": 23456,
            "url": "/images/justrayzist_YYYYMMDD_hhmmss_001.png",
        },
    ),
    ApiExample(
        method="GET",
        path="/client-jobs",
        description="Return the current active generation or upscale job for the requesting client.",
        requires_client=True,
        request=None,
        response={
            "active_job": {
                "job_id": "pending_1712345678901_abcd1234",
                "kind": "generate",
                "status": "generating",
                "prompt": "A cinematic skyline at sunrise",
                "width": 1024,
                "height": 1024,
                "pack": "Rayzist_bf16",
                "seed": 123456,
                "enhance_prompt": False,
                "procedural_creativity": 0,
                "started_at": "2026-03-25T12:34:56+00:00",
            }
        },
        include_in_readme=False,
    ),
    ApiExample(
        method="POST",
        path="/client-jobs/cancel",
        description="Cancel the current active client-scoped job or a specific active job id.",
        requires_client=True,
        request={"job_id": "pending_1712345678901_abcd1234"},
        response={
            "status": "ok",
            "cancel_requested": True,
            "job_id": "pending_1712345678901_abcd1234",
            "message": "Cancellation requested.",
        },
        include_in_readme=False,
    ),
    ApiExample(
        method="GET",
        path="/images?prompt=skyline&color=blue&favorite=true&limit=50&offset=0&newest_first=true",
        description="List images for the current client scope, with optional prompt, color, and favorite filtering.",
        requires_client=True,
        request=None,
        response={
            "count": 1,
            "limit": 50,
            "offset": 0,
            "items": [{"filename": "justrayzist_YYYYMMDD_hhmmss_000.png", "favorite": 1}],
            "color_cache": {
                "active": False,
                "version": "dominant_v6",
                "target_version": "dominant_v6",
                "needs_rebuild": False,
                "last_error": None,
            },
        },
        include_in_readme=False,
    ),
    ApiExample(
        method="GET",
        path="/images/{filename}?client_id=<client-id>",
        description="Download one image by filename. Use the query parameter for direct links and image tags.",
        requires_client=True,
        request=None,
        response="PNG binary response",
        include_in_readme=False,
    ),
    ApiExample(
        method="POST",
        path="/images/{filename}/favorite",
        description="Set or clear the favorite flag for one client-scoped image.",
        requires_client=True,
        request={"favorite": True},
        response={
            "status": "ok",
            "filename": "justrayzist_YYYYMMDD_hhmmss_000.png",
            "favorite": True,
            "item": {"filename": "justrayzist_YYYYMMDD_hhmmss_000.png", "favorite": 1},
        },
        include_in_readme=False,
    ),
    ApiExample(
        method="POST",
        path="/images/download-zip",
        description="Download a ZIP archive containing the selected client-scoped images.",
        requires_client=True,
        request={"filenames": ["justrayzist_YYYYMMDD_hhmmss_000.png", "justrayzist_YYYYMMDD_hhmmss_001.png"]},
        response="ZIP binary response (attachment filename: <client>_selection.zip)",
    ),
    ApiExample(
        method="DELETE",
        path="/images/{filename}?confirm=DELETE",
        description="Delete one image and its index entry in the current client scope.",
        requires_client=True,
        request={"confirm": "DELETE"},
        response={"status": "ok", "deleted_files": 1, "deleted_rows": 1, "remaining_rows": 0, "filename": "..."},
        include_in_readme=False,
    ),
    ApiExample(
        method="DELETE",
        path="/gallery?confirm=DELETE",
        description="Delete all gallery images for the current client scope.",
        requires_client=True,
        request={"confirm": "DELETE"},
        response={"status": "ok", "deleted_files": 42, "deleted_rows": 42, "remaining_rows": 0},
        include_in_readme=False,
    ),
    ApiExample(
        method="POST",
        path="/gallery/rebuild",
        description="Rebuild the current client-scoped gallery index after manual PNG copies, replacements, or deletions in the gallery folder.",
        requires_client=True,
        request=None,
        response={
            "status": "ok",
            "owner_id": "example-client",
            "scanned_files": 12,
            "indexed": 2,
            "updated": 10,
            "removed_missing": 1,
            "total_items": 12,
        },
        include_in_readme=False,
    ),
    ApiExample(
        method="GET",
        path="/gallery/import-sources",
        description="List gallery import candidates from the legacy root or other userspaces.",
        requires_client=True,
        request=None,
        response={"count": 2, "items": [{"source_id": "__legacy_root__", "image_count": 10}]},
        include_in_readme=False,
    ),
    ApiExample(
        method="POST",
        path="/gallery/import",
        description="Copy PNGs from another gallery source into the current client userspace.",
        requires_client=True,
        request={"source_id": "__legacy_root__", "dry_run": False},
        response={
            "status": "ok",
            "source_id": "__legacy_root__",
            "target_owner_id": "example-client",
            "imported": 12,
            "skipped": 0,
            "failed": 0,
        },
        include_in_readme=False,
    ),
    ApiExample(
        method="POST",
        path="/server/kill",
        description="Request local server shutdown.",
        requires_client=False,
        request={},
        response={"status": "ok", "message": "Server shutdown initiated."},
        include_in_readme=False,
    ),
)


def api_manifest_payload() -> dict[str, Any]:
    items = [entry.to_dict() for entry in API_EXAMPLES]
    return {"count": len(items), "items": items}


def _format_example_block(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2)


def render_route_summary_markdown(*, include_usage_only: bool) -> str:
    lines = []
    for entry in API_EXAMPLES:
        if include_usage_only and not entry.include_in_usage:
            continue
        if not include_usage_only and not entry.include_in_readme:
            continue
        lines.append(f"- `{entry.method} {entry.path}`")
    return "\n".join(lines)


def render_examples_markdown(*, include_usage_only: bool) -> str:
    lines: list[str] = []
    for entry in API_EXAMPLES:
        if include_usage_only and not entry.include_in_usage:
            continue
        if not include_usage_only and not entry.include_in_readme:
            continue
        lines.append(f"### `{entry.method} {entry.path}`")
        lines.append("")
        lines.append(entry.description)
        lines.append("")
        if entry.requires_client:
            lines.append("Requires `X-JustRayzist-Client`.")
            lines.append("")
        if entry.request is not None:
            lines.append("Sample request body:")
            lines.append("")
            if isinstance(entry.request, str):
                lines.append("```text")
            else:
                lines.append("```json")
            lines.append(_format_example_block(entry.request))
            lines.append("```")
            lines.append("")
        lines.append("Sample response:")
        lines.append("")
        if isinstance(entry.response, str):
            lines.append("```text")
            lines.append(entry.response)
            lines.append("```")
        else:
            lines.append("```json")
            lines.append(_format_example_block(entry.response))
            lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip()

