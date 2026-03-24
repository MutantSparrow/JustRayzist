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
    request: dict[str, Any] | None
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
        method="POST",
        path="/generate",
        description="Generate one image from prompt and dimensions in the current client scope.",
        requires_client=True,
        request={
            "prompt": "A cinematic skyline at sunrise",
            "width": 1024,
            "height": 1024,
            "pack": "Rayzist_bf16",
            "seed": 123456,
            "scheduler_mode": "euler",
            "enhance_prompt": False,
            "procedural_creativity": 0,
        },
        response={
            "filename": "justrayzist_YYYYMMDD_hhmmss_000.png",
            "output_path": "S:\\STABLEDIFFUSION\\JustRayzist\\outputs\\example-client\\justrayzist_YYYYMMDD_hhmmss_000.png",
            "prompt": "A cinematic skyline at sunrise",
            "width": 1024,
            "height": 1024,
            "duration_ms": 12345,
            "url": "/images/justrayzist_YYYYMMDD_hhmmss_000.png",
            "prompt_enhanced": False,
            "scheduler_mode": "euler",
            "procedural_creativity": 0,
        },
    ),
    ApiExample(
        method="POST",
        path="/upscale",
        description="Upscale one gallery image with the app's mixed-model fast upscale flow.",
        requires_client=True,
        request={
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
            "upscale_engine": "x2_seedvr2_blend",
            "duration_ms": 23456,
            "url": "/images/justrayzist_YYYYMMDD_hhmmss_001.png",
        },
    ),
    ApiExample(
        method="GET",
        path="/images?prompt=skyline&limit=50&offset=0&newest_first=true",
        description="List images for the current client scope.",
        requires_client=True,
        request=None,
        response={
            "count": 1,
            "limit": 50,
            "offset": 0,
            "items": [{"filename": "justrayzist_YYYYMMDD_hhmmss_000.png"}],
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
