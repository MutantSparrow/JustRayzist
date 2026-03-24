from __future__ import annotations

import json
import logging
import os
import csv
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from uuid import uuid4

try:
    import typer
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    if exc.name == "typer":
        raise SystemExit(
            "Missing dependency: typer.\n"
            "Install project dependencies with one of:\n"
            "  python -m pip install -e .\n"
            "  python -m pip install -e .[dev]"
        ) from exc
    raise

from app.config import load_settings
from app.api.inference_service import InferenceService
from app.core.logging import configure_logging
from app.core.backends import SUPPORTED_BACKENDS
from app.core.model_registry import (
    ModelPackValidationError,
    discover_model_packs,
    load_model_pack,
    load_model_pack_by_name,
)

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

cli = typer.Typer(add_completion=False, help="JustRayzist CLI")

DEFAULT_FP8_SUITE_PACKS = [
    "Rayzist_bf16",
    "Rayzist_bf16__auto_fp8_storage",
    "Rayzist_fp8_full",
]

DEFAULT_FP8_SUITE_PROMPTS = [
    (
        "portrait",
        "close-up portrait of a weathered astronaut, cinematic rim light, pores and skin texture, "
        "shallow depth of field, 85mm lens, ultra detailed",
    ),
    (
        "action",
        "explosive rooftop chase at sunset, heroine leaping between collapsing buildings, sparks, debris, "
        "motion blur, dramatic perspective, cinematic action scene",
    ),
    (
        "anime",
        "anime key visual of a lone swordswoman under neon rain, dynamic three-quarter shot, glowing city "
        "reflections, sharp cel shading, expressive eyes, dramatic composition",
    ),
]

PROMPT_GRID_SCENARIOS = [
    {"label": "forced_high", "source": "forced", "profile_override": "high"},
    {"label": "forced_balanced", "source": "forced", "profile_override": "balanced"},
    {"label": "forced_constrained", "source": "forced", "profile_override": "constrained"},
    {"label": "auto", "source": "auto", "profile_override": None},
]


def _load_pack_or_exit(settings, pack_name: str):
    try:
        return load_model_pack_by_name(settings.paths.model_packs_dir, pack_name)
    except ModelPackValidationError as exc:
        typer.echo(f"Model pack error: {exc}")
        raise typer.Exit(code=1) from exc


def _load_runtime_pack_or_exit(settings, pack_name: str):
    try:
        inference = InferenceService(settings=settings)
        _, effective_pack, _ = inference.resolve_runtime_pack(
            pack_name,
            apply_resource_tier_policy=False,
        )
        return effective_pack
    except ModelPackValidationError as exc:
        typer.echo(f"Model pack error: {exc}")
        raise typer.Exit(code=1) from exc


def _assert_supported_backend_or_exit(model_pack) -> None:
    backends = [
        str(name).strip().lower()
        for name in model_pack.backend_preference
        if str(name).strip()
    ]
    if not any(name in SUPPORTED_BACKENDS for name in backends):
        typer.echo(
            "Unsupported backend preference list "
            f"{model_pack.backend_preference!r} in bootstrap version. "
            f"Include one of: {sorted(SUPPORTED_BACKENDS)}."
        )
        raise typer.Exit(code=1)


def _memory_source_and_bytes(result) -> tuple[str | None, int | None]:
    if result.cuda_memory_after is not None:
        return "cuda_reserved_bytes", result.cuda_memory_after.reserved_bytes
    if result.process_memory_after is not None:
        return "process_rss_bytes", result.process_memory_after.rss_bytes
    return None, None


def _resolve_cli_path(root: Path, candidate: Path) -> Path:
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def _preview_output_dir(root: Path, stem: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = root / stem / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def _latent_stats(tensor) -> dict[str, float]:
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def _normalize_tensor_to_uint8(tensor, *, low_quantile: float = 0.02, high_quantile: float = 0.98):
    import torch

    channel = tensor.detach().to(dtype=torch.float32, device="cpu")
    flat = channel.reshape(-1)
    low = float(torch.quantile(flat, low_quantile).item())
    high = float(torch.quantile(flat, high_quantile).item())
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        low = float(channel.min().item())
        high = float(channel.max().item())
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        normalized = torch.zeros_like(channel, dtype=torch.float32)
    else:
        normalized = ((channel - low) / (high - low)).clamp(0.0, 1.0)
    return (normalized * 255.0).round().to(dtype=torch.uint8)


def _build_latent_composite_image(latent_tensor, size: int):
    from PIL import Image
    import torch

    latent = latent_tensor.detach().to(dtype=torch.float32, device="cpu").squeeze(0)
    groups = (
        latent[0:5].mean(dim=0),
        latent[5:10].mean(dim=0),
        latent[10:16].mean(dim=0),
    )
    rgb = torch.stack(
        [_normalize_tensor_to_uint8(channel) for channel in groups],
        dim=-1,
    ).numpy()
    image = Image.fromarray(rgb, mode="RGB")
    return image.resize((size, size), Image.Resampling.NEAREST)


def _build_latent_channel_grid_image(latent_tensor, tile_size: int):
    from PIL import Image, ImageDraw, ImageFont

    latent = latent_tensor.detach().to(dtype=latent_tensor.dtype, device="cpu").squeeze(0)
    channels = int(latent.shape[0])
    columns = 4
    rows = int(math.ceil(channels / columns))
    label_h = 16
    canvas = Image.new("RGB", (columns * tile_size, rows * (tile_size + label_h)), color=(18, 20, 24))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index in range(channels):
        row = index // columns
        col = index % columns
        x = col * tile_size
        y = row * (tile_size + label_h)
        tile = _normalize_tensor_to_uint8(latent[index]).numpy()
        image = Image.fromarray(tile, mode="L").convert("RGB")
        image = image.resize((tile_size, tile_size), Image.Resampling.NEAREST)
        canvas.paste(image, (x, y))
        draw.text((x + 4, y + tile_size + 2), f"ch {index:02d}", fill=(210, 210, 210), font=font)
    return canvas


def _image_mse_psnr(baseline_image, candidate_image) -> tuple[float, float]:
    from PIL import Image, ImageChops, ImageStat

    baseline_rgb = baseline_image.convert("RGB")
    candidate_rgb = candidate_image.convert("RGB")
    if candidate_rgb.size != baseline_rgb.size:
        candidate_rgb = candidate_rgb.resize(
            baseline_rgb.size,
            Image.Resampling.BICUBIC,
        )
    diff = ImageChops.difference(baseline_rgb, candidate_rgb)
    stats = ImageStat.Stat(diff)
    pixel_count = max(1, baseline_rgb.width * baseline_rgb.height)
    channel_count = max(1, len(diff.getbands()))
    mse = float(sum(channel_sum2 / pixel_count for channel_sum2 in stats.sum2) / channel_count)
    if mse <= 0.0:
        return 0.0, float("inf")
    psnr = 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)
    return mse, float(psnr)


def _result_reserved_bytes(result) -> int | None:
    if result.cuda_memory_after is None:
        return None
    return int(result.cuda_memory_after.reserved_bytes)


def _result_max_reserved_bytes(result) -> int | None:
    if result.cuda_memory_after is None:
        return None
    return int(result.cuda_memory_after.max_reserved_bytes)


def _summarize_compare_side(rows: list[dict[str, object]], prefix: str) -> dict[str, float | int | None]:
    duration_values = [
        int(row[f"{prefix}_duration_ms"])
        for row in rows
        if row.get(f"{prefix}_status") == "success" and row.get(f"{prefix}_duration_ms") is not None
    ]
    reserved_values = [
        int(row[f"{prefix}_reserved_bytes"])
        for row in rows
        if row.get(f"{prefix}_status") == "success" and row.get(f"{prefix}_reserved_bytes") is not None
    ]
    max_reserved_values = [
        int(row[f"{prefix}_max_reserved_bytes"])
        for row in rows
        if row.get(f"{prefix}_status") == "success" and row.get(f"{prefix}_max_reserved_bytes") is not None
    ]
    return {
        "median_duration_ms": float(statistics.median(duration_values)) if duration_values else None,
        "median_reserved_bytes": float(statistics.median(reserved_values)) if reserved_values else None,
        "max_reserved_bytes": max(max_reserved_values) if max_reserved_values else None,
    }


def _bytes_to_mb(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) / (1024 * 1024):.1f}"


def _run_pack_compare_generation(
    *,
    session,
    settings,
    model_pack,
    prompt: str,
    width: int,
    height: int,
    steps: int | None,
    guidance_scale: float | None,
    seed: int | None,
    enhance_prompt: bool,
    output_dir: Path,
    run_label: str,
    pair_id: str,
    role: str,
    recycle_before: bool = False,
    recycle_after: bool = False,
    record_metric: bool = True,
):
    from app.core.memory import now_perf
    from app.core.worker import GenerationRequest
    from app.storage import append_generation_metric, build_output_path, save_png_with_metadata

    if recycle_before:
        session.recycle(f"pack-compare {role} {run_label} pre-run recycle")

    wall_started = now_perf()
    record: dict[str, object] = {
        "pair_id": pair_id,
        "run_label": run_label,
        "role": role,
        "pack": model_pack.name,
        "seed": seed,
        "status": "pending",
        "duration_ms": None,
        "reserved_bytes": None,
        "max_reserved_bytes": None,
        "execution_mode": "",
        "runtime_profile": settings.runtime_profile.name,
        "output_path": "",
        "error": "",
    }
    result = None
    try:
        result = session.generate(
            GenerationRequest(
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                enhance_prompt=enhance_prompt,
            )
        )
        saved_path = save_png_with_metadata(
            image=result.image,
            prompt=result.prompt_effective,
            settings=settings,
            output_path=build_output_path(
                output_dir,
                prefix=f"packcompare_{role}_{run_label}_{pair_id}",
            ),
            extra_metadata={
                "mode": "pack_compare",
                "compare_pair_id": pair_id,
                "compare_run_label": run_label,
                "compare_role": role,
                "model_pack": model_pack.name,
                "duration_ms": result.duration_ms,
                "seed": seed,
                "runtime_profile": result.runtime_profile,
                "execution_mode": result.execution_mode,
            },
        )
        record["status"] = "success"
        record["duration_ms"] = int(result.duration_ms)
        record["reserved_bytes"] = _result_reserved_bytes(result)
        record["max_reserved_bytes"] = _result_max_reserved_bytes(result)
        record["execution_mode"] = str(result.execution_mode or "")
        record["runtime_profile"] = str(result.runtime_profile or settings.runtime_profile.name)
        record["output_path"] = str(saved_path)
        if record_metric:
            append_generation_metric(
                settings=settings,
                payload={
                    "mode": "pack_compare_run",
                    "compare_pair_id": pair_id,
                    "compare_run_label": run_label,
                    "compare_role": role,
                    "model_pack": model_pack.name,
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "output_path": str(saved_path),
                    **result.telemetry_dict(),
                },
            )
    except Exception as exc:  # noqa: BLE001
        record["status"] = "error"
        record["duration_ms"] = int((now_perf() - wall_started) * 1000)
        record["error"] = str(exc)
    finally:
        if recycle_after:
            session.recycle(f"pack-compare {role} {run_label} post-run recycle")

    return record, result


def _build_pack_compare_pair_row(
    *,
    pair_id: str,
    run_label: str,
    profile_name: str,
    prompt: str,
    width: int,
    height: int,
    seed: int | None,
    baseline_pack_name: str,
    candidate_pack_name: str,
    baseline_record: dict[str, object],
    candidate_record: dict[str, object],
    baseline_result,
    candidate_result,
) -> dict[str, object]:
    mse = None
    psnr = None
    if baseline_result is not None and candidate_result is not None:
        mse, psnr = _image_mse_psnr(baseline_result.image, candidate_result.image)

    return {
        "pair_id": pair_id,
        "run_label": run_label,
        "profile": profile_name,
        "prompt": prompt,
        "width": width,
        "height": height,
        "seed": seed,
        "baseline_pack": baseline_pack_name,
        "candidate_pack": candidate_pack_name,
        "baseline_status": baseline_record["status"],
        "candidate_status": candidate_record["status"],
        "baseline_duration_ms": baseline_record["duration_ms"],
        "candidate_duration_ms": candidate_record["duration_ms"],
        "baseline_reserved_bytes": baseline_record["reserved_bytes"],
        "candidate_reserved_bytes": candidate_record["reserved_bytes"],
        "baseline_max_reserved_bytes": baseline_record["max_reserved_bytes"],
        "candidate_max_reserved_bytes": candidate_record["max_reserved_bytes"],
        "baseline_execution_mode": baseline_record["execution_mode"],
        "candidate_execution_mode": candidate_record["execution_mode"],
        "baseline_output_path": baseline_record["output_path"],
        "candidate_output_path": candidate_record["output_path"],
        "baseline_error": baseline_record["error"],
        "candidate_error": candidate_record["error"],
        "mse": mse,
        "psnr_db": psnr,
    }


def _draw_wrapped_text(draw, text: str, *, x: int, y: int, width: int, fill, font, line_height: int) -> int:
    words = text.split()
    if not words:
        return y
    current = words[0]
    lines: list[str] = []
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textlength(candidate, font=font) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += line_height
    return y


def _slugify_label(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return normalized.strip("_") or "item"


def _format_optional_number(value: object, *, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return "inf"
    return f"{float(value):.{digits}f}"


def _summarize_suite_pack(rows: list[dict[str, object]], pack_name: str) -> dict[str, float | int | None]:
    duration_values = [
        int(row["duration_ms"])
        for row in rows
        if row.get("pack") == pack_name and row.get("status") == "success" and row.get("duration_ms") is not None
    ]
    reserved_values = [
        int(row["reserved_bytes"])
        for row in rows
        if row.get("pack") == pack_name and row.get("status") == "success" and row.get("reserved_bytes") is not None
    ]
    max_reserved_values = [
        int(row["max_reserved_bytes"])
        for row in rows
        if row.get("pack") == pack_name and row.get("status") == "success" and row.get("max_reserved_bytes") is not None
    ]
    mse_values = [
        float(row["mse"])
        for row in rows
        if row.get("pack") == pack_name and row.get("mse") is not None
    ]
    psnr_values = [
        float(row["psnr_db"])
        for row in rows
        if row.get("pack") == pack_name
        and row.get("psnr_db") is not None
        and math.isfinite(float(row["psnr_db"]))
    ]
    return {
        "median_duration_ms": float(statistics.median(duration_values)) if duration_values else None,
        "median_reserved_bytes": float(statistics.median(reserved_values)) if reserved_values else None,
        "max_reserved_bytes": max(max_reserved_values) if max_reserved_values else None,
        "avg_mse": (sum(mse_values) / len(mse_values)) if mse_values else None,
        "avg_psnr_db": (sum(psnr_values) / len(psnr_values)) if psnr_values else None,
    }


def _build_pack_suite_row(
    *,
    suite_id: str,
    prompt_key: str,
    prompt: str,
    run_label: str,
    seed: int | None,
    width: int,
    height: int,
    baseline_pack_name: str,
    baseline_record: dict[str, object],
    baseline_result,
    pack_name: str,
    record: dict[str, object],
    result,
) -> dict[str, object]:
    mse = None
    psnr = None
    if (
        pack_name != baseline_pack_name
        and baseline_result is not None
        and result is not None
        and record.get("status") == "success"
    ):
        mse, psnr = _image_mse_psnr(baseline_result.image, result.image)

    return {
        "suite_id": suite_id,
        "prompt_key": prompt_key,
        "prompt": prompt,
        "run_label": run_label,
        "seed": seed,
        "width": width,
        "height": height,
        "baseline_pack": baseline_pack_name,
        "pack": pack_name,
        "status": record["status"],
        "duration_ms": record["duration_ms"],
        "reserved_bytes": record["reserved_bytes"],
        "max_reserved_bytes": record["max_reserved_bytes"],
        "execution_mode": record["execution_mode"],
        "output_path": record["output_path"],
        "error": record["error"],
        "baseline_output_path": baseline_record["output_path"],
        "mse": mse,
        "psnr_db": psnr,
    }


def _build_benchmark_contact_sheet(
    *,
    rows: list[dict[str, object]],
    pack_order: list[str],
    run_order: list[str],
    title: str,
    prompt: str,
 ) -> "PILImage":
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    sidebar_w = 180
    thumb_size = 220
    label_h = 82
    gutter = 18
    header_h = 108
    bg = (18, 20, 24)
    fg = (240, 240, 240)
    subtle = (170, 175, 184)
    tile_bg = (32, 35, 41)
    tile_failed = (58, 28, 32)
    tile_missing = (28, 30, 36)
    border = (72, 78, 90)
    width = sidebar_w + gutter + len(run_order) * (thumb_size + gutter) + gutter
    height = header_h + gutter + len(pack_order) * (thumb_size + label_h + gutter) + gutter
    sheet = Image.new("RGB", (width, height), color=bg)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()

    draw.text((gutter, 18), title, fill=fg, font=font)
    _draw_wrapped_text(
        draw,
        prompt,
        x=gutter,
        y=38,
        width=width - (gutter * 2),
        fill=subtle,
        font=small_font,
        line_height=14,
    )

    row_index = {(str(row["pack"]), str(row["run_label"])): row for row in rows}
    for col, run_label in enumerate(run_order):
        x = sidebar_w + gutter + col * (thumb_size + gutter)
        draw.text((x, header_h - 24), run_label, fill=fg, font=font)

    for row_idx, pack_name in enumerate(pack_order):
        y = header_h + gutter + row_idx * (thumb_size + label_h + gutter)
        draw.text((gutter, y + 4), pack_name, fill=fg, font=font)
        for col, run_label in enumerate(run_order):
            x = sidebar_w + gutter + col * (thumb_size + gutter)
            row = row_index.get((pack_name, run_label))
            tile = Image.new("RGB", (thumb_size, thumb_size), color=tile_bg)
            if row and row.get("status") == "success" and row.get("output_path"):
                with Image.open(str(row["output_path"])) as image:
                    tile = ImageOps.fit(image.convert("RGB"), (thumb_size, thumb_size), Image.Resampling.LANCZOS)
            else:
                tile_color = tile_missing
                if row and row.get("status") == "error":
                    tile_color = tile_failed
                tile = Image.new("RGB", (thumb_size, thumb_size), color=tile_color)
                placeholder_draw = ImageDraw.Draw(tile)
                if row and row.get("status") == "error":
                    placeholder_draw.text((12, 12), "FAILED", fill=fg, font=font)
                    _draw_wrapped_text(
                        placeholder_draw,
                        str(row.get("error") or "runtime failure"),
                        x=12,
                        y=34,
                        width=thumb_size - 24,
                        fill=subtle,
                        font=small_font,
                        line_height=13,
                    )
                else:
                    placeholder_draw.text((12, 12), "no image", fill=subtle, font=font)
            sheet.paste(tile, (x, y))
            draw.rectangle((x - 1, y - 1, x + thumb_size, y + thumb_size), outline=border, width=1)

            caption_y = y + thumb_size + 6
            if row is None:
                draw.text((x, caption_y), "missing row", fill=subtle, font=small_font)
                continue
            duration_text = f"{row['duration_ms']} ms" if row.get("duration_ms") is not None else "duration n/a"
            memory_text = (
                f"VRAM {_bytes_to_mb(row.get('reserved_bytes'))}/{_bytes_to_mb(row.get('max_reserved_bytes'))} MB"
            )
            draw.text((x, caption_y), f"seed {row.get('seed')} | {duration_text}", fill=fg, font=small_font)
            draw.text((x, caption_y + 14), memory_text, fill=subtle, font=small_font)
            if row.get("status") == "error":
                error_text = str(row.get("error") or "runtime failure").strip()
                error_short = error_text if len(error_text) <= 38 else f"{error_text[:35]}..."
                draw.text((x, caption_y + 28), f"FAILED | {error_short}", fill=subtle, font=small_font)
            elif row.get("pack") != row.get("baseline_pack"):
                quality = f"MSE {_format_optional_number(row.get('mse'))} | PSNR {_format_optional_number(row.get('psnr_db'))}"
                draw.text((x, caption_y + 28), quality, fill=subtle, font=small_font)
            elif row.get("error"):
                draw.text((x, caption_y + 28), str(row["error"])[:34], fill=subtle, font=small_font)
    return sheet


def _build_procedural_preview_panel(
    *,
    seed: int,
    creativity: int,
    recipe: str,
    width: int,
    height: int,
    raw_latent,
    mixed_latent,
):
    from PIL import Image, ImageDraw, ImageFont

    composite_size = 320
    grid_tile_size = 72
    panel_w = 760
    panel_h = 420
    bg = (18, 20, 24)
    fg = (240, 240, 240)
    subtle = (170, 175, 184)
    panel = Image.new("RGB", (panel_w, panel_h), color=bg)
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()

    composite = _build_latent_composite_image(raw_latent, composite_size)
    grid = _build_latent_channel_grid_image(raw_latent, grid_tile_size)
    panel.paste(composite, (20, 54))
    panel.paste(grid, (380, 54))

    draw.text(
        (20, 16),
        f"Procedural latent preview | creativity {creativity} | seed {seed} | output {width}x{height}",
        fill=fg,
        font=font,
    )
    draw.text((20, 34), recipe, fill=subtle, font=small_font)
    draw.text((20, 382), f"raw: {_latent_stats(raw_latent)}", fill=subtle, font=small_font)
    draw.text((380, 382), f"mixed: {_latent_stats(mixed_latent)}", fill=subtle, font=small_font)
    return panel, composite


def _build_preview_contact_sheet(entries: list[dict], *, title: str):
    from PIL import Image, ImageDraw, ImageFont

    columns = min(4, max(1, len(entries)))
    rows = int(math.ceil(len(entries) / columns))
    thumb_size = 220
    gutter = 18
    label_h = 58
    header_h = 54
    width = gutter + (columns * (thumb_size + gutter))
    height = header_h + gutter + rows * (thumb_size + label_h + gutter)
    bg = (18, 20, 24)
    fg = (240, 240, 240)
    subtle = (170, 175, 184)
    sheet = Image.new("RGB", (width, height), color=bg)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()

    draw.text((gutter, 18), title, fill=fg, font=font)
    for index, entry in enumerate(entries):
        row = index // columns
        col = index % columns
        x = gutter + col * (thumb_size + gutter)
        y = header_h + gutter + row * (thumb_size + label_h + gutter)
        thumb = entry["composite"].resize((thumb_size, thumb_size), Image.Resampling.NEAREST)
        sheet.paste(thumb, (x, y))
        draw.rectangle((x - 1, y - 1, x + thumb_size, y + thumb_size), outline=(65, 70, 80), width=1)
        draw.text((x, y + thumb_size + 8), f"seed {entry['seed']}", fill=fg, font=font)
        recipe = str(entry["recipe"])
        short_recipe = recipe if len(recipe) <= 36 else f"{recipe[:33]}..."
        draw.text((x, y + thumb_size + 24), short_recipe, fill=subtle, font=small_font)
    return sheet


def _load_prompt_grid_prompts(
    *,
    prompt_values: Optional[list[str]],
    prompt_file: Optional[Path],
    root: Path,
) -> list[tuple[str, str]]:
    if prompt_values and prompt_file is not None:
        raise ValueError("Use either repeated --prompt or --prompt-file, not both.")
    if not prompt_values and prompt_file is None:
        raise ValueError("Provide exactly 3 prompts via --prompt or --prompt-file.")

    prompts: list[tuple[str, str]] = []
    if prompt_file is not None:
        resolved_path = _resolve_cli_path(root, prompt_file)
        if not resolved_path.exists() or not resolved_path.is_file():
            raise ValueError(f"Prompt file not found: {resolved_path}")
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Prompt file must be a JSON list.")
        for index, item in enumerate(payload):
            if isinstance(item, str):
                key = f"prompt_{index + 1}"
                prompt_text = item.strip()
            elif isinstance(item, dict):
                key = _slugify_label(str(item.get("key") or f"prompt_{index + 1}"))
                prompt_text = str(item.get("prompt") or "").strip()
            else:
                raise ValueError("Prompt file items must be strings or {key, prompt} objects.")
            if not prompt_text:
                raise ValueError(f"Prompt {index + 1} is empty.")
            prompts.append((key, prompt_text))
    else:
        for index, prompt_text in enumerate(prompt_values or []):
            cleaned = str(prompt_text or "").strip()
            if not cleaned:
                raise ValueError(f"Prompt {index + 1} is empty.")
            prompts.append((f"prompt_{index + 1}", cleaned))

    if len(prompts) != 3:
        raise ValueError(f"Prompt grid benchmark requires exactly 3 prompts, got {len(prompts)}.")
    return prompts


def _build_prompt_grid_contact_sheet(
    *,
    rows: list[dict[str, object]],
    stage: str,
    scenarios: list[dict[str, str | None]],
    prompts: list[tuple[str, str]],
    title: str,
 ) -> "PILImage":
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    sidebar_w = 200
    thumb_size = 220
    label_h = 118
    gutter = 18
    header_h = 124
    bg = (18, 20, 24)
    fg = (240, 240, 240)
    subtle = (170, 175, 184)
    tile_bg = (32, 35, 41)
    tile_failed = (64, 28, 32)
    border = (72, 78, 90)
    sheet_w = sidebar_w + gutter + len(prompts) * (thumb_size + gutter) + gutter
    sheet_h = header_h + gutter + len(scenarios) * (thumb_size + label_h + gutter) + gutter
    sheet = Image.new("RGB", (sheet_w, sheet_h), color=bg)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()

    draw.text((gutter, 18), title, fill=fg, font=font)
    header_y = 42
    for column_index, (prompt_key, prompt_text) in enumerate(prompts):
        x = sidebar_w + gutter + column_index * (thumb_size + gutter)
        draw.text((x, header_y), prompt_key, fill=fg, font=font)
        _draw_wrapped_text(
            draw,
            prompt_text,
            x=x,
            y=header_y + 16,
            width=thumb_size,
            fill=subtle,
            font=small_font,
            line_height=12,
        )

    row_index = {
        (str(row.get("scenario_label")), str(row.get("prompt_key"))): row
        for row in rows
        if str(row.get("row_type")) == stage
    }
    for row_number, scenario in enumerate(scenarios):
        scenario_label = str(scenario["label"])
        y = header_h + gutter + row_number * (thumb_size + label_h + gutter)
        draw.text((gutter, y + 4), scenario_label, fill=fg, font=font)
        for column_index, (prompt_key, _) in enumerate(prompts):
            x = sidebar_w + gutter + column_index * (thumb_size + gutter)
            row = row_index.get((scenario_label, prompt_key))
            tile = Image.new("RGB", (thumb_size, thumb_size), color=tile_bg)
            if row and row.get("status") == "success" and row.get("output_path"):
                with Image.open(str(row["output_path"])) as image:
                    tile = ImageOps.fit(
                        image.convert("RGB"),
                        (thumb_size, thumb_size),
                        Image.Resampling.LANCZOS,
                    )
            elif row and row.get("status") != "success":
                tile = Image.new("RGB", (thumb_size, thumb_size), color=tile_failed)
                tile_draw = ImageDraw.Draw(tile)
                tile_draw.text((12, 12), "FAILED", fill=fg, font=font)
                _draw_wrapped_text(
                    tile_draw,
                    str(row.get("error") or "unknown error"),
                    x=12,
                    y=32,
                    width=thumb_size - 24,
                    fill=subtle,
                    font=small_font,
                    line_height=12,
                )
            sheet.paste(tile, (x, y))
            draw.rectangle((x - 1, y - 1, x + thumb_size, y + thumb_size), outline=border, width=1)

            caption_y = y + thumb_size + 8
            if row is None:
                draw.text((x, caption_y), "missing", fill=subtle, font=small_font)
                continue
            if stage == "generation":
                draw.text(
                    (x, caption_y),
                    f"seed {row.get('seed')} | {row.get('effective_pack')}",
                    fill=fg,
                    font=small_font,
                )
                draw.text(
                    (x, caption_y + 14),
                    f"tier {row.get('effective_resource_tier')} | mode {row.get('execution_mode_before_generate')}",
                    fill=subtle,
                    font=small_font,
                )
                draw.text(
                    (x, caption_y + 28),
                    f"{row.get('duration_ms')} ms | max {_bytes_to_mb(row.get('max_reserved_bytes'))} MB",
                    fill=subtle,
                    font=small_font,
                )
                draw.text(
                    (x, caption_y + 42),
                    f"fallback {row.get('preflight_fallback_triggered')}",
                    fill=subtle,
                    font=small_font,
                )
            else:
                draw.text(
                    (x, caption_y),
                    f"seed {row.get('seed')} | {row.get('duration_ms')} ms",
                    fill=fg,
                    font=small_font,
                )
                draw.text(
                    (x, caption_y + 14),
                    f"tile {row.get('tile_size')} / {row.get('tile_overlap')}",
                    fill=subtle,
                    font=small_font,
                )
                draw.text(
                    (x, caption_y + 28),
                    f"{row.get('precision')} | max {_bytes_to_mb(row.get('max_reserved_bytes'))} MB",
                    fill=subtle,
                    font=small_font,
                )
    return sheet


def _draw_prompt_grid_bar_panel(
    *,
    draw,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    labels: list[str],
    values: list[float | None],
    fill_color,
    font,
    small_font,
) -> None:
    fg = (240, 240, 240)
    subtle = (170, 175, 184)
    panel_bg = (26, 29, 34)
    axis = (88, 94, 108)
    draw.rounded_rectangle((x, y, x + width, y + height), radius=10, fill=panel_bg, outline=axis, width=1)
    draw.text((x + 12, y + 10), title, fill=fg, font=font)
    chart_x = x + 44
    chart_y = y + 34
    chart_w = width - 56
    chart_h = height - 72
    draw.line((chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h), fill=axis, width=1)
    bar_count = max(1, len(labels))
    slot_w = max(20, chart_w // bar_count)
    bar_w = max(10, slot_w - 8)
    finite_values = [float(value) for value in values if value is not None]
    max_value = max(finite_values) if finite_values else 1.0
    if max_value <= 0:
        max_value = 1.0
    for index, label in enumerate(labels):
        value = values[index]
        bar_x = chart_x + index * slot_w + 4
        if value is None:
            draw.text((bar_x, chart_y + chart_h - 16), "n/a", fill=subtle, font=small_font)
        else:
            bar_h = int((float(value) / max_value) * max(24, chart_h - 18))
            draw.rectangle(
                (bar_x, chart_y + chart_h - bar_h, bar_x + bar_w, chart_y + chart_h),
                fill=fill_color,
            )
            draw.text((bar_x, chart_y + chart_h - bar_h - 14), str(int(value)), fill=subtle, font=small_font)
        draw.text((bar_x, chart_y + chart_h + 4), label, fill=subtle, font=small_font)


def _build_prompt_grid_dashboard(
    *,
    summary_rows: list[dict[str, object]],
    title: str,
 ) -> "PILImage":
    from PIL import Image, ImageDraw, ImageFont

    width = 1720
    height = 1180
    bg = (18, 20, 24)
    fg = (240, 240, 240)
    subtle = (170, 175, 184)
    panel_x = 20
    panel_w = 820
    panel_h = 300
    sheet = Image.new("RGB", (width, height), color=bg)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()
    draw.text((20, 18), title, fill=fg, font=font)

    labels = [
        f"{row['prompt_key']}:{str(row['scenario_label']).replace('forced_', '')}"
        for row in summary_rows
    ]
    generation_values = [
        float(row["generation_duration_ms"]) if row.get("generation_duration_ms") is not None else None
        for row in summary_rows
    ]
    generation_vram_values = [
        round(float(row["generation_max_reserved_bytes"]) / (1024 * 1024))
        if row.get("generation_max_reserved_bytes") is not None
        else None
        for row in summary_rows
    ]
    upscale_values = [
        float(row["upscale_duration_ms"]) if row.get("upscale_duration_ms") is not None else None
        for row in summary_rows
    ]
    _draw_prompt_grid_bar_panel(
        draw=draw,
        x=panel_x,
        y=56,
        width=panel_w,
        height=panel_h,
        title="Generation Duration (ms)",
        labels=labels,
        values=generation_values,
        fill_color=(71, 128, 255),
        font=font,
        small_font=small_font,
    )
    _draw_prompt_grid_bar_panel(
        draw=draw,
        x=panel_x + panel_w + 20,
        y=56,
        width=panel_w,
        height=panel_h,
        title="Generation Max Reserved (MB)",
        labels=labels,
        values=generation_vram_values,
        fill_color=(93, 196, 127),
        font=font,
        small_font=small_font,
    )
    _draw_prompt_grid_bar_panel(
        draw=draw,
        x=panel_x,
        y=56 + panel_h + 20,
        width=panel_w,
        height=panel_h,
        title="x2 Upscale Duration (ms)",
        labels=labels,
        values=upscale_values,
        fill_color=(232, 171, 58),
        font=font,
        small_font=small_font,
    )

    table_x = panel_x + panel_w + 20
    table_y = 56 + panel_h + 20
    table_w = panel_w
    table_h = panel_h + 220
    draw.rounded_rectangle(
        (table_x, table_y, table_x + table_w, table_y + table_h),
        radius=10,
        fill=(26, 29, 34),
        outline=(88, 94, 108),
        width=1,
    )
    draw.text((table_x + 12, table_y + 10), "Effective Tier / Mode Audit", fill=fg, font=font)
    row_y = table_y + 34
    for row in summary_rows:
        summary_text = (
            f"{row['prompt_key']} | {row['scenario_label']} | "
            f"pack={row.get('effective_pack')} | tier={row.get('effective_resource_tier')} | "
            f"mode={row.get('execution_mode_before_generate')}->{row.get('execution_mode_after_generate')} | "
            f"fallback={row.get('preflight_fallback_triggered')}"
        )
        row_y = _draw_wrapped_text(
            draw,
            summary_text,
            x=table_x + 12,
            y=row_y,
            width=table_w - 24,
            fill=subtle,
            font=small_font,
            line_height=12,
        )
        row_y += 6
    return sheet


@cli.callback()
def callback(log_level: str = typer.Option("INFO", "--log-level")) -> None:
    configure_logging(log_level)


@cli.command("status")
def status() -> None:
    settings = load_settings()
    typer.echo(json.dumps(settings.to_dict(), indent=2))


@cli.command("doctor")
def doctor() -> None:
    settings = load_settings()
    pack_paths = discover_model_packs(settings.paths.model_packs_dir)
    report = {
        "offline_mode": settings.offline_mode,
        "hf_hub_offline": "HF_HUB_OFFLINE" in os.environ,
        "transformers_offline": "TRANSFORMERS_OFFLINE" in os.environ,
        "paths_exist": {
            "models_dir": settings.paths.models_dir.exists(),
            "model_packs_dir": settings.paths.model_packs_dir.exists(),
            "outputs_dir": settings.paths.outputs_dir.exists(),
            "data_dir": settings.paths.data_dir.exists(),
        },
        "model_pack_count": len(pack_paths),
        "runtime_profile": settings.runtime_profile.name,
        "resource_tier": settings.resource_tier_controller.current().name,
        "resource_tier_override": settings.resource_tier_override,
        "auto_resource_tier": settings.auto_resource_tier,
    }
    typer.echo(json.dumps(report, indent=2))


@cli.command("validate-models")
def validate_models(
    all_packs: bool = typer.Option(
        False,
        "--all",
        help="Validate all discovered packs, including disabled ones.",
    ),
) -> None:
    settings = load_settings()
    pack_paths = discover_model_packs(settings.paths.model_packs_dir)
    if not pack_paths:
        typer.echo("No model packs found.")
        raise typer.Exit(code=1)

    selected_pack_paths = pack_paths
    if not all_packs:
        selected_pack_paths = []
        for pack_path in pack_paths:
            raw_manifest = pack_path.read_text(encoding="utf-8")
            if re.search(r"(?mi)^\s*enabled:\s*false\s*$", raw_manifest):
                continue
            selected_pack_paths.append(pack_path)
        if not selected_pack_paths:
            typer.echo("No enabled model packs found. Use --all to validate disabled packs too.")
            raise typer.Exit(code=1)

    failed = 0
    for pack_path in selected_pack_paths:
        try:
            pack = load_model_pack(pack_path)
            typer.echo(
                f"[OK] {pack.name}: {pack.architecture}, "
                f"{len(pack.components)} component(s), backend={pack.backend_preference[0]}"
            )
        except ModelPackValidationError as exc:
            failed += 1
            typer.echo(f"[FAIL] {pack_path}: {exc}")

    if failed:
        raise typer.Exit(code=1)


@cli.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(37717, "--port"),
    verbose_logs: bool = typer.Option(
        False,
        "--verbose-logs/--quiet-logs",
        help="Enable verbose HTTP access/dependency logs. Default is quiet logs.",
    ),
) -> None:
    import uvicorn

    configure_logging(verbose_logs=verbose_logs)
    load_settings()
    logging.getLogger(__name__).info("Starting web server on http://%s:%d", host, port)
    uvicorn_log_level = str(os.environ.get("JUSTRAYZIST_LOG_LEVEL", "INFO")).strip().lower()
    uvicorn.run(
        "app.api.main:app",
        host=host,
        port=port,
        reload=False,
        log_level=uvicorn_log_level,
        access_log=verbose_logs,
    )


@cli.command("generate")
def generate(
    prompt: str = typer.Option(..., "--prompt"),
    pack: str = typer.Option(..., "--pack", help="Model pack name or folder name"),
    width: int = typer.Option(1024, "--width"),
    height: int = typer.Option(1024, "--height"),
    steps: Optional[int] = typer.Option(None, "--steps"),
    guidance_scale: Optional[float] = typer.Option(None, "--guidance-scale"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    enhance_prompt: bool = typer.Option(
        False,
        "--enhance-prompt/--no-enhance-prompt",
        help="Use loaded text_encoder to rewrite prompt before image generation.",
    ),
    output: Optional[Path] = typer.Option(None, "--output"),
) -> None:
    from app.core.worker import GenerationRequest, GenerationSession
    from app.storage import append_generation_metric, save_png_with_metadata

    settings = load_settings()
    model_pack = _load_pack_or_exit(settings, pack)
    _assert_supported_backend_or_exit(model_pack)
    session = GenerationSession(settings=settings, model_pack=model_pack)

    try:
        result = session.generate(
            GenerationRequest(
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                enhance_prompt=enhance_prompt,
            )
        )
    except ImportError as exc:
        typer.echo(
            f"Missing dependency during generation: {exc}. "
            "Run RunMeFirst.bat (recommended) or repair with scripts/bootstrap_env.ps1."
        )
        raise typer.Exit(code=2) from exc
    except Exception as exc:
        typer.echo(f"Generation failed: {exc}")
        raise typer.Exit(code=1) from exc

    saved_path = save_png_with_metadata(
        image=result.image,
        prompt=result.prompt_effective,
        settings=settings,
        output_path=output,
        extra_metadata={
            "prompt_original": result.prompt_original,
            "prompt_effective": result.prompt_effective,
            "prompt_enhanced": result.prompt_enhanced,
            "width": width,
            "height": height,
            "steps": result.steps,
            "guidance_scale": result.guidance_scale,
            "backend": result.backend,
            "device": result.device,
            "model_pack": model_pack.name,
            "duration_ms": result.duration_ms,
            "runtime_profile": result.runtime_profile,
            "execution_mode": result.execution_mode,
            **result.telemetry_dict(),
        },
    )
    metrics_file = append_generation_metric(
        settings=settings,
        payload={
            "prompt": result.prompt_effective,
            "prompt_original": result.prompt_original,
            "prompt_effective": result.prompt_effective,
            "prompt_enhanced": result.prompt_enhanced,
            "width": width,
            "height": height,
            "output_path": str(saved_path),
            "model_pack": model_pack.name,
            **result.telemetry_dict(),
        },
    )
    typer.echo(f"Saved: {saved_path}")
    if result.prompt_enhanced:
        typer.echo(f"Prompt enhanced: {result.prompt_effective}")
    typer.echo(f"Metrics: {metrics_file}")


@cli.command("prompt-grid-benchmark")
def prompt_grid_benchmark(
    prompt: Optional[list[str]] = typer.Option(
        None,
        "--prompt",
        help="Repeat exactly 3 times, or use --prompt-file.",
    ),
    prompt_file: Optional[Path] = typer.Option(
        None,
        "--prompt-file",
        help="JSON file containing exactly 3 prompts as strings or {key, prompt} objects.",
    ),
    pack: str = typer.Option("Rayzist_bf16", "--pack"),
    width: int = typer.Option(1024, "--width"),
    height: int = typer.Option(1024, "--height"),
    seed_start: int = typer.Option(1, "--seed-start"),
    checkpoint: Path = typer.Option(
        Path("models/upscaler/2x_RealESRGAN_x2plus.pth"),
        "--checkpoint",
        help="Path to the x2 upscaler checkpoint.",
    ),
    enhance_prompt: bool = typer.Option(
        False,
        "--enhance-prompt/--no-enhance-prompt",
        help="Use loaded text_encoder to rewrite prompt before image generation.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Destination directory for prompt grid outputs. Defaults to outputs/prompt_grid_benchmark/<timestamp>/.",
    ),
) -> None:
    import gc

    import torch

    from app.core.upscale import upscale_image
    from app.core.worker import GenerationRequest, GenerationSession
    from app.storage import append_generation_metric, build_output_path, save_png_with_metadata

    seed_settings = load_settings()
    root = seed_settings.paths.root_dir
    try:
        prompts = _load_prompt_grid_prompts(
            prompt_values=prompt,
            prompt_file=prompt_file,
            root=root,
        )
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1)

    checkpoint_path = _resolve_cli_path(root, checkpoint)
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        typer.echo(f"Upscaler checkpoint not found: {checkpoint_path}")
        raise typer.Exit(code=1)

    suite_key = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    destination_root = (
        _resolve_cli_path(root, output_dir)
        if output_dir is not None
        else seed_settings.paths.outputs_dir / "prompt_grid_benchmark" / suite_key
    )
    destination_root.mkdir(parents=True, exist_ok=True)
    report_csv = seed_settings.paths.data_dir / f"prompt_grid_benchmark_{suite_key}.csv"
    report_jsonl = seed_settings.paths.data_dir / f"prompt_grid_benchmark_{suite_key}.jsonl"
    report_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    typer.echo(
        f"Prompt grid benchmark: pack={pack}, prompts={len(prompts)}, scenarios={len(PROMPT_GRID_SCENARIOS)}"
    )

    for prompt_index, (prompt_key, prompt_text) in enumerate(prompts):
        seed = int(seed_start) + prompt_index
        prompt_dir = destination_root / f"{prompt_index + 1:02d}_{prompt_key}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        typer.echo(f"[{prompt_key}] seed={seed}")

        for scenario in PROMPT_GRID_SCENARIOS:
            scenario_label = str(scenario["label"])
            profile_override = scenario["profile_override"]
            settings = load_settings(profile_name=profile_override)
            inference = InferenceService(settings=settings)
            scenario_dir = prompt_dir / scenario_label
            scenario_dir.mkdir(parents=True, exist_ok=True)

            selected_pack, effective_pack, detected_tier = inference.resolve_runtime_pack(pack)
            _assert_supported_backend_or_exit(selected_pack)
            _assert_supported_backend_or_exit(effective_pack)
            session = GenerationSession(
                settings=settings,
                model_pack=effective_pack,
                resource_tier=detected_tier,
            )

            generation_result = None
            generation_row: dict[str, object] = {
                "row_type": "generation",
                "suite_id": suite_key,
                "scenario_label": scenario_label,
                "scenario_source": str(scenario["source"]),
                "requested_resource_tier": profile_override or "auto",
                "prompt_key": prompt_key,
                "prompt_index": prompt_index,
                "prompt": prompt_text,
                "seed": seed,
                "selected_pack": selected_pack.name,
                "effective_pack": effective_pack.name,
                "derived_strategy": effective_pack.derived_strategy,
                "detected_resource_tier": detected_tier.name,
                "effective_resource_tier": detected_tier.name,
                "status": "pending",
                "duration_ms": None,
                "reserved_bytes": None,
                "max_reserved_bytes": None,
                "runtime_profile": settings.runtime_profile.name,
                "resource_tier": detected_tier.name,
                "execution_mode": "",
                "execution_mode_initial": "",
                "execution_mode_before_generate": "",
                "execution_mode_after_generate": "",
                "preflight_fallback_triggered": False,
                "cuda_free_before_load_bytes": None,
                "cuda_free_after_load_bytes": None,
                "cuda_free_before_generate_bytes": None,
                "cuda_free_after_generate_bytes": None,
                "output_path": "",
                "error": "",
            }
            try:
                generation_result = session.generate(
                    GenerationRequest(
                        prompt=prompt_text,
                        width=width,
                        height=height,
                        seed=seed,
                        enhance_prompt=enhance_prompt,
                    )
                )
                generation_path = save_png_with_metadata(
                    image=generation_result.image,
                    prompt=generation_result.prompt_effective,
                    settings=settings,
                    output_path=build_output_path(
                        scenario_dir,
                        prefix=f"generation_{scenario_label}_{prompt_key}",
                    ),
                    extra_metadata={
                        "mode": "prompt_grid_benchmark_generation",
                        "suite_id": suite_key,
                        "scenario_label": scenario_label,
                        "scenario_source": str(scenario["source"]),
                        "requested_resource_tier": profile_override or "auto",
                        "selected_pack": selected_pack.name,
                        "effective_pack": effective_pack.name,
                        "derived_strategy": effective_pack.derived_strategy,
                        "prompt_key": prompt_key,
                        "seed": seed,
                        **generation_result.telemetry_dict(),
                    },
                )
                generation_row.update(
                    {
                        "status": "success",
                        "duration_ms": int(generation_result.duration_ms),
                        "reserved_bytes": _result_reserved_bytes(generation_result),
                        "max_reserved_bytes": _result_max_reserved_bytes(generation_result),
                        "runtime_profile": str(generation_result.runtime_profile or settings.runtime_profile.name),
                        "resource_tier": str(generation_result.resource_tier or detected_tier.name),
                        "effective_resource_tier": str(generation_result.resource_tier or detected_tier.name),
                        "execution_mode": str(generation_result.execution_mode or ""),
                        "execution_mode_initial": str(generation_result.execution_mode_initial or ""),
                        "execution_mode_before_generate": str(
                            generation_result.execution_mode_before_generate or ""
                        ),
                        "execution_mode_after_generate": str(
                            generation_result.execution_mode_after_generate or ""
                        ),
                        "preflight_fallback_triggered": bool(
                            generation_result.preflight_fallback_triggered
                        ),
                        "cuda_free_before_load_bytes": generation_result.cuda_free_before_load_bytes,
                        "cuda_free_after_load_bytes": generation_result.cuda_free_after_load_bytes,
                        "cuda_free_before_generate_bytes": generation_result.cuda_free_before_generate_bytes,
                        "cuda_free_after_generate_bytes": generation_result.cuda_free_after_generate_bytes,
                        "output_path": str(generation_path),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                generation_row["status"] = "error"
                generation_row["error"] = str(exc)
            finally:
                rows.append(generation_row)
                append_generation_metric(
                    settings=settings,
                    payload={"mode": "prompt_grid_benchmark_generation_row", **generation_row},
                )

            upscale_row: dict[str, object] = {
                "row_type": "upscale",
                "suite_id": suite_key,
                "scenario_label": scenario_label,
                "scenario_source": str(scenario["source"]),
                "requested_resource_tier": profile_override or "auto",
                "prompt_key": prompt_key,
                "prompt_index": prompt_index,
                "prompt": prompt_text,
                "seed": seed,
                "selected_pack": selected_pack.name,
                "effective_pack": effective_pack.name,
                "derived_strategy": effective_pack.derived_strategy,
                "detected_resource_tier": detected_tier.name,
                "effective_resource_tier": generation_row["effective_resource_tier"],
                "status": "skipped",
                "duration_ms": None,
                "reserved_bytes": None,
                "max_reserved_bytes": None,
                "precision": "",
                "tile_size": None,
                "tile_overlap": None,
                "scale_factor": None,
                "output_path": "",
                "error": "",
            }
            if generation_result is not None and generation_row["status"] == "success":
                try:
                    upscale_result = upscale_image(
                        image=generation_result.image,
                        checkpoint_path=checkpoint_path,
                        profile_name=str(generation_row["effective_resource_tier"]),
                    )
                    upscale_path = save_png_with_metadata(
                        image=upscale_result.image,
                        prompt=generation_result.prompt_effective,
                        settings=settings,
                        output_path=build_output_path(
                            scenario_dir,
                            prefix=f"upscale_{scenario_label}_{prompt_key}",
                        ),
                        extra_metadata={
                            "mode": "prompt_grid_benchmark_upscale",
                            "suite_id": suite_key,
                            "scenario_label": scenario_label,
                            "scenario_source": str(scenario["source"]),
                            "requested_resource_tier": profile_override or "auto",
                            "selected_pack": selected_pack.name,
                            "effective_pack": effective_pack.name,
                            "derived_strategy": effective_pack.derived_strategy,
                            "prompt_key": prompt_key,
                            "seed": seed,
                            "source_generation_output": generation_row["output_path"],
                            **upscale_result.telemetry_dict(),
                        },
                    )
                    upscale_row.update(
                        {
                            "status": "success",
                            "duration_ms": int(upscale_result.duration_ms),
                            "reserved_bytes": (
                                upscale_result.cuda_memory_after.reserved_bytes
                                if upscale_result.cuda_memory_after is not None
                                else None
                            ),
                            "max_reserved_bytes": (
                                upscale_result.cuda_memory_after.max_reserved_bytes
                                if upscale_result.cuda_memory_after is not None
                                else None
                            ),
                            "precision": str(upscale_result.precision),
                            "tile_size": int(upscale_result.tile_size),
                            "tile_overlap": int(upscale_result.tile_overlap),
                            "scale_factor": int(upscale_result.scale_factor),
                            "output_path": str(upscale_path),
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    upscale_row["status"] = "error"
                    upscale_row["error"] = str(exc)
            rows.append(upscale_row)
            append_generation_metric(
                settings=settings,
                payload={"mode": "prompt_grid_benchmark_upscale_row", **upscale_row},
            )

            summary_row = {
                "row_type": "summary",
                "suite_id": suite_key,
                "scenario_label": scenario_label,
                "scenario_source": str(scenario["source"]),
                "requested_resource_tier": profile_override or "auto",
                "prompt_key": prompt_key,
                "prompt_index": prompt_index,
                "prompt": prompt_text,
                "seed": seed,
                "selected_pack": selected_pack.name,
                "effective_pack": effective_pack.name,
                "derived_strategy": effective_pack.derived_strategy,
                "detected_resource_tier": detected_tier.name,
                "effective_resource_tier": generation_row["effective_resource_tier"],
                "generation_status": generation_row["status"],
                "generation_duration_ms": generation_row["duration_ms"],
                "generation_max_reserved_bytes": generation_row["max_reserved_bytes"],
                "upscale_status": upscale_row["status"],
                "upscale_duration_ms": upscale_row["duration_ms"],
                "upscale_max_reserved_bytes": upscale_row["max_reserved_bytes"],
                "execution_mode_initial": generation_row["execution_mode_initial"],
                "execution_mode_before_generate": generation_row["execution_mode_before_generate"],
                "execution_mode_after_generate": generation_row["execution_mode_after_generate"],
                "preflight_fallback_triggered": generation_row["preflight_fallback_triggered"],
                "generation_output_path": generation_row["output_path"],
                "upscale_output_path": upscale_row["output_path"],
                "error": generation_row["error"] or upscale_row["error"],
            }
            summary_rows.append(summary_row)
            rows.append(summary_row)
            append_generation_metric(
                settings=settings,
                payload={"mode": "prompt_grid_benchmark_summary_row", **summary_row},
            )

            typer.echo(
                f"  {scenario_label}: gen={generation_row['status']} {generation_row['duration_ms']} ms | "
                f"upscale={upscale_row['status']} {upscale_row['duration_ms']} ms | "
                f"tier={summary_row['effective_resource_tier']} | pack={effective_pack.name}"
            )

            session.recycle(f"prompt-grid-benchmark cleanup {scenario_label} {prompt_key}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    generation_contact_sheet = _build_prompt_grid_contact_sheet(
        rows=rows,
        stage="generation",
        scenarios=PROMPT_GRID_SCENARIOS,
        prompts=prompts,
        title=f"Prompt Grid Benchmark | Generation | {pack}",
    )
    generation_contact_sheet_path = destination_root / "generation_contact_sheet.png"
    generation_contact_sheet.save(generation_contact_sheet_path, format="PNG")

    upscale_contact_sheet = _build_prompt_grid_contact_sheet(
        rows=rows,
        stage="upscale",
        scenarios=PROMPT_GRID_SCENARIOS,
        prompts=prompts,
        title=f"Prompt Grid Benchmark | x2 Upscale | {pack}",
    )
    upscale_contact_sheet_path = destination_root / "upscale_contact_sheet.png"
    upscale_contact_sheet.save(upscale_contact_sheet_path, format="PNG")

    dashboard = _build_prompt_grid_dashboard(
        summary_rows=summary_rows,
        title=f"Prompt Grid Benchmark Dashboard | {pack}",
    )
    dashboard_path = destination_root / "benchmark_dashboard.png"
    dashboard.save(dashboard_path, format="PNG")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with report_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with report_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    manifest_path = destination_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "suite_id": suite_key,
                "pack": pack,
                "prompts": [{"key": key, "prompt": text} for key, text in prompts],
                "scenarios": PROMPT_GRID_SCENARIOS,
                "generation_contact_sheet": str(generation_contact_sheet_path),
                "upscale_contact_sheet": str(upscale_contact_sheet_path),
                "dashboard": str(dashboard_path),
                "report_csv": str(report_csv),
                "report_jsonl": str(report_jsonl),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    typer.echo(f"Generation contact sheet: {generation_contact_sheet_path}")
    typer.echo(f"Upscale contact sheet: {upscale_contact_sheet_path}")
    typer.echo(f"Dashboard: {dashboard_path}")
    typer.echo(f"Report CSV: {report_csv}")
    typer.echo(f"Report JSONL: {report_jsonl}")
    typer.echo(f"Manifest: {manifest_path}")


@cli.command("pack-compare")
def pack_compare(
    prompt: str = typer.Option(..., "--prompt"),
    baseline_pack: str = typer.Option("Rayzist_bf16", "--baseline-pack"),
    candidate_pack: str = typer.Option("Rayzist_bf16__auto_fp8_storage", "--candidate-pack"),
    iterations: int = typer.Option(5, "--iterations", min=1),
    width: int = typer.Option(1024, "--width"),
    height: int = typer.Option(1024, "--height"),
    steps: Optional[int] = typer.Option(None, "--steps"),
    guidance_scale: Optional[float] = typer.Option(None, "--guidance-scale"),
    seed_start: Optional[int] = typer.Option(1, "--seed-start"),
    enhance_prompt: bool = typer.Option(
        False,
        "--enhance-prompt/--no-enhance-prompt",
        help="Use loaded text_encoder to rewrite prompt before image generation.",
    ),
    warmup: bool = typer.Option(True, "--warmup/--no-warmup"),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Destination directory for paired benchmark outputs. Defaults to outputs/.",
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Engineering-only runtime override for controlled benchmark comparisons.",
    ),
) -> None:
    from app.core.worker import GenerationSession
    from app.storage import append_generation_metric

    settings = load_settings(profile_name=profile)
    baseline_model_pack = _load_runtime_pack_or_exit(settings, baseline_pack)
    candidate_model_pack = _load_runtime_pack_or_exit(settings, candidate_pack)
    _assert_supported_backend_or_exit(baseline_model_pack)
    _assert_supported_backend_or_exit(candidate_model_pack)

    destination_dir = (
        _resolve_cli_path(settings.paths.root_dir, output_dir)
        if output_dir is not None
        else settings.paths.outputs_dir
    )
    destination_dir.mkdir(parents=True, exist_ok=True)

    report_key = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_csv = settings.paths.data_dir / f"pack_compare_{report_key}.csv"
    report_jsonl = settings.paths.data_dir / f"pack_compare_{report_key}.jsonl"
    report_csv.parent.mkdir(parents=True, exist_ok=True)

    typer.echo(
        f"Pack compare: baseline={baseline_model_pack.name}, candidate={candidate_model_pack.name}, "
        f"runtime_profile={settings.runtime_profile.name}, iterations={iterations}"
    )

    baseline_session = GenerationSession(settings=settings, model_pack=baseline_model_pack)
    candidate_session = GenerationSession(settings=settings, model_pack=candidate_model_pack)
    pair_rows: list[dict[str, object]] = []

    def _pair_seed(offset: int) -> int | None:
        if seed_start is None:
            return None
        return seed_start + offset

    cold_pair_id = uuid4().hex[:10]
    baseline_cold_record, baseline_cold_result = _run_pack_compare_generation(
        session=baseline_session,
        settings=settings,
        model_pack=baseline_model_pack,
        prompt=prompt,
        width=width,
        height=height,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=_pair_seed(0),
        enhance_prompt=enhance_prompt,
        output_dir=destination_dir,
        run_label="cold_1",
        pair_id=cold_pair_id,
        role="baseline",
        recycle_before=True,
        recycle_after=True,
    )
    candidate_cold_record, candidate_cold_result = _run_pack_compare_generation(
        session=candidate_session,
        settings=settings,
        model_pack=candidate_model_pack,
        prompt=prompt,
        width=width,
        height=height,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=_pair_seed(0),
        enhance_prompt=enhance_prompt,
        output_dir=destination_dir,
        run_label="cold_1",
        pair_id=cold_pair_id,
        role="candidate",
        recycle_before=True,
        recycle_after=True,
    )
    cold_row = _build_pack_compare_pair_row(
        pair_id=cold_pair_id,
        run_label="cold_1",
        profile_name=settings.runtime_profile.name,
        prompt=prompt,
        width=width,
        height=height,
        seed=_pair_seed(0),
        baseline_pack_name=baseline_model_pack.name,
        candidate_pack_name=candidate_model_pack.name,
        baseline_record=baseline_cold_record,
        candidate_record=candidate_cold_record,
        baseline_result=baseline_cold_result,
        candidate_result=candidate_cold_result,
    )
    pair_rows.append(cold_row)
    append_generation_metric(
        settings=settings,
        payload={
            "mode": "pack_compare_pair",
            **cold_row,
        },
    )

    if warmup:
        typer.echo("Warmup baseline session...")
        _run_pack_compare_generation(
            session=baseline_session,
            settings=settings,
            model_pack=baseline_model_pack,
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=_pair_seed(0),
            enhance_prompt=enhance_prompt,
            output_dir=destination_dir,
            run_label="warmup_baseline",
            pair_id=uuid4().hex[:10],
            role="baseline",
            recycle_before=False,
            recycle_after=False,
            record_metric=False,
        )
        typer.echo("Warmup candidate session...")
        _run_pack_compare_generation(
            session=candidate_session,
            settings=settings,
            model_pack=candidate_model_pack,
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=_pair_seed(0),
            enhance_prompt=enhance_prompt,
            output_dir=destination_dir,
            run_label="warmup_candidate",
            pair_id=uuid4().hex[:10],
            role="candidate",
            recycle_before=False,
            recycle_after=False,
            record_metric=False,
        )

    for index in range(iterations):
        run_label = f"warm_{index + 1}"
        pair_id = uuid4().hex[:10]
        seed = _pair_seed(index + 1)
        baseline_record, baseline_result = _run_pack_compare_generation(
            session=baseline_session,
            settings=settings,
            model_pack=baseline_model_pack,
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            enhance_prompt=enhance_prompt,
            output_dir=destination_dir,
            run_label=run_label,
            pair_id=pair_id,
            role="baseline",
        )
        candidate_record, candidate_result = _run_pack_compare_generation(
            session=candidate_session,
            settings=settings,
            model_pack=candidate_model_pack,
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            enhance_prompt=enhance_prompt,
            output_dir=destination_dir,
            run_label=run_label,
            pair_id=pair_id,
            role="candidate",
        )
        row = _build_pack_compare_pair_row(
            pair_id=pair_id,
            run_label=run_label,
            profile_name=settings.runtime_profile.name,
            prompt=prompt,
            width=width,
            height=height,
            seed=seed,
            baseline_pack_name=baseline_model_pack.name,
            candidate_pack_name=candidate_model_pack.name,
            baseline_record=baseline_record,
            candidate_record=candidate_record,
            baseline_result=baseline_result,
            candidate_result=candidate_result,
        )
        pair_rows.append(row)
        append_generation_metric(
            settings=settings,
            payload={
                "mode": "pack_compare_pair",
                **row,
            },
        )
        typer.echo(
            f"[{run_label}] baseline={row['baseline_duration_ms']} ms "
            f"candidate={row['candidate_duration_ms']} ms mse={row['mse']} psnr={row['psnr_db']}"
        )

    baseline_summary = _summarize_compare_side(pair_rows, "baseline")
    candidate_summary = _summarize_compare_side(pair_rows, "candidate")
    mse_values = [
        float(row["mse"])
        for row in pair_rows
        if row.get("mse") is not None
    ]
    psnr_values = [
        float(row["psnr_db"])
        for row in pair_rows
        if row.get("psnr_db") is not None and math.isfinite(float(row["psnr_db"]))
    ]

    fieldnames = [
        "pair_id",
        "run_label",
        "profile",
        "prompt",
        "width",
        "height",
        "seed",
        "baseline_pack",
        "candidate_pack",
        "baseline_status",
        "candidate_status",
        "baseline_duration_ms",
        "candidate_duration_ms",
        "baseline_reserved_bytes",
        "candidate_reserved_bytes",
        "baseline_max_reserved_bytes",
        "candidate_max_reserved_bytes",
        "baseline_execution_mode",
        "candidate_execution_mode",
        "baseline_output_path",
        "candidate_output_path",
        "baseline_error",
        "candidate_error",
        "mse",
        "psnr_db",
    ]
    with report_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in pair_rows:
            writer.writerow(row)

    with report_jsonl.open("w", encoding="utf-8") as handle:
        for row in pair_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    typer.echo("")
    typer.echo("Pack compare summary:")
    typer.echo(
        f"baseline  median_ms={baseline_summary['median_duration_ms']} "
        f"median_reserved_mb={_bytes_to_mb(baseline_summary['median_reserved_bytes'])} "
        f"max_reserved_mb={_bytes_to_mb(baseline_summary['max_reserved_bytes'])}"
    )
    typer.echo(
        f"candidate median_ms={candidate_summary['median_duration_ms']} "
        f"median_reserved_mb={_bytes_to_mb(candidate_summary['median_reserved_bytes'])} "
        f"max_reserved_mb={_bytes_to_mb(candidate_summary['max_reserved_bytes'])}"
    )
    typer.echo(
        f"quality   avg_mse={round(sum(mse_values) / len(mse_values), 4) if mse_values else 'n/a'} "
        f"avg_psnr_db={round(sum(psnr_values) / len(psnr_values), 4) if psnr_values else 'n/a'}"
    )
    typer.echo(f"Report CSV: {report_csv}")
    typer.echo(f"Report JSONL: {report_jsonl}")


@cli.command("pack-compare-suite")
def pack_compare_suite(
    iterations: int = typer.Option(3, "--iterations", min=1),
    width: int = typer.Option(1024, "--width"),
    height: int = typer.Option(1024, "--height"),
    steps: Optional[int] = typer.Option(None, "--steps"),
    guidance_scale: Optional[float] = typer.Option(None, "--guidance-scale"),
    seed_start: Optional[int] = typer.Option(1, "--seed-start"),
    enhance_prompt: bool = typer.Option(
        False,
        "--enhance-prompt/--no-enhance-prompt",
        help="Use loaded text_encoder to rewrite prompt before image generation.",
    ),
    warmup: bool = typer.Option(True, "--warmup/--no-warmup"),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Destination directory for benchmark suite outputs. Defaults to outputs/pack_compare_suite/<timestamp>/.",
    ),
    pack: Optional[list[str]] = typer.Option(
        None,
        "--pack",
        help="Optional repeated pack override. The first pack is treated as the baseline.",
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Engineering-only runtime override for controlled benchmark comparisons.",
    ),
) -> None:
    from app.core.worker import GenerationSession
    from app.storage import append_generation_metric

    settings = load_settings(profile_name=profile)
    pack_names = list(pack) if pack else list(DEFAULT_FP8_SUITE_PACKS)
    if len(pack_names) < 2:
        typer.echo("pack-compare-suite requires at least two packs.")
        raise typer.Exit(code=1)

    model_packs = [_load_runtime_pack_or_exit(settings, pack_name) for pack_name in pack_names]
    for model_pack in model_packs:
        _assert_supported_backend_or_exit(model_pack)

    suite_key = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    destination_root = (
        _resolve_cli_path(settings.paths.root_dir, output_dir)
        if output_dir is not None
        else settings.paths.outputs_dir / "pack_compare_suite" / suite_key
    )
    destination_root.mkdir(parents=True, exist_ok=True)
    report_csv = settings.paths.data_dir / f"pack_compare_suite_{suite_key}.csv"
    report_jsonl = settings.paths.data_dir / f"pack_compare_suite_{suite_key}.jsonl"
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    sessions = {model_pack.name: GenerationSession(settings=settings, model_pack=model_pack) for model_pack in model_packs}
    rows: list[dict[str, object]] = []
    contact_sheet_paths: list[Path] = []
    run_order = ["cold_1", *[f"warm_{index + 1}" for index in range(iterations)]]
    baseline_pack_name = model_packs[0].name

    def _suite_seed(prompt_index: int, offset: int) -> int | None:
        if seed_start is None:
            return None
        return seed_start + (prompt_index * (iterations + 1)) + offset

    typer.echo(
        f"Pack compare suite: packs={', '.join(pack_names)} "
        f"runtime_profile={settings.runtime_profile.name} iterations={iterations}"
    )

    try:
        for prompt_index, (prompt_key, prompt_text) in enumerate(DEFAULT_FP8_SUITE_PROMPTS):
            prompt_output_dir = destination_root / f"{prompt_index + 1:02d}_{prompt_key}"
            prompt_output_dir.mkdir(parents=True, exist_ok=True)
            prompt_rows: list[dict[str, object]] = []
            typer.echo("")
            typer.echo(f"[{prompt_key}] {prompt_text}")

            def _run_round(
                *,
                run_label: str,
                seed: int | None,
                recycle_before: bool,
                recycle_after: bool,
                record_metric: bool,
            ) -> None:
                baseline_model_pack = model_packs[0]
                baseline_slug = _slugify_label(baseline_model_pack.name)
                baseline_record, baseline_result = _run_pack_compare_generation(
                    session=sessions[baseline_model_pack.name],
                    settings=settings,
                    model_pack=baseline_model_pack,
                    prompt=prompt_text,
                    width=width,
                    height=height,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    enhance_prompt=enhance_prompt,
                    output_dir=prompt_output_dir,
                    run_label=run_label,
                    pair_id=f"{suite_key}_{prompt_key}_{run_label}",
                    role=baseline_slug,
                    recycle_before=recycle_before,
                    recycle_after=recycle_after,
                    record_metric=record_metric,
                )
                if record_metric:
                    baseline_row = _build_pack_suite_row(
                        suite_id=suite_key,
                        prompt_key=prompt_key,
                        prompt=prompt_text,
                        run_label=run_label,
                        seed=seed,
                        width=width,
                        height=height,
                        baseline_pack_name=baseline_pack_name,
                        baseline_record=baseline_record,
                        baseline_result=baseline_result,
                        pack_name=baseline_model_pack.name,
                        record=baseline_record,
                        result=baseline_result,
                    )
                    rows.append(baseline_row)
                    prompt_rows.append(baseline_row)
                    append_generation_metric(
                        settings=settings,
                        payload={"mode": "pack_compare_suite_row", **baseline_row},
                    )
                typer.echo(
                    f"  {run_label} | {baseline_model_pack.name}: {baseline_record['status']} "
                    f"{baseline_record['duration_ms']} ms"
                )

                for candidate_pack in model_packs[1:]:
                    role_slug = _slugify_label(candidate_pack.name)
                    record, result = _run_pack_compare_generation(
                        session=sessions[candidate_pack.name],
                        settings=settings,
                        model_pack=candidate_pack,
                        prompt=prompt_text,
                        width=width,
                        height=height,
                        steps=steps,
                        guidance_scale=guidance_scale,
                        seed=seed,
                        enhance_prompt=enhance_prompt,
                        output_dir=prompt_output_dir,
                        run_label=run_label,
                        pair_id=f"{suite_key}_{prompt_key}_{run_label}",
                        role=role_slug,
                        recycle_before=recycle_before,
                        recycle_after=recycle_after,
                        record_metric=record_metric,
                    )
                    if record_metric:
                        row = _build_pack_suite_row(
                            suite_id=suite_key,
                            prompt_key=prompt_key,
                            prompt=prompt_text,
                            run_label=run_label,
                            seed=seed,
                            width=width,
                            height=height,
                            baseline_pack_name=baseline_pack_name,
                            baseline_record=baseline_record,
                            baseline_result=baseline_result,
                            pack_name=candidate_pack.name,
                            record=record,
                            result=result,
                        )
                        rows.append(row)
                        prompt_rows.append(row)
                        append_generation_metric(
                            settings=settings,
                            payload={"mode": "pack_compare_suite_row", **row},
                        )
                        typer.echo(
                            f"  {run_label} | {candidate_pack.name}: {row['status']} "
                            f"{row['duration_ms']} ms mse={row['mse']} psnr={row['psnr_db']}"
                        )
                    else:
                        typer.echo(
                            f"  {run_label} | {candidate_pack.name}: {record['status']} {record['duration_ms']} ms"
                        )

            _run_round(
                run_label="cold_1",
                seed=_suite_seed(prompt_index, 0),
                recycle_before=True,
                recycle_after=True,
                record_metric=True,
            )

            if warmup:
                for model_pack in model_packs:
                    typer.echo(f"  warmup | {model_pack.name}")
                    _run_pack_compare_generation(
                        session=sessions[model_pack.name],
                        settings=settings,
                        model_pack=model_pack,
                        prompt=prompt_text,
                        width=width,
                        height=height,
                        steps=steps,
                        guidance_scale=guidance_scale,
                        seed=_suite_seed(prompt_index, 0),
                        enhance_prompt=enhance_prompt,
                        output_dir=prompt_output_dir,
                        run_label=f"warmup_{_slugify_label(model_pack.name)}",
                        pair_id=f"{suite_key}_{prompt_key}_warmup",
                        role=_slugify_label(model_pack.name),
                        recycle_before=False,
                        recycle_after=False,
                        record_metric=False,
                    )

            for iteration_index in range(iterations):
                _run_round(
                    run_label=f"warm_{iteration_index + 1}",
                    seed=_suite_seed(prompt_index, iteration_index + 1),
                    recycle_before=False,
                    recycle_after=False,
                    record_metric=True,
                )

            contact_sheet = _build_benchmark_contact_sheet(
                rows=prompt_rows,
                pack_order=[model_pack.name for model_pack in model_packs],
                run_order=run_order,
                title=f"FP8 benchmark suite | {prompt_key}",
                prompt=prompt_text,
            )
            contact_sheet_path = prompt_output_dir / f"contact_sheet_{prompt_key}.png"
            contact_sheet.save(contact_sheet_path)
            contact_sheet_paths.append(contact_sheet_path)
            typer.echo(f"  contact sheet: {contact_sheet_path}")
    finally:
        for pack_name, session in sessions.items():
            if hasattr(session, "recycle"):
                session.recycle(f"pack-compare-suite cleanup {pack_name}")

    fieldnames = [
        "suite_id",
        "prompt_key",
        "prompt",
        "run_label",
        "seed",
        "width",
        "height",
        "baseline_pack",
        "pack",
        "status",
        "duration_ms",
        "reserved_bytes",
        "max_reserved_bytes",
        "execution_mode",
        "output_path",
        "baseline_output_path",
        "error",
        "mse",
        "psnr_db",
    ]
    with report_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with report_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    typer.echo("")
    typer.echo("Pack compare suite summary:")
    for pack_name in pack_names:
        summary = _summarize_suite_pack(rows, pack_name)
        typer.echo(
            f"{pack_name}: median_ms={_format_optional_number(summary['median_duration_ms'])} "
            f"median_reserved_mb={_bytes_to_mb(summary['median_reserved_bytes'])} "
            f"max_reserved_mb={_bytes_to_mb(summary['max_reserved_bytes'])} "
            f"avg_mse={_format_optional_number(summary['avg_mse'])} "
            f"avg_psnr_db={_format_optional_number(summary['avg_psnr_db'])}"
        )
    typer.echo(f"Report CSV: {report_csv}")
    typer.echo(f"Report JSONL: {report_jsonl}")
    for contact_sheet_path in contact_sheet_paths:
        typer.echo(f"Contact sheet: {contact_sheet_path}")


@cli.command("procedural-latent-preview")
def procedural_latent_preview(
    width: int = typer.Option(1024, "--width"),
    height: int = typer.Option(1024, "--height"),
    count: int = typer.Option(16, "--count", min=1, max=64),
    seed_start: int = typer.Option(1, "--seed-start"),
    creativity: int = typer.Option(1, "--creativity", min=1, max=3),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir"),
) -> None:
    import torch

    from app.core.backends.diffusers_zimage import DiffusersZImageBackend

    if width <= 0 or height <= 0:
        typer.echo("Width and height must be positive.")
        raise typer.Exit(code=1)

    settings = load_settings()
    root_output = settings.paths.outputs_dir.resolve()
    resolved_output_dir = (
        _resolve_cli_path(Path.cwd(), output_dir)
        if output_dir is not None
        else _preview_output_dir(root_output, "procedural-latent-preview")
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    latent_pipe_stub = type("LatentPipeStub", (), {"vae_scale_factor": 8})()
    latent_height, latent_width = DiffusersZImageBackend._resolve_latent_spatial_shape(
        latent_pipe_stub,
        width=width,
        height=height,
    )

    entries: list[dict[str, object]] = []
    manifest_items: list[dict[str, object]] = []
    typer.echo(
        f"Generating {count} procedural latent previews at creativity {creativity} "
        f"for output {width}x{height} "
        f"(latent {latent_width}x{latent_height}) into {resolved_output_dir}"
    )

    for seed in range(int(seed_start), int(seed_start) + int(count)):
        raw_latent, recipe = DiffusersZImageBackend._build_procedural_latent_tensor(
            expected_channels=16,
            target_height=latent_height,
            target_width=latent_width,
            seed=seed,
            creativity=creativity,
            torch_module=torch,
        )
        mixed_latent, alpha, preprocess = DiffusersZImageBackend._normalize_and_mix_latent(
            latent_tensor=raw_latent,
            seed=seed,
            torch_module=torch,
            noise_mix=DiffusersZImageBackend._PROCEDURAL_LATENT_NOISE_MIX,
            preprocess=DiffusersZImageBackend._PROCEDURAL_LATENT_PREPROCESS,
        )
        panel, composite = _build_procedural_preview_panel(
            seed=seed,
            creativity=creativity,
            recipe=recipe,
            width=width,
            height=height,
            raw_latent=raw_latent,
            mixed_latent=mixed_latent,
        )
        preview_path = resolved_output_dir / f"seed_{seed:06d}_preview.png"
        panel.save(preview_path, format="PNG")
        manifest_items.append(
            {
                "seed": seed,
                "recipe": recipe,
                "latent_shape": list(raw_latent.shape),
                "creativity": creativity,
                "width": width,
                "height": height,
                "raw_stats": _latent_stats(raw_latent),
                "mixed_stats": _latent_stats(mixed_latent),
                "mixed_alpha": alpha,
                "mixed_preprocess": preprocess,
                "preview_path": str(preview_path),
            }
        )
        entries.append(
            {
                "seed": seed,
                "creativity": creativity,
                "recipe": recipe,
                "composite": composite,
            }
        )
        typer.echo(f"[OK] seed {seed}: {preview_path.name} | {recipe}")

    contact_sheet = _build_preview_contact_sheet(
        entries,
        title=f"Procedural latent preview | creativity {creativity} | {count} seeds | {width}x{height}",
    )
    contact_sheet_path = resolved_output_dir / "contact_sheet.png"
    contact_sheet.save(contact_sheet_path, format="PNG")

    manifest_path = resolved_output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "count": count,
                "seed_start": seed_start,
                "creativity": creativity,
                "width": width,
                "height": height,
                "latent_width": latent_width,
                "latent_height": latent_height,
                "items": manifest_items,
                "contact_sheet": str(contact_sheet_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    typer.echo(f"Contact sheet: {contact_sheet_path}")
    typer.echo(f"Manifest: {manifest_path}")


@cli.command("upscale-test")
def upscale_test(
    input_image: Path = typer.Option(
        Path("outputs/_Upscale_test.png"),
        "--input-image",
        help="Input image for superscaling tests.",
    ),
    checkpoint: Path = typer.Option(
        Path("models/upscaler/2x_RealESRGAN_x2plus.pth"),
        "--checkpoint",
        help="Path to upscaler checkpoint (.pth or .safetensors).",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Destination directory for upscaled images. Defaults to outputs/.",
    ),
    profiles: str = typer.Option(
        "high,balanced,constrained",
        "--profiles",
        help="Engineering-only comma-separated resource tiers to test (high, balanced, constrained).",
    ),
    tile_size: Optional[int] = typer.Option(
        None,
        "--tile-size",
        min=0,
        help="Override upscaler tile size. Use 0 for full-frame (no tiling).",
    ),
    tile_overlap: Optional[int] = typer.Option(
        None,
        "--tile-overlap",
        min=0,
        help="Override upscaler tile overlap. Ignored when tile size is 0.",
    ),
) -> None:
    import gc

    import torch

    from app.core.upscale import run_upscale_test
    from app.storage import append_generation_metric, build_output_path, save_png_with_metadata

    seed_settings = load_settings()
    root = seed_settings.paths.root_dir
    input_path = _resolve_cli_path(root, input_image)
    checkpoint_path = _resolve_cli_path(root, checkpoint)

    if not input_path.exists() or not input_path.is_file():
        typer.echo(f"Input image not found: {input_path}")
        raise typer.Exit(code=1)
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        typer.echo(f"Upscaler checkpoint not found: {checkpoint_path}")
        raise typer.Exit(code=1)

    requested_profiles: list[str] = []
    for entry in profiles.split(","):
        name = entry.strip().lower()
        if name and name not in requested_profiles:
            requested_profiles.append(name)
    if not requested_profiles:
        typer.echo("No resource tiers requested. Provide --profiles high,balanced,constrained")
        raise typer.Exit(code=1)

    failures = 0
    for profile_name in requested_profiles:
        try:
            settings = load_settings(profile_name=profile_name)
            result = run_upscale_test(
                input_image_path=input_path,
                checkpoint_path=checkpoint_path,
                profile_name=settings.runtime_profile.name,
                tile_size_override=tile_size,
                tile_overlap_override=tile_overlap,
            )
            destination_dir = (
                _resolve_cli_path(root, output_dir) if output_dir else settings.paths.outputs_dir
            )
            output_path = build_output_path(
                destination_dir,
                prefix=f"upscale_{settings.runtime_profile.name}",
            )
            saved_path = save_png_with_metadata(
                image=result.image,
                prompt=f"Upscale test from {input_path.name}",
                settings=settings,
                output_path=output_path,
                extra_metadata={
                    "mode": "upscale_test",
                    "profile": settings.runtime_profile.name,
                    "source_image": str(input_path),
                    "upscaler_checkpoint": str(checkpoint_path),
                    **result.telemetry_dict(),
                },
            )
            metrics_file = append_generation_metric(
                settings=settings,
                payload={
                    "mode": "upscale_test",
                    "profile": settings.runtime_profile.name,
                    "source_image": str(input_path),
                    "output_path": str(saved_path),
                    "upscaler_checkpoint": str(checkpoint_path),
                    **result.telemetry_dict(),
                },
            )
            typer.echo(
                f"[{settings.runtime_profile.name}] {result.source_width}x{result.source_height} -> "
                f"{result.output_width}x{result.output_height} in {result.duration_ms} ms "
                f"(device={result.device}, precision={result.precision}, tile={result.tile_size}); "
                f"saved={saved_path}"
            )
            typer.echo(f"[{settings.runtime_profile.name}] Metrics: {metrics_file}")
        except Exception as exc:
            failures += 1
            typer.echo(f"[{profile_name}] upscale test failed: {exc}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if failures:
        raise typer.Exit(code=1)


@cli.command("upscale-refine")
def upscale_refine(
    input_image: Path = typer.Option(..., "--input-image", help="Input image to upscale and refine."),
    prompt: str = typer.Option(..., "--prompt"),
    pack: str = typer.Option(..., "--pack", help="Model pack name or folder name"),
    checkpoint: Path = typer.Option(
        Path("models/upscaler/2x_RealESRGAN_x2plus.pth"),
        "--checkpoint",
        help="Path to upscaler checkpoint (.pth or .safetensors).",
    ),
    strength: float = typer.Option(0.20, "--strength", min=0.01, max=0.99),
    refine_steps: int = typer.Option(6, "--refine-steps", min=1, max=50),
    refine_tile_size: Optional[int] = typer.Option(
        None,
        "--refine-tile-size",
        min=0,
        help="Override img2img tile size. Use 0 for full-frame refine.",
    ),
    refine_tile_overlap: int = typer.Option(
        64,
        "--refine-tile-overlap",
        min=8,
        help="Tile overlap for img2img refine when tiling is active.",
    ),
    scheduler_mode: str = typer.Option("euler", "--scheduler-mode"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    enhance_prompt: bool = typer.Option(
        False,
        "--enhance-prompt/--no-enhance-prompt",
        help="Use loaded text_encoder to rewrite prompt before refine pass.",
    ),
    output: Optional[Path] = typer.Option(None, "--output"),
) -> None:
    from PIL import Image

    from app.core.worker import GenerationRequest, GenerationSession
    from app.storage import append_generation_metric, save_png_with_metadata

    seed_settings = load_settings()
    root = seed_settings.paths.root_dir
    input_path = _resolve_cli_path(root, input_image)
    checkpoint_path = _resolve_cli_path(root, checkpoint)

    if not input_path.exists() or not input_path.is_file():
        typer.echo(f"Input image not found: {input_path}")
        raise typer.Exit(code=1)
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        typer.echo(f"Upscaler checkpoint not found: {checkpoint_path}")
        raise typer.Exit(code=1)

    model_pack = _load_pack_or_exit(seed_settings, pack)
    _assert_supported_backend_or_exit(model_pack)
    session = GenerationSession(settings=seed_settings, model_pack=model_pack)
    with Image.open(input_path) as source_file:
        source_image = source_file.convert("RGB")

    try:
        result = session.upscale_and_refine(
            input_image=source_image,
            request=GenerationRequest(
                prompt=prompt,
                width=source_image.width,
                height=source_image.height,
                seed=seed,
                scheduler_mode=scheduler_mode,
                enhance_prompt=enhance_prompt,
                refine_strength=strength,
                refine_steps=refine_steps,
                refine_tile_size=refine_tile_size,
                refine_tile_overlap=refine_tile_overlap,
                upscaler_checkpoint=checkpoint_path,
            ),
        )
    except Exception as exc:
        typer.echo(f"Upscale+refine failed: {exc}")
        raise typer.Exit(code=1) from exc

    final_width, final_height = result.image.size
    saved_path = save_png_with_metadata(
        image=result.image,
        prompt=result.prompt_effective,
        settings=seed_settings,
        output_path=output,
        extra_metadata={
            "mode": result.mode,
            "prompt_original": result.prompt_original,
            "prompt_effective": result.prompt_effective,
            "prompt_enhanced": result.prompt_enhanced,
            "source_image": str(input_path),
            "source_width": source_image.width,
            "source_height": source_image.height,
            "width": final_width,
            "height": final_height,
            "steps": result.steps,
            "guidance_scale": result.guidance_scale,
            "backend": result.backend,
            "device": result.device,
            "model_pack": model_pack.name,
            **result.telemetry_dict(),
            "duration_ms": result.duration_ms,
            "upscale_duration_ms": result.upscale_duration_ms,
            "refine_duration_ms": result.refine_duration_ms,
            "refine_strength": result.refine_strength,
            "refine_tile_size": result.refine_tile_size,
            "refine_tile_overlap": result.refine_tile_overlap,
            "refine_tile_size_requested": result.refine_tile_size_requested,
            "refine_tile_size_effective": result.refine_tile_size_effective,
            "refine_tile_overlap_effective": result.refine_tile_overlap_effective,
            "refine_fallback_used": result.refine_fallback_used,
            "refine_fallback_attempt_count": result.refine_fallback_attempt_count,
            "upscaler_checkpoint": str(checkpoint_path),
            "runtime_profile": result.runtime_profile,
            "execution_mode": result.execution_mode,
        },
    )
    metrics_file = append_generation_metric(
        settings=seed_settings,
        payload={
            "mode": result.mode,
            "prompt": result.prompt_effective,
            "prompt_original": result.prompt_original,
            "prompt_effective": result.prompt_effective,
            "prompt_enhanced": result.prompt_enhanced,
            "source_image": str(input_path),
            "source_width": source_image.width,
            "source_height": source_image.height,
            "width": final_width,
            "height": final_height,
            "output_path": str(saved_path),
            "model_pack": model_pack.name,
            "upscaler_checkpoint": str(checkpoint_path),
            **result.telemetry_dict(),
        },
    )

    typer.echo(
        f"Saved: {saved_path} ({source_image.width}x{source_image.height} -> {final_width}x{final_height})"
    )
    typer.echo(
        f"Mode={result.mode}, total={result.duration_ms} ms, "
        f"upscale={result.upscale_duration_ms} ms, "
        f"refine={result.refine_duration_ms} ms, "
        f"tile={result.refine_tile_size}"
    )
    if result.prompt_enhanced:
        typer.echo(f"Prompt enhanced: {result.prompt_effective}")
    typer.echo(f"Metrics: {metrics_file}")


@cli.command("seedvr2-benchmark")
def seedvr2_benchmark(
    input_image: Path = typer.Option(
        Path("outputs/_Upscale_test.png"),
        "--input-image",
        help="Input image used for A/B benchmark runs.",
    ),
    checkpoint: Path = typer.Option(
        Path("models/upscaler/2x_RealESRGAN_x2plus.pth"),
        "--checkpoint",
        help="Baseline x2plus checkpoint path for comparison.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Destination directory for benchmark outputs. Defaults to outputs/.",
    ),
    profiles: str = typer.Option(
        "high,balanced,constrained",
        "--profiles",
        help="Engineering-only comma-separated resource tiers to test (high, balanced, constrained).",
    ),
    mode: str = typer.Option(
        "both",
        "--mode",
        help="Benchmark mode: cold, warm, or both.",
    ),
    runs: int = typer.Option(
        3,
        "--runs",
        min=1,
        max=20,
        help="Measured runs per mode/profile.",
    ),
    warmup_runs: int = typer.Option(
        1,
        "--warmup-runs",
        min=0,
        max=10,
        help="Warmup runs before warm measurements.",
    ),
    target_median_seconds: int = typer.Option(
        45,
        "--target-median-seconds",
        min=10,
        max=600,
        help="Warm median target for pass/fail evaluation.",
    ),
    timeout_seconds: int = typer.Option(
        240,
        "--timeout-seconds",
        min=30,
        max=3600,
        help="Per-run timeout for SeedVR2 execution.",
    ),
    max_consecutive_failures: int = typer.Option(
        3,
        "--max-consecutive-failures",
        min=1,
        max=20,
        help="Abort benchmark after this many consecutive failures.",
    ),
) -> None:
    import gc
    import statistics
    import time

    import torch
    from PIL import Image

    from app.core.seedvr2 import clear_seedvr2_runtime_cache, upscale_with_seedvr2
    from app.core.upscale import run_upscale_test
    from app.storage import append_generation_metric, build_output_path, save_png_with_metadata

    seed_settings = load_settings()
    root = seed_settings.paths.root_dir
    input_path = _resolve_cli_path(root, input_image)
    checkpoint_path = _resolve_cli_path(root, checkpoint)
    if not input_path.exists() or not input_path.is_file():
        typer.echo(f"Input image not found: {input_path}")
        raise typer.Exit(code=1)
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        typer.echo(f"Baseline checkpoint not found: {checkpoint_path}")
        raise typer.Exit(code=1)

    requested_profiles: list[str] = []
    for entry in profiles.split(","):
        name = entry.strip().lower()
        if name and name not in requested_profiles:
            requested_profiles.append(name)
    if not requested_profiles:
        typer.echo("No resource tiers requested. Provide --profiles high,balanced,constrained")
        raise typer.Exit(code=1)

    mode_normalized = mode.strip().lower()
    if mode_normalized not in {"cold", "warm", "both"}:
        typer.echo("Invalid --mode. Allowed: cold, warm, both")
        raise typer.Exit(code=1)

    started_utc = datetime.now(timezone.utc)
    report_key = started_utc.strftime("%Y%m%d_%H%M%S")
    report_csv = seed_settings.paths.data_dir / f"seedvr2_benchmark_{report_key}.csv"
    report_jsonl = seed_settings.paths.data_dir / f"seedvr2_benchmark_{report_key}.jsonl"
    report_csv.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    consecutive_failures = 0
    aborted = False
    warm_medians_by_profile: dict[str, float] = {}

    for profile_name in requested_profiles:
        profile_settings = load_settings(profile_name=profile_name)
        destination_dir = (
            _resolve_cli_path(root, output_dir) if output_dir else profile_settings.paths.outputs_dir
        )
        base_pair_id = uuid4().hex[:10]

        # Baseline x2plus: one run per profile for reference.
        for engine_name in ("x2plus_baseline",):
            row: dict[str, object] = {
                "benchmark_pair_id": base_pair_id,
                "profile": profile_settings.runtime_profile.name,
                "engine": engine_name,
                "run_label": "baseline_1",
                "status": "pending",
                "duration_ms": None,
                "upscale_infer_ms": None,
                "source_image": str(input_path),
                "output_path": "",
                "error": "",
            }
            wall_started = time.perf_counter()
            try:
                if engine_name == "x2plus_baseline":
                    baseline = run_upscale_test(
                        input_image_path=input_path,
                        checkpoint_path=checkpoint_path,
                        profile_name=profile_settings.runtime_profile.name,
                    )
                    output_image = baseline.image
                    infer_duration_ms = int(baseline.duration_ms)
                    telemetry: dict[str, object] = {
                        "upscale_engine": "x2plus_baseline",
                        "upscale_model_repo": "",
                        "upscale_model_revision": "",
                        "upscale_dtype": baseline.precision,
                        "upscale_vram_peak_mb": None,
                        "upscale_infer_ms": infer_duration_ms,
                        "upscale_success": True,
                        **baseline.telemetry_dict(),
                    }
                else:
                    with Image.open(input_path) as input_file:
                        seedvr2 = upscale_with_seedvr2(
                            image=input_file.convert("RGB"),
                            settings=profile_settings,
                            runtime_profile=profile_settings.runtime_profile.name,
                            timeout_seconds=timeout_seconds,
                        )
                    output_image = seedvr2.image
                    infer_duration_ms = int(seedvr2.infer_ms)
                    telemetry = seedvr2.telemetry_dict()

                saved_path = save_png_with_metadata(
                    image=output_image,
                    prompt=f"SeedVR2 benchmark from {input_path.name}",
                    settings=profile_settings,
                    output_path=build_output_path(
                        destination_dir,
                        prefix=f"benchmark_{profile_settings.runtime_profile.name}_{engine_name}_{base_pair_id}",
                    ),
                    extra_metadata={
                        "mode": "seedvr2_benchmark",
                        "benchmark_pair_id": base_pair_id,
                        "benchmark_engine": engine_name,
                        "benchmark_run_label": "baseline_1",
                        "source_image": str(input_path),
                        "profile": profile_settings.runtime_profile.name,
                        **telemetry,
                    },
                )
                total_duration_ms = int((time.perf_counter() - wall_started) * 1000)
                row["status"] = "success"
                row["duration_ms"] = total_duration_ms
                row["output_path"] = str(saved_path)
                row["error"] = ""
                row["upscale_infer_ms"] = infer_duration_ms
                append_generation_metric(
                    settings=profile_settings,
                    payload={
                        "mode": "seedvr2_benchmark",
                        "benchmark_pair_id": base_pair_id,
                        "benchmark_engine": engine_name,
                        "benchmark_run_label": "baseline_1",
                        "profile": profile_settings.runtime_profile.name,
                        "source_image": str(input_path),
                        "output_path": str(saved_path),
                        "duration_ms": total_duration_ms,
                        **telemetry,
                    },
                )
                consecutive_failures = 0
            except TimeoutError as exc:
                total_duration_ms = int((time.perf_counter() - wall_started) * 1000)
                row["status"] = "timeout"
                row["duration_ms"] = total_duration_ms
                row["error"] = str(exc)
                consecutive_failures += 1
            except Exception as exc:  # noqa: BLE001
                total_duration_ms = int((time.perf_counter() - wall_started) * 1000)
                row["status"] = "error"
                row["duration_ms"] = total_duration_ms
                row["error"] = str(exc)
                consecutive_failures += 1
            finally:
                records.append(row)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if consecutive_failures >= max_consecutive_failures:
                aborted = True
                typer.echo(
                    "Aborting benchmark after "
                    f"{consecutive_failures} consecutive failures."
                )
                break
        if aborted:
            break

        def _record_seedvr2_run(
            *,
            run_pair_id: str,
            run_label: str,
            reuse_runner: bool,
        ) -> None:
            nonlocal consecutive_failures, aborted
            row: dict[str, object] = {
                "benchmark_pair_id": run_pair_id,
                "profile": profile_settings.runtime_profile.name,
                "engine": "seedvr2",
                "run_label": run_label,
                "status": "pending",
                "duration_ms": None,
                "upscale_infer_ms": None,
                "source_image": str(input_path),
                "output_path": "",
                "error": "",
            }
            wall_started = time.perf_counter()
            try:
                with Image.open(input_path) as input_file:
                    seedvr2 = upscale_with_seedvr2(
                        image=input_file.convert("RGB"),
                        settings=profile_settings,
                        runtime_profile=profile_settings.runtime_profile.name,
                        timeout_seconds=timeout_seconds,
                        reuse_runner=reuse_runner,
                    )
                output_image = seedvr2.image
                infer_duration_ms = int(seedvr2.infer_ms)
                telemetry = seedvr2.telemetry_dict()
                saved_path = save_png_with_metadata(
                    image=output_image,
                    prompt=f"SeedVR2 benchmark from {input_path.name}",
                    settings=profile_settings,
                    output_path=build_output_path(
                        destination_dir,
                        prefix=(
                            f"benchmark_{profile_settings.runtime_profile.name}_seedvr2_"
                            f"{run_label}_{run_pair_id}"
                        ),
                    ),
                    extra_metadata={
                        "mode": "seedvr2_benchmark",
                        "benchmark_pair_id": run_pair_id,
                        "benchmark_engine": "seedvr2",
                        "benchmark_run_label": run_label,
                        "source_image": str(input_path),
                        "profile": profile_settings.runtime_profile.name,
                        **telemetry,
                    },
                )
                total_duration_ms = int((time.perf_counter() - wall_started) * 1000)
                row["status"] = "success"
                row["duration_ms"] = total_duration_ms
                row["output_path"] = str(saved_path)
                row["error"] = ""
                row["upscale_infer_ms"] = infer_duration_ms
                append_generation_metric(
                    settings=profile_settings,
                    payload={
                        "mode": "seedvr2_benchmark",
                        "benchmark_pair_id": run_pair_id,
                        "benchmark_engine": "seedvr2",
                        "benchmark_run_label": run_label,
                        "profile": profile_settings.runtime_profile.name,
                        "source_image": str(input_path),
                        "output_path": str(saved_path),
                        "duration_ms": total_duration_ms,
                        **telemetry,
                    },
                )
                consecutive_failures = 0
            except TimeoutError as exc:
                total_duration_ms = int((time.perf_counter() - wall_started) * 1000)
                row["status"] = "timeout"
                row["duration_ms"] = total_duration_ms
                row["error"] = str(exc)
                consecutive_failures += 1
            except Exception as exc:  # noqa: BLE001
                total_duration_ms = int((time.perf_counter() - wall_started) * 1000)
                row["status"] = "error"
                row["duration_ms"] = total_duration_ms
                row["error"] = str(exc)
                consecutive_failures += 1
            finally:
                records.append(row)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if consecutive_failures >= max_consecutive_failures:
                aborted = True
                typer.echo(
                    "Aborting benchmark after "
                    f"{consecutive_failures} consecutive failures."
                )

        if mode_normalized in {"cold", "both"}:
            typer.echo(f"Running cold SeedVR2 series for profile '{profile_settings.runtime_profile.name}'...")
            for run_idx in range(runs):
                clear_seedvr2_runtime_cache(profile_settings.runtime_profile.name)
                _record_seedvr2_run(
                    run_pair_id=uuid4().hex[:10],
                    run_label=f"cold_{run_idx + 1}",
                    reuse_runner=False,
                )
                if aborted:
                    break

        if aborted:
            break

        if mode_normalized in {"warm", "both"}:
            typer.echo(f"Running warm SeedVR2 series for profile '{profile_settings.runtime_profile.name}'...")
            clear_seedvr2_runtime_cache(profile_settings.runtime_profile.name)
            for warmup_idx in range(warmup_runs):
                _record_seedvr2_run(
                    run_pair_id=uuid4().hex[:10],
                    run_label=f"warmup_{warmup_idx + 1}",
                    reuse_runner=True,
                )
                if aborted:
                    break
            if aborted:
                break
            for run_idx in range(runs):
                _record_seedvr2_run(
                    run_pair_id=uuid4().hex[:10],
                    run_label=f"warm_{run_idx + 1}",
                    reuse_runner=True,
                )
                if aborted:
                    break
            if aborted:
                break

            warm_success_ms = [
                int(row["duration_ms"])
                for row in records
                if row.get("profile") == profile_settings.runtime_profile.name
                and row.get("engine") == "seedvr2"
                and str(row.get("run_label", "")).startswith("warm_")
                and row.get("status") == "success"
                and row.get("duration_ms") is not None
            ]
            if warm_success_ms:
                warm_medians_by_profile[profile_settings.runtime_profile.name] = float(
                    statistics.median(warm_success_ms)
                )

    fieldnames = [
        "benchmark_pair_id",
        "profile",
        "engine",
        "run_label",
        "status",
        "duration_ms",
        "upscale_infer_ms",
        "source_image",
        "output_path",
        "error",
    ]
    with report_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    with report_jsonl.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    typer.echo("")
    typer.echo("SeedVR2 benchmark results:")
    typer.echo("profile      engine            run_label      status    duration_ms  output")
    typer.echo("-----------  ----------------  -------------  --------  -----------  ----------------")
    for row in records:
        typer.echo(
            f"{str(row['profile']):11}  "
            f"{str(row['engine']):16}  "
            f"{str(row['run_label']):13}  "
            f"{str(row['status']):8}  "
            f"{str(row['duration_ms']):11}  "
            f"{str(row['output_path'])}"
        )
    if warm_medians_by_profile:
        typer.echo("")
        typer.echo("Warm median summary:")
        target_ms = target_median_seconds * 1000
        for profile_name in requested_profiles:
            if profile_name not in warm_medians_by_profile:
                continue
            median_ms = int(warm_medians_by_profile[profile_name])
            verdict = "PASS" if median_ms <= target_ms else "FAIL"
            typer.echo(
                f"  {profile_name:11} median={median_ms}ms "
                f"target<={target_ms}ms [{verdict}]"
            )
    typer.echo(f"CSV report: {report_csv}")
    typer.echo(f"JSONL report: {report_jsonl}")

    warm_target_failed = False
    if mode_normalized in {"warm", "both"}:
        target_ms = target_median_seconds * 1000
        for profile_name, median_ms in warm_medians_by_profile.items():
            if median_ms > target_ms:
                warm_target_failed = True
                typer.echo(
                    f"Warm median target failed for profile '{profile_name}': "
                    f"{int(median_ms)}ms > {target_ms}ms"
                )
    if aborted or any(str(row.get("status")) != "success" for row in records) or warm_target_failed:
        raise typer.Exit(code=1)


@cli.command("seedvr2-blend-benchmark")
def seedvr2_blend_benchmark(
    inputs: str = typer.Option(
        "",
        "--inputs",
        help="Comma-separated source image paths. If omitted, auto-selects latest two 1024x1024 justrayzist PNGs.",
    ),
    profile: str = typer.Option(
        "high",
        "--profile",
        help="Engineering-only resource tier override for x2 and SeedVR2 passes.",
    ),
    checkpoint: Path = typer.Option(
        Path("models/upscaler/2x_RealESRGAN_x2plus.pth"),
        "--checkpoint",
        help="Baseline x2plus checkpoint path.",
    ),
    alphas: str = typer.Option(
        "25,50,75",
        "--alphas",
        help="Comma-separated alpha percentages for x2-over-seed blending.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Destination directory for benchmark outputs. Defaults to outputs/.",
    ),
    timeout_seconds: int = typer.Option(
        240,
        "--timeout-seconds",
        min=30,
        max=3600,
        help="Per-run timeout for SeedVR2 execution.",
    ),
    max_consecutive_failures: int = typer.Option(
        3,
        "--max-consecutive-failures",
        min=1,
        max=20,
        help="Abort benchmark after this many consecutive failures.",
    ),
) -> None:
    import gc
    import time

    import torch
    from PIL import Image

    from app.core.seedvr2 import clear_seedvr2_runtime_cache, upscale_with_seedvr2
    from app.core.upscale import run_upscale_test
    from app.storage import append_generation_metric, build_output_path, save_png_with_metadata

    def _parse_alpha_list(raw: str) -> list[int]:
        values: list[int] = []
        for chunk in raw.split(","):
            token = chunk.strip()
            if not token:
                continue
            try:
                parsed = int(token)
            except ValueError as exc:
                raise ValueError(f"Invalid alpha value '{token}'. Use integers in [0,100].") from exc
            if parsed < 0 or parsed > 100:
                raise ValueError(f"Alpha value out of range: {parsed}. Allowed: 0..100.")
            if parsed not in values:
                values.append(parsed)
        if not values:
            raise ValueError("No alpha values provided. Example: --alphas 25,50,75")
        return values

    def _resolve_input_paths(root_dir: Path, raw_inputs: str) -> list[Path]:
        if raw_inputs.strip():
            resolved: list[Path] = []
            for chunk in raw_inputs.split(","):
                token = chunk.strip()
                if not token:
                    continue
                path = _resolve_cli_path(root_dir, Path(token))
                if not path.exists() or not path.is_file():
                    raise ValueError(f"Input image not found: {path}")
                resolved.append(path)
            if not resolved:
                raise ValueError("No valid input images provided in --inputs.")
            return resolved

        outputs_dir = root_dir / "outputs"
        candidates = sorted(
            outputs_dir.glob("justrayzist_*.png"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        selected: list[Path] = []
        for candidate in candidates:
            try:
                with Image.open(candidate) as img:
                    if img.size == (1024, 1024):
                        selected.append(candidate)
            except Exception:
                continue
            if len(selected) >= 2:
                break
        if len(selected) < 2:
            raise ValueError(
                "Unable to auto-select two 1024x1024 images from outputs/. "
                "Provide --inputs explicitly."
            )
        return selected

    seed_settings = load_settings()
    root = seed_settings.paths.root_dir
    profile_settings = load_settings(profile_name=profile)
    destination_dir = (
        _resolve_cli_path(root, output_dir) if output_dir else profile_settings.paths.outputs_dir
    )

    checkpoint_path = _resolve_cli_path(root, checkpoint)
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        typer.echo(f"Upscaler checkpoint not found: {checkpoint_path}")
        raise typer.Exit(code=1)

    try:
        alpha_values = _parse_alpha_list(alphas)
        input_paths = _resolve_input_paths(root, inputs)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1)

    started_utc = datetime.now(timezone.utc)
    report_key = started_utc.strftime("%Y%m%d_%H%M%S")
    report_csv = seed_settings.paths.data_dir / f"seedvr2_blend_benchmark_{report_key}.csv"
    report_jsonl = seed_settings.paths.data_dir / f"seedvr2_blend_benchmark_{report_key}.jsonl"
    report_csv.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    consecutive_failures = 0
    aborted = False

    def _record_row(row: dict[str, object]) -> None:
        nonlocal consecutive_failures, aborted
        records.append(row)
        if str(row.get("status")) == "success":
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                aborted = True
                typer.echo(
                    "Aborting benchmark after "
                    f"{consecutive_failures} consecutive failures."
                )

    for source_path in input_paths:
        if aborted:
            break

        run_id = uuid4().hex[:10]
        source_stem = source_path.stem
        seed_result = None
        x2_result = None

        # Stage 1: x2 baseline pass
        x2_row: dict[str, object] = {
            "run_id": run_id,
            "source_image": str(source_path),
            "profile": profile_settings.runtime_profile.name,
            "stage": "x2",
            "alpha_percent": None,
            "status": "pending",
            "duration_ms": None,
            "infer_ms": None,
            "output_path": "",
            "error": "",
        }
        x2_started = time.perf_counter()
        try:
            x2_result = run_upscale_test(
                input_image_path=source_path,
                checkpoint_path=checkpoint_path,
                profile_name=profile_settings.runtime_profile.name,
            )
            x2_saved = save_png_with_metadata(
                image=x2_result.image,
                prompt=f"Blend benchmark x2 from {source_path.name}",
                settings=profile_settings,
                output_path=build_output_path(
                    destination_dir,
                    prefix=f"blendbench_{profile_settings.runtime_profile.name}_{source_stem}_{run_id}_x2",
                ),
                extra_metadata={
                    "mode": "seedvr2_blend_benchmark",
                    "run_id": run_id,
                    "stage": "x2",
                    "source_image": str(source_path),
                    "profile": profile_settings.runtime_profile.name,
                    **x2_result.telemetry_dict(),
                },
            )
            x2_duration_ms = int((time.perf_counter() - x2_started) * 1000)
            x2_row["status"] = "success"
            x2_row["duration_ms"] = x2_duration_ms
            x2_row["infer_ms"] = int(x2_result.duration_ms)
            x2_row["output_path"] = str(x2_saved)
            append_generation_metric(
                settings=profile_settings,
                payload={
                    "mode": "seedvr2_blend_benchmark",
                    "run_id": run_id,
                    "stage": "x2",
                    "source_image": str(source_path),
                    "output_path": str(x2_saved),
                    "profile": profile_settings.runtime_profile.name,
                    "duration_ms": x2_duration_ms,
                    **x2_result.telemetry_dict(),
                },
            )
        except Exception as exc:  # noqa: BLE001
            x2_duration_ms = int((time.perf_counter() - x2_started) * 1000)
            x2_row["status"] = "error"
            x2_row["duration_ms"] = x2_duration_ms
            x2_row["error"] = str(exc)
        finally:
            _record_row(x2_row)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if aborted:
            break

        # Stage 2: seed pass
        seed_row: dict[str, object] = {
            "run_id": run_id,
            "source_image": str(source_path),
            "profile": profile_settings.runtime_profile.name,
            "stage": "seedvr2",
            "alpha_percent": None,
            "status": "pending",
            "duration_ms": None,
            "infer_ms": None,
            "output_path": "",
            "error": "",
        }
        seed_started = time.perf_counter()
        try:
            clear_seedvr2_runtime_cache(profile_settings.runtime_profile.name)
            with Image.open(source_path) as source_file:
                seed_result = upscale_with_seedvr2(
                    image=source_file.convert("RGB"),
                    settings=profile_settings,
                    runtime_profile=profile_settings.runtime_profile.name,
                    timeout_seconds=timeout_seconds,
                    reuse_runner=True,
                )
            seed_saved = save_png_with_metadata(
                image=seed_result.image,
                prompt=f"Blend benchmark seedvr2 from {source_path.name}",
                settings=profile_settings,
                output_path=build_output_path(
                    destination_dir,
                    prefix=f"blendbench_{profile_settings.runtime_profile.name}_{source_stem}_{run_id}_seedvr2",
                ),
                extra_metadata={
                    "mode": "seedvr2_blend_benchmark",
                    "run_id": run_id,
                    "stage": "seedvr2",
                    "source_image": str(source_path),
                    "profile": profile_settings.runtime_profile.name,
                    **seed_result.telemetry_dict(),
                },
            )
            seed_duration_ms = int((time.perf_counter() - seed_started) * 1000)
            seed_row["status"] = "success"
            seed_row["duration_ms"] = seed_duration_ms
            seed_row["infer_ms"] = int(seed_result.infer_ms)
            seed_row["output_path"] = str(seed_saved)
            append_generation_metric(
                settings=profile_settings,
                payload={
                    "mode": "seedvr2_blend_benchmark",
                    "run_id": run_id,
                    "stage": "seedvr2",
                    "source_image": str(source_path),
                    "output_path": str(seed_saved),
                    "profile": profile_settings.runtime_profile.name,
                    "duration_ms": seed_duration_ms,
                    **seed_result.telemetry_dict(),
                },
            )
        except Exception as exc:  # noqa: BLE001
            seed_duration_ms = int((time.perf_counter() - seed_started) * 1000)
            seed_row["status"] = "error"
            seed_row["duration_ms"] = seed_duration_ms
            seed_row["error"] = str(exc)
        finally:
            _record_row(seed_row)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if aborted:
            break

        # Stage 3: blended outputs (x2 over seed)
        for alpha_percent in alpha_values:
            blend_row: dict[str, object] = {
                "run_id": run_id,
                "source_image": str(source_path),
                "profile": profile_settings.runtime_profile.name,
                "stage": "blend",
                "alpha_percent": alpha_percent,
                "status": "pending",
                "duration_ms": None,
                "infer_ms": None,
                "output_path": "",
                "error": "",
            }
            blend_started = time.perf_counter()
            try:
                if x2_result is None or seed_result is None:
                    raise RuntimeError("Blend prerequisites missing: x2 and SeedVR2 passes must succeed.")
                if x2_result.image.size != seed_result.image.size:
                    raise RuntimeError(
                        "Blend size mismatch: "
                        f"x2={x2_result.image.size}, seedvr2={seed_result.image.size}"
                    )
                alpha_value = alpha_percent / 100.0
                blended = Image.blend(seed_result.image, x2_result.image, alpha_value)
                blend_saved = save_png_with_metadata(
                    image=blended,
                    prompt=f"Blend benchmark alpha {alpha_percent}% from {source_path.name}",
                    settings=profile_settings,
                    output_path=build_output_path(
                        destination_dir,
                        prefix=(
                            f"blendbench_{profile_settings.runtime_profile.name}_"
                            f"{source_stem}_{run_id}_a{alpha_percent:02d}"
                        ),
                    ),
                    extra_metadata={
                        "mode": "seedvr2_blend_benchmark",
                        "run_id": run_id,
                        "stage": "blend",
                        "alpha_percent": alpha_percent,
                        "source_image": str(source_path),
                        "profile": profile_settings.runtime_profile.name,
                        "blend_top": "x2plus_baseline",
                        "blend_base": "seedvr2",
                    },
                )
                blend_duration_ms = int((time.perf_counter() - blend_started) * 1000)
                blend_row["status"] = "success"
                blend_row["duration_ms"] = blend_duration_ms
                blend_row["output_path"] = str(blend_saved)
                append_generation_metric(
                    settings=profile_settings,
                    payload={
                        "mode": "seedvr2_blend_benchmark",
                        "run_id": run_id,
                        "stage": "blend",
                        "alpha_percent": alpha_percent,
                        "source_image": str(source_path),
                        "output_path": str(blend_saved),
                        "profile": profile_settings.runtime_profile.name,
                        "duration_ms": blend_duration_ms,
                        "blend_top": "x2plus_baseline",
                        "blend_base": "seedvr2",
                    },
                )
            except Exception as exc:  # noqa: BLE001
                blend_duration_ms = int((time.perf_counter() - blend_started) * 1000)
                blend_row["status"] = "error"
                blend_row["duration_ms"] = blend_duration_ms
                blend_row["error"] = str(exc)
            finally:
                _record_row(blend_row)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if aborted:
                break

    fieldnames = [
        "run_id",
        "source_image",
        "profile",
        "stage",
        "alpha_percent",
        "status",
        "duration_ms",
        "infer_ms",
        "output_path",
        "error",
    ]
    with report_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    with report_jsonl.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    typer.echo("")
    typer.echo("SeedVR2 blend benchmark results:")
    typer.echo("stage    alpha  status    duration_ms  source -> output")
    typer.echo("-------  -----  --------  -----------  ----------------")
    for row in records:
        alpha_label = "-" if row.get("alpha_percent") is None else str(row.get("alpha_percent"))
        typer.echo(
            f"{str(row['stage']):7}  "
            f"{alpha_label:5}  "
            f"{str(row['status']):8}  "
            f"{str(row['duration_ms']):11}  "
            f"{Path(str(row['source_image'])).name} -> {str(row['output_path'])}"
        )
    typer.echo(f"CSV report: {report_csv}")
    typer.echo(f"JSONL report: {report_jsonl}")

    if aborted or any(str(row.get("status")) != "success" for row in records):
        raise typer.Exit(code=1)


@cli.command("soak")
def soak(
    prompt: str = typer.Option(..., "--prompt"),
    pack: str = typer.Option(..., "--pack", help="Model pack name or folder name"),
    iterations: int = typer.Option(10, "--iterations", min=1),
    width: int = typer.Option(1024, "--width"),
    height: int = typer.Option(1024, "--height"),
    steps: Optional[int] = typer.Option(None, "--steps"),
    guidance_scale: Optional[float] = typer.Option(None, "--guidance-scale"),
    seed_start: Optional[int] = typer.Option(1, "--seed-start"),
    enhance_prompt: bool = typer.Option(
        False,
        "--enhance-prompt/--no-enhance-prompt",
        help="Use loaded text_encoder to rewrite prompt before generation.",
    ),
    recycle_every: Optional[int] = typer.Option(
        None,
        "--recycle-every",
        help="Override recycle cadence. Default follows the stable runtime baseline.",
    ),
    drift_threshold_mb: Optional[int] = typer.Option(
        None,
        "--drift-threshold-mb",
        help="Override drift threshold in MB. Default follows the stable runtime baseline.",
    ),
    warmup: bool = typer.Option(True, "--warmup/--no-warmup"),
    save_images: bool = typer.Option(False, "--save-images/--no-save-images"),
) -> None:
    from app.core.worker import GenerationRequest, GenerationSession
    from app.storage import append_generation_metric, save_png_with_metadata

    settings = load_settings()
    effective_drift_threshold_mb = (
        drift_threshold_mb
        if drift_threshold_mb is not None
        else settings.runtime_profile.default_soak_drift_threshold_mb
    )
    effective_recycle_every = (
        recycle_every
        if recycle_every is not None
        else settings.runtime_profile.default_soak_recycle_every
    )
    if effective_drift_threshold_mb < 1:
        typer.echo("--drift-threshold-mb must be >= 1.")
        raise typer.Exit(code=1)
    if effective_recycle_every < 0:
        typer.echo("--recycle-every must be >= 0.")
        raise typer.Exit(code=1)

    model_pack = _load_pack_or_exit(settings, pack)
    _assert_supported_backend_or_exit(model_pack)
    session = GenerationSession(settings=settings, model_pack=model_pack)
    session_id = f"soak_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    typer.echo(f"Session: {session_id}")
    typer.echo(
        f"Soak policy: runtime_profile={settings.runtime_profile.name}, "
        f"resource_tier={settings.resource_tier_controller.current().name}, "
        f"drift-threshold={effective_drift_threshold_mb}MB, "
        f"recycle-every={effective_recycle_every}"
    )

    baseline_source: str | None = None
    baseline_bytes: int | None = None

    if warmup:
        typer.echo("Warmup run starting...")
        warmup_seed = seed_start if seed_start is not None else None
        warmup_result = session.generate(
            GenerationRequest(
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=warmup_seed,
                enhance_prompt=enhance_prompt,
            )
        )
        baseline_source, baseline_bytes = _memory_source_and_bytes(warmup_result)
        append_generation_metric(
            settings=settings,
            payload={
                "mode": "soak_warmup",
                "session_id": session_id,
                "model_pack": model_pack.name,
                "prompt": prompt,
                "width": width,
                "height": height,
                **warmup_result.telemetry_dict(),
            },
        )
        typer.echo(
            f"Warmup completed in {warmup_result.duration_ms} ms "
            f"(memory source: {baseline_source or 'none'})."
        )

    failure_count = 0
    for iteration in range(1, iterations + 1):
        seed = (seed_start + iteration - 1) if seed_start is not None else None
        try:
            result = session.generate(
                GenerationRequest(
                    prompt=prompt,
                    width=width,
                    height=height,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    enhance_prompt=enhance_prompt,
                )
            )
        except Exception as exc:
            failure_count += 1
            append_generation_metric(
                settings=settings,
                payload={
                    "mode": "soak_error",
                    "session_id": session_id,
                    "iteration": iteration,
                    "model_pack": model_pack.name,
                    "prompt": prompt,
                    "error": str(exc),
                },
            )
            typer.echo(f"[{iteration}/{iterations}] error: {exc}")
            break

        memory_source, memory_bytes = _memory_source_and_bytes(result)
        if baseline_bytes is None and memory_bytes is not None:
            baseline_source = memory_source
            baseline_bytes = memory_bytes

        drift_mb = None
        if (
            baseline_bytes is not None
            and memory_bytes is not None
            and baseline_source == memory_source
        ):
            drift_mb = round((memory_bytes - baseline_bytes) / (1024 * 1024), 2)

        saved_path = None
        if save_images:
            saved_path = save_png_with_metadata(
                image=result.image,
                prompt=prompt,
                settings=settings,
                extra_metadata={
                    "prompt_original": result.prompt_original,
                    "prompt_effective": result.prompt_effective,
                    "prompt_enhanced": result.prompt_enhanced,
                    "width": width,
                    "height": height,
                    "steps": result.steps,
                    "guidance_scale": result.guidance_scale,
                    "backend": result.backend,
                    "device": result.device,
                    "model_pack": model_pack.name,
                    "duration_ms": result.duration_ms,
                    "mode": "soak",
                    "iteration": iteration,
                },
            )

        recycle_reason: str | None = None
        if (
            drift_mb is not None
            and drift_mb > effective_drift_threshold_mb
            and iteration < iterations
        ):
            recycle_reason = (
                f"memory drift {drift_mb}MB exceeded threshold {effective_drift_threshold_mb}MB"
            )
        if (
            effective_recycle_every > 0
            and (iteration % effective_recycle_every == 0)
            and iteration < iterations
        ):
            period_reason = f"periodic recycle every {effective_recycle_every} iterations"
            recycle_reason = f"{recycle_reason}; {period_reason}" if recycle_reason else period_reason

        append_generation_metric(
            settings=settings,
            payload={
                "mode": "soak",
                "session_id": session_id,
                "iteration": iteration,
                "model_pack": model_pack.name,
                "prompt": prompt,
                "prompt_original": result.prompt_original,
                "prompt_effective": result.prompt_effective,
                "prompt_enhanced": result.prompt_enhanced,
                "width": width,
                "height": height,
                "memory_source": memory_source,
                "memory_drift_mb": drift_mb,
                "recycle_reason": recycle_reason,
                "output_path": str(saved_path) if saved_path else None,
                **result.telemetry_dict(),
            },
        )

        typer.echo(
            f"[{iteration}/{iterations}] {result.duration_ms} ms, "
            f"drift={drift_mb if drift_mb is not None else 'n/a'} MB, "
            f"saved={'yes' if saved_path else 'no'}"
        )

        if recycle_reason:
            session.recycle(recycle_reason)
            baseline_source = None
            baseline_bytes = None

        del result

    typer.echo(
        f"Soak complete. iterations={iterations}, failures={failure_count}, "
        f"recycles={session.stats.recycle_count}."
    )
    append_generation_metric(
        settings=settings,
        payload={
            "mode": "soak_summary",
            "session_id": session_id,
            "model_pack": model_pack.name,
            "prompt": prompt,
            "iterations_requested": iterations,
            "iterations_completed": session.stats.generation_count - (1 if warmup else 0),
            "failures": failure_count,
            "recycles": session.stats.recycle_count,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "drift_threshold_mb": effective_drift_threshold_mb,
            "recycle_every": effective_recycle_every,
            "runtime_profile": settings.runtime_profile.name,
            "resource_tier": settings.resource_tier_controller.current().name,
        },
    )


@cli.command("soak-report")
def soak_report(
    session_id: Optional[str] = typer.Option(None, "--session-id"),
    list_sessions: bool = typer.Option(False, "--list-sessions"),
    metrics_path: Optional[Path] = typer.Option(None, "--metrics-path"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    from app.storage.soak_report import (
        group_soak_sessions,
        latest_session_id,
        load_metrics_jsonl,
        summarize_session,
    )

    settings = load_settings()
    path = metrics_path or (settings.paths.data_dir / "generation_metrics.jsonl")
    rows = load_metrics_jsonl(path)
    grouped = group_soak_sessions(rows)
    if not grouped:
        typer.echo(f"No soak metrics found in {path}")
        raise typer.Exit(code=1)

    if list_sessions:
        summaries = [summarize_session(name, records).to_dict() for name, records in grouped.items()]
        summaries.sort(key=lambda item: item.get("ended_at") or "", reverse=True)
        if as_json:
            typer.echo(json.dumps({"metrics_path": str(path), "sessions": summaries}, indent=2))
            return
        typer.echo(f"Metrics file: {path}")
        for item in summaries:
            typer.echo(
                f"- {item['session_id']}: iterations={item['iteration_count']}, "
                f"errors={item['error_count']}, recycles={item['recycle_count']}, "
                f"ended_at={item['ended_at']}"
            )
        return

    selected = session_id or latest_session_id(grouped)
    if not selected or selected not in grouped:
        typer.echo(f"Session not found: {selected}")
        typer.echo("Use --list-sessions to inspect available session IDs.")
        raise typer.Exit(code=1)

    summary = summarize_session(selected, grouped[selected]).to_dict()
    if as_json:
        typer.echo(json.dumps({"metrics_path": str(path), "summary": summary}, indent=2))
        return

    typer.echo(f"Metrics file: {path}")
    typer.echo(f"Session: {summary['session_id']}")
    typer.echo(f"Time window: {summary['started_at']} -> {summary['ended_at']}")
    typer.echo(
        "Iterations: "
        f"{summary['iteration_count']} (warmup={summary['warmup_count']}, errors={summary['error_count']})"
    )
    typer.echo(f"Recycle events: {summary['recycle_count']}")
    typer.echo(
        "Latency ms: "
        f"avg={summary['duration_avg_ms']}, "
        f"p50={summary['duration_p50_ms']}, "
        f"p95={summary['duration_p95_ms']}, "
        f"p99={summary['duration_p99_ms']}"
    )
    typer.echo(
        "Drift MB: "
        f"first={summary['drift_first_mb']}, "
        f"last={summary['drift_last_mb']}, "
        f"min={summary['drift_min_mb']}, "
        f"max={summary['drift_max_mb']}, "
        f"slope/iter={summary['drift_slope_mb_per_iteration']}"
    )


def run() -> None:
    cli()


if __name__ == "__main__":
    run()
