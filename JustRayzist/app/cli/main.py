from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
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
from app.core.logging import configure_logging
from app.core.model_registry import (
    ModelPackValidationError,
    discover_model_packs,
    load_model_pack,
    load_model_pack_by_name,
)

cli = typer.Typer(add_completion=False, help="JustRayzist CLI")


def _load_pack_or_exit(settings, pack_name: str):
    try:
        return load_model_pack_by_name(settings.paths.model_packs_dir, pack_name)
    except ModelPackValidationError as exc:
        typer.echo(f"Model pack error: {exc}")
        raise typer.Exit(code=1)


def _assert_supported_backend_or_exit(model_pack) -> None:
    backend_name = (model_pack.backend_preference[0] if model_pack.backend_preference else "").lower()
    if backend_name not in {"diffusers", "diffusers_zimage"}:
        typer.echo(
            f"Unsupported backend '{backend_name}' in bootstrap version. "
            "Use backend_preference: ['diffusers']."
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


@cli.callback()
def callback(log_level: str = typer.Option("INFO", "--log-level")) -> None:
    configure_logging(log_level)


@cli.command("status")
def status(profile: Optional[str] = typer.Option(None, "--profile")) -> None:
    settings = load_settings(profile_name=profile)
    typer.echo(json.dumps(settings.to_dict(), indent=2))


@cli.command("doctor")
def doctor(profile: Optional[str] = typer.Option(None, "--profile")) -> None:
    settings = load_settings(profile_name=profile)
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
        "profile": settings.runtime_profile.name,
    }
    typer.echo(json.dumps(report, indent=2))


@cli.command("validate-models")
def validate_models(profile: Optional[str] = typer.Option(None, "--profile")) -> None:
    settings = load_settings(profile_name=profile)
    pack_paths = discover_model_packs(settings.paths.model_packs_dir)
    if not pack_paths:
        typer.echo("No model packs found.")
        raise typer.Exit(code=1)

    failed = 0
    for pack_path in pack_paths:
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
    profile: Optional[str] = typer.Option(None, "--profile"),
) -> None:
    import uvicorn

    if profile:
        os.environ["JUSTRAYZIST_PROFILE"] = profile
    load_settings(profile_name=profile)
    logging.getLogger(__name__).info("Starting web server on http://%s:%d", host, port)
    uvicorn.run("app.api.main:app", host=host, port=port, reload=False)


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
    profile: Optional[str] = typer.Option(None, "--profile"),
) -> None:
    from app.core.worker import GenerationRequest, GenerationSession
    from app.storage import append_generation_metric, save_png_with_metadata

    settings = load_settings(profile_name=profile)
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
            "Install dependencies from pyproject.toml first."
        )
        raise typer.Exit(code=2)
    except Exception as exc:
        typer.echo(f"Generation failed: {exc}")
        raise typer.Exit(code=1)

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
        help="Path to upscaler .pth checkpoint.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Destination directory for upscaled images. Defaults to outputs/.",
    ),
    profiles: str = typer.Option(
        "high,balanced,constrained",
        "--profiles",
        help="Comma-separated profile list to test (high, balanced, constrained).",
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
        typer.echo("No profiles requested. Provide --profiles high,balanced,constrained")
        raise typer.Exit(code=1)

    failures = 0
    for profile_name in requested_profiles:
        try:
            settings = load_settings(profile_name=profile_name)
            result = run_upscale_test(
                input_image_path=input_path,
                checkpoint_path=checkpoint_path,
                profile_name=settings.runtime_profile.name,
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
        help="Path to upscaler .pth checkpoint.",
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
    sharpen: bool = typer.Option(
        True,
        "--sharpen/--no-sharpen",
        help="Apply luma-only unsharp mask between upscale and img2img refine.",
    ),
    sharpen_sigma: float = typer.Option(
        1.0,
        "--sharpen-sigma",
        min=0.0,
        help="Unsharp sigma/radius on Y channel (YCbCr).",
    ),
    sharpen_amount: float = typer.Option(
        0.35,
        "--sharpen-amount",
        min=0.0,
        max=4.0,
        help="Unsharp amount for Y channel.",
    ),
    sharpen_threshold: int = typer.Option(
        3,
        "--sharpen-threshold",
        min=0,
        max=255,
        help="Unsharp threshold in 8-bit scale (noise suppression).",
    ),
    scheduler_mode: str = typer.Option("euler", "--scheduler-mode"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    enhance_prompt: bool = typer.Option(
        False,
        "--enhance-prompt/--no-enhance-prompt",
        help="Use loaded text_encoder to rewrite prompt before refine pass.",
    ),
    output: Optional[Path] = typer.Option(None, "--output"),
    profile: Optional[str] = typer.Option(None, "--profile"),
) -> None:
    from PIL import Image

    from app.core.worker import GenerationRequest, GenerationSession
    from app.storage import append_generation_metric, save_png_with_metadata

    seed_settings = load_settings(profile_name=profile)
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
    source_image = Image.open(input_path).convert("RGB")

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
                sharpen_enabled=sharpen,
                sharpen_sigma=sharpen_sigma,
                sharpen_amount=sharpen_amount,
                sharpen_threshold=sharpen_threshold,
                upscaler_checkpoint=checkpoint_path,
            ),
        )
    except Exception as exc:
        typer.echo(f"Upscale+refine failed: {exc}")
        raise typer.Exit(code=1)

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
            "duration_ms": result.duration_ms,
            "upscale_duration_ms": result.upscale_duration_ms,
            "sharpen_duration_ms": result.sharpen_duration_ms,
            "sharpen_enabled": result.sharpen_enabled,
            "sharpen_sigma": result.sharpen_sigma,
            "sharpen_amount": result.sharpen_amount,
            "sharpen_threshold": result.sharpen_threshold,
            "refine_duration_ms": result.refine_duration_ms,
            "refine_strength": result.refine_strength,
            "refine_tile_size": result.refine_tile_size,
            "refine_tile_overlap": result.refine_tile_overlap,
            "upscaler_checkpoint": str(checkpoint_path),
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
        f"upscale={result.upscale_duration_ms} ms, sharpen={result.sharpen_duration_ms} ms, "
        f"refine={result.refine_duration_ms} ms, "
        f"tile={result.refine_tile_size}"
    )
    if result.prompt_enhanced:
        typer.echo(f"Prompt enhanced: {result.prompt_effective}")
    typer.echo(f"Metrics: {metrics_file}")


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
        help="Override recycle cadence. Default is profile-specific.",
    ),
    drift_threshold_mb: Optional[int] = typer.Option(
        None,
        "--drift-threshold-mb",
        help="Override drift threshold in MB. Default is profile-specific.",
    ),
    warmup: bool = typer.Option(True, "--warmup/--no-warmup"),
    save_images: bool = typer.Option(False, "--save-images/--no-save-images"),
    profile: Optional[str] = typer.Option(None, "--profile"),
) -> None:
    from app.core.worker import GenerationRequest, GenerationSession
    from app.storage import append_generation_metric, save_png_with_metadata

    settings = load_settings(profile_name=profile)
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
        f"Soak policy: profile={settings.runtime_profile.name}, "
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
            "profile": settings.runtime_profile.name,
        },
    )


@cli.command("soak-report")
def soak_report(
    session_id: Optional[str] = typer.Option(None, "--session-id"),
    list_sessions: bool = typer.Option(False, "--list-sessions"),
    metrics_path: Optional[Path] = typer.Option(None, "--metrics-path"),
    as_json: bool = typer.Option(False, "--json"),
    profile: Optional[str] = typer.Option(None, "--profile"),
) -> None:
    from app.storage.soak_report import (
        group_soak_sessions,
        latest_session_id,
        load_metrics_jsonl,
        summarize_session,
    )

    settings = load_settings(profile_name=profile)
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
