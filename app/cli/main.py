from __future__ import annotations

import json
import logging
import os
import csv
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
    supported = {"diffusers", "diffusers_zimage"}
    backends = [
        str(name).strip().lower()
        for name in model_pack.backend_preference
        if str(name).strip()
    ]
    if not any(name in supported for name in backends):
        typer.echo(
            "Unsupported backend preference list "
            f"{model_pack.backend_preference!r} in bootstrap version. "
            f"Include one of: {sorted(supported)}."
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
            "Run RunMeFirst.bat (recommended) or repair with scripts/bootstrap_env.ps1."
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
            "runtime_profile": result.runtime_profile,
            "execution_mode": result.execution_mode,
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
        help="Comma-separated profile list to test (high, balanced, constrained).",
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
        help="Comma-separated profile list to test (high, balanced, constrained).",
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
        typer.echo("No profiles requested. Provide --profiles high,balanced,constrained")
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
        help="Runtime profile to use for x2 and SeedVR2 passes.",
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
