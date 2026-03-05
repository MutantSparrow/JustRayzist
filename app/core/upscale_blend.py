from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from app.config.settings import AppSettings
from app.core.memory import now_perf
from app.core.seedvr2 import SEEDVR2_DEFAULT_TIMEOUT_SECONDS, upscale_with_seedvr2
from app.core.upscale import upscale_image

DEFAULT_BLEND_ALPHA = 0.50
DEFAULT_UPSCALER_CHECKPOINT = "2x_RealESRGAN_x2plus.pth"
UPSCALE_ENGINE_NAME = "x2_seedvr2_blend"


def _resolve_blend_alpha(override: float | None = None) -> float:
    if override is not None:
        value = float(override)
    else:
        raw = os.getenv("JUSTRAYZIST_UPSCALE_BLEND_ALPHA", "").strip()
        value = DEFAULT_BLEND_ALPHA if not raw else float(raw)
    if value < 0.0 or value > 1.0:
        raise ValueError("Blend alpha must be within [0.0, 1.0].")
    return float(value)


@dataclass
class BlendUpscaleResult:
    image: Image.Image
    duration_ms: int
    output_width: int
    output_height: int
    device: str
    engine: str
    blend_alpha: float
    x2_duration_ms: int
    x2_infer_ms: int
    x2_tile_size: int
    x2_tile_overlap: int
    x2_precision: str
    x2_architecture: str
    seed_duration_ms: int
    seed_infer_ms: int
    seed_offload_mode: str
    seed_fallback_tier: int
    seed_vae_encode_tiled: bool
    seed_vae_decode_tiled: bool
    seed_attempt_count: int
    seed_attempts: list[dict[str, Any]]
    seed_policy_source: str
    seed_timeout_hit: bool
    blend_duration_ms: int

    def telemetry_dict(self) -> dict[str, Any]:
        return {
            "upscale_engine": self.engine,
            "upscale_blend_alpha": self.blend_alpha,
            "upscale_blend_alpha_percent": int(round(self.blend_alpha * 100.0)),
            "upscale_total_ms": self.duration_ms,
            "upscale_output_width": self.output_width,
            "upscale_output_height": self.output_height,
            "upscale_device": self.device,
            "upscale_x2_duration_ms": self.x2_duration_ms,
            "upscale_x2_infer_ms": self.x2_infer_ms,
            "upscale_x2_tile_size": self.x2_tile_size,
            "upscale_x2_tile_overlap": self.x2_tile_overlap,
            "upscale_x2_precision": self.x2_precision,
            "upscale_x2_architecture": self.x2_architecture,
            "upscale_seed_duration_ms": self.seed_duration_ms,
            "upscale_seed_infer_ms": self.seed_infer_ms,
            "upscale_seed_offload_mode": self.seed_offload_mode,
            "upscale_seed_fallback_tier": self.seed_fallback_tier,
            "upscale_seed_vae_encode_tiled": self.seed_vae_encode_tiled,
            "upscale_seed_vae_decode_tiled": self.seed_vae_decode_tiled,
            "upscale_seed_attempt_count": self.seed_attempt_count,
            "upscale_seed_attempts": self.seed_attempts,
            "upscale_seed_policy_source": self.seed_policy_source,
            "upscale_seed_timeout_hit": self.seed_timeout_hit,
            "upscale_blend_duration_ms": self.blend_duration_ms,
            "upscale_success": True,
        }


def upscale_with_x2_seed_blend(
    *,
    image: Image.Image,
    settings: AppSettings,
    runtime_profile: str,
    seed: int | None = None,
    blend_alpha: float | None = None,
    seed_timeout_seconds: int = SEEDVR2_DEFAULT_TIMEOUT_SECONDS,
    upscaler_checkpoint: Path | None = None,
) -> BlendUpscaleResult:
    if not isinstance(image, Image.Image):
        raise ValueError("image must be a PIL.Image.Image instance.")

    checkpoint = upscaler_checkpoint or (
        settings.paths.models_dir / "upscaler" / DEFAULT_UPSCALER_CHECKPOINT
    )
    if not checkpoint.exists():
        raise RuntimeError(
            "x2 upscaler checkpoint not found. Run .\\RunMeFirst.bat to fetch default assets."
        )

    effective_alpha = _resolve_blend_alpha(blend_alpha)
    source = image.convert("RGB")

    total_started = now_perf()

    x2_started = now_perf()
    x2_result = upscale_image(
        image=source,
        checkpoint_path=checkpoint,
        profile_name=runtime_profile,
    )
    x2_duration_ms = int((now_perf() - x2_started) * 1000)

    seed_started = now_perf()
    seed_result = upscale_with_seedvr2(
        image=source,
        settings=settings,
        runtime_profile=runtime_profile,
        seed=seed,
        timeout_seconds=seed_timeout_seconds,
        reuse_runner=True,
    )
    seed_duration_ms = int((now_perf() - seed_started) * 1000)

    if x2_result.image.size != seed_result.image.size:
        raise RuntimeError(
            "Upscale stage size mismatch. "
            f"x2={x2_result.image.size}, seedvr2={seed_result.image.size}"
        )

    blend_started = now_perf()
    blended_image = Image.blend(seed_result.image, x2_result.image, effective_alpha).convert("RGB")
    blend_duration_ms = int((now_perf() - blend_started) * 1000)
    duration_ms = int((now_perf() - total_started) * 1000)

    return BlendUpscaleResult(
        image=blended_image,
        duration_ms=duration_ms,
        output_width=int(blended_image.width),
        output_height=int(blended_image.height),
        device=seed_result.device,
        engine=UPSCALE_ENGINE_NAME,
        blend_alpha=effective_alpha,
        x2_duration_ms=x2_duration_ms,
        x2_infer_ms=int(x2_result.duration_ms),
        x2_tile_size=int(x2_result.tile_size),
        x2_tile_overlap=int(x2_result.tile_overlap),
        x2_precision=str(x2_result.precision),
        x2_architecture=str(x2_result.architecture),
        seed_duration_ms=seed_duration_ms,
        seed_infer_ms=int(seed_result.infer_ms),
        seed_offload_mode=str(seed_result.offload_mode),
        seed_fallback_tier=int(seed_result.fallback_tier),
        seed_vae_encode_tiled=bool(seed_result.vae_encode_tiled),
        seed_vae_decode_tiled=bool(seed_result.vae_decode_tiled),
        seed_attempt_count=int(seed_result.attempt_count),
        seed_attempts=list(seed_result.attempts),
        seed_policy_source=str(seed_result.policy_source),
        seed_timeout_hit=bool(seed_result.timeout_hit),
        blend_duration_ms=blend_duration_ms,
    )
