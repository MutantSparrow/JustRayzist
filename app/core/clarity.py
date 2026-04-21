from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from PIL import Image
from PIL import ImageFilter

CLARITY_ENGINE_NAME = "fs_unsharp_downscale"
CLARITY_FS_METHOD = "linear"
CLARITY_FS_TYPE = "gaussian"
CLARITY_FS_INTENSITY = 8
CLARITY_FINAL_UNSHARP_RADIUS = 1.1
CLARITY_FINAL_UNSHARP_PERCENT = 110
CLARITY_FINAL_UNSHARP_THRESHOLD = 2
CLARITY_UNSHARP_STAGE = "pre_downscale"


@dataclass(frozen=True)
class ClarityVariantConfig:
    label: str
    pre_downscale_unsharp: bool = True
    post_downscale_unsharp: bool = False


CLARITY_BENCHMARK_VARIANTS: dict[str, ClarityVariantConfig] = {
    "current": ClarityVariantConfig(
        label="current",
        pre_downscale_unsharp=True,
        post_downscale_unsharp=False,
    ),
    "fs_downsize_final_unsharp": ClarityVariantConfig(
        label="fs_downsize_final_unsharp",
        pre_downscale_unsharp=False,
        post_downscale_unsharp=True,
    ),
    "fs_only": ClarityVariantConfig(
        label="fs_only",
        pre_downscale_unsharp=False,
        post_downscale_unsharp=False,
    ),
}


def resolve_clarity_variant(variant: str | ClarityVariantConfig | None = None) -> ClarityVariantConfig:
    if isinstance(variant, ClarityVariantConfig):
        return variant
    key = str(variant or "current").strip().lower() or "current"
    if key not in CLARITY_BENCHMARK_VARIANTS:
        raise ValueError(
            f"Unsupported clarity variant: {variant}. Choose one of: {', '.join(sorted(CLARITY_BENCHMARK_VARIANTS))}."
        )
    return CLARITY_BENCHMARK_VARIANTS[key]


@dataclass
class ClarityResult:
    image: Image.Image
    duration_ms: int
    source_width: int
    source_height: int
    working_width: int
    working_height: int
    engine_name: str
    variant_label: str
    device: str
    step_timings_ms: dict[str, int]

    def telemetry_dict(self) -> dict[str, Any]:
        return {
            "clarity_engine": self.engine_name,
            "clarity_variant": self.variant_label,
            "working_width": self.working_width,
            "working_height": self.working_height,
            "clarity_duration_ms": self.duration_ms,
            **self.step_timings_ms,
        }


def _measure_ms(started: float) -> int:
    return int((perf_counter() - started) * 1000)


def ensure_clarity_runtime_dependencies() -> None:
    return None


def _gaussian_low_pass(array: np.ndarray, intensity: int) -> np.ndarray:
    kernel_size = max(1, int(intensity) - 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size <= 1:
        return array.copy()
    return cv2.GaussianBlur(array, (kernel_size, kernel_size), 0)


def fs_sharpen(image: Image.Image, *, method: str, blur_type: str, intensity: int) -> Image.Image:
    normalized_method = str(method).strip().lower()
    normalized_type = str(blur_type).strip().lower()
    if normalized_method != "linear":
        raise ValueError(f"Unsupported clarity FS method: {method}")
    if normalized_type != "gaussian":
        raise ValueError(f"Unsupported clarity FS blur type: {blur_type}")
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    low_pass = _gaussian_low_pass(image_array, intensity)
    sharpened = np.clip((2.0 * image_array) - low_pass, 0.0, 1.0)
    return Image.fromarray((sharpened * 255.0).round().astype(np.uint8), mode="RGB")


def final_unsharp(image: Image.Image) -> Image.Image:
    return image.convert("RGB").filter(
        ImageFilter.UnsharpMask(
            radius=CLARITY_FINAL_UNSHARP_RADIUS,
            percent=CLARITY_FINAL_UNSHARP_PERCENT,
            threshold=CLARITY_FINAL_UNSHARP_THRESHOLD,
        )
    )


def apply_clarity_sharpen_core(image: Image.Image) -> tuple[Image.Image, dict[str, int]]:
    fs_started = perf_counter()
    working_image = fs_sharpen(
        image.convert("RGB"),
        method=CLARITY_FS_METHOD,
        blur_type=CLARITY_FS_TYPE,
        intensity=CLARITY_FS_INTENSITY,
    )
    fs_ms = _measure_ms(fs_started)

    pre_downscale_unsharp_started = perf_counter()
    working_image = final_unsharp(working_image)
    pre_downscale_unsharp_ms = _measure_ms(pre_downscale_unsharp_started)
    return working_image, {
        "clarity_fs_ms": fs_ms,
        "clarity_pre_downscale_unsharp_ms": pre_downscale_unsharp_ms,
    }


def run_clarity_pipeline(
    *,
    image: Image.Image,
    variant: str | ClarityVariantConfig | None = None,
) -> ClarityResult:
    selected_variant = resolve_clarity_variant(variant)
    total_started = perf_counter()
    source_image = image.convert("RGB")
    source_width, source_height = source_image.size
    working_width = max(64, source_width * 2)
    working_height = max(64, source_height * 2)

    resize_started = perf_counter()
    resized_image = source_image.resize((working_width, working_height), Image.Resampling.LANCZOS)
    resize_ms = _measure_ms(resize_started)

    working_image = resized_image
    fs_ms = 0
    pre_downscale_unsharp_ms = 0
    if selected_variant.pre_downscale_unsharp:
        working_image, sharpen_telemetry = apply_clarity_sharpen_core(resized_image)
        fs_ms = int(sharpen_telemetry["clarity_fs_ms"])
        pre_downscale_unsharp_ms = int(sharpen_telemetry["clarity_pre_downscale_unsharp_ms"])
    else:
        fs_started = perf_counter()
        working_image = fs_sharpen(
            resized_image,
            method=CLARITY_FS_METHOD,
            blur_type=CLARITY_FS_TYPE,
            intensity=CLARITY_FS_INTENSITY,
        )
        fs_ms = _measure_ms(fs_started)

    downscale_started = perf_counter()
    final_image = working_image.resize((source_width, source_height), Image.Resampling.LANCZOS)
    downscale_ms = _measure_ms(downscale_started)

    post_downscale_unsharp_ms = 0
    if selected_variant.post_downscale_unsharp:
        post_downscale_unsharp_started = perf_counter()
        final_image = final_unsharp(final_image)
        post_downscale_unsharp_ms = _measure_ms(post_downscale_unsharp_started)

    return ClarityResult(
        image=final_image,
        duration_ms=_measure_ms(total_started),
        source_width=source_width,
        source_height=source_height,
        working_width=working_width,
        working_height=working_height,
        engine_name=CLARITY_ENGINE_NAME,
        variant_label=selected_variant.label,
        device="cpu",
        step_timings_ms={
            "clarity_resize_ms": resize_ms,
            "clarity_fs_ms": fs_ms,
            "clarity_pre_downscale_unsharp_ms": pre_downscale_unsharp_ms,
            "clarity_downscale_ms": downscale_ms,
            "clarity_post_downscale_unsharp_ms": post_downscale_unsharp_ms,
        },
    )
