from __future__ import annotations

import math
import random
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.config.settings import AppSettings
from app.core.clarity import (
    CLARITY_FS_INTENSITY,
    CLARITY_FS_METHOD,
    CLARITY_FS_TYPE,
    apply_clarity_sharpen_core,
    final_unsharp,
    fs_sharpen,
)
from app.core.seedvr2 import (
    SEEDVR2_DIT_FILENAME,
    SEEDVR2_VAE_FILENAME,
    SeedVR2StillImageConfig,
    _runtime_script_path,
    _seedvr2_model_dir,
    upscale_with_seedvr2_direct_x2,
)
from app.core.worker import GenerationSession
from app.core.worker.types import GenerationRequest

DEBLUR_ENGINE_NAME = "img2img_seedvr2_clarity_downscale"
DEBLUR_PROMPT = (
    "highly detailed, sharp focus, 8k, intricate details, high-resolution, skin texture, realistic, clean"
)
DEBLUR_SIMILARITY = 0.75
DEBLUR_TEXTURE_COMPARE_VARIANTS = (
    "baseline_ai",
    "orig_chroma_restore",
    "orig_highpass_restore",
    "orig_chroma_plus_highpass",
)
DEBLUR_FIDELITY_CONTENT_TYPES = ("photo", "art")
DEBLUR_FIDELITY_PHOTO_VARIANTS = (
    "baseline_ai_x2",
    "multiband_detail_transfer",
    "multiband_plus_chroma",
    "multiband_chroma_edgeaware",
)
DEBLUR_FIDELITY_ART_VARIANTS = (
    "baseline_seed_x2",
    "art_multiband_detail_transfer",
    "art_detail_plus_chroma",
    "art_detail_chroma_edgeaware",
)
DEBLUR_FIDELITY_PHOTO_SIMILARITY = 0.90
DEFAULT_UPSCALE_ENGINE_NAME = "baseline_ai_x2_fs"
DEFAULT_CLARITY_ENGINE_NAME = "multiband_chroma_edgeaware_fs_unsharp_downscale"

_IMG2IMG_MAX_PIXELS = 1_500_000
_IMG2IMG_DIM_MULTIPLE = 32
_IMG2IMG_MIN_DIM = 64
_IMG2IMG_MIN_STRENGTH = 0.05
_IMG2IMG_MAX_STRENGTH = 0.95


@dataclass
class DeblurResult:
    image: Image.Image
    duration_ms: int
    source_width: int
    source_height: int
    working_width: int
    working_height: int
    engine_name: str
    prompt: str
    similarity: float
    seed: int
    device: str
    step_timings_ms: dict[str, int]

    def telemetry_dict(self) -> dict[str, Any]:
        return {
            "deblur_engine": self.engine_name,
            "deblur_prompt": self.prompt,
            "deblur_similarity": self.similarity,
            "working_width": self.working_width,
            "working_height": self.working_height,
            "deblur_duration_ms": self.duration_ms,
            **self.step_timings_ms,
        }


@dataclass
class DeblurTextureCompareRow:
    variant: str
    image: Image.Image
    width: int
    height: int
    total_ms: int
    output_path: str = ""
    hf_luma_energy_ratio_vs_original: float = 0.0
    lab_chroma_std_ratio_vs_original: float = 0.0
    mean_abs_luma_delta_vs_original: float = 0.0
    step_timings_ms: dict[str, int] | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "width": self.width,
            "height": self.height,
            "total_ms": self.total_ms,
            "output_path": self.output_path,
            "hf_luma_energy_ratio_vs_original": self.hf_luma_energy_ratio_vs_original,
            "lab_chroma_std_ratio_vs_original": self.lab_chroma_std_ratio_vs_original,
            "mean_abs_luma_delta_vs_original": self.mean_abs_luma_delta_vs_original,
            **(self.step_timings_ms or {}),
        }


@dataclass
class DeblurAiFrontHalfResult:
    final_image: Image.Image
    source_width: int
    source_height: int
    working_width: int
    working_height: int
    seed: int
    similarity: float
    device: str
    step_timings_ms: dict[str, int]


@dataclass
class DeblurFidelityCoreResult:
    x2_image: Image.Image
    source_x2_image: Image.Image
    source_width: int
    source_height: int
    working_width: int
    working_height: int
    seed: int
    similarity: float | None
    device: str
    content_type: str
    step_timings_ms: dict[str, int]


@dataclass
class DefaultUpscaleResult:
    image: Image.Image
    duration_ms: int
    source_width: int
    source_height: int
    working_width: int
    working_height: int
    seed: int
    device: str
    engine_name: str
    step_timings_ms: dict[str, int]

    def telemetry_dict(self) -> dict[str, Any]:
        return {
            "upscale_engine": self.engine_name,
            "working_width": self.working_width,
            "working_height": self.working_height,
            "upscale_duration_ms": self.duration_ms,
            **self.step_timings_ms,
        }


@dataclass
class DefaultClarityResult:
    image: Image.Image
    duration_ms: int
    source_width: int
    source_height: int
    working_width: int
    working_height: int
    seed: int
    device: str
    engine_name: str
    variant_label: str
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


@dataclass
class DeblurFidelityCompareRow:
    variant: str
    content_type: str
    image: Image.Image
    width: int
    height: int
    total_ms: int
    output_path: str = ""
    hf_luma_energy_ratio_vs_source_x2: float = 0.0
    lab_chroma_std_ratio_vs_source_x2: float = 0.0
    mean_abs_luma_delta_vs_source_x2: float = 0.0
    gradient_energy_ratio_vs_source_x2: float = 0.0
    step_timings_ms: dict[str, int] | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "content_type": self.content_type,
            "width": self.width,
            "height": self.height,
            "total_ms": self.total_ms,
            "output_path": self.output_path,
            "hf_luma_energy_ratio_vs_source_x2": self.hf_luma_energy_ratio_vs_source_x2,
            "lab_chroma_std_ratio_vs_source_x2": self.lab_chroma_std_ratio_vs_source_x2,
            "mean_abs_luma_delta_vs_source_x2": self.mean_abs_luma_delta_vs_source_x2,
            "gradient_energy_ratio_vs_source_x2": self.gradient_energy_ratio_vs_source_x2,
            **(self.step_timings_ms or {}),
        }


def _measure_ms(started: float) -> int:
    return int((perf_counter() - started) * 1000)


def _rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _lab_float_channels(image: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(_rgb_array(image), cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab[..., 0], lab[..., 1], lab[..., 2]


def _lab_to_rgb_image(l_channel: np.ndarray, a_channel: np.ndarray, b_channel: np.ndarray) -> Image.Image:
    lab = np.stack(
        [
            np.clip(l_channel, 0.0, 255.0),
            np.clip(a_channel, 0.0, 255.0),
            np.clip(b_channel, 0.0, 255.0),
        ],
        axis=-1,
    ).astype(np.uint8)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb, mode="RGB")


def _validate_content_type(content_type: str) -> str:
    normalized = str(content_type).strip().lower() or "photo"
    if normalized not in DEBLUR_FIDELITY_CONTENT_TYPES:
        raise ValueError(
            f"Unsupported content type: {content_type}. Choose one of: {', '.join(DEBLUR_FIDELITY_CONTENT_TYPES)}."
        )
    return normalized


def _resize_source_to_x2(source_image: Image.Image) -> Image.Image:
    source_rgb = source_image.convert("RGB")
    return source_rgb.resize(
        (max(1, source_rgb.width * 2), max(1, source_rgb.height * 2)),
        Image.Resampling.LANCZOS,
    )


def _gradient_energy(luma: np.ndarray) -> float:
    grad_x = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt((grad_x * grad_x) + (grad_y * grad_y))))


def restore_original_chroma(
    original_image: Image.Image,
    ai_image: Image.Image,
    *,
    original_weight: float = 0.65,
    ai_weight: float = 0.35,
) -> Image.Image:
    ai_l, ai_a, ai_b = _lab_float_channels(ai_image)
    _orig_l, orig_a, orig_b = _lab_float_channels(original_image)
    blended_a = (orig_a * float(original_weight)) + (ai_a * float(ai_weight))
    blended_b = (orig_b * float(original_weight)) + (ai_b * float(ai_weight))
    return _lab_to_rgb_image(ai_l, blended_a, blended_b)


def restore_original_luma_highpass(
    original_image: Image.Image,
    ai_image: Image.Image,
    *,
    sigma: float = 1.0,
    amount: float = 0.45,
) -> Image.Image:
    orig_l, _orig_a, _orig_b = _lab_float_channels(original_image)
    ai_l, ai_a, ai_b = _lab_float_channels(ai_image)
    blurred = cv2.GaussianBlur(orig_l, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    highpass = orig_l - blurred
    restored_l = ai_l + (highpass * float(amount))
    return _lab_to_rgb_image(restored_l, ai_a, ai_b)


def restore_original_chroma_x2(
    source_x2_image: Image.Image,
    candidate_image: Image.Image,
    *,
    source_weight: float,
    candidate_weight: float,
) -> Image.Image:
    return restore_original_chroma(
        source_x2_image,
        candidate_image,
        original_weight=float(source_weight),
        ai_weight=float(candidate_weight),
    )


def transfer_multiband_detail(
    source_x2_image: Image.Image,
    candidate_image: Image.Image,
    *,
    mid_amount: float,
    high_amount: float,
) -> Image.Image:
    source = np.asarray(source_x2_image.convert("RGB"), dtype=np.float32) / 255.0
    candidate = np.asarray(candidate_image.convert("RGB"), dtype=np.float32) / 255.0

    source_blur_mid = cv2.GaussianBlur(source, (0, 0), sigmaX=1.0, sigmaY=1.0)
    source_blur_low = cv2.GaussianBlur(source, (0, 0), sigmaX=2.5, sigmaY=2.5)
    candidate_blur_mid = cv2.GaussianBlur(candidate, (0, 0), sigmaX=1.0, sigmaY=1.0)
    candidate_blur_low = cv2.GaussianBlur(candidate, (0, 0), sigmaX=2.5, sigmaY=2.5)

    source_high = source - source_blur_mid
    source_mid = source_blur_mid - source_blur_low
    candidate_high = candidate - candidate_blur_mid
    candidate_mid = candidate_blur_mid - candidate_blur_low

    restored = (
        candidate_blur_low
        + ((1.0 - float(mid_amount)) * candidate_mid)
        + (float(mid_amount) * source_mid)
        + ((1.0 - float(high_amount)) * candidate_high)
        + (float(high_amount) * source_high)
    )
    restored = np.clip(restored, 0.0, 1.0)
    return Image.fromarray((restored * 255.0).round().astype(np.uint8), mode="RGB")


def apply_edge_aware_sharpen(
    candidate_image: Image.Image,
    *,
    content_type: str,
) -> Image.Image:
    normalized_type = _validate_content_type(content_type)
    image_rgb = candidate_image.convert("RGB")
    image_array = np.asarray(image_rgb, dtype=np.float32) / 255.0
    gray = cv2.cvtColor((image_array * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt((grad_x * grad_x) + (grad_y * grad_y))

    if normalized_type == "photo":
        threshold_quantile = 0.75
        amount = 0.35
    else:
        threshold_quantile = 0.65
        amount = 0.50

    threshold = float(np.quantile(grad, threshold_quantile))
    grad_max = float(max(grad.max(), threshold + 1e-6))
    mask = np.clip((grad - threshold) / (grad_max - threshold), 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=1.0, sigmaY=1.0)
    mask = np.repeat(mask[..., None], 3, axis=2)

    blurred = cv2.GaussianBlur(image_array, (0, 0), sigmaX=0.8, sigmaY=0.8)
    sharpened = np.clip(image_array + ((image_array - blurred) * float(amount)), 0.0, 1.0)
    blended = np.clip((image_array * (1.0 - mask)) + (sharpened * mask), 0.0, 1.0)
    return Image.fromarray((blended * 255.0).round().astype(np.uint8), mode="RGB")


def compute_texture_metrics(
    original_image: Image.Image,
    candidate_image: Image.Image,
    *,
    sigma: float = 1.0,
) -> dict[str, float]:
    orig_l, orig_a, orig_b = _lab_float_channels(original_image)
    cand_l, cand_a, cand_b = _lab_float_channels(candidate_image)

    orig_blur = cv2.GaussianBlur(orig_l, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    cand_blur = cv2.GaussianBlur(cand_l, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    orig_hf = orig_l - orig_blur
    cand_hf = cand_l - cand_blur
    orig_hf_energy = float(np.mean(np.abs(orig_hf)))
    cand_hf_energy = float(np.mean(np.abs(cand_hf)))

    orig_chroma_std = float(np.std(np.sqrt(((orig_a - 128.0) ** 2) + ((orig_b - 128.0) ** 2))))
    cand_chroma_std = float(np.std(np.sqrt(((cand_a - 128.0) ** 2) + ((cand_b - 128.0) ** 2))))
    mean_abs_luma_delta = float(np.mean(np.abs(cand_l - orig_l)))

    return {
        "hf_luma_energy_ratio_vs_original": 0.0 if orig_hf_energy <= 1e-6 else cand_hf_energy / orig_hf_energy,
        "lab_chroma_std_ratio_vs_original": 0.0 if orig_chroma_std <= 1e-6 else cand_chroma_std / orig_chroma_std,
        "mean_abs_luma_delta_vs_original": mean_abs_luma_delta,
    }


def compute_fidelity_metrics(
    source_x2_image: Image.Image,
    candidate_image: Image.Image,
    *,
    sigma: float = 1.0,
) -> dict[str, float]:
    source_l, source_a, source_b = _lab_float_channels(source_x2_image)
    candidate_l, candidate_a, candidate_b = _lab_float_channels(candidate_image)

    source_blur = cv2.GaussianBlur(source_l, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    candidate_blur = cv2.GaussianBlur(candidate_l, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    source_hf = source_l - source_blur
    candidate_hf = candidate_l - candidate_blur
    source_hf_energy = float(np.mean(np.abs(source_hf)))
    candidate_hf_energy = float(np.mean(np.abs(candidate_hf)))

    source_chroma_std = float(np.std(np.sqrt(((source_a - 128.0) ** 2) + ((source_b - 128.0) ** 2))))
    candidate_chroma_std = float(np.std(np.sqrt(((candidate_a - 128.0) ** 2) + ((candidate_b - 128.0) ** 2))))
    mean_abs_luma_delta = float(np.mean(np.abs(candidate_l - source_l)))

    return {
        "hf_luma_energy_ratio_vs_source_x2": 0.0
        if source_hf_energy <= 1e-6
        else candidate_hf_energy / source_hf_energy,
        "lab_chroma_std_ratio_vs_source_x2": 0.0
        if source_chroma_std <= 1e-6
        else candidate_chroma_std / source_chroma_std,
        "mean_abs_luma_delta_vs_source_x2": mean_abs_luma_delta,
        "gradient_energy_ratio_vs_source_x2": 0.0
        if _gradient_energy(source_l) <= 1e-6
        else _gradient_energy(candidate_l) / _gradient_energy(source_l),
    }


def run_deblur_texture_variant(
    *,
    original_image: Image.Image,
    ai_final_image: Image.Image,
    variant: str,
) -> Image.Image:
    normalized_variant = str(variant).strip().lower()
    if normalized_variant == "baseline_ai":
        return ai_final_image.copy()
    if normalized_variant == "orig_chroma_restore":
        return restore_original_chroma(original_image, ai_final_image)
    if normalized_variant == "orig_highpass_restore":
        return restore_original_luma_highpass(original_image, ai_final_image)
    if normalized_variant == "orig_chroma_plus_highpass":
        return restore_original_chroma(
            original_image,
            restore_original_luma_highpass(original_image, ai_final_image),
        )
    raise ValueError(
        "Unsupported deblur texture variant: "
        f"{variant}. Choose one of: {', '.join(DEBLUR_TEXTURE_COMPARE_VARIANTS)}."
    )


def run_fidelity_upscale_core(
    *,
    image: Image.Image,
    settings: AppSettings,
    session: GenerationSession,
    profile_name: str,
    content_type: str = "photo",
    seed: int | None = None,
    scheduler_mode: str | None = None,
) -> DeblurFidelityCoreResult:
    ensure_deblur_prerequisites(settings)
    normalized_type = _validate_content_type(content_type)
    source_image = image.convert("RGB")
    source_x2_image = _resize_source_to_x2(source_image)
    effective_seed = int(seed) if seed is not None else random.randint(1, 2_147_483_647)

    seed_result = upscale_with_seedvr2_direct_x2(
        image=source_image,
        settings=settings,
        runtime_profile=profile_name,
        seed=effective_seed,
        still_image_config=SeedVR2StillImageConfig(
            input_noise_scale=0.0,
            latent_noise_scale=0.0,
            color_correction="lab",
        ),
    )
    seed_ms = int(getattr(seed_result, "duration_ms", 0) or getattr(seed_result, "total_ms", 0) or 0)
    working_image = seed_result.image.convert("RGB")
    effective_similarity: float | None = None
    img2img_ms = 0

    if normalized_type == "photo":
        effective_similarity = DEBLUR_FIDELITY_PHOTO_SIMILARITY
        img2img_result = session.refine_image(
            working_image,
            GenerationRequest(
                prompt=DEBLUR_PROMPT,
                width=working_image.width,
                height=working_image.height,
                seed=effective_seed,
                scheduler_mode=scheduler_mode,
                enhance_prompt=False,
                procedural_creativity=0,
                refine_strength=similarity_to_refine_strength(effective_similarity),
            ),
        )
        working_image = img2img_result.image.convert("RGB")
        img2img_ms = int(getattr(img2img_result, "duration_ms", 0) or 0)

    return DeblurFidelityCoreResult(
        x2_image=working_image,
        source_x2_image=source_x2_image,
        source_width=source_image.width,
        source_height=source_image.height,
        working_width=working_image.width,
        working_height=working_image.height,
        seed=effective_seed,
        similarity=effective_similarity,
        device=str(getattr(seed_result, "device", "unknown")),
        content_type=normalized_type,
        step_timings_ms={
            "fidelity_seed_ms": seed_ms,
            "fidelity_img2img_ms": img2img_ms,
        },
    )


def run_fidelity_compare_variant(
    *,
    source_x2_image: Image.Image,
    baseline_x2_image: Image.Image,
    content_type: str,
    variant: str,
) -> Image.Image:
    normalized_type = _validate_content_type(content_type)
    normalized_variant = str(variant).strip().lower()

    if normalized_type == "photo":
        if normalized_variant == "baseline_ai_x2":
            return baseline_x2_image.copy()
        if normalized_variant == "multiband_detail_transfer":
            return transfer_multiband_detail(
                source_x2_image,
                baseline_x2_image,
                mid_amount=0.45,
                high_amount=0.65,
            )
        if normalized_variant == "multiband_plus_chroma":
            return restore_original_chroma_x2(
                source_x2_image,
                transfer_multiband_detail(
                    source_x2_image,
                    baseline_x2_image,
                    mid_amount=0.45,
                    high_amount=0.65,
                ),
                source_weight=0.55,
                candidate_weight=0.45,
            )
        if normalized_variant == "multiband_chroma_edgeaware":
            return apply_edge_aware_sharpen(
                restore_original_chroma_x2(
                    source_x2_image,
                    transfer_multiband_detail(
                        source_x2_image,
                        baseline_x2_image,
                        mid_amount=0.45,
                        high_amount=0.65,
                    ),
                    source_weight=0.55,
                    candidate_weight=0.45,
                ),
                content_type="photo",
            )
        raise ValueError(
            f"Unsupported photo fidelity variant: {variant}. Choose one of: {', '.join(DEBLUR_FIDELITY_PHOTO_VARIANTS)}."
        )

    if normalized_variant == "baseline_seed_x2":
        return baseline_x2_image.copy()
    if normalized_variant == "art_multiband_detail_transfer":
        return transfer_multiband_detail(
            source_x2_image,
            baseline_x2_image,
            mid_amount=0.60,
            high_amount=0.80,
        )
    if normalized_variant == "art_detail_plus_chroma":
        return restore_original_chroma_x2(
            source_x2_image,
            transfer_multiband_detail(
                source_x2_image,
                baseline_x2_image,
                mid_amount=0.60,
                high_amount=0.80,
            ),
            source_weight=0.70,
            candidate_weight=0.30,
        )
    if normalized_variant == "art_detail_chroma_edgeaware":
        return apply_edge_aware_sharpen(
            restore_original_chroma_x2(
                source_x2_image,
                transfer_multiband_detail(
                    source_x2_image,
                    baseline_x2_image,
                    mid_amount=0.60,
                    high_amount=0.80,
                ),
                source_weight=0.70,
                candidate_weight=0.30,
            ),
            content_type="art",
        )
    raise ValueError(
        f"Unsupported art fidelity variant: {variant}. Choose one of: {', '.join(DEBLUR_FIDELITY_ART_VARIANTS)}."
    )


def run_default_upscale_pipeline(
    *,
    image: Image.Image,
    settings: AppSettings,
    session: GenerationSession,
    profile_name: str,
    seed: int | None = None,
    scheduler_mode: str | None = None,
) -> DefaultUpscaleResult:
    total_started = perf_counter()
    core_result = run_fidelity_upscale_core(
        image=image,
        settings=settings,
        session=session,
        profile_name=profile_name,
        content_type="photo",
        seed=seed,
        scheduler_mode=scheduler_mode,
    )
    fs_started = perf_counter()
    final_image = fs_sharpen(
        core_result.x2_image,
        method=CLARITY_FS_METHOD,
        blur_type=CLARITY_FS_TYPE,
        intensity=CLARITY_FS_INTENSITY,
    )
    fs_ms = _measure_ms(fs_started)
    return DefaultUpscaleResult(
        image=final_image,
        duration_ms=_measure_ms(total_started),
        source_width=core_result.source_width,
        source_height=core_result.source_height,
        working_width=core_result.working_width,
        working_height=core_result.working_height,
        seed=core_result.seed,
        device=core_result.device,
        engine_name=DEFAULT_UPSCALE_ENGINE_NAME,
        step_timings_ms={
            **core_result.step_timings_ms,
            "upscale_fs_ms": fs_ms,
        },
    )


def run_default_clarity_pipeline(
    *,
    image: Image.Image,
    settings: AppSettings,
    session: GenerationSession,
    profile_name: str,
    seed: int | None = None,
    scheduler_mode: str | None = None,
) -> DefaultClarityResult:
    total_started = perf_counter()
    core_result = run_fidelity_upscale_core(
        image=image,
        settings=settings,
        session=session,
        profile_name=profile_name,
        content_type="photo",
        seed=seed,
        scheduler_mode=scheduler_mode,
    )

    multiband_started = perf_counter()
    multiband_image = transfer_multiband_detail(
        core_result.source_x2_image,
        core_result.x2_image,
        mid_amount=0.45,
        high_amount=0.65,
    )
    multiband_ms = _measure_ms(multiband_started)

    chroma_started = perf_counter()
    chroma_image = restore_original_chroma_x2(
        core_result.source_x2_image,
        multiband_image,
        source_weight=0.55,
        candidate_weight=0.45,
    )
    chroma_ms = _measure_ms(chroma_started)

    edgeaware_started = perf_counter()
    edgeaware_image = apply_edge_aware_sharpen(chroma_image, content_type="photo")
    edgeaware_ms = _measure_ms(edgeaware_started)

    fs_started = perf_counter()
    fs_image = fs_sharpen(
        edgeaware_image,
        method=CLARITY_FS_METHOD,
        blur_type=CLARITY_FS_TYPE,
        intensity=CLARITY_FS_INTENSITY,
    )
    fs_ms = _measure_ms(fs_started)

    unsharp_started = perf_counter()
    unsharp_image = final_unsharp(fs_image)
    unsharp_ms = _measure_ms(unsharp_started)

    downscale_started = perf_counter()
    final_image = unsharp_image.resize(
        (core_result.source_width, core_result.source_height),
        Image.Resampling.LANCZOS,
    )
    downscale_ms = _measure_ms(downscale_started)

    return DefaultClarityResult(
        image=final_image,
        duration_ms=_measure_ms(total_started),
        source_width=core_result.source_width,
        source_height=core_result.source_height,
        working_width=core_result.working_width,
        working_height=core_result.working_height,
        seed=core_result.seed,
        device=core_result.device,
        engine_name=DEFAULT_CLARITY_ENGINE_NAME,
        variant_label="multiband_chroma_edgeaware_fs_unsharp_shrink",
        step_timings_ms={
            **core_result.step_timings_ms,
            "clarity_multiband_ms": multiband_ms,
            "clarity_chroma_ms": chroma_ms,
            "clarity_edgeaware_ms": edgeaware_ms,
            "clarity_fs_ms": fs_ms,
            "clarity_pre_downscale_unsharp_ms": unsharp_ms,
            "clarity_downscale_ms": downscale_ms,
            "clarity_post_downscale_unsharp_ms": 0,
        },
    )


def ensure_deblur_prerequisites(settings: AppSettings) -> None:
    runtime_script = _runtime_script_path(settings)
    if not runtime_script.exists():
        raise RuntimeError(f"SeedVR2 runtime script not found: {runtime_script}")

    model_dir = _seedvr2_model_dir(settings)
    dit_path = model_dir / SEEDVR2_DIT_FILENAME
    vae_path = model_dir / SEEDVR2_VAE_FILENAME
    if not dit_path.exists() or not vae_path.exists():
        raise RuntimeError(
            "SeedVR2 model files are missing. "
            f"Expected dit='{dit_path.name}' and vae='{vae_path.name}' under {model_dir}."
        )


def normalize_img2img_similarity(similarity: float | int | str | None) -> float:
    if similarity is None:
        return DEBLUR_SIMILARITY
    value = float(similarity)
    if value < 0.0 or value > 1.0:
        raise ValueError("similarity must be between 0.0 and 1.0.")
    return value


def similarity_to_refine_strength(similarity: float | int | str | None) -> float:
    normalized = normalize_img2img_similarity(similarity)
    strength = 1.0 - normalized
    return max(_IMG2IMG_MIN_STRENGTH, min(_IMG2IMG_MAX_STRENGTH, strength))


def _normalized_img2img_dimensions(width: int, height: int) -> tuple[int, int]:
    safe_width = max(1, int(width))
    safe_height = max(1, int(height))
    pixel_count = safe_width * safe_height
    scale = 1.0
    if pixel_count > _IMG2IMG_MAX_PIXELS:
        scale = math.sqrt(_IMG2IMG_MAX_PIXELS / float(pixel_count))
    scaled_width = max(1, int(round(safe_width * scale)))
    scaled_height = max(1, int(round(safe_height * scale)))

    def _snap(value: int) -> int:
        if value <= _IMG2IMG_MIN_DIM:
            return _IMG2IMG_MIN_DIM
        snapped = value - (value % _IMG2IMG_DIM_MULTIPLE)
        if snapped < _IMG2IMG_MIN_DIM:
            return _IMG2IMG_MIN_DIM
        return snapped

    scaled_width = _snap(scaled_width)
    scaled_height = _snap(scaled_height)

    while scaled_width * scaled_height > _IMG2IMG_MAX_PIXELS:
        if scaled_width >= scaled_height and scaled_width > _IMG2IMG_MIN_DIM:
            scaled_width = _snap(max(_IMG2IMG_MIN_DIM, scaled_width - _IMG2IMG_DIM_MULTIPLE))
            continue
        if scaled_height > _IMG2IMG_MIN_DIM:
            scaled_height = _snap(max(_IMG2IMG_MIN_DIM, scaled_height - _IMG2IMG_DIM_MULTIPLE))
            continue
        break
    return scaled_width, scaled_height


def normalize_img2img_reference_image(image: Image.Image) -> tuple[Image.Image, dict[str, int]]:
    if not isinstance(image, Image.Image):
        raise ValueError("reference image must be a PIL.Image.Image instance.")
    source_width, source_height = image.size
    target_width, target_height = _normalized_img2img_dimensions(source_width, source_height)
    normalized = image.convert("RGB")
    if normalized.size != (target_width, target_height):
        normalized = normalized.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return normalized, {
        "source_width": int(source_width),
        "source_height": int(source_height),
        "normalized_width": int(target_width),
        "normalized_height": int(target_height),
    }


def run_deblur_ai_front_half(
    *,
    image: Image.Image,
    settings: AppSettings,
    session: GenerationSession,
    profile_name: str,
    seed: int | None = None,
    similarity: float | int | str | None = None,
    scheduler_mode: str | None = None,
) -> DeblurAiFrontHalfResult:
    ensure_deblur_prerequisites(settings)
    source_image = image.convert("RGB")
    source_width, source_height = source_image.size
    effective_seed = int(seed) if seed is not None else random.randint(1, 2_147_483_647)
    effective_similarity = normalize_img2img_similarity(similarity)
    normalized_image, _image_info = normalize_img2img_reference_image(source_image)

    img2img_result = session.refine_image(
        normalized_image,
        GenerationRequest(
            prompt=DEBLUR_PROMPT,
            width=normalized_image.width,
            height=normalized_image.height,
            seed=effective_seed,
            scheduler_mode=scheduler_mode,
            enhance_prompt=False,
            procedural_creativity=0,
            refine_strength=similarity_to_refine_strength(effective_similarity),
        ),
    )
    img2img_ms = int(getattr(img2img_result, "duration_ms", 0) or 0)

    seed_result = upscale_with_seedvr2_direct_x2(
        image=img2img_result.image,
        settings=settings,
        runtime_profile=profile_name,
        seed=effective_seed,
        still_image_config=SeedVR2StillImageConfig(
            input_noise_scale=0.0,
            latent_noise_scale=0.0,
            color_correction="lab",
        ),
    )
    seed_ms = int(getattr(seed_result, "duration_ms", 0) or getattr(seed_result, "total_ms", 0) or 0)

    clarity_started = perf_counter()
    sharpened_image, clarity_telemetry = apply_clarity_sharpen_core(seed_result.image)
    clarity_ms = _measure_ms(clarity_started)

    downscale_started = perf_counter()
    final_image = sharpened_image.resize((source_width, source_height), Image.Resampling.LANCZOS)
    downscale_ms = _measure_ms(downscale_started)

    return DeblurAiFrontHalfResult(
        final_image=final_image,
        source_width=source_width,
        source_height=source_height,
        working_width=sharpened_image.width,
        working_height=sharpened_image.height,
        seed=effective_seed,
        similarity=effective_similarity,
        device=str(getattr(seed_result, "device", getattr(img2img_result, "device", "unknown"))),
        step_timings_ms={
            "deblur_img2img_ms": img2img_ms,
            "deblur_seed_ms": seed_ms,
            "deblur_clarity_ms": clarity_ms,
            "deblur_downscale_ms": downscale_ms,
            **clarity_telemetry,
        },
    )


def run_deblur_pipeline(
    *,
    image: Image.Image,
    settings: AppSettings,
    session: GenerationSession,
    profile_name: str,
    seed: int | None = None,
    similarity: float | int | str | None = None,
    scheduler_mode: str | None = None,
) -> DeblurResult:
    total_started = perf_counter()
    ai_result = run_deblur_ai_front_half(
        image=image,
        settings=settings,
        session=session,
        profile_name=profile_name,
        seed=seed,
        similarity=similarity,
        scheduler_mode=scheduler_mode,
    )

    return DeblurResult(
        image=ai_result.final_image,
        duration_ms=_measure_ms(total_started),
        source_width=ai_result.source_width,
        source_height=ai_result.source_height,
        working_width=ai_result.working_width,
        working_height=ai_result.working_height,
        engine_name=DEBLUR_ENGINE_NAME,
        prompt=DEBLUR_PROMPT,
        similarity=ai_result.similarity,
        seed=ai_result.seed,
        device=ai_result.device,
        step_timings_ms=ai_result.step_timings_ms,
    )
