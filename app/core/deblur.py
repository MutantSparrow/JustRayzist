from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

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
    SEEDVR2_RUNTIME_PRESET_CURRENT_BASELINE,
    SEEDVR2_RUNTIME_PRESET_HIGHRES_AUTO,
    SEEDVR2_VAE_FILENAME,
    SeedVR2StillImageConfig,
    _runtime_script_path,
    _seedvr2_model_dir,
    upscale_with_seedvr2_direct_x2,
)
from app.core.tile_seams import soften_tile_seams, tile_seam_positions
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
DEFAULT_UPSCALE_PHOTO_SIMILARITY = 0.85
DEFAULT_UPSCALE_FS_INTENSITY = 4
DEFAULT_UPSCALE_ENGINE_NAME = "content_aware_ai_x2"
DEFAULT_CLARITY_ENGINE_NAME = "multiband_chroma_edgeaware_fs_unsharp_downscale"
UPSCALE_CONTENT_MODE_VALUES = ("auto", "photo", "art")
_UPSCALE_PROMPT_ART_PATTERNS: tuple[str, ...] = (
    r"\billustration\b",
    r"\billustrated\b",
    r"\bdrawing\b",
    r"\bsketch(?:ed)?\b",
    r"\bline(?:-| )art\b",
    r"\bdigital art\b",
    r"\bconcept art\b",
    r"\banime\b",
    r"\bmanga\b",
    r"\bcomic(?: book)?\b",
    r"\bcartoon\b",
    r"\bcel(?:-| )shad(?:e|ed|ing)\b",
    r"\binked\b",
    r"\bpanel\b",
    r"\bstoryboard\b",
    r"\bgraphic novel\b",
    r"\bwebtoon\b",
    r"\bchibi\b",
    r"\bkawaii\b",
    r"\bmoe\b",
    r"\bshonen\b",
    r"\bshojo\b",
    r"\bseinen\b",
    r"\bmanhwa\b",
    r"\bfan art\b",
)
_UPSCALE_PROMPT_PHOTO_PATTERNS: tuple[str, ...] = (
    r"\bphoto\b",
    r"\bphotograph(?:y|ic)?\b",
    r"\bphotorealistic\b",
    r"\brealistic\b",
    r"\bcinematic\b",
    r"\bportrait\b",
    r"\bselfie\b",
    r"\bsnapshot\b",
    r"\bpicture\b",
    r"\bcamera\b",
    r"\bdslr\b",
    r"\bmirrorless\b",
    r"\blens\b",
    r"\bbokeh\b",
    r"\bdepth of field\b",
    r"\bexposure\b",
    r"\baperture\b",
    r"\bshutter speed\b",
    r"\biso\b",
    r"\bfocal length\b",
    r"\bclose(?:-| )up\b",
    r"\bmacro\b",
    r"\bwide(?:-| )angle\b",
    r"\bstudio lighting\b",
    r"\bfilm grain\b",
)

UpscaleContentMode = Literal["auto", "photo", "art"]
ResolvedUpscaleContentMode = Literal["photo", "art"]
UpscaleContentDecisionSource = Literal["prompt", "image", "mixed", "forced"]

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
    seed_decode_tiled: bool = False
    seed_decode_tile_size: int | None = None
    seed_decode_tile_overlap: int | None = None


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
    telemetry_extra: dict[str, Any] | None = None

    def telemetry_dict(self) -> dict[str, Any]:
        return {
            "upscale_engine": self.engine_name,
            "working_width": self.working_width,
            "working_height": self.working_height,
            "upscale_duration_ms": self.duration_ms,
            **self.step_timings_ms,
            **(self.telemetry_extra or {}),
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
class UpscaleContentModeDecision:
    mode: ResolvedUpscaleContentMode
    source: UpscaleContentDecisionSource
    prompt_art_hits: int
    prompt_photo_hits: int
    image_art_signals: int


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


def _lab_float_image(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(_rgb_array(image), cv2.COLOR_RGB2LAB).astype(np.float32)


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


def _lab_image_to_rgb_image(lab_image: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(np.clip(lab_image, 0.0, 255.0).astype(np.uint8), cv2.COLOR_LAB2RGB)
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


def _gradient_magnitude(luma: np.ndarray) -> np.ndarray:
    grad_x = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt((grad_x * grad_x) + (grad_y * grad_y))


def _tile_cell_bounds(length: int, seam_positions: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    edges = [0, *[int(value) for value in seam_positions if 0 < int(value) < int(length)], int(length)]
    bounds: list[tuple[int, int]] = []
    for start, end in zip(edges, edges[1:]):
        if end > start:
            bounds.append((int(start), int(end)))
    return tuple(bounds)


def _raised_cosine_axis_weights(axis_length: int, bounds: tuple[tuple[int, int], ...]) -> np.ndarray:
    if not bounds:
        return np.zeros((0, max(0, int(axis_length))), dtype=np.float32)
    weights = np.zeros((len(bounds), int(axis_length)), dtype=np.float32)
    centers = [0.5 * float(start + end - 1) for start, end in bounds]
    for index in range(int(axis_length)):
        position = float(index)
        if position <= centers[0]:
            weights[0, index] = 1.0
            continue
        if position >= centers[-1]:
            weights[-1, index] = 1.0
            continue
        for left_index in range(len(centers) - 1):
            left_center = float(centers[left_index])
            right_center = float(centers[left_index + 1])
            if position > right_center:
                continue
            span = max(1e-6, right_center - left_center)
            t = np.clip((position - left_center) / span, 0.0, 1.0)
            blend = 0.5 - (0.5 * math.cos(math.pi * t))
            weights[left_index, index] = 1.0 - float(blend)
            weights[left_index + 1, index] = float(blend)
            break
    return weights


def _masked_mean(values: np.ndarray, mask: np.ndarray | None) -> float:
    if mask is None or not np.any(mask):
        return float(values.mean())
    return float(values[mask].mean())


def _validate_upscale_content_mode(content_mode: UpscaleContentMode) -> UpscaleContentMode:
    normalized = str(content_mode or "auto").strip().lower() or "auto"
    if normalized not in UPSCALE_CONTENT_MODE_VALUES:
        raise ValueError(
            f"Unsupported upscale content mode: {content_mode}. Choose one of: {', '.join(UPSCALE_CONTENT_MODE_VALUES)}."
        )
    return normalized  # type: ignore[return-value]


def _score_prompt_patterns(prompt_text: str | None, patterns: tuple[str, ...]) -> int:
    normalized = str(prompt_text or "").strip().lower()
    if not normalized:
        return 0
    return sum(1 for pattern in patterns if re.search(pattern, normalized))


def _resize_for_upscale_mode_detection(image: Image.Image, *, max_edge: int = 512) -> Image.Image:
    rgb = image.convert("RGB")
    longest_edge = max(rgb.width, rgb.height)
    if longest_edge <= int(max_edge):
        return rgb
    scale = float(max_edge) / float(longest_edge)
    return rgb.resize(
        (
            max(1, int(round(float(rgb.width) * scale))),
            max(1, int(round(float(rgb.height) * scale))),
        ),
        Image.Resampling.LANCZOS,
    )


def _palette_top32_share(image: Image.Image) -> float:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    quantized = (rgb // 8).astype(np.int32)
    flattened = ((quantized[..., 0] * 32 * 32) + (quantized[..., 1] * 32) + quantized[..., 2]).reshape(-1)
    counts = np.bincount(flattened, minlength=32 * 32 * 32)
    top_k = min(32, counts.shape[0])
    if top_k <= 0:
        return 0.0
    top_indices = np.argpartition(counts, -top_k)[-top_k:]
    return float(counts[top_indices].sum()) / float(max(1, flattened.size))


def _microtexture_energy(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2, sigmaY=1.2)
    return float(np.mean(np.abs(gray - blurred)))


def _line_share(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    grad = _gradient_magnitude(gray)
    return float(np.mean(grad >= 0.18))


def _detect_upscale_content_mode_from_image(image: Image.Image) -> tuple[ResolvedUpscaleContentMode, int]:
    resized = _resize_for_upscale_mode_detection(image)
    palette_top32_share = _palette_top32_share(resized)
    microtexture_energy = _microtexture_energy(resized)
    line_share = _line_share(resized)
    image_art_signals = sum(
        (
            palette_top32_share >= 0.72,
            microtexture_energy <= 0.028,
            line_share >= 0.06,
        )
    )
    return ("art" if image_art_signals >= 2 else "photo"), int(image_art_signals)


def detect_upscale_content_mode(
    image: Image.Image,
    prompt_text: str | None = None,
) -> UpscaleContentModeDecision:
    prompt_art_hits = _score_prompt_patterns(prompt_text, _UPSCALE_PROMPT_ART_PATTERNS)
    prompt_photo_hits = _score_prompt_patterns(prompt_text, _UPSCALE_PROMPT_PHOTO_PATTERNS)
    image_mode, image_art_signals = _detect_upscale_content_mode_from_image(image)

    if prompt_photo_hits > prompt_art_hits:
        return UpscaleContentModeDecision(
            mode="photo",
            source="prompt",
            prompt_art_hits=prompt_art_hits,
            prompt_photo_hits=prompt_photo_hits,
            image_art_signals=image_art_signals,
        )
    if prompt_art_hits > prompt_photo_hits:
        return UpscaleContentModeDecision(
            mode="art",
            source="prompt",
            prompt_art_hits=prompt_art_hits,
            prompt_photo_hits=prompt_photo_hits,
            image_art_signals=image_art_signals,
        )
    if prompt_photo_hits > 0 or prompt_art_hits > 0:
        return UpscaleContentModeDecision(
            mode="photo",
            source="mixed",
            prompt_art_hits=prompt_art_hits,
            prompt_photo_hits=prompt_photo_hits,
            image_art_signals=image_art_signals,
        )
    return UpscaleContentModeDecision(
        mode=image_mode,
        source="image",
        prompt_art_hits=prompt_art_hits,
        prompt_photo_hits=prompt_photo_hits,
        image_art_signals=image_art_signals,
    )


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


def repair_photo_upscale_tile_drift(
    source_x2_image: Image.Image,
    candidate_image: Image.Image,
    *,
    tile_size: int,
    tile_overlap: int,
) -> tuple[Image.Image, float]:
    source_rgb = source_x2_image.convert("RGB")
    candidate_rgb = candidate_image.convert("RGB")
    if source_rgb.size != candidate_rgb.size:
        source_rgb = source_rgb.resize(candidate_rgb.size, Image.Resampling.LANCZOS)

    width, height = candidate_rgb.size
    vertical_seams = tile_seam_positions(width, tile_size, tile_overlap)
    horizontal_seams = tile_seam_positions(height, tile_size, tile_overlap)
    if not vertical_seams and not horizontal_seams:
        return candidate_rgb.copy(), 0.0

    sigma = float(np.clip(float(tile_overlap) / 2.0, 16.0, 48.0))
    source_lab = _lab_float_image(source_rgb)
    candidate_lab = _lab_float_image(candidate_rgb)
    source_low = cv2.GaussianBlur(source_lab, (0, 0), sigmaX=sigma, sigmaY=sigma)
    candidate_low = cv2.GaussianBlur(candidate_lab, (0, 0), sigmaX=sigma, sigmaY=sigma)
    candidate_high = candidate_lab - candidate_low
    low_delta = candidate_low - source_low

    source_luma = source_lab[..., 0]
    source_grad = _gradient_magnitude(source_luma)
    grad_threshold = float(np.quantile(source_grad, 0.70))
    smooth_mask = source_grad <= grad_threshold
    if float(smooth_mask.mean()) < 0.02:
        smooth_mask = np.ones_like(smooth_mask, dtype=bool)

    global_delta = np.array(
        [_masked_mean(low_delta[..., channel_index], smooth_mask) for channel_index in range(3)],
        dtype=np.float32,
    )

    x_bounds = _tile_cell_bounds(width, vertical_seams)
    y_bounds = _tile_cell_bounds(height, horizontal_seams)
    x_weights = _raised_cosine_axis_weights(width, x_bounds)
    y_weights = _raised_cosine_axis_weights(height, y_bounds)
    cell_deltas = np.zeros((len(y_bounds), len(x_bounds), 3), dtype=np.float32)
    for row_index, (y0, y1) in enumerate(y_bounds):
        for col_index, (x0, x1) in enumerate(x_bounds):
            cell_mask = smooth_mask[y0:y1, x0:x1]
            cell_delta = low_delta[y0:y1, x0:x1, :]
            if float(cell_mask.mean()) >= 0.02 and np.any(cell_mask):
                cell_deltas[row_index, col_index, :] = np.array(
                    [
                        _masked_mean(cell_delta[..., channel_index], cell_mask)
                        for channel_index in range(3)
                    ],
                    dtype=np.float32,
                )
            else:
                cell_deltas[row_index, col_index, :] = global_delta

    correction = np.einsum("yh,xw,yxc->hwc", y_weights, x_weights, cell_deltas, optimize=True)
    corrected_low = candidate_low - correction
    repaired_lab = corrected_low + candidate_high
    repaired_image = _lab_image_to_rgb_image(repaired_lab)
    repaired_image = soften_tile_seams(
        repaired_image,
        vertical_seams=vertical_seams,
        horizontal_seams=horizontal_seams,
        band_radius=max(12, int(tile_overlap) // 6),
    )
    return repaired_image, sigma


def repair_art_upscale_tile_drift(
    source_x2_image: Image.Image,
    candidate_image: Image.Image,
    *,
    tile_size: int,
    tile_overlap: int,
) -> tuple[Image.Image, float]:
    source_rgb = source_x2_image.convert("RGB")
    candidate_rgb = candidate_image.convert("RGB")
    if source_rgb.size != candidate_rgb.size:
        source_rgb = source_rgb.resize(candidate_rgb.size, Image.Resampling.LANCZOS)

    width, height = candidate_rgb.size
    vertical_seams = tile_seam_positions(width, tile_size, tile_overlap)
    horizontal_seams = tile_seam_positions(height, tile_size, tile_overlap)
    if not vertical_seams and not horizontal_seams:
        return candidate_rgb.copy(), 0.0

    sigma = float(np.clip(float(tile_overlap) / 2.0, 16.0, 48.0))
    source_lab = _lab_float_image(source_rgb)
    candidate_lab = _lab_float_image(candidate_rgb)
    source_luma = source_lab[..., 0]
    candidate_luma = candidate_lab[..., 0]
    source_low = cv2.GaussianBlur(source_luma, (0, 0), sigmaX=sigma, sigmaY=sigma)
    candidate_low = cv2.GaussianBlur(candidate_luma, (0, 0), sigmaX=sigma, sigmaY=sigma)
    candidate_high = candidate_luma - candidate_low
    low_delta = candidate_low - source_low

    source_grad = _gradient_magnitude(source_luma)
    grad_threshold = float(np.quantile(source_grad, 0.70))
    smooth_mask = source_grad <= grad_threshold
    if float(smooth_mask.mean()) < 0.02:
        smooth_mask = np.ones_like(smooth_mask, dtype=bool)

    global_delta = _masked_mean(low_delta, smooth_mask)
    x_bounds = _tile_cell_bounds(width, vertical_seams)
    y_bounds = _tile_cell_bounds(height, horizontal_seams)
    x_weights = _raised_cosine_axis_weights(width, x_bounds)
    y_weights = _raised_cosine_axis_weights(height, y_bounds)
    cell_deltas = np.zeros((len(y_bounds), len(x_bounds)), dtype=np.float32)
    for row_index, (y0, y1) in enumerate(y_bounds):
        for col_index, (x0, x1) in enumerate(x_bounds):
            cell_mask = smooth_mask[y0:y1, x0:x1]
            cell_delta = low_delta[y0:y1, x0:x1]
            if float(cell_mask.mean()) >= 0.02 and np.any(cell_mask):
                cell_deltas[row_index, col_index] = _masked_mean(cell_delta, cell_mask)
            else:
                cell_deltas[row_index, col_index] = global_delta

    correction = np.einsum("yh,xw,yx->hw", y_weights, x_weights, cell_deltas, optimize=True)
    repaired_lab = candidate_lab.copy()
    repaired_lab[..., 0] = (candidate_low - correction) + candidate_high
    return _lab_image_to_rgb_image(repaired_lab), sigma


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
    photo_similarity_override: float | None = None,
    seed_runtime_preset: str = SEEDVR2_RUNTIME_PRESET_HIGHRES_AUTO,
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
        runtime_preset=seed_runtime_preset,
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
        effective_similarity = (
            float(photo_similarity_override)
            if photo_similarity_override is not None
            else DEBLUR_FIDELITY_PHOTO_SIMILARITY
        )
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
        seed_decode_tiled=bool(getattr(seed_result, "vae_decode_tiled", False)),
        seed_decode_tile_size=int(getattr(seed_result, "vae_decode_tile_size", 0) or 0) or None,
        seed_decode_tile_overlap=int(getattr(seed_result, "vae_decode_tile_overlap", 0) or 0) or None,
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
    content_mode: UpscaleContentMode = "auto",
    prompt_text: str | None = None,
) -> DefaultUpscaleResult:
    total_started = perf_counter()
    normalized_content_mode = _validate_upscale_content_mode(content_mode)
    if normalized_content_mode == "auto":
        content_decision = detect_upscale_content_mode(image, prompt_text=prompt_text)
    else:
        content_decision = UpscaleContentModeDecision(
            mode=normalized_content_mode,
            source="forced",
            prompt_art_hits=0,
            prompt_photo_hits=0,
            image_art_signals=0,
        )
    core_result = run_fidelity_upscale_core(
        image=image,
        settings=settings,
        session=session,
        profile_name=profile_name,
        content_type=content_decision.mode,
        seed=seed,
        scheduler_mode=scheduler_mode,
        photo_similarity_override=(
            DEFAULT_UPSCALE_PHOTO_SIMILARITY if content_decision.mode == "photo" else None
        ),
        seed_runtime_preset=SEEDVR2_RUNTIME_PRESET_CURRENT_BASELINE,
    )
    seam_repair_started = perf_counter()
    seam_repair_applied = bool(
        core_result.seed_decode_tiled
        and core_result.seed_decode_tile_size
        and core_result.seed_decode_tile_overlap
    )
    repaired_image = core_result.x2_image
    seam_repair_sigma = 0.0
    if seam_repair_applied:
        if content_decision.mode == "art":
            repaired_image, seam_repair_sigma = repair_art_upscale_tile_drift(
                core_result.source_x2_image,
                core_result.x2_image,
                tile_size=int(core_result.seed_decode_tile_size or 0),
                tile_overlap=int(core_result.seed_decode_tile_overlap or 0),
            )
        else:
            repaired_image, seam_repair_sigma = repair_photo_upscale_tile_drift(
                core_result.source_x2_image,
                core_result.x2_image,
                tile_size=int(core_result.seed_decode_tile_size or 0),
                tile_overlap=int(core_result.seed_decode_tile_overlap or 0),
            )
    seam_repair_ms = _measure_ms(seam_repair_started)
    art_multiband_ms = 0
    art_chroma_ms = 0
    fs_ms = 0
    pipeline_variant = "photo_default"
    if content_decision.mode == "art":
        pipeline_variant = "art_preserve"
        art_multiband_started = perf_counter()
        multiband_image = transfer_multiband_detail(
            core_result.source_x2_image,
            repaired_image,
            mid_amount=0.60,
            high_amount=0.80,
        )
        art_multiband_ms = _measure_ms(art_multiband_started)
        art_chroma_started = perf_counter()
        final_image = restore_original_chroma_x2(
            core_result.source_x2_image,
            multiband_image,
            source_weight=0.75,
            candidate_weight=0.25,
        )
        art_chroma_ms = _measure_ms(art_chroma_started)
    else:
        fs_started = perf_counter()
        final_image = fs_sharpen(
            repaired_image,
            method=CLARITY_FS_METHOD,
            blur_type=CLARITY_FS_TYPE,
            intensity=DEFAULT_UPSCALE_FS_INTENSITY,
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
            "upscale_seam_repair_ms": seam_repair_ms,
            "upscale_art_multiband_ms": art_multiband_ms,
            "upscale_art_chroma_ms": art_chroma_ms,
            "upscale_fs_ms": fs_ms,
        },
        telemetry_extra={
            "upscale_seam_repair_applied": seam_repair_applied,
            "upscale_seam_repair_mode": "source_guided_lowfreq" if seam_repair_applied else "none",
            "upscale_seam_repair_sigma": seam_repair_sigma,
            "upscale_seam_repair_decode_overlap": int(core_result.seed_decode_tile_overlap or 0),
            "upscale_auto_content_mode": content_decision.mode,
            "upscale_auto_content_source": content_decision.source,
            "upscale_auto_prompt_art_hits": content_decision.prompt_art_hits,
            "upscale_auto_prompt_photo_hits": content_decision.prompt_photo_hits,
            "upscale_auto_image_art_signals": content_decision.image_art_signals,
            "upscale_pipeline_variant": pipeline_variant,
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
