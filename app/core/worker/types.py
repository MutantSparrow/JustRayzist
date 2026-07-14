from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast


def resolve_procedural_creativity(*, procedural_creativity: int | None = 0) -> int:
    level = int(procedural_creativity or 0)
    if level < 0 or level > 3:
        raise ValueError("procedural_creativity must be between 0 and 3.")
    return level


InferenceProcess = Literal["standard", "rplus"]


def resolve_inference_process(*, inference_process: str | None = "standard") -> InferenceProcess:
    normalized = str(inference_process or "standard").strip().lower()
    if normalized not in {"standard", "rplus"}:
        raise ValueError("inference_process must be 'standard' or 'rplus'.")
    return cast(InferenceProcess, normalized)


@dataclass(frozen=True)
class LoraSelection:
    id: str
    path: Path
    weight: float = 1.0
    name: str | None = None
    trigger_words: tuple[str, ...] = ()


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    width: int
    height: int
    steps: int | None = None
    guidance_scale: float | None = None
    seed: int | None = None
    scheduler_mode: str | None = None
    inference_process: InferenceProcess = "standard"
    enhance_prompt: bool = False
    procedural_creativity: int = 0
    rplus_vibrance: float = 0.0
    rplus_initial_bias_level: float = 0.0
    rplus_initial_sample_size: int | str | None = None
    refine_strength: float | None = None
    refine_steps: int | None = None
    refine_tile_size: int | None = None
    refine_tile_overlap: int = 64
    upscaler_checkpoint: Path | None = None
    loras: tuple[LoraSelection, ...] = ()
    # Optional reference image for image+text joint conditioning (image-edit style). Krea2-only:
    # its Qwen3VLModel text encoder is vision-language and can encode an image alongside the
    # prompt. Additive and defaulting to None so the frozen contract stays backward-compatible;
    # the Z-Image backend ignores it. See JustRayzist-Krea.md §2/§4/WP-5.
    context_image: Path | None = None
