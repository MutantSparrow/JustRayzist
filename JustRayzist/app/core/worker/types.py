from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    width: int
    height: int
    steps: int | None = None
    guidance_scale: float | None = None
    seed: int | None = None
    scheduler_mode: str | None = None
    enhance_prompt: bool = False
    refine_strength: float | None = None
    refine_steps: int | None = None
    refine_tile_size: int | None = None
    refine_tile_overlap: int = 64
    upscaler_checkpoint: Path | None = None
