from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def resolve_procedural_creativity(*, procedural_creativity: int | None = 0) -> int:
    level = int(procedural_creativity or 0)
    if level < 0 or level > 3:
        raise ValueError("procedural_creativity must be between 0 and 3.")
    return level


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
    enhance_prompt: bool = False
    procedural_creativity: int = 0
    refine_strength: float | None = None
    refine_steps: int | None = None
    refine_tile_size: int | None = None
    refine_tile_overlap: int = 64
    upscaler_checkpoint: Path | None = None
    loras: tuple[LoraSelection, ...] = ()
