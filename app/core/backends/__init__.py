from __future__ import annotations

from typing import Any

from app.core.backends.diffusers_zimage import DiffusersZImageBackend, GenerationResult
from app.core.backends.fp8_zimage import Fp8ZImageBackend

SUPPORTED_BACKENDS = {"diffusers", "diffusers_zimage", "fp8_zimage"}


def create_backend(*, settings: Any, model_pack: Any, resource_tier: Any):
    normalized = [
        str(name).strip().lower()
        for name in getattr(model_pack, "backend_preference", [])
        if str(name).strip()
    ]
    if not normalized:
        normalized = ["diffusers"]

    for backend_name in normalized:
        if backend_name == "fp8_zimage":
            return Fp8ZImageBackend(
                settings=settings,
                model_pack=model_pack,
                resource_tier=resource_tier,
            )
        if backend_name in {"diffusers", "diffusers_zimage"}:
            return DiffusersZImageBackend(
                settings=settings,
                model_pack=model_pack,
                resource_tier=resource_tier,
            )
    raise ValueError(
        f"Unsupported backend preference list {getattr(model_pack, 'backend_preference', None)!r}. "
        f"Include one of: {sorted(SUPPORTED_BACKENDS)}."
    )


__all__ = [
    "DiffusersZImageBackend",
    "Fp8ZImageBackend",
    "GenerationResult",
    "SUPPORTED_BACKENDS",
    "create_backend",
]
