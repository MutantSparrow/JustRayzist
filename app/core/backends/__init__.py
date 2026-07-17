from __future__ import annotations

from typing import Any

from app.core.backends.diffusers_zimage import DiffusersZImageBackend, GenerationResult
from app.core.backends.fp8_zimage import Fp8ZImageBackend

SUPPORTED_BACKENDS = {
    "diffusers",
    "diffusers_zimage",
    "fp8_zimage",
    "diffusers_krea",
    "fp8_krea",
}


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
        # Krea backends are imported lazily so the diffusers Krea classes (which require
        # diffusers >=0.39.0) are only touched when a Krea pack is actually dispatched. This
        # keeps the Z-Image path free of any Krea import cost/risk on older diffusers builds.
        if backend_name == "fp8_krea":
            from app.core.backends.diffusers_krea import Fp8KreaBackend

            return Fp8KreaBackend(
                settings=settings,
                model_pack=model_pack,
                resource_tier=resource_tier,
            )
        if backend_name == "diffusers_krea":
            from app.core.backends.diffusers_krea import DiffusersKreaBackend

            return DiffusersKreaBackend(
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
