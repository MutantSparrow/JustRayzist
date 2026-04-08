from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any

from app.config.profiles import RuntimeProfile
from app.config.settings import AppSettings
from app.core.backends import GenerationResult, create_backend
from app.core.model_registry import ModelPack
from app.core.worker.types import GenerationRequest

LOGGER = logging.getLogger(__name__)


@dataclass
class SessionStats:
    generation_count: int = 0
    recycle_count: int = 0


class GenerationSession:
    def __init__(
        self,
        settings: AppSettings,
        model_pack: ModelPack,
        resource_tier: RuntimeProfile | None = None,
    ):
        self._settings = settings
        self._model_pack = model_pack
        self._resource_tier = resource_tier or settings.resource_tier_controller.current()
        self._backend: Any | None = None
        self.stats = SessionStats()

    def _ensure_backend(self):
        if self._backend is None:
            self._backend = create_backend(
                settings=self._settings,
                model_pack=self._model_pack,
                resource_tier=self._resource_tier,
            )
        return self._backend

    def set_resource_tier(self, resource_tier: RuntimeProfile) -> None:
        if self._resource_tier.name == resource_tier.name:
            return
        self._resource_tier = resource_tier
        if self._backend is not None:
            self.recycle(f"Resource tier changed to {resource_tier.name}")

    def generate(self, request: GenerationRequest) -> GenerationResult:
        backend = self._ensure_backend()
        result = backend.generate(request)
        self.stats.generation_count += 1
        return result

    def upscale_and_refine(self, input_image: object, request: GenerationRequest) -> GenerationResult:
        backend = self._ensure_backend()
        result = backend.upscale_and_refine(input_image=input_image, request=request)
        self.stats.generation_count += 1
        return result

    def refine_image(self, input_image: object, request: GenerationRequest) -> GenerationResult:
        backend = self._ensure_backend()
        refine = getattr(backend, "refine_image", None)
        if not callable(refine):
            raise AttributeError("Active backend does not support refine_image().")
        result = refine(input_image=input_image, request=request)
        self.stats.generation_count += 1
        return result

    def suggest_wildcard_entries(
        self,
        *,
        theme: str,
        format_example: str,
        seed: int | None = None,
        existing_entries: list[str] | tuple[str, ...] | None = None,
        target_count: int = 10,
    ) -> dict[str, object]:
        backend = self._ensure_backend()
        suggest = getattr(backend, "suggest_wildcard_entries", None)
        if not callable(suggest):
            raise AttributeError("Active backend does not support wildcard suggestions.")
        return suggest(
            theme=theme,
            format_example=format_example,
            seed=seed,
            existing_entries=existing_entries,
            target_count=target_count,
        )

    def cancel_active(self) -> None:
        if self._backend is None:
            return
        cancel = getattr(self._backend, "cancel_active", None)
        if callable(cancel):
            cancel()

    def drop_lora_adapters(self, lora_ids: list[str] | None = None) -> None:
        if self._backend is None:
            return
        drop = getattr(self._backend, "drop_lora_adapters", None)
        if callable(drop):
            drop(lora_ids)

    def recycle(self, reason: str) -> None:
        LOGGER.info("Recycling generation session backend. Reason: %s", reason)
        self._backend = None
        self.stats.recycle_count += 1
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def runtime_status(self) -> dict[str, object]:
        backend = self._ensure_backend()
        if hasattr(backend, "runtime_status"):
            return dict(backend.runtime_status())
        return {"backend": "unknown", "effective_pack": self._model_pack.name}
