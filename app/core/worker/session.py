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
    switch_count: int = 0


# Resource tiers with enough headroom to keep a previously-loaded backend resident for instant
# switch-back. On tighter tiers the outgoing backend is always released before the next is built,
# so two (potentially 12B-class) model families are never resident at once.
# See JustRayzist-Krea.md WP-7.
_KEEP_RESIDENT_TIERS = frozenset({"high"})


class GenerationSession:
    def __init__(
        self,
        settings: AppSettings,
        model_pack: ModelPack,
        resource_tier: RuntimeProfile | None = None,
    ):
        self._settings = settings
        self._model_pack = model_pack
        # Auto-tier selection consults pack-specific thresholds when the caller doesn't pin a
        # profile explicitly. This lets a large model (e.g. Krea2's 12B DiT) demand more free VRAM
        # for `high` than the default the smaller Z-Image family uses.
        if resource_tier is not None:
            self._resource_tier = resource_tier
        else:
            controller = settings.resource_tier_controller
            current_for = getattr(controller, "current_for", None)
            self._resource_tier = (
                current_for(model_pack) if callable(current_for) else controller.current()
            )
        self._backend: Any | None = None
        # Cache of idle backends keyed by pack name, populated only on keep-resident tiers.
        self._resident_backends: dict[str, Any] = {}
        self.stats = SessionStats()

    def _ensure_backend(self):
        if self._backend is None:
            self._backend = create_backend(
                settings=self._settings,
                model_pack=self._model_pack,
                resource_tier=self._resource_tier,
            )
        return self._backend

    def _keep_resident(self) -> bool:
        return self._resource_tier.name in _KEEP_RESIDENT_TIERS

    @staticmethod
    def _release_backend(backend: Any | None) -> None:
        """Best-effort teardown of a backend and its VRAM."""
        if backend is None:
            return
        cancel = getattr(backend, "cancel_active", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:  # pragma: no cover - defensive teardown
                LOGGER.debug("cancel_active() failed during backend release.", exc_info=True)
        teardown = getattr(backend, "teardown", None)
        if callable(teardown):
            try:
                teardown()
            except Exception:  # pragma: no cover - defensive teardown
                LOGGER.debug("teardown() failed during backend release.", exc_info=True)

    @staticmethod
    def _free_cuda_cache() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @property
    def model_pack(self) -> ModelPack:
        return self._model_pack

    def switch_model_pack(self, model_pack: ModelPack) -> ModelPack:
        """Switch the active model family/pack at runtime, tier-adaptively.

        On a keep-resident tier (headroom) the outgoing backend is cached for instant switch-back;
        otherwise it is torn down and its VRAM released before the target backend is built. Any
        in-flight generation on the current backend is cancelled first. Returns the now-active pack.

        See JustRayzist-Krea.md WP-7.
        """
        if model_pack.name == self._model_pack.name and self._backend is not None:
            return self._model_pack

        outgoing = self._backend
        keep_resident = self._keep_resident()

        if outgoing is not None:
            if keep_resident:
                # Cancel any in-flight work, then park the backend for instant switch-back. It is
                # NOT torn down, so its weights stay resident (headroom tier only).
                cancel = getattr(outgoing, "cancel_active", None)
                if callable(cancel):
                    try:
                        cancel()
                    except Exception:  # pragma: no cover - defensive
                        LOGGER.debug("cancel_active() failed before switch.", exc_info=True)
                self._resident_backends[self._model_pack.name] = outgoing
            else:
                # Release fully (cancel + teardown + free VRAM) before loading the next family, so
                # two large models are never resident at once. _release_backend cancels for us.
                self._release_backend(outgoing)
                self._free_cuda_cache()

        self._backend = None
        self._model_pack = model_pack

        # Re-resolve the resource tier against the new pack's thresholds. A pack switch between
        # families with different VRAM footprints (e.g. Z-Image ↔ Krea2) may demand a different
        # tier — the user override is still honored via ResourceTierController.current_for().
        controller = self._settings.resource_tier_controller
        current_for = getattr(controller, "current_for", None)
        if callable(current_for):
            self._resource_tier = current_for(model_pack)

        # Reuse a cached resident backend if we have one for the target pack.
        cached = self._resident_backends.pop(model_pack.name, None)
        if cached is not None:
            self._backend = cached
        else:
            self._backend = self._ensure_backend()

        self.stats.switch_count += 1
        LOGGER.info(
            "Switched active model pack to '%s' (tier=%s, keep_resident=%s).",
            model_pack.name,
            self._resource_tier.name,
            keep_resident,
        )
        return self._model_pack

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

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        app_context: str | None = None,
        seed: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 0.75,
    ) -> dict[str, object]:
        backend = self._ensure_backend()
        chat = getattr(backend, "chat", None)
        if not callable(chat):
            raise AttributeError("Active backend does not support Rayzist Chat.")
        return chat(
            messages=messages,
            app_context=app_context,
            seed=seed,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
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
        # Drop the active backend and any resident (cached) backends so a recycle — e.g. on a
        # resource-tier change — fully releases VRAM rather than stranding a cached model family.
        if self._resident_backends:
            for cached in self._resident_backends.values():
                self._release_backend(cached)
            self._resident_backends.clear()
        self._backend = None
        self.stats.recycle_count += 1
        self._free_cuda_cache()

    def runtime_status(self) -> dict[str, object]:
        backend = self._ensure_backend()
        if hasattr(backend, "runtime_status"):
            return dict(backend.runtime_status())
        return {"backend": "unknown", "effective_pack": self._model_pack.name}
