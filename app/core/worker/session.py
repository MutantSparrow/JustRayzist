from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

from app.config.settings import AppSettings
from app.core.backends import DiffusersZImageBackend, GenerationResult
from app.core.model_registry import ModelPack
from app.core.worker.types import GenerationRequest

LOGGER = logging.getLogger(__name__)


@dataclass
class SessionStats:
    generation_count: int = 0
    recycle_count: int = 0


class GenerationSession:
    def __init__(self, settings: AppSettings, model_pack: ModelPack):
        self._settings = settings
        self._model_pack = model_pack
        self._backend: DiffusersZImageBackend | None = None
        self.stats = SessionStats()

    def _ensure_backend(self) -> DiffusersZImageBackend:
        if self._backend is None:
            self._backend = DiffusersZImageBackend(
                settings=self._settings,
                model_pack=self._model_pack,
            )
        return self._backend

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
