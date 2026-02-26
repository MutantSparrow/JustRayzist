from __future__ import annotations

import os
import random
from pathlib import Path
from threading import Lock
from typing import Any

from PIL import Image

from app.config.settings import AppSettings
from app.core.model_registry import (
    ModelPack,
    ModelPackValidationError,
    discover_model_packs,
    load_model_pack,
    load_model_pack_by_name,
)
from app.core.worker import GenerationRequest, GenerationSession
from app.storage import append_generation_metric, save_png_with_metadata
from app.storage.gallery_index import (
    delete_image,
    delete_gallery,
    get_image,
    index_image,
    list_images,
    sync_outputs_to_gallery,
)


def _assert_supported_backend(model_pack: ModelPack) -> None:
    backend_name = (model_pack.backend_preference[0] if model_pack.backend_preference else "").lower()
    if backend_name not in {"diffusers", "diffusers_zimage"}:
        raise ModelPackValidationError(
            f"Unsupported backend '{backend_name}'. Use backend_preference: ['diffusers']."
        )


class InferenceService:
    def __init__(self, settings: AppSettings):
        self._settings = settings
        self._lock = Lock()
        configured_pack = os.getenv("JUSTRAYZIST_PACK", "").strip()
        self._default_pack_name = configured_pack or None
        self._active_pack_name: str | None = None
        self._active_session: GenerationSession | None = None

    def list_model_packs(self) -> list[dict[str, str]]:
        packs: list[dict[str, str]] = []
        for pack_file in discover_model_packs(self._settings.paths.model_packs_dir):
            try:
                pack = load_model_pack(pack_file)
            except ModelPackValidationError:
                continue
            packs.append(
                {
                    "name": pack.name,
                    "path": str(pack.source_file),
                    "architecture": pack.architecture,
                }
            )
        return packs

    def sync_gallery(self) -> int:
        return sync_outputs_to_gallery(self._settings)

    def list_images(
        self,
        prompt_query: str | None = None,
        limit: int = 100,
        offset: int = 0,
        newest_first: bool = True,
    ) -> list[dict[str, Any]]:
        return list_images(
            settings=self._settings,
            prompt_query=prompt_query,
            limit=limit,
            offset=offset,
            newest_first=newest_first,
        )

    def get_image(self, filename: str) -> dict[str, Any] | None:
        return get_image(self._settings, filename)

    def delete_gallery(self, confirm_text: str) -> dict[str, int]:
        normalized = confirm_text.strip()
        if normalized.upper() != "DELETE":
            raise ValueError("Deletion rejected. Type DELETE exactly to confirm.")
        with self._lock:
            return delete_gallery(self._settings)

    def delete_image(self, filename: str, confirm_text: str) -> dict[str, int]:
        normalized = confirm_text.strip()
        if normalized.upper() != "DELETE":
            raise ValueError("Deletion rejected. Type DELETE exactly to confirm.")
        with self._lock:
            return delete_image(self._settings, filename)

    def _resolve_pack(self, pack_name: str | None) -> ModelPack:
        requested_pack_name = (pack_name or self._default_pack_name or "").strip()
        if requested_pack_name:
            pack = load_model_pack_by_name(self._settings.paths.model_packs_dir, requested_pack_name)
            _assert_supported_backend(pack)
            return pack

        pack_paths = discover_model_packs(self._settings.paths.model_packs_dir)
        if not pack_paths:
            raise ModelPackValidationError("No model packs found.")
        pack = load_model_pack(pack_paths[0])
        _assert_supported_backend(pack)
        return pack

    def _session_for_pack(self, model_pack: ModelPack) -> GenerationSession:
        if self._active_session is None:
            self._active_pack_name = model_pack.name
            self._active_session = GenerationSession(settings=self._settings, model_pack=model_pack)
            return self._active_session

        if self._active_pack_name != model_pack.name:
            self._active_session.recycle("Switching active model pack")
            self._active_pack_name = model_pack.name
            self._active_session = GenerationSession(settings=self._settings, model_pack=model_pack)
        return self._active_session

    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        pack_name: str | None = None,
        seed: int | None = None,
        scheduler_mode: str | None = None,
        enhance_prompt: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            model_pack = self._resolve_pack(pack_name)
            session = self._session_for_pack(model_pack)
            effective_seed = seed if seed is not None else random.randint(1, 2_147_483_647)

            result = session.generate(
                GenerationRequest(
                    prompt=prompt,
                    width=width,
                    height=height,
                    seed=effective_seed,
                    scheduler_mode=scheduler_mode,
                    enhance_prompt=enhance_prompt,
                )
            )

            saved_path = save_png_with_metadata(
                image=result.image,
                prompt=result.prompt_effective,
                settings=self._settings,
                extra_metadata={
                    "prompt_original": result.prompt_original,
                    "prompt_effective": result.prompt_effective,
                    "prompt_enhanced": result.prompt_enhanced,
                    "width": width,
                    "height": height,
                    "steps": result.steps,
                    "guidance_scale": result.guidance_scale,
                    "backend": result.backend,
                    "device": result.device,
                    "model_pack": model_pack.name,
                    "duration_ms": result.duration_ms,
                    "seed": result.seed,
                    "scheduler_mode": result.scheduler_mode,
                },
            )
            append_generation_metric(
                settings=self._settings,
                payload={
                    "mode": "api_generate",
                    "prompt": result.prompt_effective,
                    "prompt_original": result.prompt_original,
                    "prompt_effective": result.prompt_effective,
                    "prompt_enhanced": result.prompt_enhanced,
                    "width": width,
                    "height": height,
                    "output_path": str(saved_path),
                    "model_pack": model_pack.name,
                    **result.telemetry_dict(),
                },
            )
            image_row = index_image(self._settings, saved_path)
            image_row["url"] = f"/images/{image_row['filename']}"
            image_row["pack"] = model_pack.name
            image_row["duration_ms"] = result.duration_ms
            image_row["seed"] = result.seed
            image_row["scheduler_mode"] = result.scheduler_mode
            image_row["prompt_original"] = result.prompt_original
            image_row["prompt_effective"] = result.prompt_effective
            image_row["prompt_enhanced"] = result.prompt_enhanced
            return image_row

    def upscale(
        self,
        filename: str,
        pack_name: str | None = None,
        seed: int | None = None,
        scheduler_mode: str | None = None,
        enhance_prompt: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            safe_filename = self.sanitize_filename(filename)
            source_row = get_image(self._settings, safe_filename)
            if source_row is None:
                raise ValueError("Image not found.")

            source_output = source_row.get("output_path")
            if not source_output:
                raise ValueError("Image source path is missing.")
            source_path = Path(str(source_output)).expanduser().resolve()
            if not source_path.exists():
                raise ValueError("Image file not found on disk.")

            source_prompt = str(source_row.get("prompt") or "").strip() or "(missing prompt metadata)"
            preferred_pack = str(source_row.get("model_pack") or "").strip() or None
            resolved_pack_name = pack_name or preferred_pack
            model_pack = self._resolve_pack(resolved_pack_name)
            session = self._session_for_pack(model_pack)
            effective_seed = seed if seed is not None else random.randint(1, 2_147_483_647)

            with Image.open(source_path) as source_file:
                source_image = source_file.convert("RGB")
            source_width, source_height = source_image.size

            result = session.upscale_and_refine(
                input_image=source_image,
                request=GenerationRequest(
                    prompt=source_prompt,
                    width=source_width,
                    height=source_height,
                    seed=effective_seed,
                    scheduler_mode=scheduler_mode,
                    enhance_prompt=enhance_prompt,
                ),
            )
            final_width, final_height = result.image.size
            saved_path = save_png_with_metadata(
                image=result.image,
                prompt=result.prompt_effective,
                settings=self._settings,
                extra_metadata={
                    "mode": "api_upscale",
                    "prompt_original": result.prompt_original,
                    "prompt_effective": result.prompt_effective,
                    "prompt_enhanced": result.prompt_enhanced,
                    "source_image": str(source_path),
                    "source_filename": safe_filename,
                    "source_width": source_width,
                    "source_height": source_height,
                    "width": final_width,
                    "height": final_height,
                    "steps": result.steps,
                    "guidance_scale": result.guidance_scale,
                    "backend": result.backend,
                    "device": result.device,
                    "model_pack": model_pack.name,
                    "duration_ms": result.duration_ms,
                    "seed": result.seed,
                    "scheduler_mode": result.scheduler_mode,
                },
            )
            append_generation_metric(
                settings=self._settings,
                payload={
                    "mode": "api_upscale",
                    "prompt": result.prompt_effective,
                    "prompt_original": result.prompt_original,
                    "prompt_effective": result.prompt_effective,
                    "prompt_enhanced": result.prompt_enhanced,
                    "source_filename": safe_filename,
                    "source_width": source_width,
                    "source_height": source_height,
                    "width": final_width,
                    "height": final_height,
                    "output_path": str(saved_path),
                    "model_pack": model_pack.name,
                    **result.telemetry_dict(),
                },
            )
            image_row = index_image(self._settings, saved_path)
            image_row["url"] = f"/images/{image_row['filename']}"
            image_row["pack"] = model_pack.name
            image_row["duration_ms"] = result.duration_ms
            image_row["seed"] = result.seed
            image_row["scheduler_mode"] = result.scheduler_mode
            image_row["prompt_original"] = result.prompt_original
            image_row["prompt_effective"] = result.prompt_effective
            image_row["prompt_enhanced"] = result.prompt_enhanced
            return image_row

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        sanitized = Path(filename).name
        if sanitized != filename:
            raise ValueError("Invalid filename path.")
        return sanitized
