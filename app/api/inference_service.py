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
from app.core.upscale_blend import UPSCALE_ENGINE_NAME, upscale_with_x2_seed_blend
from app.storage import append_generation_metric, build_output_path, save_png_with_metadata
from app.storage.gallery_index import (
    delete_image,
    delete_gallery,
    get_image,
    import_gallery_source,
    index_image,
    list_import_sources,
    list_images,
    normalize_owner_id,
    sync_outputs_to_gallery,
)


def _assert_supported_backend(model_pack: ModelPack) -> None:
    supported = {"diffusers", "diffusers_zimage"}
    backends = [
        str(name).strip().lower()
        for name in model_pack.backend_preference
        if str(name).strip()
    ]
    if not any(name in supported for name in backends):
        raise ModelPackValidationError(
            "Unsupported backend preference list "
            f"{model_pack.backend_preference!r}. Include one of: {sorted(supported)}."
        )


class InferenceService:
    def __init__(self, settings: AppSettings):
        self._settings = settings
        self._lock = Lock()
        configured_pack = os.getenv("JUSTRAYZIST_PACK", "").strip()
        self._default_pack_name = configured_pack or None
        self._active_pack_name: str | None = None
        self._active_session: GenerationSession | None = None

    @staticmethod
    def sanitize_owner_id(owner_id: str) -> str:
        return normalize_owner_id(owner_id)

    def owner_output_dir(self, owner_id: str) -> Path:
        safe_owner = self.sanitize_owner_id(owner_id)
        output_dir = (self._settings.paths.outputs_dir / safe_owner).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

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
        owner_id: str,
        prompt_query: str | None = None,
        limit: int = 100,
        offset: int = 0,
        newest_first: bool = True,
    ) -> list[dict[str, Any]]:
        return list_images(
            settings=self._settings,
            owner_id=self.sanitize_owner_id(owner_id),
            prompt_query=prompt_query,
            limit=limit,
            offset=offset,
            newest_first=newest_first,
        )

    def get_image(self, filename: str, owner_id: str) -> dict[str, Any] | None:
        return get_image(self._settings, filename, owner_id=self.sanitize_owner_id(owner_id))

    def delete_gallery(self, owner_id: str, confirm_text: str) -> dict[str, int]:
        normalized = confirm_text.strip()
        if normalized.upper() != "DELETE":
            raise ValueError("Deletion rejected. Type DELETE exactly to confirm.")
        with self._lock:
            return delete_gallery(self._settings, owner_id=self.sanitize_owner_id(owner_id))

    def delete_image(self, owner_id: str, filename: str, confirm_text: str) -> dict[str, int]:
        normalized = confirm_text.strip()
        if normalized.upper() != "DELETE":
            raise ValueError("Deletion rejected. Type DELETE exactly to confirm.")
        with self._lock:
            return delete_image(
                self._settings,
                filename,
                owner_id=self.sanitize_owner_id(owner_id),
            )

    def list_import_sources(self, owner_id: str) -> list[dict[str, Any]]:
        return list_import_sources(self._settings, self.sanitize_owner_id(owner_id))

    def import_gallery(
        self,
        owner_id: str,
        source_id: str,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            return import_gallery_source(
                self._settings,
                target_owner_id=self.sanitize_owner_id(owner_id),
                source_id=source_id,
                dry_run=dry_run,
            )

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

    def resolve_output_path(self, raw_path: str) -> Path:
        resolved = Path(str(raw_path)).expanduser().resolve()
        outputs_dir = self._settings.paths.outputs_dir.resolve()
        try:
            resolved.relative_to(outputs_dir)
        except ValueError as exc:
            raise ValueError("Image path is outside managed outputs directory.") from exc
        return resolved

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
        owner_id: str,
        prompt: str,
        width: int,
        height: int,
        pack_name: str | None = None,
        seed: int | None = None,
        scheduler_mode: str | None = None,
        enhance_prompt: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            safe_owner_id = self.sanitize_owner_id(owner_id)
            model_pack = self._resolve_pack(pack_name)
            session = self._session_for_pack(model_pack)
            effective_seed = seed if seed is not None else random.randint(1, 2_147_483_647)
            output_path = build_output_path(self.owner_output_dir(safe_owner_id))

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
                output_path=output_path,
                extra_metadata={
                    "owner_id": safe_owner_id,
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
                    "runtime_profile": result.runtime_profile,
                    "execution_mode": result.execution_mode,
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
                    "owner_id": safe_owner_id,
                    "model_pack": model_pack.name,
                    **result.telemetry_dict(),
                },
            )
            image_row = index_image(self._settings, saved_path, owner_id=safe_owner_id)
            image_row["url"] = f"/images/{image_row['filename']}"
            image_row["pack"] = model_pack.name
            image_row["duration_ms"] = result.duration_ms
            image_row["seed"] = result.seed
            image_row["scheduler_mode"] = result.scheduler_mode
            image_row["prompt_original"] = result.prompt_original
            image_row["prompt_effective"] = result.prompt_effective
            image_row["prompt_enhanced"] = result.prompt_enhanced
            image_row["runtime_profile"] = result.runtime_profile
            image_row["execution_mode"] = result.execution_mode
            return image_row

    def upscale(
        self,
        owner_id: str,
        filename: str,
        pack_name: str | None = None,
        seed: int | None = None,
        scheduler_mode: str | None = None,
        enhance_prompt: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            safe_owner_id = self.sanitize_owner_id(owner_id)
            safe_filename = self.sanitize_filename(filename)
            source_row = get_image(self._settings, safe_filename, owner_id=safe_owner_id)
            if source_row is None:
                raise ValueError("Image not found.")

            source_output = source_row.get("output_path")
            if not source_output:
                raise ValueError("Image source path is missing.")
            source_path = self.resolve_output_path(str(source_output))
            if not source_path.exists():
                raise ValueError("Image file not found on disk.")

            source_prompt = str(source_row.get("prompt") or "").strip() or "(missing prompt metadata)"
            preferred_pack = str(source_row.get("model_pack") or "").strip() or None
            resolved_pack_name = pack_name or preferred_pack
            model_pack_name = (resolved_pack_name or "unknown").strip() or "unknown"
            effective_seed = seed if seed is not None else random.randint(1, 2_147_483_647)

            with Image.open(source_path) as source_file:
                source_image = source_file.convert("RGB")
            source_width, source_height = source_image.size

            result = upscale_with_x2_seed_blend(
                image=source_image,
                settings=self._settings,
                runtime_profile=self._settings.runtime_profile.name,
                seed=effective_seed,
            )
            final_width, final_height = result.output_width, result.output_height
            output_path = build_output_path(self.owner_output_dir(safe_owner_id))
            saved_path = save_png_with_metadata(
                image=result.image,
                prompt=source_prompt,
                settings=self._settings,
                output_path=output_path,
                extra_metadata={
                    "owner_id": safe_owner_id,
                    "mode": "api_upscale",
                    "prompt_original": source_prompt,
                    "prompt_effective": source_prompt,
                    "prompt_enhanced": False,
                    "source_image": str(source_path),
                    "source_filename": safe_filename,
                    "source_width": source_width,
                    "source_height": source_height,
                    "width": final_width,
                    "height": final_height,
                    "steps": 0,
                    "guidance_scale": 0.0,
                    "backend": UPSCALE_ENGINE_NAME,
                    "device": result.device,
                    "model_pack": model_pack_name,
                    "duration_ms": result.duration_ms,
                    "seed": effective_seed,
                    "scheduler_mode": scheduler_mode or "euler",
                    "runtime_profile": self._settings.runtime_profile.name,
                    "execution_mode": UPSCALE_ENGINE_NAME,
                    "request_enhance_prompt": bool(enhance_prompt),
                    **result.telemetry_dict(),
                },
            )
            append_generation_metric(
                settings=self._settings,
                payload={
                    "mode": "api_upscale",
                    "prompt": source_prompt,
                    "prompt_original": source_prompt,
                    "prompt_effective": source_prompt,
                    "prompt_enhanced": False,
                    "source_filename": safe_filename,
                    "source_width": source_width,
                    "source_height": source_height,
                    "width": final_width,
                    "height": final_height,
                    "output_path": str(saved_path),
                    "owner_id": safe_owner_id,
                    "model_pack": model_pack_name,
                    "backend": UPSCALE_ENGINE_NAME,
                    "seed": effective_seed,
                    "scheduler_mode": scheduler_mode or "euler",
                    "request_enhance_prompt": bool(enhance_prompt),
                    **result.telemetry_dict(),
                },
            )
            image_row = index_image(self._settings, saved_path, owner_id=safe_owner_id)
            image_row["url"] = f"/images/{image_row['filename']}"
            image_row["pack"] = model_pack_name
            image_row["duration_ms"] = result.duration_ms
            image_row["seed"] = effective_seed
            image_row["scheduler_mode"] = scheduler_mode or "euler"
            image_row["prompt_original"] = source_prompt
            image_row["prompt_effective"] = source_prompt
            image_row["prompt_enhanced"] = False
            image_row["runtime_profile"] = self._settings.runtime_profile.name
            image_row["execution_mode"] = UPSCALE_ENGINE_NAME
            image_row["upscale_engine"] = UPSCALE_ENGINE_NAME
            return image_row

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        sanitized = Path(filename).name
        if sanitized != filename:
            raise ValueError("Invalid filename path.")
        return sanitized
