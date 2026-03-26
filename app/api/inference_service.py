from __future__ import annotations

import logging
import os
import random
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

from PIL import Image

from app.config.profiles import RuntimeProfile
from app.config.settings import AppSettings
from app.core.cancellation import GenerationCancelledError
from app.core.backends import SUPPORTED_BACKENDS
from app.core.model_registry import (
    ModelComponent,
    ModelPack,
    ModelPackValidationError,
    discover_model_packs,
    load_model_pack,
    load_model_pack_by_name,
)
from app.core.worker import GenerationRequest, GenerationSession
from app.core.worker.types import resolve_procedural_creativity
from app.core.upscale_blend import UPSCALE_ENGINE_NAME, upscale_with_x2_seed_blend
from app.storage import append_generation_metric, build_output_path, save_png_with_metadata
from app.storage.gallery_index import (
    COLOR_CACHE_VERSION,
    delete_image,
    delete_gallery,
    gallery_color_cache_needs_rebuild,
    gallery_color_cache_version,
    get_image,
    import_gallery_source,
    index_image,
    list_import_sources,
    list_images,
    normalize_owner_id,
    rebuild_gallery_color_cache,
    sync_outputs_to_gallery,
)

LOGGER = logging.getLogger(__name__)
_DONOR_PACK_NAME = "Rayzist_bf16"
_DERIVED_FP8_STORAGE_NAME = "fp8_storage"
_DERIVED_FP8_STORAGE_SUFFIX = "__auto_fp8_storage"


def _assert_supported_backend(model_pack: ModelPack) -> None:
    backends = [
        str(name).strip().lower()
        for name in model_pack.backend_preference
        if str(name).strip()
    ]
    if not any(name in SUPPORTED_BACKENDS for name in backends):
        raise ModelPackValidationError(
            "Unsupported backend preference list "
            f"{model_pack.backend_preference!r}. Include one of: {sorted(SUPPORTED_BACKENDS)}."
        )


class InferenceService:
    def __init__(self, settings: AppSettings):
        self._settings = settings
        self._state_lock = Lock()
        self._generation_lock = Lock()
        configured_pack = os.getenv("JUSTRAYZIST_PACK", "").strip()
        self._default_pack_name = configured_pack or None
        self._active_pack_name: str | None = None
        self._active_selected_pack_name: str | None = None
        self._active_backend_name: str | None = None
        self._active_session: GenerationSession | None = None
        self._donor_pack_cache: ModelPack | None = None
        self._client_active_jobs: dict[str, dict[str, Any]] = {}
        self._client_cancel_events: dict[str, Event] = {}
        self._gallery_color_cache_rebuild_active = False
        self._gallery_color_cache_rebuild_last_error: str | None = None
        self._gallery_color_cache_rebuild_thread: Thread | None = None

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
            if not pack.user_visible or not pack.enabled:
                continue
            packs.append(
                {
                    "name": pack.name,
                    "path": str(pack.source_file),
                    "architecture": pack.architecture,
                }
            )
        return packs

    def current_resource_tier(self, *, refresh: bool = True) -> RuntimeProfile:
        controller = self._settings.resource_tier_controller
        return controller.refresh() if refresh else controller.current()

    def runtime_status(self) -> dict[str, Any]:
        tier = self.current_resource_tier(refresh=False)
        backend_status: dict[str, Any] = {}
        if self._active_session is not None:
            try:
                backend_status = self._active_session.runtime_status()
            except Exception:
                backend_status = {}
        color_cache_status = self.gallery_color_cache_status()
        return {
            "runtime_profile": self._settings.runtime_profile.name,
            "resource_tier": tier.name,
            "resource_tier_description": tier.description,
            "resource_tier_override": self._settings.resource_tier_override,
            "auto_resource_tier": self._settings.auto_resource_tier,
            "active_pack": self._active_pack_name,
            "selected_pack": self._active_selected_pack_name,
            "effective_pack": backend_status.get("effective_pack", self._active_pack_name),
            "active_backend": backend_status.get("backend", self._active_backend_name),
            "execution_mode": backend_status.get("execution_mode"),
            "execution_mode_initial": backend_status.get("execution_mode_initial"),
            "fp8_checkpoint": backend_status.get("fp8_checkpoint", False),
            "fp8_fallback_used": backend_status.get("fp8_fallback_used", False),
            "fp8_fallback_reason": backend_status.get("fp8_fallback_reason"),
            "fp8_runtime_mode": backend_status.get("fp8_runtime_mode"),
            "fp8_normalized_tensor_count": backend_status.get("fp8_normalized_tensor_count", 0),
            "fp8_storage_preserved_tensor_count": backend_status.get(
                "fp8_storage_preserved_tensor_count", 0
            ),
            "fp8_promoted_tensor_count": backend_status.get("fp8_promoted_tensor_count", 0),
            "gallery_color_cache_active": color_cache_status.get("active", False),
            "gallery_color_cache_version": color_cache_status.get("version"),
            "gallery_color_cache_target_version": color_cache_status.get("target_version"),
            "gallery_color_cache_error": color_cache_status.get("last_error"),
        }

    def client_job_status(self, owner_id: str) -> dict[str, Any]:
        safe_owner_id = self.sanitize_owner_id(owner_id)
        with self._state_lock:
            active_job = self._client_active_jobs.get(safe_owner_id)
            return {"active_job": dict(active_job) if active_job else None}

    def gallery_color_cache_status(self) -> dict[str, Any]:
        version = gallery_color_cache_version(self._settings)
        with self._state_lock:
            active = self._gallery_color_cache_rebuild_active
            last_error = self._gallery_color_cache_rebuild_last_error
        return {
            "active": active,
            "version": version,
            "target_version": COLOR_CACHE_VERSION,
            "needs_rebuild": version != COLOR_CACHE_VERSION,
            "last_error": last_error,
        }

    def start_gallery_color_cache_rebuild(self) -> bool:
        if not gallery_color_cache_needs_rebuild(self._settings):
            with self._state_lock:
                self._gallery_color_cache_rebuild_active = False
                self._gallery_color_cache_rebuild_last_error = None
            return False
        with self._state_lock:
            if self._gallery_color_cache_rebuild_active:
                return True
            self._gallery_color_cache_rebuild_active = True
            self._gallery_color_cache_rebuild_last_error = None

        def _worker() -> None:
            try:
                updated = rebuild_gallery_color_cache(self._settings)
                LOGGER.info("Gallery color cache rebuild completed: updated=%s", updated)
            except Exception as exc:
                LOGGER.exception("Gallery color cache rebuild failed.")
                with self._state_lock:
                    self._gallery_color_cache_rebuild_last_error = str(exc)
            finally:
                with self._state_lock:
                    self._gallery_color_cache_rebuild_active = False

        thread = Thread(
            target=_worker,
            daemon=True,
            name="justrayzist-gallery-color-cache-rebuild",
        )
        with self._state_lock:
            self._gallery_color_cache_rebuild_thread = thread
        thread.start()
        return True

    def _set_active_client_job_locked(self, owner_id: str, payload: dict[str, Any]) -> None:
        self._client_active_jobs[self.sanitize_owner_id(owner_id)] = dict(payload)

    def _clear_active_client_job_locked(self, owner_id: str, *, job_id: str | None = None) -> None:
        safe_owner_id = self.sanitize_owner_id(owner_id)
        active_job = self._client_active_jobs.get(safe_owner_id)
        if active_job is None:
            return
        if job_id is not None and active_job.get("job_id") != job_id:
            return
        self._client_active_jobs.pop(safe_owner_id, None)
        self._client_cancel_events.pop(safe_owner_id, None)

    def request_cancel_client_job(self, owner_id: str, *, job_id: str | None = None) -> dict[str, Any]:
        safe_owner_id = self.sanitize_owner_id(owner_id)
        session_to_cancel: GenerationSession | None = None
        requested_job_id = str(job_id or "").strip() or None
        active_job_id: str | None = None
        with self._state_lock:
            active_job = self._client_active_jobs.get(safe_owner_id)
            if active_job is None:
                return {
                    "status": "ok",
                    "cancel_requested": False,
                    "job_id": requested_job_id,
                    "message": "No active job.",
                }
            active_job_id = str(active_job.get("job_id") or "").strip() or None
            if requested_job_id is not None and requested_job_id != active_job_id:
                return {
                    "status": "ok",
                    "cancel_requested": False,
                    "job_id": requested_job_id,
                    "message": "Job is no longer active.",
                }
            active_job["status"] = "cancelling"
            cancel_event = self._client_cancel_events.get(safe_owner_id)
            if cancel_event is not None:
                cancel_event.set()
            if active_job.get("kind") == "generate" and self._active_session is not None:
                session_to_cancel = self._active_session
        if session_to_cancel is not None:
            session_to_cancel.cancel_active()
        return {
            "status": "ok",
            "cancel_requested": True,
            "job_id": active_job_id,
            "message": "Cancellation requested.",
        }

    def sync_gallery(self) -> int:
        return sync_outputs_to_gallery(self._settings)

    def list_images(
        self,
        owner_id: str,
        prompt_query: str | None = None,
        color_filter: str | None = None,
        limit: int = 100,
        offset: int = 0,
        newest_first: bool = True,
    ) -> list[dict[str, Any]]:
        return list_images(
            settings=self._settings,
            owner_id=self.sanitize_owner_id(owner_id),
            prompt_query=prompt_query,
            color_filter=color_filter,
            limit=limit,
            offset=offset,
            newest_first=newest_first,
        )

    def get_image(self, filename: str, owner_id: str) -> dict[str, Any] | None:
        return get_image(self._settings, filename, owner_id=self.sanitize_owner_id(owner_id))

    def resolve_download_images(self, owner_id: str, filenames: list[str]) -> list[tuple[str, Path]]:
        safe_owner_id = self.sanitize_owner_id(owner_id)
        resolved: list[tuple[str, Path]] = []
        seen: set[str] = set()
        for raw_filename in filenames:
            safe_filename = self.sanitize_filename(raw_filename)
            if safe_filename in seen:
                continue
            row = get_image(self._settings, safe_filename, owner_id=safe_owner_id)
            if row is None:
                raise ValueError(f"Image not found: {safe_filename}")
            output_path = row.get("output_path")
            if not output_path:
                raise ValueError(f"Image path is missing: {safe_filename}")
            resolved_path = self.resolve_output_path(str(output_path))
            if not resolved_path.exists():
                raise ValueError(f"Image file not found on disk: {safe_filename}")
            resolved.append((safe_filename, resolved_path))
            seen.add(safe_filename)
        return resolved

    def delete_gallery(self, owner_id: str, confirm_text: str) -> dict[str, int]:
        normalized = confirm_text.strip()
        if normalized.upper() != "DELETE":
            raise ValueError("Deletion rejected. Type DELETE exactly to confirm.")
        with self._state_lock:
            return delete_gallery(self._settings, owner_id=self.sanitize_owner_id(owner_id))

    def delete_image(self, owner_id: str, filename: str, confirm_text: str) -> dict[str, int]:
        normalized = confirm_text.strip()
        if normalized.upper() != "DELETE":
            raise ValueError("Deletion rejected. Type DELETE exactly to confirm.")
        with self._state_lock:
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
        with self._state_lock:
            return import_gallery_source(
                self._settings,
                target_owner_id=self.sanitize_owner_id(owner_id),
                source_id=source_id,
                dry_run=dry_run,
            )

    def _load_base_pack(self, pack_name: str | None) -> ModelPack:
        requested_pack_name = (pack_name or self._default_pack_name or "").strip()
        if requested_pack_name:
            pack = load_model_pack_by_name(self._settings.paths.model_packs_dir, requested_pack_name)
            _assert_supported_backend(pack)
            return pack

        pack_paths = discover_model_packs(self._settings.paths.model_packs_dir)
        if not pack_paths:
            raise ModelPackValidationError("No model packs found.")
        for pack_path in pack_paths:
            pack = load_model_pack(pack_path)
            if not pack.user_visible or not pack.enabled:
                continue
            _assert_supported_backend(pack)
            return pack
        raise ModelPackValidationError("No public enabled model packs found.")

    def _load_donor_pack(self) -> ModelPack:
        if self._donor_pack_cache is None:
            donor_pack = load_model_pack_by_name(self._settings.paths.model_packs_dir, _DONOR_PACK_NAME)
            _assert_supported_backend(donor_pack)
            self._donor_pack_cache = donor_pack
        return self._donor_pack_cache

    @staticmethod
    def _merge_required_configs(primary: list[Path], fallback: list[Path]) -> list[Path]:
        merged: list[Path] = []
        seen: set[Path] = set()
        for path in [*primary, *fallback]:
            resolved = path.resolve()
            if resolved in seen:
                continue
            merged.append(resolved)
            seen.add(resolved)
        return merged

    def _complete_pack_with_donor(self, model_pack: ModelPack) -> ModelPack:
        has_pipeline = model_pack.pipeline_config_dir is not None
        has_vae = "vae" in model_pack.components
        has_text_encoder = "text_encoder" in model_pack.components
        if has_pipeline and has_vae and has_text_encoder and model_pack.required_configs:
            return model_pack

        donor_pack = self._load_donor_pack()
        if model_pack.architecture != donor_pack.architecture or "transformer" not in model_pack.components:
            raise ModelPackValidationError(
                f"Model pack '{model_pack.name}' is missing required components/config and is not "
                f"compatible with donor pack '{donor_pack.name}'."
            )

        merged_components = dict(donor_pack.components)
        merged_components.update(model_pack.components)
        return replace(
            model_pack,
            backend_preference=model_pack.backend_preference or donor_pack.backend_preference,
            components=merged_components,
            pipeline_config_dir=model_pack.pipeline_config_dir or donor_pack.pipeline_config_dir,
            required_configs=self._merge_required_configs(
                model_pack.required_configs,
                donor_pack.required_configs,
            ),
            base_name=model_pack.base_name or model_pack.name,
            derived_strategy=model_pack.derived_strategy,
        )

    @staticmethod
    def _can_derive_fp8_storage(model_pack: ModelPack) -> bool:
        if model_pack.architecture != "z_image_turbo":
            return False
        transformer = model_pack.components.get("transformer")
        if transformer is None:
            return False
        if transformer.file_format != "safetensors":
            return False
        if transformer.storage_mode or transformer.storage_dtype or transformer.compute_dtype:
            return False
        if model_pack.derived_strategy == _DERIVED_FP8_STORAGE_NAME:
            return False
        return True

    def _derive_fp8_storage_pack(self, model_pack: ModelPack) -> ModelPack:
        if not self._can_derive_fp8_storage(model_pack):
            return model_pack

        transformer = model_pack.components["transformer"]
        derived_transformer = ModelComponent(
            role=transformer.role,
            path=transformer.path,
            file_format=transformer.file_format,
            storage_dtype="fp8_e4m3fn",
            compute_dtype="bfloat16",
            storage_mode="layerwise",
        )
        derived_components = dict(model_pack.components)
        derived_components["transformer"] = derived_transformer
        base_name = model_pack.base_name or model_pack.name
        return replace(
            model_pack,
            name=f"{base_name}{_DERIVED_FP8_STORAGE_SUFFIX}",
            components=derived_components,
            user_visible=False,
            enabled=False,
            base_name=base_name,
            derived_strategy=_DERIVED_FP8_STORAGE_NAME,
        )

    @staticmethod
    def _split_requested_pack_name(pack_name: str | None) -> tuple[str | None, str | None]:
        requested = (pack_name or "").strip()
        if not requested:
            return None, None
        if requested.lower().endswith(_DERIVED_FP8_STORAGE_SUFFIX.lower()):
            base_name = requested[: -len(_DERIVED_FP8_STORAGE_SUFFIX)].strip()
            if not base_name:
                raise ModelPackValidationError("Derived FP8-storage pack alias is missing a base pack name.")
            return base_name, _DERIVED_FP8_STORAGE_NAME
        return requested, None

    def _resolve_runtime_pack(
        self,
        pack_name: str | None,
        *,
        apply_resource_tier_policy: bool = True,
    ) -> tuple[ModelPack, ModelPack, RuntimeProfile]:
        resource_tier = self.current_resource_tier(refresh=True)
        requested_pack_name, requested_strategy = self._split_requested_pack_name(pack_name)
        base_pack = self._load_base_pack(requested_pack_name)
        completed_pack = self._complete_pack_with_donor(base_pack)
        effective_pack = completed_pack
        if requested_strategy == _DERIVED_FP8_STORAGE_NAME:
            effective_pack = self._derive_fp8_storage_pack(completed_pack)
        elif apply_resource_tier_policy and resource_tier.name == "constrained":
            effective_pack = self._derive_fp8_storage_pack(completed_pack)
        return base_pack, effective_pack, resource_tier

    def resolve_runtime_pack(
        self,
        pack_name: str | None,
        *,
        apply_resource_tier_policy: bool = True,
    ) -> tuple[ModelPack, ModelPack, RuntimeProfile]:
        return self._resolve_runtime_pack(
            pack_name,
            apply_resource_tier_policy=apply_resource_tier_policy,
        )

    def resolve_output_path(self, raw_path: str) -> Path:
        resolved = Path(str(raw_path)).expanduser().resolve()
        outputs_dir = self._settings.paths.outputs_dir.resolve()
        try:
            resolved.relative_to(outputs_dir)
        except ValueError as exc:
            raise ValueError("Image path is outside managed outputs directory.") from exc
        return resolved

    def _session_for_pack(
        self,
        model_pack: ModelPack,
        resource_tier: RuntimeProfile,
    ) -> GenerationSession:
        if self._active_session is None:
            self._active_selected_pack_name = model_pack.base_name or model_pack.name
            self._active_pack_name = model_pack.name
            self._active_session = GenerationSession(
                settings=self._settings,
                model_pack=model_pack,
                resource_tier=resource_tier,
            )
            self._active_backend_name = None
            return self._active_session

        if self._active_pack_name != model_pack.name:
            self._active_session.recycle("Switching active model pack")
            self._active_selected_pack_name = model_pack.base_name or model_pack.name
            self._active_pack_name = model_pack.name
            self._active_session = GenerationSession(
                settings=self._settings,
                model_pack=model_pack,
                resource_tier=resource_tier,
            )
            self._active_backend_name = None
        else:
            self._active_session.set_resource_tier(resource_tier)
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
        procedural_creativity: int = 0,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        effective_procedural_creativity = resolve_procedural_creativity(
            procedural_creativity=procedural_creativity
        )
        safe_owner_id = self.sanitize_owner_id(owner_id)
        cancel_event = Event()
        with self._state_lock:
            base_pack, effective_pack, resource_tier = self._resolve_runtime_pack(pack_name)
            session = self._session_for_pack(effective_pack, resource_tier)
            effective_seed = seed if seed is not None else random.randint(1, 2_147_483_647)
            LOGGER.info(
                "Generate request: owner=%s pack=%s effective_pack=%s tier=%s size=%dx%d seed=%s creative_mode=%s",
                safe_owner_id,
                base_pack.name,
                effective_pack.name,
                resource_tier.name,
                width,
                height,
                effective_seed,
                effective_procedural_creativity,
            )
            self._set_active_client_job_locked(
                safe_owner_id,
                {
                    "job_id": job_id,
                    "kind": "generate",
                    "status": "generating",
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "pack": base_pack.name,
                    "seed": effective_seed,
                    "enhance_prompt": enhance_prompt,
                    "procedural_creativity": effective_procedural_creativity,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            self._client_cancel_events[safe_owner_id] = cancel_event

        output_path = build_output_path(self.owner_output_dir(safe_owner_id))
        try:
            with self._generation_lock:
                if cancel_event.is_set():
                    raise GenerationCancelledError("Generation cancelled.")
                result = session.generate(
                    GenerationRequest(
                        prompt=prompt,
                        width=width,
                        height=height,
                        seed=effective_seed,
                        scheduler_mode=scheduler_mode,
                        enhance_prompt=enhance_prompt,
                        procedural_creativity=effective_procedural_creativity,
                    )
                )
                if cancel_event.is_set():
                    raise GenerationCancelledError("Generation cancelled.")
                with self._state_lock:
                    self._active_backend_name = result.backend

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
                        "model_pack": base_pack.name,
                        "selected_pack": base_pack.name,
                        "effective_pack": effective_pack.name,
                        "derived_strategy": effective_pack.derived_strategy,
                        "fp8_checkpoint": result.fp8_checkpoint,
                        "fp8_fallback_used": result.fp8_fallback_used,
                        "fp8_fallback_reason": result.fp8_fallback_reason,
                        "fp8_runtime_mode": result.fp8_runtime_mode,
                        "fp8_normalized_tensor_count": result.fp8_normalized_tensor_count,
                        "fp8_storage_preserved_tensor_count": result.fp8_storage_preserved_tensor_count,
                        "fp8_promoted_tensor_count": result.fp8_promoted_tensor_count,
                        "duration_ms": result.duration_ms,
                        "seed": result.seed,
                        "scheduler_mode": result.scheduler_mode,
                        "runtime_profile": result.runtime_profile,
                        "resource_tier": result.resource_tier,
                        "execution_mode": result.execution_mode,
                        "procedural_creativity": result.procedural_creativity,
                    },
                )
                if cancel_event.is_set():
                    try:
                        saved_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    raise GenerationCancelledError("Generation cancelled.")
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
                        "model_pack": base_pack.name,
                        "selected_pack": base_pack.name,
                        "effective_pack": effective_pack.name,
                        "derived_strategy": effective_pack.derived_strategy,
                        "resource_tier": result.resource_tier,
                        "procedural_creativity": result.procedural_creativity,
                        **result.telemetry_dict(),
                    },
                )
                image_row = index_image(self._settings, saved_path, owner_id=safe_owner_id)
                image_row["url"] = f"/images/{image_row['filename']}"
                image_row["pack"] = base_pack.name
                image_row["selected_pack"] = base_pack.name
                image_row["effective_pack"] = effective_pack.name
                image_row["derived_strategy"] = effective_pack.derived_strategy
                image_row["duration_ms"] = result.duration_ms
                image_row["seed"] = result.seed
                image_row["scheduler_mode"] = result.scheduler_mode
                image_row["prompt_original"] = result.prompt_original
                image_row["prompt_effective"] = result.prompt_effective
                image_row["prompt_enhanced"] = result.prompt_enhanced
                image_row["runtime_profile"] = result.runtime_profile
                image_row["resource_tier"] = result.resource_tier
                image_row["execution_mode"] = result.execution_mode
                image_row["backend"] = result.backend
                image_row["fp8_fallback_used"] = result.fp8_fallback_used
                image_row["fp8_fallback_reason"] = result.fp8_fallback_reason
                image_row["fp8_runtime_mode"] = result.fp8_runtime_mode
                image_row["procedural_creativity"] = result.procedural_creativity
                image_row["job_id"] = job_id
                LOGGER.info(
                    "Image created: owner=%s file=%s pack=%s effective_pack=%s tier=%s size=%dx%d seed=%s duration_ms=%s creative_mode=%s",
                    safe_owner_id,
                    image_row["filename"],
                    base_pack.name,
                    effective_pack.name,
                    result.resource_tier,
                    width,
                    height,
                    result.seed,
                    result.duration_ms,
                    result.procedural_creativity,
                )
                return image_row
        finally:
            with self._state_lock:
                self._clear_active_client_job_locked(safe_owner_id, job_id=job_id)

    def upscale(
        self,
        owner_id: str,
        filename: str,
        pack_name: str | None = None,
        seed: int | None = None,
        scheduler_mode: str | None = None,
        enhance_prompt: bool = False,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        safe_owner_id = self.sanitize_owner_id(owner_id)
        safe_filename = self.sanitize_filename(filename)
        cancel_event = Event()
        with self._state_lock:
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
            resource_tier = self.current_resource_tier(refresh=True)
            effective_seed = seed if seed is not None else random.randint(1, 2_147_483_647)
            LOGGER.info(
                "Upscale request: owner=%s source=%s pack=%s tier=%s seed=%s",
                safe_owner_id,
                safe_filename,
                model_pack_name,
                resource_tier.name,
                effective_seed,
            )
            self._set_active_client_job_locked(
                safe_owner_id,
                {
                    "job_id": job_id,
                    "kind": "upscale",
                    "status": "upscaling",
                    "filename": safe_filename,
                    "source_filename": safe_filename,
                    "width": source_row.get("width", source_row.get("source_width", 0)) or 0,
                    "height": source_row.get("height", source_row.get("source_height", 0)) or 0,
                    "pack": model_pack_name,
                    "seed": effective_seed,
                    "enhance_prompt": bool(enhance_prompt),
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            self._client_cancel_events[safe_owner_id] = cancel_event

        try:
            with self._generation_lock:
                if cancel_event.is_set():
                    raise GenerationCancelledError("Upscale cancelled.")
                with Image.open(source_path) as source_file:
                    source_image = source_file.convert("RGB")
                source_width, source_height = source_image.size
                with self._state_lock:
                    active_job = self._client_active_jobs.get(safe_owner_id)
                    if active_job is not None:
                        active_job["width"] = source_width * 2
                        active_job["height"] = source_height * 2
                if cancel_event.is_set():
                    raise GenerationCancelledError("Upscale cancelled.")

                result = upscale_with_x2_seed_blend(
                    image=source_image,
                    settings=self._settings,
                    runtime_profile=self._settings.runtime_profile.name,
                    seed=effective_seed,
                    is_cancel_requested=cancel_event.is_set,
                )
                if cancel_event.is_set():
                    raise GenerationCancelledError("Upscale cancelled.")

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
                        "resource_tier": resource_tier.name,
                        "execution_mode": UPSCALE_ENGINE_NAME,
                        "request_enhance_prompt": bool(enhance_prompt),
                        **result.telemetry_dict(),
                    },
                )
                if cancel_event.is_set():
                    try:
                        saved_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    raise GenerationCancelledError("Upscale cancelled.")

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
                        "resource_tier": resource_tier.name,
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
                image_row["resource_tier"] = resource_tier.name
                image_row["execution_mode"] = UPSCALE_ENGINE_NAME
                image_row["upscale_engine"] = UPSCALE_ENGINE_NAME
                image_row["job_id"] = job_id
                LOGGER.info(
                    "Image upscaled: owner=%s source=%s file=%s size=%dx%d seed=%s duration_ms=%s",
                    safe_owner_id,
                    safe_filename,
                    image_row["filename"],
                    final_width,
                    final_height,
                    effective_seed,
                    result.duration_ms,
                )
                with self._state_lock:
                    self._active_backend_name = UPSCALE_ENGINE_NAME
                return image_row
        finally:
            with self._state_lock:
                self._clear_active_client_job_locked(safe_owner_id, job_id=job_id)

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        sanitized = Path(filename).name
        if sanitized != filename:
            raise ValueError("Invalid filename path.")
        return sanitized
