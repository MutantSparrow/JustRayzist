from __future__ import annotations

import json
import logging
import math
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
from app.core.clarity import (
    CLARITY_ENGINE_NAME,
    CLARITY_FS_INTENSITY,
    CLARITY_FS_METHOD,
    CLARITY_FS_TYPE,
    CLARITY_FINAL_UNSHARP_PERCENT,
    CLARITY_FINAL_UNSHARP_RADIUS,
    CLARITY_FINAL_UNSHARP_THRESHOLD,
    CLARITY_UNSHARP_STAGE,
    run_clarity_pipeline,
)
from app.core.backends import SUPPORTED_BACKENDS
from app.core.deblur import (
    DEFAULT_UPSCALE_ENGINE_NAME,
    run_default_upscale_pipeline,
)
from app.core.model_registry import (
    ModelComponent,
    ModelPack,
    ModelPackValidationError,
    discover_model_packs,
    load_model_pack,
    load_model_pack_by_name,
)
from app.core.worker import GenerationRequest, GenerationSession
from app.core.worker.types import LoraSelection, resolve_inference_process, resolve_procedural_creativity
from app.storage import append_generation_metric, build_output_path, save_png_with_metadata
from app.storage.lora_library import (
    DEFAULT_MAX_ACTIVE_LORAS,
    DEFAULT_LORA_WEIGHT,
    MAX_LORA_WEIGHT,
    MIN_LORA_WEIGHT,
    create_lora_draft,
    detect_lora_draft_triggers,
    finalize_deleted_lora,
    finalize_lora_draft,
    get_lora as get_library_lora,
    get_lora_draft as get_library_lora_draft,
    list_loras as list_library_loras,
    mark_lora_deleted,
    normalize_lora_id,
    preview_path_for_lora,
    update_lora as update_library_lora,
)
from app.storage.chat_history import (
    append_chat_exchange,
    chat_messages_for_context,
    clear_chat_history,
    load_chat_history,
)
from app.storage.chat_context import load_chat_context
from app.storage.chat_rag import build_chat_document_context
from app.storage.wildcard_library import (
    create_wildcard as create_library_wildcard,
    delete_wildcard as delete_library_wildcard,
    list_wildcards as list_library_wildcards,
    normalize_wildcard_entry_value,
    update_wildcard as update_library_wildcard,
)
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
    rebuild_gallery,
    rebuild_gallery_color_cache,
    set_image_favorite,
    sync_outputs_to_gallery,
)

LOGGER = logging.getLogger(__name__)
_DONOR_PACK_NAME = "Rayzist_bf16"
_DERIVED_FP8_STORAGE_NAME = "fp8_storage"
_DERIVED_FP8_STORAGE_SUFFIX = "__auto_fp8_storage"
_LORA_CAPABLE_BACKENDS = {"diffusers", "diffusers_zimage", "fp8_zimage"}
_IMG2IMG_MAX_PIXELS = 1_500_000
_IMG2IMG_DIM_MULTIPLE = 32
_IMG2IMG_MIN_DIM = 64
_IMG2IMG_MIN_STRENGTH = 0.05
_IMG2IMG_MAX_STRENGTH = 0.95
_IMG2IMG_DEFAULT_SIMILARITY = 0.80
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
        self._lora_usage_counts: dict[str, int] = {}
        self._pending_lora_deletions: set[str] = set()
        self._gallery_color_cache_rebuild_active = False
        self._gallery_color_cache_rebuild_last_error: str | None = None
        self._gallery_color_cache_rebuild_thread: Thread | None = None

    @staticmethod
    def sanitize_owner_id(owner_id: str) -> str:
        return normalize_owner_id(owner_id)

    @staticmethod
    def sanitize_lora_id(lora_id: str) -> str:
        return normalize_lora_id(lora_id)

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            text = str(value).strip()
        except Exception:
            return None
        if not text:
            return None
        try:
            return int(text)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def normalize_img2img_similarity(similarity: float | int | str | None) -> float:
        if similarity is None:
            return _IMG2IMG_DEFAULT_SIMILARITY
        try:
            value = float(similarity)
        except (TypeError, ValueError) as exc:
            raise ValueError("similarity must be a number between 0.0 and 1.0.") from exc
        if value < 0.0 or value > 1.0:
            raise ValueError("similarity must be between 0.0 and 1.0.")
        return value

    @classmethod
    def similarity_to_refine_strength(cls, similarity: float | int | str | None) -> float:
        normalized = cls.normalize_img2img_similarity(similarity)
        strength = 1.0 - normalized
        return max(_IMG2IMG_MIN_STRENGTH, min(_IMG2IMG_MAX_STRENGTH, strength))

    @staticmethod
    def _normalized_img2img_dimensions(width: int, height: int) -> tuple[int, int]:
        safe_width = max(1, int(width))
        safe_height = max(1, int(height))
        pixel_count = safe_width * safe_height
        scale = 1.0
        if pixel_count > _IMG2IMG_MAX_PIXELS:
            scale = math.sqrt(_IMG2IMG_MAX_PIXELS / float(pixel_count))
        scaled_width = max(1, int(round(safe_width * scale)))
        scaled_height = max(1, int(round(safe_height * scale)))

        def _snap(value: int) -> int:
            if value <= _IMG2IMG_MIN_DIM:
                return _IMG2IMG_MIN_DIM
            snapped = value - (value % _IMG2IMG_DIM_MULTIPLE)
            if snapped < _IMG2IMG_MIN_DIM:
                return _IMG2IMG_MIN_DIM
            return snapped

        scaled_width = _snap(scaled_width)
        scaled_height = _snap(scaled_height)

        while scaled_width * scaled_height > _IMG2IMG_MAX_PIXELS:
            if scaled_width >= scaled_height and scaled_width > _IMG2IMG_MIN_DIM:
                scaled_width = _snap(max(_IMG2IMG_MIN_DIM, scaled_width - _IMG2IMG_DIM_MULTIPLE))
                continue
            if scaled_height > _IMG2IMG_MIN_DIM:
                scaled_height = _snap(max(_IMG2IMG_MIN_DIM, scaled_height - _IMG2IMG_DIM_MULTIPLE))
                continue
            break
        return scaled_width, scaled_height

    @classmethod
    def normalize_img2img_reference_image(
        cls,
        image: Image.Image,
    ) -> tuple[Image.Image, dict[str, int]]:
        if not isinstance(image, Image.Image):
            raise ValueError("reference image must be a PIL.Image.Image instance.")
        source_width, source_height = image.size
        target_width, target_height = cls._normalized_img2img_dimensions(source_width, source_height)
        normalized = image.convert("RGB")
        if normalized.size != (target_width, target_height):
            normalized = normalized.resize((target_width, target_height), Image.Resampling.LANCZOS)
        return normalized, {
            "source_width": int(source_width),
            "source_height": int(source_height),
            "normalized_width": int(target_width),
            "normalized_height": int(target_height),
        }

    def owner_output_dir(self, owner_id: str) -> Path:
        safe_owner = self.sanitize_owner_id(owner_id)
        output_dir = (self._settings.paths.outputs_dir / safe_owner).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @staticmethod
    def _pack_supports_loras(model_pack: ModelPack) -> bool:
        backends = [
            str(name).strip().lower()
            for name in getattr(model_pack, "backend_preference", [])
            if str(name).strip()
        ]
        if not backends:
            backends = ["diffusers"]
        return any(name in _LORA_CAPABLE_BACKENDS for name in backends)

    def lora_capabilities(self, pack_name: str | None = None) -> dict[str, Any]:
        try:
            _base_pack, effective_pack, _resource_tier = self._resolve_runtime_pack(pack_name)
            supported = self._pack_supports_loras(effective_pack)
            active_pack = effective_pack.base_name or effective_pack.name
        except Exception:
            supported = False
            active_pack = None
        return {
            "supported": supported,
            "active_pack": active_pack,
            "max_active": DEFAULT_MAX_ACTIVE_LORAS,
            "min_weight": MIN_LORA_WEIGHT,
            "max_weight": MAX_LORA_WEIGHT,
            "default_weight": DEFAULT_LORA_WEIGHT,
        }

    def wildcard_capabilities(self, pack_name: str | None = None) -> dict[str, Any]:
        active_pack = None
        suggestions_supported = False
        try:
            _base_pack, effective_pack, _resource_tier = self._resolve_runtime_pack(pack_name)
            active_pack = effective_pack.base_name or effective_pack.name
            suggestions_supported = True
            with self._state_lock:
                active_session = self._active_session if self._active_pack_name == effective_pack.name else None
            if active_session is not None:
                try:
                    runtime_status = active_session.runtime_status()
                    suggestions_supported = bool(runtime_status.get("wildcard_suggestions_capable", True))
                except Exception:
                    pass
        except Exception:
            pass
        return {
            "supported": True,
            "active_pack": active_pack,
            "suggestions_supported": suggestions_supported,
        }

    @staticmethod
    def _text_encoder_label_for_pack(model_pack: ModelPack) -> str:
        component = model_pack.components.get("text_encoder") if model_pack.components else None
        if component is not None:
            return str(component.path.name or component.path)
        return model_pack.base_name or model_pack.name

    def chat_capabilities(self, pack_name: str | None = None) -> dict[str, Any]:
        active_pack = None
        encoder = None
        supported = False
        try:
            _base_pack, effective_pack, _resource_tier = self._resolve_runtime_pack(pack_name)
            active_pack = effective_pack.base_name or effective_pack.name
            encoder = self._text_encoder_label_for_pack(effective_pack)
            supported = "text_encoder" in effective_pack.components
            with self._state_lock:
                active_session = self._active_session if self._active_pack_name == effective_pack.name else None
            if active_session is not None:
                try:
                    runtime_status = active_session.runtime_status()
                    supported = bool(runtime_status.get("chat_capable", supported))
                    encoder = str(runtime_status.get("text_encoder_label") or encoder)
                except Exception:
                    pass
        except Exception:
            pass
        return {
            "supported": supported,
            "active_pack": active_pack,
            "encoder": encoder,
        }

    def chat_history(self, *, owner_id: str) -> dict[str, Any]:
        safe_owner_id = self.sanitize_owner_id(owner_id)
        return {
            "status": "ok",
            "history": load_chat_history(self._settings, safe_owner_id),
            "capabilities": self.chat_capabilities(),
        }

    def clear_chat_history(self, *, owner_id: str) -> dict[str, Any]:
        safe_owner_id = self.sanitize_owner_id(owner_id)
        return {
            "status": "ok",
            "history": clear_chat_history(self._settings, safe_owner_id),
            "capabilities": self.chat_capabilities(),
        }

    @staticmethod
    def _chat_context_text(value: Any, *, limit: int = 1200) -> str:
        text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if len(text) > limit:
            return text[:limit].rstrip()
        return text

    @classmethod
    def _chat_context_list(cls, value: Any, *, limit: int = 8) -> list[str]:
        if not isinstance(value, list):
            return []
        items: list[str] = []
        for raw in value:
            if isinstance(raw, dict):
                text = cls._chat_context_text(
                    raw.get("display_name") or raw.get("name") or raw.get("id") or raw.get("token"),
                    limit=120,
                )
            else:
                text = cls._chat_context_text(raw, limit=120)
            if text:
                items.append(text)
            if len(items) >= limit:
                break
        return items

    def _build_chat_app_context(
        self,
        *,
        app_state: Any,
        effective_pack: ModelPack,
        user_message: str,
    ) -> str:
        parts = [load_chat_context(self._settings)]
        doc_context = build_chat_document_context(self._settings, user_message)
        if doc_context:
            parts.append(doc_context)
        state_payload: dict[str, Any] = {
            "active_pack": effective_pack.base_name or effective_pack.name,
            "encoder": self._text_encoder_label_for_pack(effective_pack),
        }
        if isinstance(app_state, dict):
            current_prompt = self._chat_context_text(app_state.get("current_prompt"), limit=1600)
            if current_prompt:
                state_payload["current_prompt_box"] = current_prompt
            for key in (
                "resolution",
                "prompt_enhance",
                "rplus_enabled",
                "creative_mode",
                "reference_image_active",
                "queue_status",
            ):
                value = app_state.get(key)
                if isinstance(value, (str, int, float, bool)) or value is None:
                    state_payload[key] = value
            active_loras = self._chat_context_list(app_state.get("active_loras"), limit=6)
            if active_loras:
                state_payload["active_loras"] = active_loras
            wildcard_examples = self._chat_context_list(app_state.get("wildcard_examples"), limit=8)
            if wildcard_examples:
                state_payload["wildcard_examples"] = wildcard_examples
        parts.append("Current visible app state:\n" + json.dumps(state_payload, indent=2, ensure_ascii=True))
        return "\n\n".join(part.strip() for part in parts if str(part or "").strip())

    def chat(
        self,
        *,
        owner_id: str,
        message: str,
        pack_name: str | None = None,
        app_state: Any = None,
        seed: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 0.75,
    ) -> dict[str, Any]:
        safe_owner_id = self.sanitize_owner_id(owner_id)
        user_message = str(message or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not user_message:
            raise ValueError("Chat message is required.")
        if len(user_message) > 4000:
            raise ValueError("Chat message is too long.")

        history = load_chat_history(self._settings, safe_owner_id)
        context_messages = chat_messages_for_context(history, max_exchanges=10)
        context_messages.append({"role": "user", "content": user_message})
        with self._state_lock:
            _base_pack, effective_pack, resource_tier = self._resolve_runtime_pack(pack_name)
            session = self._session_for_pack(effective_pack, resource_tier)
            effective_seed = int(seed) if seed is not None else random.randint(1, 2_147_483_647)
            app_context = self._build_chat_app_context(
                app_state=app_state,
                effective_pack=effective_pack,
                user_message=user_message,
            )

        assistant_error = False
        try:
            with self._generation_lock:
                result = session.chat(
                    messages=context_messages,
                    app_context=app_context,
                    seed=effective_seed,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                with self._state_lock:
                    try:
                        runtime_status = session.runtime_status()
                        self._active_backend_name = str(runtime_status.get("backend") or self._active_backend_name)
                    except Exception:
                        pass
            assistant_content = str(result.get("response") or "").strip() or "I could not generate a response."
        except Exception as exc:
            assistant_error = True
            assistant_content = f"Rayzist Chat failed: {exc}"
            result = {
                "response": assistant_content,
                "seed": effective_seed,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "encoder": self.chat_capabilities(pack_name).get("encoder"),
            }

        assistant_actions = result.get("actions") if not assistant_error else []
        saved_history = append_chat_exchange(
            self._settings,
            safe_owner_id,
            user_content=user_message,
            assistant_content=assistant_content,
            assistant_error=assistant_error,
            assistant_actions=assistant_actions,
        )
        exchange = (saved_history.get("exchanges") or [])[-1] if saved_history.get("exchanges") else None
        return {
            "status": "error" if assistant_error else "ok",
            "exchange": exchange,
            "history": saved_history,
            "capabilities": self.chat_capabilities(pack_name),
            "seed": result.get("seed"),
            "encoder": result.get("encoder"),
            "actions": assistant_actions,
        }

    def list_loras(self) -> list[dict[str, Any]]:
        return list_library_loras(self._settings)

    def list_wildcards(self) -> list[dict[str, Any]]:
        return list_library_wildcards(self._settings)

    def create_wildcard(self, *, display_name: Any, token: Any, content_text: Any) -> dict[str, Any]:
        return create_library_wildcard(
            self._settings,
            display_name=display_name,
            token=token,
            content_text=content_text,
        )

    def update_wildcard(
        self,
        *,
        wildcard_id: str,
        display_name: Any,
        token: Any,
        content_text: Any,
    ) -> dict[str, Any]:
        return update_library_wildcard(
            self._settings,
            wildcard_id=wildcard_id,
            display_name=display_name,
            token=token,
            content_text=content_text,
        )

    def delete_wildcard(self, wildcard_id: str) -> dict[str, Any]:
        return delete_library_wildcard(self._settings, wildcard_id)

    def suggest_wildcard_entries(
        self,
        *,
        theme: str,
        format_example: str,
        existing_entries: list[str] | None = None,
        seed: int | None = None,
        pack_name: str | None = None,
    ) -> dict[str, Any]:
        normalized_existing = [
            normalize_wildcard_entry_value(item)
            for item in (existing_entries or [])
            if normalize_wildcard_entry_value(item)
        ]
        with self._state_lock:
            _base_pack, effective_pack, resource_tier = self._resolve_runtime_pack(pack_name)
            session = self._session_for_pack(effective_pack, resource_tier)
            effective_seed = int(seed) if seed is not None else random.randint(1, 2_147_483_647)

        with self._generation_lock:
            result = session.suggest_wildcard_entries(
                theme=theme,
                format_example=format_example,
                seed=effective_seed,
                existing_entries=normalized_existing,
                target_count=10,
            )
            with self._state_lock:
                try:
                    self._active_backend_name = str(session.runtime_status().get("backend") or self._active_backend_name)
                except Exception:
                    pass
        return dict(result)

    def create_lora_draft(
        self,
        *,
        filename: str,
        content: bytes | None = None,
        content_file: Any | None = None,
    ) -> dict[str, Any]:
        return create_lora_draft(self._settings, filename=filename, content=content, content_file=content_file)

    def get_lora_draft(self, draft_id: str) -> dict[str, Any] | None:
        return get_library_lora_draft(self._settings, normalize_lora_id(draft_id))

    def detect_lora_draft_triggers(self, draft_id: str) -> dict[str, Any]:
        return detect_lora_draft_triggers(self._settings, normalize_lora_id(draft_id))

    def finalize_lora_draft(
        self,
        *,
        draft_id: str,
        display_name: Any,
        trigger_words: Any,
        preview_content: bytes | None = None,
    ) -> dict[str, Any]:
        return finalize_lora_draft(
            self._settings,
            draft_id=normalize_lora_id(draft_id),
            display_name=display_name,
            trigger_words=trigger_words,
            preview_content=preview_content,
        )

    def update_lora(
        self,
        *,
        lora_id: str,
        display_name: Any,
        trigger_words: Any,
        preview_content: bytes | None = None,
    ) -> dict[str, Any]:
        return update_library_lora(
            self._settings,
            lora_id=normalize_lora_id(lora_id),
            display_name=display_name,
            trigger_words=trigger_words,
            preview_content=preview_content,
        )

    def preview_lora_path(self, lora_id: str) -> Path:
        return preview_path_for_lora(self._settings, self.sanitize_lora_id(lora_id))

    def delete_lora(self, lora_id: str) -> dict[str, Any]:
        normalized = self.sanitize_lora_id(lora_id)
        adapter_cleanup_ids: list[str] = []
        with self._state_lock:
            record = get_library_lora(self._settings, normalized, include_deleted=True)
            if record is None:
                raise FileNotFoundError(f"LoRA not found: {normalized}")
            active_uses = int(self._lora_usage_counts.get(normalized, 0))
            pending_cleanup = active_uses > 0
            mark_lora_deleted(self._settings, normalized, pending_cleanup=pending_cleanup)
            if pending_cleanup:
                self._pending_lora_deletions.add(normalized)
                result = {"id": normalized, "deleted_files": 0, "deletion_state": "deferred"}
            else:
                result = finalize_deleted_lora(self._settings, normalized)
                result["deletion_state"] = "immediate"
                if self._active_session is not None:
                    adapter_cleanup_ids.append(normalized)
        if adapter_cleanup_ids:
            self._drop_session_lora_adapters(adapter_cleanup_ids)
        return result

    def _retain_generation_loras_locked(self, loras: tuple[LoraSelection, ...]) -> tuple[str, ...]:
        retained_ids: list[str] = []
        for lora in loras:
            lora_id = self.sanitize_lora_id(lora.id)
            self._lora_usage_counts[lora_id] = int(self._lora_usage_counts.get(lora_id, 0)) + 1
            retained_ids.append(lora_id)
        return tuple(retained_ids)

    def _release_generation_loras_locked(self, retained_ids: tuple[str, ...]) -> list[str]:
        ready_for_cleanup: list[str] = []
        for lora_id in retained_ids:
            current = int(self._lora_usage_counts.get(lora_id, 0))
            if current <= 1:
                self._lora_usage_counts.pop(lora_id, None)
                if lora_id in self._pending_lora_deletions:
                    self._pending_lora_deletions.discard(lora_id)
                    ready_for_cleanup.append(lora_id)
                continue
            self._lora_usage_counts[lora_id] = current - 1
        return ready_for_cleanup

    def _drop_session_lora_adapters(self, lora_ids: list[str]) -> None:
        if not lora_ids:
            return
        with self._generation_lock:
            with self._state_lock:
                if self._active_session is not None:
                    self._active_session.drop_lora_adapters(lora_ids)

    def _finalize_deferred_lora_cleanup(self, lora_ids: list[str]) -> None:
        if not lora_ids:
            return
        for lora_id in lora_ids:
            try:
                finalize_deleted_lora(self._settings, lora_id)
            except FileNotFoundError:
                continue
        self._drop_session_lora_adapters(lora_ids)

    def _resolve_generation_loras(
        self,
        raw_loras: list[dict[str, Any]] | None,
        *,
        pack_name: str | None = None,
    ) -> tuple[LoraSelection, ...]:
        if not raw_loras:
            return ()
        if len(raw_loras) > DEFAULT_MAX_ACTIVE_LORAS:
            raise ValueError(f"No more than {DEFAULT_MAX_ACTIVE_LORAS} LoRAs can be active at once.")
        if not self.lora_capabilities(pack_name).get("supported", False):
            raise ValueError("The active model pack does not support LoRA adapters.")

        resolved: list[LoraSelection] = []
        seen_ids: set[str] = set()
        for raw_item in raw_loras:
            if not isinstance(raw_item, dict):
                raise ValueError("LoRA selection entries must be objects.")
            lora_id = self.sanitize_lora_id(str(raw_item.get("id") or ""))
            if lora_id in seen_ids:
                raise ValueError(f"Duplicate LoRA selection: {lora_id}")
            weight = float(raw_item.get("weight", DEFAULT_LORA_WEIGHT))
            if weight < MIN_LORA_WEIGHT or weight > MAX_LORA_WEIGHT:
                raise ValueError(
                    f"LoRA weight for '{lora_id}' must be between {MIN_LORA_WEIGHT:.1f} and {MAX_LORA_WEIGHT:.1f}."
                )
            record = get_library_lora(self._settings, lora_id)
            if record is None:
                raise ValueError(f"LoRA not found: {lora_id}")
            resolved.append(
                LoraSelection(
                    id=lora_id,
                    path=Path(str(record["path"])),
                    weight=weight,
                    name=str(record.get("display_name") or lora_id),
                    trigger_words=tuple(str(item) for item in record.get("trigger_words") or []),
                )
            )
            seen_ids.add(lora_id)
        return tuple(resolved)

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
            "lora_capable": backend_status.get("lora_capable", self.lora_capabilities().get("supported", False)),
            "wildcard_suggestions_capable": backend_status.get(
                "wildcard_suggestions_capable",
                self.wildcard_capabilities().get("suggestions_supported", False),
            ),
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
                LOGGER.debug("Gallery color cache rebuild completed: updated=%s", updated)
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
            if active_job.get("kind") in {"generate", "img2img"} and self._active_session is not None:
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

    def rebuild_gallery(self, owner_id: str) -> dict[str, int | str]:
        return rebuild_gallery(self._settings, self.sanitize_owner_id(owner_id))

    def list_images(
        self,
        owner_id: str,
        prompt_query: str | None = None,
        color_filter: str | None = None,
        favorites_only: bool = False,
        limit: int = 100,
        offset: int = 0,
        newest_first: bool = True,
    ) -> list[dict[str, Any]]:
        return list_images(
            settings=self._settings,
            owner_id=self.sanitize_owner_id(owner_id),
            prompt_query=prompt_query,
            color_filter=color_filter,
            favorites_only=favorites_only,
            limit=limit,
            offset=offset,
            newest_first=newest_first,
        )

    def get_image(self, filename: str, owner_id: str) -> dict[str, Any] | None:
        return get_image(self._settings, filename, owner_id=self.sanitize_owner_id(owner_id))

    def set_image_favorite(self, owner_id: str, filename: str, favorite: bool) -> dict[str, Any]:
        return set_image_favorite(
            self._settings,
            filename,
            favorite,
            owner_id=self.sanitize_owner_id(owner_id),
        )

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

    @staticmethod
    def _prefix_payload_keys(prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
        normalized_prefix = str(prefix).strip()
        return {
            f"{normalized_prefix}{key}": value
            for key, value in payload.items()
            if str(key).strip()
        }

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
        steps: int | None = None,
        pack_name: str | None = None,
        seed: int | None = None,
        scheduler_mode: str | None = None,
        inference_process: str | None = "standard",
        enhance_prompt: bool = False,
        procedural_creativity: int = 0,
        rplus_vibrance: float = 0.0,
        rplus_initial_bias_level: float = 0.0,
        loras: list[dict[str, Any]] | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        effective_procedural_creativity = resolve_procedural_creativity(
            procedural_creativity=procedural_creativity
        )
        effective_inference_process = resolve_inference_process(
            inference_process=inference_process
        )
        resolved_loras: tuple[LoraSelection, ...] = ()
        retained_lora_ids: tuple[str, ...] = ()
        safe_owner_id = self.sanitize_owner_id(owner_id)
        cancel_event = Event()
        with self._state_lock:
            resolved_loras = self._resolve_generation_loras(loras, pack_name=pack_name)
            base_pack, effective_pack, resource_tier = self._resolve_runtime_pack(pack_name)
            session = self._session_for_pack(effective_pack, resource_tier)
            effective_seed = seed if seed is not None else random.randint(1, 2_147_483_647)
            retained_lora_ids = self._retain_generation_loras_locked(resolved_loras)
            LOGGER.debug(
                "Generate request: owner=%s pack=%s effective_pack=%s tier=%s size=%dx%d seed=%s creative_mode=%s inference_process=%s",
                safe_owner_id,
                base_pack.name,
                effective_pack.name,
                resource_tier.name,
                width,
                height,
                effective_seed,
                effective_procedural_creativity,
                effective_inference_process,
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
                    "steps": steps,
                    "inference_process": effective_inference_process,
                    "rplus_vibrance": rplus_vibrance,
                    "rplus_initial_bias_level": rplus_initial_bias_level,
                    "lora_count": len(resolved_loras),
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
                        steps=steps,
                        seed=effective_seed,
                        scheduler_mode=scheduler_mode,
                        inference_process=effective_inference_process,
                        enhance_prompt=enhance_prompt,
                        procedural_creativity=effective_procedural_creativity,
                        rplus_vibrance=rplus_vibrance,
                        rplus_initial_bias_level=rplus_initial_bias_level,
                        loras=resolved_loras,
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
                    meta_mode="generate",
                    extra_metadata={
                        "owner_id": safe_owner_id,
                        "prompt_original": result.prompt_original,
                        "prompt_wildcard_resolved": result.prompt_wildcard_resolved or result.prompt_original,
                        "prompt_effective_base": result.prompt_effective_base or result.prompt_effective,
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
                        "inference_process": result.inference_process,
                        "runtime_profile": result.runtime_profile,
                        "resource_tier": result.resource_tier,
                        "execution_mode": result.execution_mode,
                        "procedural_creativity": result.procedural_creativity,
                        "wildcards_json": json.dumps(result.wildcards),
                        "wildcard_count": result.wildcard_count,
                        "loras_json": json.dumps(result.loras),
                        "lora_count": result.lora_count,
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
                        "prompt_wildcard_resolved": result.prompt_wildcard_resolved or result.prompt_original,
                        "prompt_effective_base": result.prompt_effective_base or result.prompt_effective,
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
                        "wildcards": [dict(item) for item in result.wildcards],
                        "wildcard_count": result.wildcard_count,
                        "loras": [dict(item) for item in result.loras],
                        "lora_count": result.lora_count,
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
                image_row["prompt_wildcard_resolved"] = result.prompt_wildcard_resolved or result.prompt_original
                image_row["prompt_effective_base"] = result.prompt_effective_base or result.prompt_effective
                image_row["prompt_effective"] = result.prompt_effective
                image_row["prompt_enhanced"] = result.prompt_enhanced
                image_row["runtime_profile"] = result.runtime_profile
                image_row["resource_tier"] = result.resource_tier
                image_row["execution_mode"] = result.execution_mode
                image_row["backend"] = result.backend
                image_row["inference_process"] = result.inference_process
                image_row["fp8_fallback_used"] = result.fp8_fallback_used
                image_row["fp8_fallback_reason"] = result.fp8_fallback_reason
                image_row["fp8_runtime_mode"] = result.fp8_runtime_mode
                image_row["procedural_creativity"] = result.procedural_creativity
                image_row["wildcards"] = [dict(item) for item in result.wildcards]
                image_row["wildcards_json"] = json.dumps(result.wildcards)
                image_row["wildcard_count"] = result.wildcard_count
                image_row["loras"] = [dict(item) for item in result.loras]
                image_row["lora_count"] = result.lora_count
                image_row["job_id"] = job_id
                LOGGER.debug(
                    "Image created: owner=%s file=%s pack=%s effective_pack=%s tier=%s size=%dx%d seed=%s duration_ms=%s creative_mode=%s loras=%s",
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
                    result.lora_count,
                )
                return image_row
        finally:
            deferred_cleanup_ids: list[str] = []
            with self._state_lock:
                self._clear_active_client_job_locked(safe_owner_id, job_id=job_id)
                deferred_cleanup_ids = self._release_generation_loras_locked(retained_lora_ids)
            self._finalize_deferred_lora_cleanup(deferred_cleanup_ids)

    def img2img(
        self,
        owner_id: str,
        prompt: str,
        image: Image.Image,
        image_filename: str | None = None,
        pack_name: str | None = None,
        seed: int | None = None,
        scheduler_mode: str | None = None,
        enhance_prompt: bool = False,
        similarity: float | int | str | None = None,
        loras: list[dict[str, Any]] | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        normalized_similarity = self.normalize_img2img_similarity(similarity)
        refine_strength = self.similarity_to_refine_strength(normalized_similarity)
        normalized_image, image_info = self.normalize_img2img_reference_image(image)
        reference_filename = Path(str(image_filename or "reference.png")).name.strip() or "reference.png"
        resolved_loras: tuple[LoraSelection, ...] = ()
        retained_lora_ids: tuple[str, ...] = ()
        safe_owner_id = self.sanitize_owner_id(owner_id)
        cancel_event = Event()

        with self._state_lock:
            resolved_loras = self._resolve_generation_loras(loras, pack_name=pack_name)
            base_pack, effective_pack, resource_tier = self._resolve_runtime_pack(pack_name)
            session = self._session_for_pack(effective_pack, resource_tier)
            effective_seed = seed if seed is not None else random.randint(1, 2_147_483_647)
            retained_lora_ids = self._retain_generation_loras_locked(resolved_loras)
            LOGGER.debug(
                "Img2img request: owner=%s pack=%s effective_pack=%s tier=%s ref=%s size=%dx%d seed=%s similarity=%.2f",
                safe_owner_id,
                base_pack.name,
                effective_pack.name,
                resource_tier.name,
                reference_filename,
                image_info["normalized_width"],
                image_info["normalized_height"],
                effective_seed,
                normalized_similarity,
            )
            self._set_active_client_job_locked(
                safe_owner_id,
                {
                    "job_id": job_id,
                    "kind": "img2img",
                    "status": "generating",
                    "prompt": prompt,
                    "width": image_info["normalized_width"],
                    "height": image_info["normalized_height"],
                    "pack": base_pack.name,
                    "seed": effective_seed,
                    "enhance_prompt": enhance_prompt,
                    "similarity": normalized_similarity,
                    "source_filename": reference_filename,
                    "lora_count": len(resolved_loras),
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            self._client_cancel_events[safe_owner_id] = cancel_event

        output_path = build_output_path(self.owner_output_dir(safe_owner_id))
        try:
            with self._generation_lock:
                if cancel_event.is_set():
                    raise GenerationCancelledError("Img2img cancelled.")
                result = session.refine_image(
                    normalized_image,
                    GenerationRequest(
                        prompt=prompt,
                        width=image_info["normalized_width"],
                        height=image_info["normalized_height"],
                        seed=effective_seed,
                        scheduler_mode=scheduler_mode,
                        enhance_prompt=enhance_prompt,
                        procedural_creativity=0,
                        refine_strength=refine_strength,
                        loras=resolved_loras,
                    ),
                )
                if cancel_event.is_set():
                    raise GenerationCancelledError("Img2img cancelled.")
                with self._state_lock:
                    self._active_backend_name = result.backend

                final_width, final_height = result.image.size
                saved_path = save_png_with_metadata(
                    image=result.image,
                    prompt=result.prompt_effective,
                    settings=self._settings,
                    output_path=output_path,
                    meta_mode="img2img",
                    extra_metadata={
                        "owner_id": safe_owner_id,
                        "prompt_original": result.prompt_original,
                        "prompt_wildcard_resolved": result.prompt_wildcard_resolved or result.prompt_original,
                        "prompt_effective_base": result.prompt_effective_base or result.prompt_effective,
                        "prompt_effective": result.prompt_effective,
                        "prompt_enhanced": result.prompt_enhanced,
                        "width": final_width,
                        "height": final_height,
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
                        "mode": "img2img",
                        "refine_strength": result.refine_strength,
                        "refine_pass_count": result.refine_pass_count,
                        "refine_pass1_steps": result.refine_pass1_steps,
                        "refine_pass2_steps": result.refine_pass2_steps,
                        "refine_pass2_strength": result.refine_pass2_strength,
                        "similarity": normalized_similarity,
                        "source_filename": reference_filename,
                        "source_width": image_info["normalized_width"],
                        "source_height": image_info["normalized_height"],
                        "source_original_width": image_info["source_width"],
                        "source_original_height": image_info["source_height"],
                        "wildcards_json": json.dumps(result.wildcards),
                        "wildcard_count": result.wildcard_count,
                        "loras_json": json.dumps(result.loras),
                        "lora_count": result.lora_count,
                    },
                )
                if cancel_event.is_set():
                    try:
                        saved_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    raise GenerationCancelledError("Img2img cancelled.")

                append_generation_metric(
                    settings=self._settings,
                    payload={
                        "mode": "api_img2img",
                        "prompt": result.prompt_effective,
                        "prompt_original": result.prompt_original,
                        "prompt_wildcard_resolved": result.prompt_wildcard_resolved or result.prompt_original,
                        "prompt_effective_base": result.prompt_effective_base or result.prompt_effective,
                        "prompt_effective": result.prompt_effective,
                        "prompt_enhanced": result.prompt_enhanced,
                        "width": final_width,
                        "height": final_height,
                        "output_path": str(saved_path),
                        "owner_id": safe_owner_id,
                        "model_pack": base_pack.name,
                        "selected_pack": base_pack.name,
                        "effective_pack": effective_pack.name,
                        "derived_strategy": effective_pack.derived_strategy,
                        "resource_tier": result.resource_tier,
                        "source_filename": reference_filename,
                        "source_width": image_info["normalized_width"],
                        "source_height": image_info["normalized_height"],
                        "source_original_width": image_info["source_width"],
                        "source_original_height": image_info["source_height"],
                        "similarity": normalized_similarity,
                        "wildcards": [dict(item) for item in result.wildcards],
                        "wildcard_count": result.wildcard_count,
                        "loras": [dict(item) for item in result.loras],
                        "lora_count": result.lora_count,
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
                image_row["prompt_wildcard_resolved"] = result.prompt_wildcard_resolved or result.prompt_original
                image_row["prompt_effective_base"] = result.prompt_effective_base or result.prompt_effective
                image_row["prompt_effective"] = result.prompt_effective
                image_row["prompt_enhanced"] = result.prompt_enhanced
                image_row["runtime_profile"] = result.runtime_profile
                image_row["resource_tier"] = result.resource_tier
                image_row["execution_mode"] = result.execution_mode
                image_row["backend"] = result.backend
                image_row["fp8_fallback_used"] = result.fp8_fallback_used
                image_row["fp8_fallback_reason"] = result.fp8_fallback_reason
                image_row["fp8_runtime_mode"] = result.fp8_runtime_mode
                image_row["refine_pass_count"] = result.refine_pass_count
                image_row["refine_pass1_steps"] = result.refine_pass1_steps
                image_row["refine_pass2_steps"] = result.refine_pass2_steps
                image_row["refine_pass2_strength"] = result.refine_pass2_strength
                image_row["similarity"] = normalized_similarity
                image_row["mode"] = "img2img"
                image_row["source_filename"] = reference_filename
                image_row["source_width"] = image_info["normalized_width"]
                image_row["source_height"] = image_info["normalized_height"]
                image_row["wildcards"] = [dict(item) for item in result.wildcards]
                image_row["wildcards_json"] = json.dumps(result.wildcards)
                image_row["wildcard_count"] = result.wildcard_count
                image_row["loras"] = [dict(item) for item in result.loras]
                image_row["lora_count"] = result.lora_count
                image_row["job_id"] = job_id
                LOGGER.debug(
                    "Image refined: owner=%s file=%s pack=%s effective_pack=%s tier=%s size=%dx%d seed=%s duration_ms=%s similarity=%.2f",
                    safe_owner_id,
                    image_row["filename"],
                    base_pack.name,
                    effective_pack.name,
                    result.resource_tier,
                    final_width,
                    final_height,
                    result.seed,
                    result.duration_ms,
                    normalized_similarity,
                )
                return image_row
        finally:
            deferred_cleanup_ids: list[str] = []
            with self._state_lock:
                self._clear_active_client_job_locked(safe_owner_id, job_id=job_id)
                deferred_cleanup_ids = self._release_generation_loras_locked(retained_lora_ids)
            self._finalize_deferred_lora_cleanup(deferred_cleanup_ids)

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
            source_generation_seed = self._optional_int(source_row.get("seed"))
            preferred_pack = str(source_row.get("model_pack") or "").strip() or None
            base_pack, effective_pack, resource_tier = self._resolve_runtime_pack(pack_name or preferred_pack)
            session = self._session_for_pack(effective_pack, resource_tier)
            model_pack_name = base_pack.name
            effective_seed = seed if seed is not None else random.randint(1, 2_147_483_647)
            LOGGER.debug(
                "Upscale request: owner=%s source=%s engine=%s scale=2 pack=%s tier=%s seed=%s",
                safe_owner_id,
                safe_filename,
                DEFAULT_UPSCALE_ENGINE_NAME,
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
                target_width = max(64, int(source_width) * 2)
                target_height = max(64, int(source_height) * 2)
                with self._state_lock:
                    active_job = self._client_active_jobs.get(safe_owner_id)
                    if active_job is not None:
                        active_job["width"] = target_width
                        active_job["height"] = target_height
                if cancel_event.is_set():
                    raise GenerationCancelledError("Upscale cancelled.")
                result = run_default_upscale_pipeline(
                    image=source_image,
                    settings=self._settings,
                    session=session,
                    profile_name=self._settings.runtime_profile.name,
                    seed=effective_seed,
                    scheduler_mode=scheduler_mode,
                    prompt_text=source_prompt,
                )
                final_image = result.image
                final_width, final_height = final_image.size
                effective_engine = result.engine_name
                prompt_effective = source_prompt
                prompt_enhanced = False
                duration_ms = int(result.duration_ms)
                telemetry = result.telemetry_dict()
                metadata_payload = {
                    "owner_id": safe_owner_id,
                    "mode": "api_upscale",
                    "prompt_original": source_prompt,
                    "prompt_effective": source_prompt,
                    "prompt_enhanced": False,
                    "source_image": str(source_path),
                    "source_filename": safe_filename,
                    "source_width": source_width,
                    "source_height": source_height,
                    "source_generation_seed": source_generation_seed,
                    "width": final_width,
                    "height": final_height,
                    "steps": 0,
                    "guidance_scale": 0.0,
                    "backend": effective_engine,
                    "device": telemetry.get("device"),
                    "model_pack": model_pack_name,
                    "duration_ms": duration_ms,
                    "seed": effective_seed,
                    "scheduler_mode": scheduler_mode or "euler",
                    "runtime_profile": self._settings.runtime_profile.name,
                    "resource_tier": resource_tier.name,
                    "execution_mode": effective_engine,
                    "request_enhance_prompt": bool(enhance_prompt),
                    **telemetry,
                }
                metrics_payload = {
                    "mode": "api_upscale",
                    "prompt": source_prompt,
                    "prompt_original": source_prompt,
                    "prompt_effective": source_prompt,
                    "prompt_enhanced": False,
                    "source_filename": safe_filename,
                    "source_width": source_width,
                    "source_height": source_height,
                    "source_generation_seed": source_generation_seed,
                    "width": final_width,
                    "height": final_height,
                    "model_pack": model_pack_name,
                    "backend": effective_engine,
                    "seed": effective_seed,
                    "scheduler_mode": scheduler_mode or "euler",
                    "resource_tier": resource_tier.name,
                    "request_enhance_prompt": bool(enhance_prompt),
                    **telemetry,
                }
                if cancel_event.is_set():
                    raise GenerationCancelledError("Upscale cancelled.")

                output_path = build_output_path(self.owner_output_dir(safe_owner_id))
                saved_path = save_png_with_metadata(
                    image=final_image,
                    prompt=prompt_effective,
                    settings=self._settings,
                    output_path=output_path,
                    meta_mode="upscale",
                    extra_metadata=metadata_payload,
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
                        **metrics_payload,
                        "output_path": str(saved_path),
                    },
                )
                image_row = index_image(self._settings, saved_path, owner_id=safe_owner_id)
                image_row["url"] = f"/images/{image_row['filename']}"
                image_row["pack"] = model_pack_name
                image_row["duration_ms"] = duration_ms
                image_row["seed"] = effective_seed
                image_row["scheduler_mode"] = scheduler_mode or "euler"
                image_row["prompt_original"] = source_prompt
                image_row["prompt_effective"] = prompt_effective
                image_row["prompt_enhanced"] = prompt_enhanced
                image_row["runtime_profile"] = self._settings.runtime_profile.name
                image_row["resource_tier"] = resource_tier.name
                image_row["execution_mode"] = effective_engine
                image_row["upscale_engine"] = effective_engine
                image_row["upscale_auto_content_mode"] = telemetry.get("upscale_auto_content_mode")
                image_row["job_id"] = job_id
                LOGGER.debug(
                    "Image upscaled: owner=%s source=%s engine=%s scale=2 file=%s size=%dx%d seed=%s duration_ms=%s",
                    safe_owner_id,
                    safe_filename,
                    effective_engine,
                    image_row["filename"],
                    final_width,
                    final_height,
                    effective_seed,
                    duration_ms,
                )
                with self._state_lock:
                    self._active_backend_name = effective_engine
                return image_row
        finally:
            with self._state_lock:
                self._clear_active_client_job_locked(safe_owner_id, job_id=job_id)

    def clarity(
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
            source_generation_seed = self._optional_int(source_row.get("seed"))
            preferred_pack = str(source_row.get("model_pack") or "").strip() or None
            model_pack_name = str(pack_name or preferred_pack or "unknown").strip() or "unknown"
            resource_tier = self.current_resource_tier(refresh=True)
            effective_seed = seed if seed is not None else random.randint(1, 2_147_483_647)
            LOGGER.debug(
                "Clarity request: owner=%s source=%s engine=%s pack=%s tier=%s seed=%s",
                safe_owner_id,
                safe_filename,
                CLARITY_ENGINE_NAME,
                model_pack_name,
                resource_tier.name,
                effective_seed,
            )
            self._set_active_client_job_locked(
                safe_owner_id,
                {
                    "job_id": job_id,
                    "kind": "clarity",
                    "status": "clarifying",
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
                    raise GenerationCancelledError("Clarity cancelled.")
                with Image.open(source_path) as source_file:
                    source_image = source_file.convert("RGB")
                source_width, source_height = source_image.size
                if cancel_event.is_set():
                    raise GenerationCancelledError("Clarity cancelled.")
                clarity_result = run_clarity_pipeline(image=source_image)
                if cancel_event.is_set():
                    raise GenerationCancelledError("Clarity cancelled.")

                final_image = clarity_result.image
                final_width, final_height = final_image.size
                prompt_effective = source_prompt
                prompt_enhanced = False
                duration_ms = int(clarity_result.duration_ms)
                clarity_telemetry = clarity_result.telemetry_dict()
                metadata_payload = {
                    "owner_id": safe_owner_id,
                    "mode": "api_clarity",
                    "prompt_original": source_prompt,
                    "prompt_effective": source_prompt,
                    "prompt_enhanced": False,
                    "source_image": str(source_path),
                    "source_filename": safe_filename,
                    "source_width": source_width,
                    "source_height": source_height,
                    "source_generation_seed": source_generation_seed,
                    "working_width": clarity_result.working_width,
                    "working_height": clarity_result.working_height,
                    "width": final_width,
                    "height": final_height,
                    "steps": 0,
                    "guidance_scale": 0.0,
                    "backend": clarity_result.engine_name,
                    "device": clarity_result.device,
                    "model_pack": model_pack_name,
                    "duration_ms": duration_ms,
                    "seed": effective_seed,
                    "scheduler_mode": scheduler_mode or "euler",
                    "runtime_profile": self._settings.runtime_profile.name,
                    "resource_tier": resource_tier.name,
                    "execution_mode": clarity_result.engine_name,
                    "clarity_engine": clarity_result.engine_name,
                    "request_enhance_prompt": bool(enhance_prompt),
                    "clarity_fs_method": CLARITY_FS_METHOD,
                    "clarity_fs_type": CLARITY_FS_TYPE,
                    "clarity_fs_intensity": CLARITY_FS_INTENSITY,
                    "clarity_unsharp_stage": CLARITY_UNSHARP_STAGE,
                    "clarity_unsharp_radius": CLARITY_FINAL_UNSHARP_RADIUS,
                    "clarity_unsharp_percent": CLARITY_FINAL_UNSHARP_PERCENT,
                    "clarity_unsharp_threshold": CLARITY_FINAL_UNSHARP_THRESHOLD,
                    **clarity_telemetry,
                }
                metrics_payload = {
                    "mode": "api_clarity",
                    "prompt": source_prompt,
                    "prompt_original": source_prompt,
                    "prompt_effective": source_prompt,
                    "prompt_enhanced": False,
                    "source_filename": safe_filename,
                    "source_width": source_width,
                    "source_height": source_height,
                    "source_generation_seed": source_generation_seed,
                    "working_width": clarity_result.working_width,
                    "working_height": clarity_result.working_height,
                    "width": final_width,
                    "height": final_height,
                    "model_pack": model_pack_name,
                    "backend": clarity_result.engine_name,
                    "seed": effective_seed,
                    "scheduler_mode": scheduler_mode or "euler",
                    "resource_tier": resource_tier.name,
                    "clarity_engine": clarity_result.engine_name,
                    "request_enhance_prompt": bool(enhance_prompt),
                    "clarity_fs_method": CLARITY_FS_METHOD,
                    "clarity_fs_type": CLARITY_FS_TYPE,
                    "clarity_fs_intensity": CLARITY_FS_INTENSITY,
                    "clarity_unsharp_stage": CLARITY_UNSHARP_STAGE,
                    "clarity_unsharp_radius": CLARITY_FINAL_UNSHARP_RADIUS,
                    "clarity_unsharp_percent": CLARITY_FINAL_UNSHARP_PERCENT,
                    "clarity_unsharp_threshold": CLARITY_FINAL_UNSHARP_THRESHOLD,
                    **clarity_telemetry,
                }

                output_path = build_output_path(self.owner_output_dir(safe_owner_id))
                saved_path = save_png_with_metadata(
                    image=final_image,
                    prompt=prompt_effective,
                    settings=self._settings,
                    output_path=output_path,
                    meta_mode="clarity",
                    extra_metadata=metadata_payload,
                )
                if cancel_event.is_set():
                    try:
                        saved_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    raise GenerationCancelledError("Clarity cancelled.")

                append_generation_metric(
                    settings=self._settings,
                    payload={
                        **metrics_payload,
                        "output_path": str(saved_path),
                    },
                )
                image_row = index_image(self._settings, saved_path, owner_id=safe_owner_id)
                image_row["url"] = f"/images/{image_row['filename']}"
                image_row["pack"] = model_pack_name
                image_row["duration_ms"] = duration_ms
                image_row["seed"] = effective_seed
                image_row["scheduler_mode"] = scheduler_mode or "euler"
                image_row["prompt_original"] = source_prompt
                image_row["prompt_effective"] = prompt_effective
                image_row["prompt_enhanced"] = prompt_enhanced
                image_row["runtime_profile"] = self._settings.runtime_profile.name
                image_row["resource_tier"] = resource_tier.name
                image_row["execution_mode"] = clarity_result.engine_name
                image_row["clarity_engine"] = clarity_result.engine_name
                image_row["mode"] = "api_clarity"
                image_row["source_filename"] = safe_filename
                image_row["source_width"] = source_width
                image_row["source_height"] = source_height
                image_row["working_width"] = clarity_result.working_width
                image_row["working_height"] = clarity_result.working_height
                image_row["job_id"] = job_id
                image_row.update(clarity_telemetry)
                LOGGER.debug(
                    "Image clarified: owner=%s source=%s engine=%s file=%s size=%dx%d seed=%s duration_ms=%s",
                    safe_owner_id,
                    safe_filename,
                    CLARITY_ENGINE_NAME,
                    image_row["filename"],
                    final_width,
                    final_height,
                    effective_seed,
                    duration_ms,
                )
                with self._state_lock:
                    self._active_backend_name = CLARITY_ENGINE_NAME
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
