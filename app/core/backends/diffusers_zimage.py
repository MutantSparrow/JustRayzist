from __future__ import annotations

import logging
import math
import re
import string
import inspect
import random
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from PIL import Image
from safetensors.torch import load_file as load_safetensors_file

from app.config.profiles import RuntimeProfile
from app.config.settings import AppSettings
from app.core.memory import (
    CudaMemorySnapshot,
    ProcessMemorySnapshot,
    cuda_memory_snapshot,
    now_perf,
    process_memory_snapshot,
)
from app.core.model_registry import ModelPack
from app.core.pipeline_factory import LoadedZImagePipeline, build_zimage_pipeline
from app.core.upscale import upscale_image
from app.core.worker.types import GenerationRequest, LoraSelection, resolve_procedural_creativity

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VramPreflightResult:
    checked: bool
    cleanup_attempted: bool
    passed_before_cleanup: bool | None
    passed_after_cleanup: bool | None
    free_before_bytes: int | None
    free_after_cleanup_bytes: int | None
    threshold_bytes: int | None

    @property
    def passed(self) -> bool:
        return bool(self.passed_after_cleanup)


@dataclass
class GenerationResult:
    image: Any
    seed: int | None
    steps: int
    guidance_scale: float
    scheduler_mode: str
    backend: str
    device: str
    duration_ms: int
    prompt_original: str
    prompt_effective: str
    prompt_enhanced: bool
    prompt_effective_base: str | None = None
    mode: str = "text2img"
    upscale_duration_ms: int | None = None
    refine_duration_ms: int | None = None
    refine_strength: float | None = None
    refine_tile_size: int | None = None
    refine_tile_overlap: int | None = None
    refine_tile_size_requested: int | None = None
    refine_tile_size_effective: int | None = None
    refine_tile_overlap_effective: int | None = None
    refine_fallback_used: bool | None = None
    refine_fallback_attempt_count: int | None = None
    input_image_width: int | None = None
    input_image_height: int | None = None
    cuda_memory_before: CudaMemorySnapshot | None = None
    cuda_memory_after: CudaMemorySnapshot | None = None
    process_memory_before: ProcessMemorySnapshot | None = None
    process_memory_after: ProcessMemorySnapshot | None = None
    runtime_profile: str | None = None
    resource_tier: str | None = None
    execution_mode: str | None = None
    execution_mode_initial: str | None = None
    execution_mode_before_generate: str | None = None
    execution_mode_after_generate: str | None = None
    cuda_total_bytes: int | None = None
    cuda_reserved_after_load_bytes: int | None = None
    cuda_free_before_load_bytes: int | None = None
    cuda_free_after_load_bytes: int | None = None
    cuda_free_before_generate_bytes: int | None = None
    cuda_free_after_generate_bytes: int | None = None
    preflight_checked: bool = False
    preflight_cleanup_attempted: bool = False
    preflight_passed_before_cleanup: bool | None = None
    preflight_passed_after_cleanup: bool | None = None
    preflight_free_before_bytes: int | None = None
    preflight_free_after_cleanup_bytes: int | None = None
    preflight_threshold_bytes: int | None = None
    preflight_fallback_triggered: bool = False
    procedural_latent_enabled: bool = False
    procedural_creativity: int = 0
    procedural_latent_recipe: str | None = None
    procedural_latent_alpha: float | None = None
    procedural_latent_preprocess: str | None = None
    procedural_latent_scheduler_forced: bool = False
    selected_pack: str | None = None
    effective_pack: str | None = None
    fp8_checkpoint: bool = False
    fp8_fallback_used: bool = False
    fp8_fallback_reason: str | None = None
    fp8_runtime_mode: str | None = None
    fp8_normalized_tensor_count: int = 0
    fp8_storage_preserved_tensor_count: int = 0
    fp8_promoted_tensor_count: int = 0
    fp8_normalized_tensor_names: tuple[str, ...] = ()
    loras: tuple[dict[str, Any], ...] = ()
    lora_count: int = 0
    lora_trigger_words: tuple[str, ...] = ()

    def telemetry_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "scheduler_mode": self.scheduler_mode,
            "backend": self.backend,
            "device": self.device,
            "duration_ms": self.duration_ms,
            "prompt_original": self.prompt_original,
            "prompt_effective": self.prompt_effective,
            "prompt_enhanced": self.prompt_enhanced,
            "prompt_effective_base": self.prompt_effective_base,
            "mode": self.mode,
            "upscale_duration_ms": self.upscale_duration_ms,
            "refine_duration_ms": self.refine_duration_ms,
            "refine_strength": self.refine_strength,
            "refine_tile_size": self.refine_tile_size,
            "refine_tile_overlap": self.refine_tile_overlap,
            "refine_tile_size_requested": self.refine_tile_size_requested,
            "refine_tile_size_effective": self.refine_tile_size_effective,
            "refine_tile_overlap_effective": self.refine_tile_overlap_effective,
            "refine_fallback_used": self.refine_fallback_used,
            "refine_fallback_attempt_count": self.refine_fallback_attempt_count,
            "input_image_width": self.input_image_width,
            "input_image_height": self.input_image_height,
            "cuda_memory_before": (
                self.cuda_memory_before.to_dict() if self.cuda_memory_before else None
            ),
            "cuda_memory_after": (
                self.cuda_memory_after.to_dict() if self.cuda_memory_after else None
            ),
            "process_memory_before": (
                self.process_memory_before.to_dict() if self.process_memory_before else None
            ),
            "process_memory_after": (
                self.process_memory_after.to_dict() if self.process_memory_after else None
            ),
            "runtime_profile": self.runtime_profile,
            "resource_tier": self.resource_tier,
            "execution_mode": self.execution_mode,
            "execution_mode_initial": self.execution_mode_initial,
            "execution_mode_before_generate": self.execution_mode_before_generate,
            "execution_mode_after_generate": self.execution_mode_after_generate,
            "cuda_total_bytes": self.cuda_total_bytes,
            "cuda_reserved_after_load_bytes": self.cuda_reserved_after_load_bytes,
            "cuda_free_before_load_bytes": self.cuda_free_before_load_bytes,
            "cuda_free_after_load_bytes": self.cuda_free_after_load_bytes,
            "cuda_free_before_generate_bytes": self.cuda_free_before_generate_bytes,
            "cuda_free_after_generate_bytes": self.cuda_free_after_generate_bytes,
            "preflight_checked": self.preflight_checked,
            "preflight_cleanup_attempted": self.preflight_cleanup_attempted,
            "preflight_passed_before_cleanup": self.preflight_passed_before_cleanup,
            "preflight_passed_after_cleanup": self.preflight_passed_after_cleanup,
            "preflight_free_before_bytes": self.preflight_free_before_bytes,
            "preflight_free_after_cleanup_bytes": self.preflight_free_after_cleanup_bytes,
            "preflight_threshold_bytes": self.preflight_threshold_bytes,
            "preflight_fallback_triggered": self.preflight_fallback_triggered,
            "procedural_latent_enabled": self.procedural_latent_enabled,
            "procedural_creativity": self.procedural_creativity,
            "procedural_latent_recipe": self.procedural_latent_recipe,
            "procedural_latent_alpha": self.procedural_latent_alpha,
            "procedural_latent_preprocess": self.procedural_latent_preprocess,
            "procedural_latent_scheduler_forced": self.procedural_latent_scheduler_forced,
            "selected_pack": self.selected_pack,
            "effective_pack": self.effective_pack,
            "fp8_checkpoint": self.fp8_checkpoint,
            "fp8_fallback_used": self.fp8_fallback_used,
            "fp8_fallback_reason": self.fp8_fallback_reason,
            "fp8_runtime_mode": self.fp8_runtime_mode,
            "fp8_normalized_tensor_count": self.fp8_normalized_tensor_count,
            "fp8_storage_preserved_tensor_count": self.fp8_storage_preserved_tensor_count,
            "fp8_promoted_tensor_count": self.fp8_promoted_tensor_count,
            "fp8_normalized_tensor_names": list(self.fp8_normalized_tensor_names),
            "loras": [dict(item) for item in self.loras],
            "lora_count": self.lora_count,
            "lora_trigger_words": list(self.lora_trigger_words),
        }


class DiffusersZImageBackend:
    BACKEND_NAME = "diffusers_zimage"
    _SCHEDULER_EULER = "euler"
    _SCHEDULER_DPM_MANUAL = "dpm"
    _SCHEDULER_DPM_EXP_LIGHT = "dpm_exp_light"
    _SCHEDULER_DPM_DDIM = "dpm_ddim"
    _HIGH_MODE_AUTO = "auto"
    _HIGH_MODE_FULL_CUDA = "full_cuda"
    _HIGH_MODE_MODEL_OFFLOAD = "model_offload"
    _HIGH_RUNTIME_PRESSURE_RATIO = 0.90
    _HIGH_RUNTIME_PRESSURE_HITS_TO_FALLBACK = 1

    _REFINE_TILE_SNAP = 64
    _REFINE_GRID_DIVISOR_BY_PROFILE: dict[str, int] = {
        "high": 2,
        "balanced": 3,
        "constrained": 4,
    }
    _REFINE_TILE_CAP_BY_PROFILE: dict[str, int] = {
        "high": 896,
        "balanced": 1024,
        "constrained": 896,
    }
    _REFINE_HIGH_FULL_FRAME_MAX_DIM = 1024
    _REFINE_FALLBACK_MIN_TILE_BY_PROFILE: dict[str, int] = {
        "high": 512,
        "balanced": 640,
        "constrained": 512,
    }
    _REFINE_FALLBACK_STEP_FACTORS: tuple[float, ...] = (0.8, 0.64, 0.5)
    _RANDOM_LATENT_STD_EPS = 1e-6
    _PROCEDURAL_LATENT_NOISE_MIX = 0.95
    _PROCEDURAL_LATENT_NOISE_MIX_LEVEL3 = 0.91
    _PROCEDURAL_LATENT_PREPROCESS = "procedural_normalize_mix"
    _PROCEDURAL_LATENT_RECIPE_VERSION = "proc_v4"
    _PROMPT_ENHANCEMENT_MAX_OUTPUT_CHARS = 4000
    _PROMPT_ENHANCEMENT_PRIMARY_MAX_NEW_TOKENS = 120
    _PROMPT_ENHANCEMENT_RETRY_MAX_NEW_TOKENS = 160
    _PROMPT_ENHANCEMENT_PIPELINE_MAX_SEQUENCE_LENGTH = 512
    _PROMPT_ENHANCEMENT_PIPELINE_SAFE_TOKEN_BUDGET = 480
    _PROMPT_STYLE_PATTERNS: tuple[str, ...] = (
        r"\banime\b",
        r"\bmanga\b",
        r"\bcartoon\b",
        r"\billustration\b",
        r"\bcomic(?: book)?\b",
        r"\bcel(?:-| )shad(?:e|ed|ing)\b",
        r"\bpixel art\b",
        r"\boil painting\b",
        r"\bwatercolou?r\b",
        r"\bgouache\b",
        r"\bpastel\b",
        r"\bconcept art\b",
        r"\bmatte painting\b",
        r"\bphotograph(?:y|ic)?\b",
        r"\bphoto(?:realistic)?\b",
        r"\bcinematic\b",
        r"\beditorial\b",
        r"\b3d render\b",
        r"\b3d\b",
        r"\bcgi\b",
        r"\bdigital painting\b",
        r"\bsketch\b",
        r"\bcharcoal\b",
        r"\bink(?: drawing)?\b",
    )
    _PROMPT_PRIORITY_KEYWORDS: tuple[str, ...] = (
        "lighting",
        "light",
        "rim light",
        "backlight",
        "composition",
        "framing",
        "camera",
        "lens",
        "shot on",
        "close-up",
        "portrait",
        "wide shot",
        "environment",
        "background",
        "interior",
        "exterior",
        "studio",
        "sunset",
        "night",
        "material",
        "texture",
        "color palette",
    )

    def __init__(
        self,
        settings: AppSettings,
        model_pack: ModelPack,
        resource_tier: RuntimeProfile | None = None,
    ):
        self._settings = settings
        self._model_pack = model_pack
        self._resource_tier = resource_tier
        self._loaded: LoadedZImagePipeline | None = None
        self._img2img_pipe: Any | None = None
        self._active_scheduler_mode_by_pipe: dict[int, str] = {}
        self._base_scheduler_config_by_pipe: dict[int, dict[str, Any]] = {}
        self._effective_execution_mode: str = "unknown"
        self._initial_execution_mode: str = "unknown"
        self._backend_name: str = self.BACKEND_NAME
        self._cuda_total_bytes: int | None = None
        self._cuda_reserved_after_load_bytes: int | None = None
        self._cuda_free_before_load_bytes: int | None = None
        self._cuda_free_after_load_bytes: int | None = None
        self._fp8_checkpoint = False
        self._fp8_fallback_used = False
        self._fp8_fallback_reason: str | None = None
        self._fp8_runtime_mode: str | None = None
        self._fp8_normalized_tensor_count = 0
        self._fp8_storage_preserved_tensor_count = 0
        self._fp8_promoted_tensor_count = 0
        self._fp8_normalized_tensor_names: tuple[str, ...] = ()
        self._last_preflight: VramPreflightResult = VramPreflightResult(
            checked=False,
            cleanup_attempted=False,
            passed_before_cleanup=None,
            passed_after_cleanup=None,
            free_before_bytes=None,
            free_after_cleanup_bytes=None,
            threshold_bytes=None,
        )
        self._preflight_fallback_triggered = False
        self._high_runtime_fallback_latched = False
        self._high_runtime_pressure_hits = 0

    @staticmethod
    def _interrupt_pipe(pipe: Any | None) -> None:
        if pipe is None:
            return
        if hasattr(pipe, "_interrupt"):
            try:
                setattr(pipe, "_interrupt", True)
            except Exception:
                return

    def cancel_active(self) -> None:
        loaded = self._loaded
        if loaded is not None:
            self._interrupt_pipe(loaded.pipeline)
        self._interrupt_pipe(self._img2img_pipe)

    def _build_pipeline(self) -> LoadedZImagePipeline:
        return build_zimage_pipeline(
            self._model_pack,
            self._resource_profile(),
        )

    def set_resource_tier(self, profile: RuntimeProfile) -> None:
        self._resource_tier = profile

    def _resource_profile(self) -> RuntimeProfile:
        if self._resource_tier is not None:
            return self._resource_tier
        controller = getattr(self._settings, "resource_tier_controller", None)
        if controller is not None:
            try:
                return controller.current()
            except Exception:
                pass
        profile = getattr(self._settings, "resource_tier", None)
        if profile is not None:
            return profile
        return self._settings.runtime_profile

    @staticmethod
    def _snap_up(value: int, multiple: int) -> int:
        if value <= 0:
            return 0
        return int(math.ceil(value / multiple) * multiple)

    @classmethod
    def _build_stepdown_tiles(cls, start_tile: int, min_tile: int) -> list[int]:
        if start_tile <= 0:
            return []
        tiles: list[int] = []
        current = start_tile
        for factor in cls._REFINE_FALLBACK_STEP_FACTORS:
            candidate = cls._snap_up(int(current * factor), cls._REFINE_TILE_SNAP)
            candidate = max(min_tile, candidate)
            if candidate < current and candidate not in tiles:
                tiles.append(candidate)
                current = candidate
        return tiles

    def _default_execution_mode_for_profile(self) -> str:
        profile = self._resource_profile()
        if profile.enable_sequential_offload:
            return "sequential_offload"
        if profile.enable_cpu_offload:
            return self._HIGH_MODE_MODEL_OFFLOAD
        return self._HIGH_MODE_FULL_CUDA

    def _normalize_high_force_mode(self) -> str:
        resource_profile = self._resource_profile()
        if resource_profile.name != "high":
            return self._HIGH_MODE_AUTO

        raw_mode = str(getattr(resource_profile, "high_force_mode", "auto") or "auto")
        normalized = raw_mode.strip().lower()
        if normalized in {
            self._HIGH_MODE_AUTO,
            self._HIGH_MODE_FULL_CUDA,
            self._HIGH_MODE_MODEL_OFFLOAD,
        }:
            return normalized

        LOGGER.warning(
            "Invalid high_force_mode '%s'. Falling back to auto.",
            raw_mode,
        )
        return self._HIGH_MODE_AUTO

    @staticmethod
    def _cuda_capacity_snapshot(torch_module: Any) -> tuple[int | None, int | None]:
        try:
            if not torch_module.cuda.is_available():
                return None, None
            device = int(torch_module.cuda.current_device())
            total_bytes = int(torch_module.cuda.get_device_properties(device).total_memory)
            reserved_bytes = int(torch_module.cuda.memory_reserved(device))
            return total_bytes, reserved_bytes
        except Exception:
            return None, None

    @staticmethod
    def _cuda_free_total_snapshot(torch_module: Any) -> tuple[int | None, int | None]:
        try:
            if not torch_module.cuda.is_available():
                return None, None
            device = int(torch_module.cuda.current_device())
            free_bytes, total_bytes = torch_module.cuda.mem_get_info(device)
            return int(free_bytes), int(total_bytes)
        except Exception:
            return None, None

    @staticmethod
    def _ratio(numerator: int | None, denominator: int | None) -> float | None:
        if numerator is None or denominator is None or denominator <= 0:
            return None
        return float(numerator) / float(denominator)

    def _min_free_vram_threshold_bytes(self) -> int | None:
        threshold_gb = getattr(self._resource_profile(), "min_free_vram_gb", None)
        if threshold_gb is None:
            return None
        try:
            threshold = int(threshold_gb)
        except (TypeError, ValueError):
            return None
        if threshold <= 0:
            return None
        return threshold * 1024 * 1024 * 1024

    def _run_vram_preflight(self, torch_module: Any) -> VramPreflightResult:
        started = now_perf()
        threshold_bytes = self._min_free_vram_threshold_bytes()
        if threshold_bytes is None or not torch_module.cuda.is_available():
            result = VramPreflightResult(
                checked=False,
                cleanup_attempted=False,
                passed_before_cleanup=None,
                passed_after_cleanup=None,
                free_before_bytes=None,
                free_after_cleanup_bytes=None,
                threshold_bytes=threshold_bytes,
            )
            self._last_preflight = result
            return result

        free_before, _ = self._cuda_free_total_snapshot(torch_module)
        passed_before = free_before is not None and free_before >= threshold_bytes
        cleanup_attempted = False
        free_after = free_before
        passed_after = passed_before

        if not passed_before:
            cleanup_attempted = True
            self._clear_cuda_cache(torch_module)
            free_after, _ = self._cuda_free_total_snapshot(torch_module)
            passed_after = free_after is not None and free_after >= threshold_bytes

        result = VramPreflightResult(
            checked=True,
            cleanup_attempted=cleanup_attempted,
            passed_before_cleanup=passed_before,
            passed_after_cleanup=passed_after,
            free_before_bytes=free_before,
            free_after_cleanup_bytes=free_after,
            threshold_bytes=threshold_bytes,
        )
        elapsed_ms = int((now_perf() - started) * 1000)
        if elapsed_ms > 50:
            LOGGER.debug(
                "VRAM preflight exceeded budget: elapsed_ms=%s threshold_bytes=%s free_before_bytes=%s free_after_cleanup_bytes=%s cleanup_attempted=%s",
                elapsed_ms,
                threshold_bytes,
                free_before,
                free_after,
                cleanup_attempted,
            )
        self._last_preflight = result
        return result

    def _apply_pipe_execution_mode(self, pipe: Any, mode: str) -> str:
        if mode == "sequential_offload":
            if hasattr(pipe, "enable_sequential_cpu_offload"):
                pipe.enable_sequential_cpu_offload()
                return "sequential_offload"
            LOGGER.warning(
                "Requested sequential_offload mode but pipeline does not support it; using model_offload."
            )
            mode = self._HIGH_MODE_MODEL_OFFLOAD

        if mode == self._HIGH_MODE_MODEL_OFFLOAD:
            if hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
                return self._HIGH_MODE_MODEL_OFFLOAD
            LOGGER.warning(
                "Requested model_offload mode but pipeline does not support it; falling back to full_cuda."
            )
            mode = self._HIGH_MODE_FULL_CUDA

        if mode == self._HIGH_MODE_FULL_CUDA:
            pipe.to("cuda")
            return self._HIGH_MODE_FULL_CUDA

        return mode

    def _resolve_high_startup_mode(
        self,
        *,
        total_bytes: int | None,
        reserved_bytes: int | None,
        preflight: VramPreflightResult,
    ) -> str:
        force_mode = self._normalize_high_force_mode()
        if force_mode in {self._HIGH_MODE_FULL_CUDA, self._HIGH_MODE_MODEL_OFFLOAD}:
            return force_mode

        if force_mode != self._HIGH_MODE_AUTO:
            return self._HIGH_MODE_MODEL_OFFLOAD

        if preflight.checked and not preflight.passed:
            return self._HIGH_MODE_MODEL_OFFLOAD

        if total_bytes is None or reserved_bytes is None:
            LOGGER.warning(
                "Unable to read CUDA capacity snapshot for high profile startup; using model_offload."
            )
            return self._HIGH_MODE_MODEL_OFFLOAD

        threshold = float(
            getattr(self._resource_profile(), "high_reserved_vram_ratio_threshold", 0.82)
        )
        threshold = max(0.50, min(0.98, threshold))
        ratio = self._ratio(reserved_bytes, total_bytes) or 1.0
        if ratio > threshold:
            return self._HIGH_MODE_MODEL_OFFLOAD
        return self._HIGH_MODE_FULL_CUDA

    def _initialize_execution_mode(self, pipe: Any) -> None:
        import torch

        if not torch.cuda.is_available():
            self._effective_execution_mode = "cpu"
            self._initial_execution_mode = "cpu"
            self._cuda_total_bytes = None
            self._cuda_reserved_after_load_bytes = None
            self._cuda_free_before_load_bytes = None
            self._cuda_free_after_load_bytes = None
            self._last_preflight = VramPreflightResult(
                checked=False,
                cleanup_attempted=False,
                passed_before_cleanup=None,
                passed_after_cleanup=None,
                free_before_bytes=None,
                free_after_cleanup_bytes=None,
                threshold_bytes=self._min_free_vram_threshold_bytes(),
            )
            return

        free_before, total_from_meminfo = self._cuda_free_total_snapshot(torch)
        total_bytes, reserved_bytes = self._cuda_capacity_snapshot(torch)
        profile_name = self._resource_profile().name
        selected_mode = self._default_execution_mode_for_profile()
        startup_ratio = self._ratio(reserved_bytes, total_bytes)
        startup_preflight = self._run_vram_preflight(torch)

        if profile_name == "high":
            selected_mode = self._resolve_high_startup_mode(
                total_bytes=total_bytes,
                reserved_bytes=reserved_bytes,
                preflight=startup_preflight,
            )
            if selected_mode == self._HIGH_MODE_MODEL_OFFLOAD:
                try:
                    self._apply_pipe_execution_mode(pipe, self._HIGH_MODE_MODEL_OFFLOAD)
                    self._clear_cuda_cache(torch)
                except Exception as exc:
                    LOGGER.warning(
                        "High profile startup offload selection failed, keeping full_cuda. %s",
                        exc,
                    )
                    selected_mode = self._HIGH_MODE_FULL_CUDA
            self._high_runtime_fallback_latched = selected_mode == self._HIGH_MODE_MODEL_OFFLOAD

        self._effective_execution_mode = selected_mode
        self._initial_execution_mode = selected_mode
        total_after, reserved_after = self._cuda_capacity_snapshot(torch)
        free_after, total_after_meminfo = self._cuda_free_total_snapshot(torch)
        self._cuda_total_bytes = (
            total_after
            if total_after is not None
            else total_from_meminfo
            if total_from_meminfo is not None
            else total_after_meminfo
            if total_after_meminfo is not None
            else total_bytes
        )
        self._cuda_reserved_after_load_bytes = (
            reserved_after if reserved_after is not None else reserved_bytes
        )
        self._cuda_free_before_load_bytes = free_before
        self._cuda_free_after_load_bytes = free_after
        after_ratio = self._ratio(self._cuda_reserved_after_load_bytes, self._cuda_total_bytes)
        LOGGER.debug(
            "Execution mode initialized: profile=%s mode=%s startup_reserved_ratio=%s reserved_after_load_ratio=%s total_vram_bytes=%s reserved_after_load_bytes=%s free_before_load_bytes=%s free_after_load_bytes=%s",
            profile_name,
            self._effective_execution_mode,
            f"{startup_ratio:.3f}" if startup_ratio is not None else "n/a",
            f"{after_ratio:.3f}" if after_ratio is not None else "n/a",
            self._cuda_total_bytes if self._cuda_total_bytes is not None else "n/a",
            self._cuda_reserved_after_load_bytes
            if self._cuda_reserved_after_load_bytes is not None
            else "n/a",
            self._cuda_free_before_load_bytes if self._cuda_free_before_load_bytes is not None else "n/a",
            self._cuda_free_after_load_bytes if self._cuda_free_after_load_bytes is not None else "n/a",
        )

    def _apply_high_runtime_fallback_if_needed(
        self,
        *,
        post_mem: CudaMemorySnapshot | None,
        torch_module: Any,
    ) -> None:
        if self._resource_profile().name != "high":
            return
        if self._effective_execution_mode != self._HIGH_MODE_FULL_CUDA:
            return
        if self._high_runtime_fallback_latched:
            return
        if not torch_module.cuda.is_available():
            return

        total_bytes = self._cuda_total_bytes
        if total_bytes is None:
            total_bytes, _ = self._cuda_capacity_snapshot(torch_module)
            self._cuda_total_bytes = total_bytes
        if total_bytes is None or total_bytes <= 0:
            return

        reserved_bytes = post_mem.reserved_bytes if post_mem is not None else None
        if reserved_bytes is None:
            _, reserved_bytes = self._cuda_capacity_snapshot(torch_module)
        ratio = self._ratio(reserved_bytes, total_bytes)
        if ratio is None:
            return

        if ratio >= self._HIGH_RUNTIME_PRESSURE_RATIO:
            self._high_runtime_pressure_hits += 1
        else:
            self._high_runtime_pressure_hits = 0
            return

        if self._high_runtime_pressure_hits < self._HIGH_RUNTIME_PRESSURE_HITS_TO_FALLBACK:
            return

        loaded = self._loaded
        if loaded is None:
            return

        try:
            applied_mode = self._apply_pipe_execution_mode(
                loaded.pipeline, self._HIGH_MODE_MODEL_OFFLOAD
            )
            if applied_mode != self._HIGH_MODE_MODEL_OFFLOAD:
                return

            if self._img2img_pipe is not None:
                self._apply_pipe_execution_mode(self._img2img_pipe, self._HIGH_MODE_MODEL_OFFLOAD)

            self._effective_execution_mode = self._HIGH_MODE_MODEL_OFFLOAD
            self._high_runtime_fallback_latched = True
            self._high_runtime_pressure_hits = 0
            self._clear_cuda_cache(torch_module)
            total_after, reserved_after = self._cuda_capacity_snapshot(torch_module)
            if total_after is not None:
                self._cuda_total_bytes = total_after
            if reserved_after is not None:
                self._cuda_reserved_after_load_bytes = reserved_after
            LOGGER.warning(
                "High profile runtime fallback activated: switched to model_offload after high CUDA pressure (reserved_ratio=%.3f).",
                ratio,
            )
        except Exception as exc:
            LOGGER.warning(
                "High profile runtime fallback failed; keeping full_cuda. %s",
                exc,
            )

    @staticmethod
    def _normalize_scheduler_mode(mode: str | None) -> str | None:
        if mode is None:
            return None
        normalized = str(mode).strip().lower()
        if normalized not in {"euler", "dpm"}:
            raise ValueError("Unsupported scheduler_mode. Use 'euler' or 'dpm'.")
        return normalized

    @classmethod
    def _resolve_generate_scheduler_mode(
        cls,
        *,
        requested_mode: str | None,
        procedural_creativity: int,
    ) -> tuple[str, bool]:
        if requested_mode is not None:
            return requested_mode, False
        if procedural_creativity <= 1:
            return cls._SCHEDULER_EULER, False
        if procedural_creativity == 2:
            return cls._SCHEDULER_DPM_EXP_LIGHT, False
        return cls._SCHEDULER_DPM_DDIM, False

    def _apply_scheduler_mode(self, pipe: Any, mode: str) -> str:
        pipe_id = id(pipe)
        if self._active_scheduler_mode_by_pipe.get(pipe_id) == mode:
            return mode

        from diffusers import DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler

        base_config = self._base_scheduler_config_by_pipe.setdefault(pipe_id, dict(pipe.scheduler.config))
        requested_mode = mode

        def _build_euler_scheduler() -> Any:
            shift = base_config.get("shift")
            if shift is None:
                shift = base_config.get("flow_shift", 3.0)
            if base_config.get("use_dynamic_shifting", False):
                LOGGER.warning(
                    "FlowMatch Euler dynamic shifting is incompatible with the current Z-Image pipeline; "
                    "forcing static shift for %s.",
                    getattr(self._model_pack, "name", "<unknown-pack>"),
                )
            return FlowMatchEulerDiscreteScheduler.from_config(
                base_config,
                shift=shift,
                use_dynamic_shifting=False,
            )

        if mode in {
            self._SCHEDULER_DPM_MANUAL,
            self._SCHEDULER_DPM_EXP_LIGHT,
            self._SCHEDULER_DPM_DDIM,
        }:
            base_flow_shift = base_config.get("flow_shift")
            if base_flow_shift is None:
                base_flow_shift = base_config.get("shift", 3.0)
            dpm_kwargs: dict[str, Any] = {
                "algorithm_type": "sde-dpmsolver++",
                "solver_order": 2,
                "prediction_type": "flow_prediction",
                "use_flow_sigmas": True,
            }
            if mode == self._SCHEDULER_DPM_EXP_LIGHT:
                dpm_kwargs["flow_shift"] = min(float(base_flow_shift), 1.75)
                dpm_kwargs["use_dynamic_shifting"] = True
                dpm_kwargs["time_shift_type"] = "exponential"
                dpm_kwargs["timestep_spacing"] = "leading"
            elif mode == self._SCHEDULER_DPM_DDIM:
                dpm_kwargs["flow_shift"] = max(float(base_flow_shift), 3.0)
                dpm_kwargs["use_dynamic_shifting"] = True
                dpm_kwargs["time_shift_type"] = "exponential"
                dpm_kwargs["timestep_spacing"] = "leading"
            else:
                dpm_kwargs["flow_shift"] = float(base_flow_shift)
                dpm_kwargs["use_dynamic_shifting"] = True
                dpm_kwargs["time_shift_type"] = "exponential"
                dpm_kwargs["timestep_spacing"] = "leading"
            scheduler = DPMSolverMultistepScheduler.from_config(base_config, **dpm_kwargs)
            is_img2img_pipe = "img2img" in pipe.__class__.__name__.lower()
            if is_img2img_pipe and not hasattr(scheduler, "scale_noise"):
                LOGGER.warning(
                    "Requested DPM scheduler is incompatible with %s (missing scale_noise). Falling back to Euler.",
                    pipe.__class__.__name__,
                )
                scheduler = _build_euler_scheduler()
                mode = "euler"
        else:
            scheduler = _build_euler_scheduler()

        pipe.scheduler = scheduler
        self._active_scheduler_mode_by_pipe[pipe_id] = mode
        if mode != requested_mode:
            LOGGER.debug("Scheduler mode requested=%s applied=%s", requested_mode, mode)
        else:
            LOGGER.debug("Scheduler mode set to %s", mode)
        return mode

    @staticmethod
    def _is_scheduler_incompatibility_error(exc: Exception) -> bool:
        message = str(exc).lower()
        indicators = (
            "scheduler",
            "timestep",
            "sigma",
            "flowmatch",
            "divide by zero",
            "nan",
            "non-finite",
            "invalid value",
        )
        return any(indicator in message for indicator in indicators)

    def _resolve_scheduler_retry_mode(self, current_mode: str, exc: Exception) -> str | None:
        if current_mode != self._SCHEDULER_EULER:
            return None
        if not self._is_scheduler_incompatibility_error(exc):
            return None
        return self._SCHEDULER_DPM_EXP_LIGHT

    def _ensure_loaded(self) -> LoadedZImagePipeline:
        if self._loaded is None:
            LOGGER.info("Loading pipeline for model pack '%s'", self._model_pack.name)
            self._loaded = self._build_pipeline()
            self._backend_name = self._loaded.backend_name
            self._fp8_checkpoint = self._loaded.fp8_checkpoint
            self._fp8_fallback_used = self._loaded.fp8_fallback_used
            self._fp8_fallback_reason = self._loaded.fp8_fallback_reason
            self._fp8_runtime_mode = self._loaded.fp8_runtime_mode
            self._fp8_normalized_tensor_count = self._loaded.fp8_normalized_tensor_count
            self._fp8_storage_preserved_tensor_count = self._loaded.fp8_storage_preserved_tensor_count
            self._fp8_promoted_tensor_count = self._loaded.fp8_promoted_tensor_count
            self._fp8_normalized_tensor_names = self._loaded.fp8_normalized_tensor_names
            self._initialize_execution_mode(self._loaded.pipeline)
        return self._loaded

    def runtime_status(self) -> dict[str, Any]:
        selected_pack = getattr(self._model_pack, "base_name", None) or getattr(self._model_pack, "name", None)
        lora_capable = True
        if self._loaded is not None:
            lora_capable = self._pipe_supports_lora(self._loaded.pipeline)
        return {
            "backend": self._backend_name,
            "execution_mode": self._effective_execution_mode,
            "execution_mode_initial": self._initial_execution_mode,
            "selected_pack": selected_pack,
            "effective_pack": getattr(self._model_pack, "name", None),
            "fp8_checkpoint": self._fp8_checkpoint,
            "fp8_fallback_used": self._fp8_fallback_used,
            "fp8_fallback_reason": self._fp8_fallback_reason,
            "fp8_runtime_mode": self._fp8_runtime_mode,
            "fp8_normalized_tensor_count": self._fp8_normalized_tensor_count,
            "fp8_storage_preserved_tensor_count": self._fp8_storage_preserved_tensor_count,
            "fp8_promoted_tensor_count": self._fp8_promoted_tensor_count,
            "fp8_normalized_tensor_names": list(self._fp8_normalized_tensor_names),
            "lora_capable": lora_capable,
        }

    def _ensure_img2img_pipe(self) -> Any:
        if self._img2img_pipe is not None:
            return self._img2img_pipe

        try:
            from diffusers import ZImageImg2ImgPipeline
        except ImportError as exc:
            raise ImportError(
                "Installed diffusers build is missing ZImageImg2ImgPipeline. "
                "Run RunMeFirst.bat to repair the environment."
            ) from exc

        loaded = self._ensure_loaded()
        base_pipe = loaded.pipeline
        pipe = ZImageImg2ImgPipeline(
            scheduler=base_pipe.scheduler,
            vae=base_pipe.vae,
            text_encoder=base_pipe.text_encoder,
            tokenizer=base_pipe.tokenizer,
            transformer=base_pipe.transformer,
        )
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)

        if loaded.device == "cuda":
            target_mode = self._effective_execution_mode
            if target_mode in {"unknown", "cpu"}:
                target_mode = self._default_execution_mode_for_profile()
            applied_mode = self._apply_pipe_execution_mode(pipe, target_mode)
            if applied_mode != target_mode:
                LOGGER.debug(
                    "Img2img pipe execution mode adjusted from %s to %s.",
                    target_mode,
                    applied_mode,
                )
        self._img2img_pipe = pipe
        return self._img2img_pipe

    @staticmethod
    def _clear_cuda_cache(torch_module: Any) -> None:
        try:
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _build_generator(torch_module: Any, device: str, seed: int | None) -> Any:
        if seed is None:
            return None
        return torch_module.Generator(device=device).manual_seed(int(seed))

    @staticmethod
    def _resolve_latent_spatial_shape(pipe: Any, width: int, height: int) -> tuple[int, int]:
        vae_scale_factor = int(getattr(pipe, "vae_scale_factor", 8) or 8)
        base = max(1, vae_scale_factor * 2)
        latent_height = 2 * (int(height) // base)
        latent_width = 2 * (int(width) // base)
        if latent_height <= 0 or latent_width <= 0:
            raise ValueError(
                f"Unsupported output size for latent injection: {width}x{height} "
                f"(vae_scale_factor={vae_scale_factor})."
            )
        return latent_height, latent_width

    @classmethod
    def _normalize_and_mix_latent(
        cls,
        *,
        latent_tensor: Any,
        seed: int | None,
        torch_module: Any,
        noise_mix: float,
        preprocess: str,
    ) -> tuple[Any, float, str]:
        # Match diffusion start-noise expectations while preserving latent-file influence.
        latent = latent_tensor.to(dtype=torch_module.float32, device="cpu")
        latent_mean = latent.mean(dim=(1, 2, 3), keepdim=True)
        latent_std = latent.std(dim=(1, 2, 3), keepdim=True, unbiased=False).clamp_min(
            cls._RANDOM_LATENT_STD_EPS
        )
        normalized = (latent - latent_mean) / latent_std

        noise_generator = cls._build_generator(torch_module, "cpu", seed)
        noise = torch_module.randn(
            normalized.shape,
            generator=noise_generator,
            dtype=normalized.dtype,
            device=normalized.device,
        )

        noise_mix = max(0.0, min(1.0, float(noise_mix)))
        latent_mix = 1.0 - noise_mix
        mixed = (latent_mix * normalized) + (noise_mix * noise)
        mixed_mean = mixed.mean(dim=(1, 2, 3), keepdim=True)
        mixed_std = mixed.std(dim=(1, 2, 3), keepdim=True, unbiased=False).clamp_min(
            cls._RANDOM_LATENT_STD_EPS
        )
        mixed = (mixed - mixed_mean) / mixed_std
        return mixed.contiguous(), noise_mix, preprocess

    @staticmethod
    def _sample_scalar(
        torch_module: Any,
        *,
        generator: Any,
        low: float,
        high: float,
    ) -> float:
        value = torch_module.rand((1,), generator=generator, dtype=torch_module.float32).item()
        return float(low + ((high - low) * value))

    @staticmethod
    def _sample_int(
        torch_module: Any,
        *,
        generator: Any,
        low: int,
        high: int,
    ) -> int:
        return int(
            torch_module.randint(
                int(low),
                int(high) + 1,
                (1,),
                generator=generator,
                dtype=torch_module.int64,
            ).item()
        )

    @staticmethod
    def _blur2d(torch_module: Any, tensor: Any, kernel_size: int) -> Any:
        kernel = max(1, int(kernel_size))
        if kernel <= 1:
            return tensor
        if kernel % 2 == 0:
            kernel += 1
        pad = kernel // 2
        padded = torch_module.nn.functional.pad(tensor, (pad, pad, pad, pad), mode="reflect")
        return torch_module.nn.functional.avg_pool2d(padded, kernel_size=kernel, stride=1)

    @classmethod
    def _build_grain_stack(
        cls,
        *,
        torch_module: Any,
        generator: Any,
        channels: int,
        target_height: int,
        target_width: int,
        base_divisor: int,
        blur_kernel: int,
        micro_freq_low: float,
        micro_freq_high: float,
    ) -> Any:
        base_h = max(6, target_height // max(2, int(base_divisor)))
        base_w = max(6, target_width // max(2, int(base_divisor)))
        base = torch_module.nn.functional.interpolate(
            torch_module.randn((1, channels, base_h, base_w), generator=generator, dtype=torch_module.float32),
            size=(target_height, target_width),
            mode="bicubic",
            align_corners=False,
        )
        fine = torch_module.randn(
            (1, channels, target_height, target_width),
            generator=generator,
            dtype=torch_module.float32,
        )
        fine = fine - cls._blur2d(torch_module, fine, blur_kernel + 2)
        fine = cls._blur2d(torch_module, fine, blur_kernel)

        yy = torch_module.linspace(-1.0, 1.0, target_height, dtype=torch_module.float32).view(1, 1, target_height, 1)
        xx = torch_module.linspace(-1.0, 1.0, target_width, dtype=torch_module.float32).view(1, 1, 1, target_width)
        theta = torch_module.rand((1, channels, 1, 1), generator=generator, dtype=torch_module.float32) * (
            2.0 * math.pi
        )
        freq = micro_freq_low + (
            torch_module.rand((1, channels, 1, 1), generator=generator, dtype=torch_module.float32)
            * (micro_freq_high - micro_freq_low)
        )
        phase = torch_module.rand((1, channels, 1, 1), generator=generator, dtype=torch_module.float32) * (
            2.0 * math.pi
        )
        micro = torch_module.sin((freq * ((torch_module.cos(theta) * xx) + (torch_module.sin(theta) * yy))) + phase)
        micro = micro + 0.5 * torch_module.cos((freq * 0.73 * xx) - phase)
        return (0.55 * base) + (0.30 * fine) + (0.15 * micro)

    @classmethod
    def _build_level3_shape_bands(
        cls,
        *,
        torch_module: Any,
        generator: Any,
        target_height: int,
        target_width: int,
        warped_x: Any,
        warped_y: Any,
        shape_count: int,
        clean_noise_gain: float,
        soft_blur_kernel: int,
    ) -> tuple[Any, int]:
        band_fields = torch_module.zeros((1, 3, target_height, target_width), dtype=torch_module.float32)
        applied_shapes = 0
        families = ("circle", "diamond", "box", "triangle", "wedge")
        dominant_band = cls._sample_int(torch_module, generator=generator, low=0, high=2)
        for shape_index in range(int(shape_count)):
            family = families[shape_index % len(families)]
            band_index = (
                dominant_band
                if shape_index < max(1, int(shape_count) - 1)
                else cls._sample_int(torch_module, generator=generator, low=0, high=2)
            )
            center_x = cls._sample_scalar(torch_module, generator=generator, low=-0.60, high=0.60)
            center_y = cls._sample_scalar(torch_module, generator=generator, low=-0.60, high=0.60)
            scale_x = cls._sample_scalar(torch_module, generator=generator, low=0.42, high=1.05)
            scale_y = cls._sample_scalar(torch_module, generator=generator, low=0.42, high=1.05)
            rotation = cls._sample_scalar(torch_module, generator=generator, low=0.0, high=2.0 * math.pi)
            amplitude = cls._sample_scalar(torch_module, generator=generator, low=-1.45, high=1.45)
            if band_index == dominant_band:
                amplitude *= 1.18
            is_hard = (shape_index % 3) != 1
            hardness = cls._sample_scalar(
                torch_module,
                generator=generator,
                low=8.0 if is_hard else 2.2,
                high=16.0 if is_hard else 4.6,
            )

            local_x = warped_x - center_x
            local_y = warped_y - center_y
            cos_r = math.cos(rotation)
            sin_r = math.sin(rotation)
            rot_x = ((cos_r * local_x) + (sin_r * local_y)) / max(scale_x, 1e-3)
            rot_y = ((-sin_r * local_x) + (cos_r * local_y)) / max(scale_y, 1e-3)

            if family == "circle":
                raw_field = 1.0 - torch_module.sqrt((rot_x**2) + (rot_y**2) + 1e-6)
            elif family == "diamond":
                raw_field = 1.0 - (rot_x.abs() + rot_y.abs())
            elif family == "box":
                raw_field = 1.0 - torch_module.maximum(rot_x.abs(), rot_y.abs())
            elif family == "triangle":
                ax, ay = 0.0, -1.0
                bx, by = -1.0, 1.0
                cx, cy = 1.0, 1.0
                denom = ((by - cy) * (ax - cx)) + ((cx - bx) * (ay - cy))
                w1 = (((by - cy) * (rot_x - cx)) + ((cx - bx) * (rot_y - cy))) / denom
                w2 = (((cy - ay) * (rot_x - cx)) + ((ax - cx) * (rot_y - cy))) / denom
                w3 = 1.0 - w1 - w2
                raw_field = torch_module.minimum(torch_module.minimum(w1, w2), w3)
            else:
                radius = torch_module.sqrt((rot_x**2) + (rot_y**2) + 1e-6)
                angle = torch_module.atan2(rot_y, rot_x)
                center_angle = cls._sample_scalar(torch_module, generator=generator, low=-math.pi, high=math.pi)
                width = cls._sample_scalar(torch_module, generator=generator, low=0.45, high=1.35)
                angular_delta = torch_module.atan2(
                    torch_module.sin(angle - center_angle),
                    torch_module.cos(angle - center_angle),
                ).abs()
                raw_field = torch_module.minimum(1.0 - radius, width - angular_delta)

            shaped = torch_module.tanh(hardness * raw_field)
            if not is_hard:
                shaped = cls._blur2d(torch_module, shaped, soft_blur_kernel)
            contribution = amplitude * shaped
            current_band = band_fields[:, band_index : band_index + 1, :, :]
            if is_hard or band_index == dominant_band:
                band_fields[:, band_index : band_index + 1, :, :] = torch_module.where(
                    contribution.abs() >= current_band.abs(),
                    contribution,
                    current_band,
                )
            else:
                band_fields[:, band_index : band_index + 1, :, :] = current_band + contribution
            applied_shapes += 1

        clean_noise = torch_module.randn(
            (1, 3, target_height, target_width),
            generator=generator,
            dtype=torch_module.float32,
        )
        clean_noise = cls._blur2d(torch_module, clean_noise, soft_blur_kernel)
        band_fields = band_fields + (clean_noise_gain * clean_noise)
        band_fields = (0.92 * band_fields) + (0.08 * cls._blur2d(torch_module, band_fields, soft_blur_kernel + 2))
        return band_fields, applied_shapes

    @staticmethod
    def _quantize_level3_band_fields(
        torch_module: Any,
        *,
        band_fields: Any,
    ) -> tuple[Any, int]:
        palette = torch_module.tensor(
            [
                [1.00, 1.00, 1.00],    # white
                [-1.00, -1.00, -1.00], # black
                [1.00, -0.88, -0.92],  # red
                [-0.94, 1.00, 1.00],   # cyan
                [1.00, 0.96, -0.90],   # yellow
                [-0.92, -0.88, 1.00],  # blue
            ],
            dtype=torch_module.float32,
        )
        normalized = torch_module.tanh(1.55 * band_fields[:, :3, :, :])
        palette_view = palette.view(1, 6, 3, 1, 1)
        distances = ((normalized.unsqueeze(1) - palette_view) ** 2).sum(dim=2)
        labels = distances.argmin(dim=1)
        one_hot = torch_module.nn.functional.one_hot(labels, num_classes=6).permute(0, 3, 1, 2).to(
            dtype=torch_module.float32
        )
        quantized = torch_module.einsum("bkhw,kc->bchw", one_hot, palette)
        return quantized.contiguous(), int(palette.shape[0])

    @classmethod
    def _build_rigid_primitive_mask(
        cls,
        *,
        torch_module: Any,
        generator: Any,
        coords_x: Any,
        coords_y: Any,
        family: str,
        center_x: float,
        center_y: float,
        scale_x: float,
        scale_y: float,
        rotation: float,
    ) -> Any:
        local_x = coords_x - center_x
        local_y = coords_y - center_y
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        rot_x = ((cos_r * local_x) + (sin_r * local_y)) / max(scale_x, 1e-3)
        rot_y = ((-sin_r * local_x) + (cos_r * local_y)) / max(scale_y, 1e-3)

        if family == "circle":
            raw_field = 1.0 - torch_module.sqrt((rot_x**2) + (rot_y**2) + 1e-6)
        elif family == "box":
            raw_field = 1.0 - torch_module.maximum(rot_x.abs(), rot_y.abs())
        elif family == "triangle":
            ax, ay = 0.0, -1.0
            bx, by = -1.0, 1.0
            cx, cy = 1.0, 1.0
            denom = ((by - cy) * (ax - cx)) + ((cx - bx) * (ay - cy))
            w1 = (((by - cy) * (rot_x - cx)) + ((cx - bx) * (rot_y - cy))) / denom
            w2 = (((cy - ay) * (rot_x - cx)) + ((ax - cx) * (rot_y - cy))) / denom
            w3 = 1.0 - w1 - w2
            raw_field = torch_module.minimum(torch_module.minimum(w1, w2), w3)
        elif family == "trapeze":
            top_half = cls._sample_scalar(torch_module, generator=generator, low=0.28, high=0.72)
            y_norm = (rot_y + 1.0) * 0.5
            half_width = top_half + ((1.0 - top_half) * y_norm)
            raw_field = torch_module.minimum(1.0 - rot_y.abs(), half_width - rot_x.abs())
        elif family == "lozenge":
            raw_field = 1.0 - ((0.78 * rot_x.abs()) + (1.22 * rot_y.abs()))
        else:
            raise ValueError(f"Unsupported primitive family: {family}")

        return (raw_field >= 0.0).to(dtype=torch_module.float32)

    @classmethod
    def _apply_level3_final_overlays(
        cls,
        *,
        torch_module: Any,
        generator: Any,
        latent: Any,
        coords_x: Any,
        coords_y: Any,
        overlay_count: int,
    ) -> tuple[Any, int]:
        overlaid = latent.clone()
        families = ("circle", "box", "triangle", "trapeze", "lozenge")
        applied = 0

        for index in range(int(overlay_count)):
            family = families[index % len(families)]
            center_x = cls._sample_scalar(torch_module, generator=generator, low=-0.58, high=0.58)
            center_y = cls._sample_scalar(torch_module, generator=generator, low=-0.58, high=0.58)
            scale_x = cls._sample_scalar(torch_module, generator=generator, low=0.30, high=0.82)
            scale_y = cls._sample_scalar(torch_module, generator=generator, low=0.30, high=0.82)
            rotation = cls._sample_scalar(torch_module, generator=generator, low=0.0, high=2.0 * math.pi)
            alpha = cls._sample_scalar(torch_module, generator=generator, low=0.70, high=0.85)
            polarity = 1.0 if cls._sample_int(torch_module, generator=generator, low=0, high=1) == 1 else -1.0
            mask = cls._build_rigid_primitive_mask(
                torch_module=torch_module,
                generator=generator,
                coords_x=coords_x,
                coords_y=coords_y,
                family=family,
                center_x=center_x,
                center_y=center_y,
                scale_x=scale_x,
                scale_y=scale_y,
                rotation=rotation,
            )
            if mask.max().item() <= 0.0:
                continue

            alpha_mask = alpha * mask
            target_value = polarity * torch_module.ones_like(overlaid)
            overlaid = (1.0 - alpha_mask) * overlaid + (alpha_mask * target_value)
            applied += 1

        return overlaid.contiguous(), applied

    @classmethod
    def _procedural_profile(cls, creativity: int) -> dict[str, Any]:
        profiles: dict[int, dict[str, Any]] = {
            1: {
                "warp": (0.04, 0.12),
                "low_gain": (0.26, 0.46),
                "mid_gain": (0.16, 0.34),
                "stripe_gain": (0.03, 0.09),
                "stripe_freq": (2.6, 6.4),
                "flow_gain": (0.03, 0.09),
                "flow_freq": (3.0, 7.2),
                "blob_count": (1, 2),
                "blob_sigma": (0.18, 0.40),
                "blob_amplitude": (-0.45, 0.45),
                "blob_gain": (0.03, 0.10),
                "channel_gain": (0.94, 1.06),
                "channel_weight": (0.92, 1.08),
                "mono_mix": (0.84, 0.95),
                "sign_flip_prob": 0.05,
                "blotch_gain": (0.0, 0.03),
                "ring_gain": (0.0, 0.02),
                "split_gain": (0.0, 0.02),
                "low_div": 16,
                "mid_div": 4,
                "warp_div": 9,
                "blotch_div": 3,
                "band_div": 6,
                "band_gain": (0.0, 0.02),
                "band_sharpness": (1.1, 1.8),
                "grain_base_div": 10,
                "grain_blur_kernel": 3,
                "grain_freq": (8.0, 16.0),
                "grain_gain": (0.24, 0.42),
                "contrast_power": (1.02, 1.12),
            },
            2: {
                "warp": (0.10, 0.24),
                "low_gain": (0.55, 0.90),
                "mid_gain": (0.18, 0.40),
                "stripe_gain": (0.08, 0.24),
                "stripe_freq": (1.5, 6.0),
                "flow_gain": (0.08, 0.20),
                "flow_freq": (2.0, 6.5),
                "blob_count": (3, 5),
                "blob_sigma": (0.22, 0.60),
                "blob_amplitude": (-1.0, 1.0),
                "blob_gain": (0.14, 0.40),
                "channel_gain": (0.72, 1.32),
                "channel_weight": (0.70, 1.35),
                "mono_mix": (0.10, 0.25),
                "sign_flip_prob": 0.35,
                "blotch_gain": (0.18, 0.42),
                "ring_gain": (0.0, 0.10),
                "split_gain": (0.0, 0.12),
                "low_div": 12,
                "mid_div": 6,
                "warp_div": 10,
                "blotch_div": 6,
                "band_div": 9,
                "band_gain": (0.06, 0.22),
                "band_sharpness": (1.6, 2.6),
                "cell_count": (10, 18),
                "cell_gain": (0.06, 0.16),
                "cell_amplitude": (-0.85, 0.85),
                "contrast_power": (1.10, 1.26),
            },
            3: {
                "warp": (0.24, 0.42),
                "low_gain": (0.75, 1.15),
                "mid_gain": (0.22, 0.55),
                "stripe_gain": (0.16, 0.42),
                "stripe_freq": (1.0, 7.5),
                "flow_gain": (0.14, 0.34),
                "flow_freq": (2.0, 7.5),
                "blob_count": (4, 8),
                "blob_sigma": (0.28, 0.82),
                "blob_amplitude": (-1.4, 1.4),
                "blob_gain": (0.28, 0.70),
                "channel_gain": (0.45, 1.85),
                "channel_weight": (0.45, 1.75),
                "mono_mix": (0.00, 0.06),
                "sign_flip_prob": 0.72,
                "blotch_gain": (0.46, 0.96),
                "ring_gain": (0.18, 0.45),
                "split_gain": (0.28, 0.68),
                "low_div": 9,
                "mid_div": 7,
                "warp_div": 12,
                "blotch_div": 12,
                "band_div": 14,
                "band_gain": (0.28, 0.70),
                "band_sharpness": (2.4, 4.0),
                "shape_count": (1, 3),
                "shape_gain": (1.02, 1.55),
                "shape_noise_gain": (0.00, 0.02),
                "shape_blur_kernel": (3, 3),
                "overlay_count": (1, 3),
                "contrast_power": (1.35, 1.72),
            },
        }
        if creativity not in profiles:
            raise ValueError("procedural_creativity must be between 1 and 3.")
        return profiles[creativity]

    @classmethod
    def _build_procedural_latent_tensor(
        cls,
        *,
        expected_channels: int,
        target_height: int,
        target_width: int,
        seed: int | None,
        creativity: int,
        torch_module: Any,
    ) -> tuple[Any, str]:
        profile = cls._procedural_profile(creativity)
        base_seed = int(seed if seed is not None else random.SystemRandom().randrange(1, 2_147_483_647))
        generator = cls._build_generator(torch_module, "cpu", base_seed ^ 0x5A17C0DE)
        if generator is None:
            generator = cls._build_generator(
                torch_module,
                "cpu",
                random.SystemRandom().randrange(1, 2_147_483_647),
            )

        latent_shape = (1, int(expected_channels), int(target_height), int(target_width))
        latent = torch_module.zeros(latent_shape, dtype=torch_module.float32, device="cpu")

        yy = torch_module.linspace(-1.0, 1.0, target_height, dtype=torch_module.float32).view(1, 1, target_height, 1)
        xx = torch_module.linspace(-1.0, 1.0, target_width, dtype=torch_module.float32).view(1, 1, 1, target_width)

        low_h = max(4, target_height // int(profile["low_div"]))
        low_w = max(4, target_width // int(profile["low_div"]))
        mid_h = max(8, target_height // int(profile["mid_div"]))
        mid_w = max(8, target_width // int(profile["mid_div"]))
        warp_h = max(6, target_height // int(profile["warp_div"]))
        warp_w = max(6, target_width // int(profile["warp_div"]))

        low_noise = torch_module.nn.functional.interpolate(
            torch_module.randn((1, expected_channels, low_h, low_w), generator=generator, dtype=torch_module.float32),
            size=(target_height, target_width),
            mode="bicubic",
            align_corners=False,
        )
        mid_noise = torch_module.nn.functional.interpolate(
            torch_module.randn((1, expected_channels, mid_h, mid_w), generator=generator, dtype=torch_module.float32),
            size=(target_height, target_width),
            mode="bicubic",
            align_corners=False,
        )
        warp_x = torch_module.nn.functional.interpolate(
            torch_module.randn((1, 1, warp_h, warp_w), generator=generator, dtype=torch_module.float32),
            size=(target_height, target_width),
            mode="bicubic",
            align_corners=False,
        ).tanh()
        warp_y = torch_module.nn.functional.interpolate(
            torch_module.randn((1, 1, warp_h, warp_w), generator=generator, dtype=torch_module.float32),
            size=(target_height, target_width),
            mode="bicubic",
            align_corners=False,
        ).tanh()

        warp_strength = cls._sample_scalar(
            torch_module,
            generator=generator,
            low=profile["warp"][0],
            high=profile["warp"][1],
        )
        warped_x = xx + (warp_strength * warp_x)
        warped_y = yy + (warp_strength * warp_y)

        low_gain = cls._sample_scalar(
            torch_module,
            generator=generator,
            low=profile["low_gain"][0],
            high=profile["low_gain"][1],
        )
        mid_gain = cls._sample_scalar(
            torch_module,
            generator=generator,
            low=profile["mid_gain"][0],
            high=profile["mid_gain"][1],
        )
        latent = latent + (low_gain * low_noise) + (mid_gain * mid_noise)

        theta = torch_module.rand((1, expected_channels, 1, 1), generator=generator, dtype=torch_module.float32) * (
            2.0 * math.pi
        )
        stripe_freq = (
            profile["stripe_freq"][0]
            + (
                torch_module.rand((1, expected_channels, 1, 1), generator=generator, dtype=torch_module.float32)
                * (profile["stripe_freq"][1] - profile["stripe_freq"][0])
            )
        )
        stripe_phase = torch_module.rand(
            (1, expected_channels, 1, 1),
            generator=generator,
            dtype=torch_module.float32,
        ) * (2.0 * math.pi)
        stripe_wave = torch_module.sin(
            (stripe_freq * ((torch_module.cos(theta) * warped_x) + (torch_module.sin(theta) * warped_y)))
            + stripe_phase
        )
        stripe_gain = cls._sample_scalar(
            torch_module,
            generator=generator,
            low=profile["stripe_gain"][0],
            high=profile["stripe_gain"][1],
        )
        latent = latent + (stripe_gain * stripe_wave)

        flow_freq_x = cls._sample_scalar(
            torch_module,
            generator=generator,
            low=profile["flow_freq"][0],
            high=profile["flow_freq"][1],
        )
        flow_freq_y = cls._sample_scalar(
            torch_module,
            generator=generator,
            low=profile["flow_freq"][0],
            high=profile["flow_freq"][1],
        )
        flow_phase_x = cls._sample_scalar(torch_module, generator=generator, low=0.0, high=2.0 * math.pi)
        flow_phase_y = cls._sample_scalar(torch_module, generator=generator, low=0.0, high=2.0 * math.pi)
        channel_phase = torch_module.rand(
            (1, expected_channels, 1, 1),
            generator=generator,
            dtype=torch_module.float32,
        ) * (2.0 * math.pi)
        flow_wave = torch_module.sin((flow_freq_x * warped_x) + flow_phase_x + channel_phase) + torch_module.cos(
            (flow_freq_y * warped_y) + flow_phase_y - channel_phase
        )
        flow_gain = cls._sample_scalar(
            torch_module,
            generator=generator,
            low=profile["flow_gain"][0],
            high=profile["flow_gain"][1],
        )
        latent = latent + (flow_gain * flow_wave)

        blob_count = cls._sample_int(
            torch_module,
            generator=generator,
            low=profile["blob_count"][0],
            high=profile["blob_count"][1],
        )
        blob_field = torch_module.zeros((1, 1, target_height, target_width), dtype=torch_module.float32)
        for _ in range(blob_count):
            center_x = cls._sample_scalar(torch_module, generator=generator, low=-0.80, high=0.80)
            center_y = cls._sample_scalar(torch_module, generator=generator, low=-0.80, high=0.80)
            sigma = cls._sample_scalar(
                torch_module,
                generator=generator,
                low=profile["blob_sigma"][0],
                high=profile["blob_sigma"][1],
            )
            amplitude = cls._sample_scalar(
                torch_module,
                generator=generator,
                low=profile["blob_amplitude"][0],
                high=profile["blob_amplitude"][1],
            )
            distance_sq = ((warped_x - center_x) ** 2) + ((warped_y - center_y) ** 2)
            blob_field = blob_field + (amplitude * torch_module.exp(-distance_sq / max(0.02, 2.0 * sigma * sigma)))
        blob_gain = cls._sample_scalar(
            torch_module,
            generator=generator,
            low=profile["blob_gain"][0],
            high=profile["blob_gain"][1],
        )
        blob_channel_weight = (
            profile["channel_weight"][0]
            + (
                torch_module.rand((1, expected_channels, 1, 1), generator=generator, dtype=torch_module.float32)
                * (profile["channel_weight"][1] - profile["channel_weight"][0])
            )
        )
        latent = latent + (blob_gain * blob_field * blob_channel_weight)

        blotch_h = max(4, target_height // int(profile["blotch_div"]))
        blotch_w = max(4, target_width // int(profile["blotch_div"]))
        blotch_field = torch_module.nn.functional.interpolate(
            torch_module.randn((1, 1, blotch_h, blotch_w), generator=generator, dtype=torch_module.float32),
            size=(target_height, target_width),
            mode="bicubic",
            align_corners=False,
        ).tanh()
        blotch_gain = cls._sample_scalar(
            torch_module,
            generator=generator,
            low=profile["blotch_gain"][0],
            high=profile["blotch_gain"][1],
        )
        latent = latent + (blotch_gain * blotch_field * blob_channel_weight)

        cell_count = 0
        grain_gain = 0.0
        shape_count = 0
        shape_gain = 0.0
        quantized_palette_size = 0
        overlay_count = 0

        if creativity == 1:
            grain_stack = cls._build_grain_stack(
                torch_module=torch_module,
                generator=generator,
                channels=expected_channels,
                target_height=target_height,
                target_width=target_width,
                base_divisor=int(profile["grain_base_div"]),
                blur_kernel=int(profile["grain_blur_kernel"]),
                micro_freq_low=profile["grain_freq"][0],
                micro_freq_high=profile["grain_freq"][1],
            )
            grain_gain = cls._sample_scalar(
                torch_module,
                generator=generator,
                low=profile["grain_gain"][0],
                high=profile["grain_gain"][1],
            )
            latent = latent + (grain_gain * grain_stack)
            band_fields = torch_module.zeros((1, 3, target_height, target_width), dtype=torch_module.float32)
        elif creativity == 3:
            sampled_shape_count = cls._sample_int(
                torch_module,
                generator=generator,
                low=profile["shape_count"][0],
                high=profile["shape_count"][1],
            )
            shape_noise_gain = cls._sample_scalar(
                torch_module,
                generator=generator,
                low=profile["shape_noise_gain"][0],
                high=profile["shape_noise_gain"][1],
            )
            shape_blur_kernel = cls._sample_int(
                torch_module,
                generator=generator,
                low=profile["shape_blur_kernel"][0],
                high=profile["shape_blur_kernel"][1],
            )
            band_fields, shape_count = cls._build_level3_shape_bands(
                torch_module=torch_module,
                generator=generator,
                target_height=target_height,
                target_width=target_width,
                warped_x=warped_x,
                warped_y=warped_y,
                shape_count=sampled_shape_count,
                clean_noise_gain=shape_noise_gain,
                soft_blur_kernel=shape_blur_kernel,
            )
            band_fields, quantized_palette_size = cls._quantize_level3_band_fields(torch_module, band_fields=band_fields)
            sampled_overlay_count = cls._sample_int(
                torch_module,
                generator=generator,
                low=profile["overlay_count"][0],
                high=profile["overlay_count"][1],
            )
            overlay_count = sampled_overlay_count
            shape_gain = cls._sample_scalar(
                torch_module,
                generator=generator,
                low=profile["shape_gain"][0],
                high=profile["shape_gain"][1],
            )
        else:
            cell_count = cls._sample_int(
                torch_module,
                generator=generator,
                low=profile["cell_count"][0],
                high=profile["cell_count"][1],
            )
            cell_distances: list[Any] = []
            cell_values: list[float] = []
            for _ in range(cell_count):
                center_x = cls._sample_scalar(torch_module, generator=generator, low=-0.95, high=0.95)
                center_y = cls._sample_scalar(torch_module, generator=generator, low=-0.95, high=0.95)
                cell_distances.append(((warped_x - center_x) ** 2) + ((warped_y - center_y) ** 2))
                cell_values.append(
                    cls._sample_scalar(
                        torch_module,
                        generator=generator,
                        low=profile["cell_amplitude"][0],
                        high=profile["cell_amplitude"][1],
                    )
                )
            cell_distance_stack = torch_module.cat(cell_distances, dim=1)
            nearest_cell = cell_distance_stack.argmin(dim=1, keepdim=True)
            cell_value_tensor = torch_module.tensor(cell_values, dtype=torch_module.float32).view(1, cell_count, 1, 1)
            cell_field = torch_module.gather(
                cell_value_tensor.expand(1, cell_count, target_height, target_width),
                dim=1,
                index=nearest_cell,
            )
            cell_gain = cls._sample_scalar(
                torch_module,
                generator=generator,
                low=profile["cell_gain"][0],
                high=profile["cell_gain"][1],
            )
            latent = latent + (cell_gain * cell_field)

            band_h = max(3, target_height // int(profile["band_div"]))
            band_w = max(3, target_width // int(profile["band_div"]))
            band_fields = torch_module.nn.functional.interpolate(
                torch_module.randn((1, 3, band_h, band_w), generator=generator, dtype=torch_module.float32),
                size=(target_height, target_width),
                mode="bicubic",
                align_corners=False,
            )
            band_sharpness = cls._sample_scalar(
                torch_module,
                generator=generator,
                low=profile["band_sharpness"][0],
                high=profile["band_sharpness"][1],
            )
            band_fields = torch_module.tanh(band_sharpness * band_fields)

        band_gain = cls._sample_scalar(
            torch_module,
            generator=generator,
            low=profile["band_gain"][0],
            high=profile["band_gain"][1],
        )
        band_slices = ((0, 5), (5, 10), (10, expected_channels))
        for band_index, (start, end) in enumerate(band_slices):
            if start >= expected_channels:
                break
            end = min(end, expected_channels)
            latent[:, start:end, :, :] = (
                latent[:, start:end, :, :]
                + (
                    (band_gain if creativity != 3 else shape_gain)
                    * band_fields[:, band_index : band_index + 1, :, :]
                )
            )

        ring_gain = cls._sample_scalar(
            torch_module,
            generator=generator,
            low=profile["ring_gain"][0],
            high=profile["ring_gain"][1],
        )
        if ring_gain > 0.0:
            ring_count = 2 if creativity >= 3 else 1
            ring_field = torch_module.zeros((1, 1, target_height, target_width), dtype=torch_module.float32)
            for _ in range(ring_count):
                center_x = cls._sample_scalar(torch_module, generator=generator, low=-0.75, high=0.75)
                center_y = cls._sample_scalar(torch_module, generator=generator, low=-0.75, high=0.75)
                sigma = cls._sample_scalar(torch_module, generator=generator, low=0.22, high=0.72)
                ring_freq = cls._sample_scalar(
                    torch_module,
                    generator=generator,
                    low=1.2,
                    high=3.4 if creativity >= 3 else 2.4,
                )
                radius = torch_module.sqrt(((warped_x - center_x) ** 2) + ((warped_y - center_y) ** 2))
                envelope = torch_module.exp(-(radius**2) / max(0.04, 2.0 * sigma * sigma))
                ring_field = ring_field + (torch_module.cos((radius * ring_freq * math.pi)) * envelope)
            latent = latent + (ring_gain * ring_field * blob_channel_weight)

        split_gain = cls._sample_scalar(
            torch_module,
            generator=generator,
            low=profile["split_gain"][0],
            high=profile["split_gain"][1],
        )
        if split_gain > 0.0:
            split_theta = cls._sample_scalar(torch_module, generator=generator, low=0.0, high=2.0 * math.pi)
            split_bias = cls._sample_scalar(torch_module, generator=generator, low=-0.35, high=0.35)
            split_sharpness = 4.0 if creativity >= 3 else 2.2
            split_plane = (
                (torch_module.cos(torch_module.tensor(split_theta)) * warped_x)
                + (torch_module.sin(torch_module.tensor(split_theta)) * warped_y)
                + split_bias
            )
            split_field = torch_module.tanh(split_sharpness * split_plane)
            latent = latent + (split_gain * split_field * blob_channel_weight)

        channel_gain = (
            profile["channel_gain"][0]
            + (
                torch_module.rand((1, expected_channels, 1, 1), generator=generator, dtype=torch_module.float32)
                * (profile["channel_gain"][1] - profile["channel_gain"][0])
            )
        )
        sign_flip_prob = float(profile["sign_flip_prob"])
        channel_sign = torch_module.where(
            torch_module.rand((1, expected_channels, 1, 1), generator=generator, dtype=torch_module.float32)
            < sign_flip_prob,
            -torch_module.ones((1, expected_channels, 1, 1), dtype=torch_module.float32),
            torch_module.ones((1, expected_channels, 1, 1), dtype=torch_module.float32),
        )
        latent = latent * channel_gain * channel_sign

        mono_mix = cls._sample_scalar(
            torch_module,
            generator=generator,
            low=profile["mono_mix"][0],
            high=profile["mono_mix"][1],
        )
        if mono_mix > 0.0:
            mono_field = latent.mean(dim=1, keepdim=True).expand_as(latent)
            latent = (mono_mix * mono_field) + ((1.0 - mono_mix) * latent)

        contrast_power = cls._sample_scalar(
            torch_module,
            generator=generator,
            low=profile["contrast_power"][0],
            high=profile["contrast_power"][1],
        )
        latent = torch_module.sign(latent) * torch_module.pow(latent.abs().clamp_min(1e-6), contrast_power)

        if creativity == 3 and overlay_count > 0:
            latent, overlay_count = cls._apply_level3_final_overlays(
                torch_module=torch_module,
                generator=generator,
                latent=latent,
                coords_x=xx,
                coords_y=yy,
                overlay_count=overlay_count,
            )

        band_perms: list[Any] = []
        for start, end in band_slices:
            if start >= expected_channels:
                break
            end = min(end, expected_channels)
            band_perms.append(torch_module.randperm(end - start, generator=generator, dtype=torch_module.int64) + start)
        perm = torch_module.cat(band_perms, dim=0)
        latent = latent[:, perm, :, :].contiguous()

        recipe = (
            f"{cls._PROCEDURAL_LATENT_RECIPE_VERSION}"
            f"/lvl{creativity}"
            f"/warp{warp_strength:.2f}"
            f"/low{low_gain:.2f}"
            f"/mid{mid_gain:.2f}"
            f"/stripe{stripe_gain:.2f}"
            f"/flow{flow_gain:.2f}"
            f"/blobs{blob_count}"
            f"/grain{grain_gain:.2f}"
            f"/cells{cell_count}"
            f"/shapes{shape_count}"
            f"/quant{quantized_palette_size}"
            f"/overlay_final{overlay_count}"
            f"/mono{mono_mix:.2f}"
            f"/blotch{blotch_gain:.2f}"
            f"/band{band_gain if creativity != 3 else shape_gain:.2f}"
            f"/ring{ring_gain:.2f}"
            f"/split{split_gain:.2f}"
        )
        return latent.contiguous(), recipe

    def _resolve_procedural_latents(
        self,
        *,
        pipe: Any,
        request: GenerationRequest,
        torch_module: Any,
    ) -> tuple[Any | None, str | None, float | None, str | None]:
        creativity = resolve_procedural_creativity(procedural_creativity=request.procedural_creativity)
        if creativity <= 0:
            return None, None, None, None

        latent_height, latent_width = self._resolve_latent_spatial_shape(
            pipe=pipe,
            width=request.width,
            height=request.height,
        )
        expected_channels = int(getattr(getattr(pipe, "transformer", None), "in_channels", 16))
        latent_tensor, recipe = self._build_procedural_latent_tensor(
            expected_channels=expected_channels,
            target_height=latent_height,
            target_width=latent_width,
            seed=request.seed,
            creativity=creativity,
            torch_module=torch_module,
        )
        noise_mix = (
            self._PROCEDURAL_LATENT_NOISE_MIX_LEVEL3
            if creativity >= 3
            else self._PROCEDURAL_LATENT_NOISE_MIX
        )
        latent_tensor, alpha, preprocess = self._normalize_and_mix_latent(
            latent_tensor=latent_tensor,
            seed=request.seed,
            torch_module=torch_module,
            noise_mix=noise_mix,
            preprocess=self._PROCEDURAL_LATENT_PREPROCESS,
        )
        LOGGER.debug(
            "Procedural latent injection enabled: recipe=%s seed=%s target_latent_shape=%sx%s preprocess=%s alpha=%.3f",
            recipe,
            request.seed if request.seed is not None else "none",
            latent_width,
            latent_height,
            preprocess,
            alpha,
        )
        return latent_tensor, recipe, alpha, preprocess

    @staticmethod
    def _resolve_module_device(module: Any) -> Any:
        if hasattr(module, "device"):
            return module.device
        try:
            return next(module.parameters()).device
        except Exception:
            return None

    @staticmethod
    def _build_rewrite_prompt(tokenizer: Any, prompt: str) -> str:
        system = (
            "Rewrite the input as exactly one stronger image-generation prompt. Preserve the user's intent and "
            "preserve any explicit medium or style exactly. If the user says anime, keep it anime. If the user "
            "says oil painting, keep it painterly. If the user says photograph, cinematic, editorial, or 3D render, "
            "keep that visual mode and do not drift into a conflicting style. Expand only with concrete visible "
            "details that improve subject clarity, environment, lighting, composition, materials, mood, and camera "
            "when relevant. Prefer clear structured natural language over tag soup. Avoid filler adjectives unless "
            "they describe something visible. Output the rewritten prompt only, with no analysis or explanation."
        )
        user_message = (
            "Rewrite this image prompt for better visual fidelity and specificity.\n\n"
            f"Original prompt: {prompt}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                if isinstance(rendered, str) and rendered.strip():
                    return rendered
            except TypeError:
                try:
                    rendered = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    if isinstance(rendered, str) and rendered.strip():
                        return rendered
                except Exception:
                    pass
            except Exception:
                pass
        return f"{system}\n\n{user_message}\n\nRewritten prompt:"

    @staticmethod
    def _extract_rewritten_prompt(full_text: str, input_text: str) -> str:
        candidate = full_text[len(input_text) :].strip() if full_text.startswith(input_text) else full_text.strip()
        candidate = re.sub(r"<think>.*?</think>\s*", "", candidate, flags=re.DOTALL).strip()
        if "Rewritten prompt:" in candidate:
            candidate = candidate.split("Rewritten prompt:", 1)[-1].strip()
        candidate = candidate.splitlines()[0].strip() if candidate else ""
        return candidate

    @staticmethod
    def _rewrite_quality_ok(original: str, rewritten: str) -> bool:
        return DiffusersZImageBackend._rewrite_rejection_reason(original, rewritten) == "ok"

    @staticmethod
    def _rewrite_rejection_reason(original: str, rewritten: str) -> str:
        original_text = original.strip()
        text = rewritten.strip()
        if not text:
            return "empty"
        if len(text) < 8:
            return "too_short"
        if len(text) > 4000:
            return "too_long"
        if re.search(r"(.)\1{10,}", text):
            return "repeated_characters"

        letters = sum(1 for ch in text if ch.isalpha())
        if letters < max(3, int(len(text) * 0.15)):
            return "too_few_letters"

        punctuation = sum(1 for ch in text if ch in string.punctuation)
        if punctuation / max(1, len(text)) > 0.45:
            return "too_much_punctuation"

        words = re.findall(r"[A-Za-z0-9_'-]+", text.lower())
        if words:
            unique_ratio = len(set(words)) / len(words)
            if len(words) >= 6 and unique_ratio < 0.34:
                return "low_lexical_diversity"

        if text == original_text:
            return "unchanged"
        return "ok"

    @staticmethod
    def _render_pipeline_prompt(tokenizer: Any, prompt: str) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                if isinstance(rendered, str) and rendered.strip():
                    return rendered
            except TypeError:
                try:
                    rendered = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    if isinstance(rendered, str) and rendered.strip():
                        return rendered
                except Exception:
                    pass
            except Exception:
                pass
        return prompt

    @classmethod
    def _pipeline_prompt_token_length(cls, tokenizer: Any, prompt: str) -> int:
        rendered = cls._render_pipeline_prompt(tokenizer, prompt)
        encoded = tokenizer(rendered, return_tensors="pt", truncation=False)
        input_ids = getattr(encoded, "input_ids", None)
        if input_ids is None and isinstance(encoded, dict):
            input_ids = encoded.get("input_ids")
        if input_ids is None:
            raise ValueError("Tokenizer did not return input_ids while measuring prompt length.")
        return int(input_ids.shape[-1])

    @staticmethod
    def _split_prompt_clauses(text: str) -> list[str]:
        if not text:
            return []
        clauses = re.split(r"(?<=[,;:.])\s+|\n+", text)
        cleaned: list[str] = []
        seen: set[str] = set()
        for clause in clauses:
            normalized = clause.strip(" ,;:.")
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(normalized)
        return cleaned

    @classmethod
    def _extract_style_constraints(cls, text: str) -> tuple[str, ...]:
        matches: list[str] = []
        seen: set[str] = set()
        for pattern in cls._PROMPT_STYLE_PATTERNS:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                value = match.group(0).strip()
                key = value.lower()
                if key in seen:
                    continue
                seen.add(key)
                matches.append(value)
        return tuple(matches)

    @classmethod
    def _compress_prompt_to_token_budget(
        cls,
        *,
        tokenizer: Any,
        original_prompt: str,
        candidate_prompt: str,
        max_tokens: int | None = None,
    ) -> str | None:
        budget = int(max_tokens or cls._PROMPT_ENHANCEMENT_PIPELINE_SAFE_TOKEN_BUDGET)
        candidate = re.sub(r"\s+", " ", candidate_prompt).strip()
        if not candidate:
            return None
        try:
            if cls._pipeline_prompt_token_length(tokenizer, candidate) <= budget:
                return candidate
        except Exception:
            return candidate

        style_constraints = cls._extract_style_constraints(original_prompt)
        prefix_parts: list[str] = []
        lower_candidate = candidate.lower()
        for style in style_constraints:
            if style.lower() not in lower_candidate:
                prefix_parts.append(style)
        prefix = ", ".join(prefix_parts).strip()

        clauses = cls._split_prompt_clauses(candidate)
        if not clauses:
            return None

        prioritized: list[str] = []
        seen: set[str] = set()

        def push(part: str) -> None:
            normalized = part.strip(" ,;:.")
            if not normalized:
                return
            key = normalized.lower()
            if key in seen:
                return
            seen.add(key)
            prioritized.append(normalized)

        if prefix:
            push(prefix)
        push(clauses[0])
        for clause in clauses[1:]:
            clause_lower = clause.lower()
            if any(keyword in clause_lower for keyword in cls._PROMPT_PRIORITY_KEYWORDS):
                push(clause)
        for clause in clauses[1:]:
            push(clause)

        assembled: list[str] = []
        best: str | None = None
        separator = ", "
        for part in prioritized:
            trial = separator.join(assembled + [part]).strip()
            try:
                token_length = cls._pipeline_prompt_token_length(tokenizer, trial)
            except Exception:
                token_length = 0
            if token_length <= budget:
                assembled.append(part)
                best = trial

        if best is None:
            return None

        for style in style_constraints:
            if style.lower() not in best.lower():
                return None
        return best

    @classmethod
    def _fit_prompt_to_budget(
        cls,
        *,
        tokenizer: Any,
        original_prompt: str,
        enhanced_prompt: str,
    ) -> tuple[str, bool]:
        fitted = cls._compress_prompt_to_token_budget(
            tokenizer=tokenizer,
            original_prompt=original_prompt,
            candidate_prompt=enhanced_prompt,
        )
        if fitted is not None and cls._rewrite_quality_ok(original_prompt, fitted):
            return fitted[: cls._PROMPT_ENHANCEMENT_MAX_OUTPUT_CHARS], True

        fallback_original = cls._compress_prompt_to_token_budget(
            tokenizer=tokenizer,
            original_prompt=original_prompt,
            candidate_prompt=original_prompt,
        )
        if fallback_original is not None:
            return fallback_original[: cls._PROMPT_ENHANCEMENT_MAX_OUTPUT_CHARS], False
        return original_prompt[: cls._PROMPT_ENHANCEMENT_MAX_OUTPUT_CHARS], False

    @staticmethod
    @contextmanager
    def _seeded_rng_context(torch_module: Any, seed: int | None):
        if seed is None:
            yield
            return

        cuda_devices: list[int] = []
        try:
            if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
                cuda_devices = [int(torch_module.cuda.current_device())]
        except Exception:
            cuda_devices = []

        with torch_module.random.fork_rng(devices=cuda_devices, enabled=True):
            torch_module.manual_seed(int(seed))
            if cuda_devices:
                torch_module.cuda.manual_seed_all(int(seed))
            yield

    def _enhance_prompt(
        self,
        pipe: Any,
        prompt: str,
        torch_module: Any,
        *,
        seed: int | None = None,
    ) -> str:
        tokenizer = getattr(pipe, "tokenizer", None)
        text_encoder = getattr(pipe, "text_encoder", None)
        if tokenizer is None or text_encoder is None:
            LOGGER.debug("Prompt enhancement skipped: text_encoder/tokenizer unavailable.")
            return prompt

        rewrite_input = self._build_rewrite_prompt(tokenizer, prompt)
        try:
            encoded = tokenizer(rewrite_input, return_tensors="pt")
        except Exception as exc:
            LOGGER.warning("Prompt enhancement tokenizer failed; using original prompt. %s", exc)
            return prompt

        model_device = self._resolve_module_device(text_encoder)
        model_device_type = str(getattr(model_device, "type", ""))
        if model_device is not None and model_device_type != "meta":
            encoded = {key: value.to(model_device) for key, value in encoded.items()}

        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self._PROMPT_ENHANCEMENT_PRIMARY_MAX_NEW_TOKENS,
            "do_sample": False,
        }
        if pad_token_id is not None:
            generate_kwargs["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            generate_kwargs["eos_token_id"] = eos_token_id

        rewritten, rejection_reason = self._run_rewrite_attempt(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            encoded=encoded,
            prompt=prompt,
            torch_module=torch_module,
            generate_kwargs=generate_kwargs,
            enhancement_seed=seed,
        )
        if rejection_reason == "ok":
            return rewritten[: self._PROMPT_ENHANCEMENT_MAX_OUTPUT_CHARS]

        retryable_reasons = {
            "repeated_characters",
            "too_much_punctuation",
            "low_lexical_diversity",
            "too_few_letters",
        }
        if rejection_reason in retryable_reasons:
            LOGGER.debug(
                "Prompt enhancement retrying with sampled decode after %s.",
                rejection_reason,
            )
            retry_kwargs = dict(generate_kwargs)
            retry_kwargs["do_sample"] = True
            retry_kwargs["temperature"] = 0.72
            retry_kwargs["top_p"] = 0.92
            retry_kwargs["max_new_tokens"] = self._PROMPT_ENHANCEMENT_RETRY_MAX_NEW_TOKENS
            rewritten_retry, retry_reason = self._run_rewrite_attempt(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                encoded=encoded,
                prompt=prompt,
                torch_module=torch_module,
                generate_kwargs=retry_kwargs,
                enhancement_seed=seed,
            )
            if retry_reason == "ok":
                return rewritten_retry[: self._PROMPT_ENHANCEMENT_MAX_OUTPUT_CHARS]
            rejection_reason = retry_reason

        if rejection_reason in {"empty", "too_short", "unchanged"}:
            LOGGER.debug(
                "Prompt enhancement skipped (%s); using original prompt.",
                rejection_reason,
            )
        else:
            LOGGER.warning(
                "Prompt enhancement output rejected (%s); using original prompt.",
                rejection_reason,
            )
        return prompt

    def _prepare_pipe_for_prompt_enhancement(self, pipe: Any) -> bool:
        profile = self._resource_profile()
        if (
            not profile.enable_sequential_offload
            or not profile.enable_cpu_offload
            or not hasattr(pipe, "enable_model_cpu_offload")
            or not hasattr(pipe, "enable_sequential_cpu_offload")
        ):
            return False
        try:
            pipe.enable_model_cpu_offload()
            LOGGER.debug(
                "Switched to model CPU offload for prompt enhancement (sequential offload is restored before image generation)."
            )
            return True
        except Exception as exc:
            LOGGER.warning(
                "Failed to switch offload mode for prompt enhancement; keeping sequential offload. %s",
                exc,
            )
            return False

    @staticmethod
    def _restore_pipe_after_prompt_enhancement(pipe: Any) -> None:
        if not hasattr(pipe, "enable_sequential_cpu_offload"):
            return
        try:
            pipe.enable_sequential_cpu_offload()
        except Exception as exc:
            LOGGER.warning("Failed to restore sequential offload after prompt enhancement. %s", exc)

    def _resolve_effective_prompt(
        self,
        *,
        pipe: Any,
        prompt: str,
        enhance_prompt: bool,
        seed: int | None,
        torch_module: Any,
    ) -> tuple[str, str, bool]:
        prompt_original = prompt
        prompt_effective = prompt_original
        prompt_enhanced = False
        if not enhance_prompt:
            return prompt_original, prompt_effective, prompt_enhanced

        restore_sequential = False
        loaded = self._ensure_loaded()
        if loaded.device == "cuda":
            restore_sequential = self._prepare_pipe_for_prompt_enhancement(pipe)
        try:
            enhanced_candidate = self._enhance_prompt(
                pipe,
                prompt_original,
                torch_module,
                seed=seed,
            )
        finally:
            if restore_sequential:
                self._restore_pipe_after_prompt_enhancement(pipe)

        tokenizer = getattr(pipe, "tokenizer", None)
        if self._rewrite_quality_ok(prompt_original, enhanced_candidate):
            if tokenizer is not None:
                prompt_effective, prompt_enhanced = self._fit_prompt_to_budget(
                    tokenizer=tokenizer,
                    original_prompt=prompt_original,
                    enhanced_prompt=enhanced_candidate,
                )
            else:
                prompt_effective = enhanced_candidate[: self._PROMPT_ENHANCEMENT_MAX_OUTPUT_CHARS]
                prompt_enhanced = True
        else:
            if enhanced_candidate.strip() != prompt_original.strip():
                LOGGER.warning(
                    "Prompt enhancement candidate rejected by final guard; using original prompt."
                )
            prompt_effective = prompt_original
            prompt_enhanced = False
        return prompt_original, prompt_effective, prompt_enhanced

    @staticmethod
    def _pipe_supports_lora(pipe: Any) -> bool:
        required = ("load_lora_weights", "set_adapters", "fuse_lora", "unfuse_lora", "delete_adapters")
        return all(hasattr(pipe, name) for name in required)

    @staticmethod
    def _normalize_lora_trigger_phrase(raw_value: str) -> str:
        normalized = re.sub(r"\s+", " ", str(raw_value or "").strip())
        return normalized.strip(",;| ")

    @classmethod
    def _append_lora_triggers(
        cls,
        prompt: str,
        loras: tuple[LoraSelection, ...],
    ) -> tuple[str, tuple[str, ...]]:
        base_prompt = re.sub(r"\s+", " ", str(prompt or "").strip())
        if not loras:
            return base_prompt, ()

        prompt_lower = base_prompt.lower()
        additions: list[str] = []
        seen: set[str] = set()
        for lora in loras:
            for raw_trigger in lora.trigger_words:
                trigger = cls._normalize_lora_trigger_phrase(raw_trigger)
                lowered = trigger.lower()
                if not trigger or lowered in seen:
                    continue
                seen.add(lowered)
                if lowered in prompt_lower:
                    continue
                additions.append(trigger)

        if not additions:
            return base_prompt, ()
        if not base_prompt:
            return ", ".join(additions), tuple(additions)
        return f"{base_prompt}, {', '.join(additions)}", tuple(additions)

    @staticmethod
    def _split_zimage_lora_key(key: str) -> tuple[str, str]:
        suffixes = (
            ".lora_down.weight",
            ".lora_up.weight",
            ".lora_A.weight",
            ".lora_B.weight",
            ".lora.down.weight",
            ".lora.up.weight",
            ".alpha",
        )
        for suffix in suffixes:
            if key.endswith(suffix):
                return key[: -len(suffix)], suffix
        return key, ""

    @staticmethod
    def _convert_lora_unet_key_to_legacy_zimage_key(key: str) -> str:
        base, suffix = DiffusersZImageBackend._split_zimage_lora_key(key.removeprefix("lora_unet_"))
        protected = {
            ("to", "q"),
            ("to", "k"),
            ("to", "v"),
            ("to", "out"),
            ("feed", "forward"),
        }
        protected_by_length: dict[int, set[tuple[str, ...]]] = {}
        for ngram in protected:
            protected_by_length.setdefault(len(ngram), set()).add(ngram)

        parts = base.split("_")
        merged: list[str] = []
        index = 0
        candidate_lengths = sorted(protected_by_length.keys(), reverse=True)
        while index < len(parts):
            matched = False
            for candidate_length in candidate_lengths:
                if index + candidate_length > len(parts):
                    continue
                window = tuple(parts[index : index + candidate_length])
                if window in protected_by_length[candidate_length]:
                    merged.append("_".join(window))
                    index += candidate_length
                    matched = True
                    break
            if matched:
                continue
            merged.append(parts[index])
            index += 1
        return ".".join(merged) + suffix

    @staticmethod
    def _normalize_legacy_zimage_lora_key(key: str) -> str:
        updated_key = str(key).replace(".lora.down.weight", ".lora_down.weight").replace(
            ".lora.up.weight", ".lora_up.weight"
        )
        updated_key = re.sub(
            r"\.out(?=\.(?:lora_down|lora_up|lora_A|lora_B|alpha)\b)",
            ".to_out.0",
            updated_key,
        )
        return updated_key

    @classmethod
    def _normalize_legacy_zimage_lora_state_dict(cls, state_dict: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        normalized: dict[str, Any] = {}
        changed = False
        for key, value in state_dict.items():
            normalized_key = cls._normalize_legacy_zimage_lora_key(str(key))
            if normalized_key != str(key):
                changed = True
            normalized[normalized_key] = value
        return normalized, changed

    @staticmethod
    def _apply_zimage_alpha_scale(down_weight: Any, up_weight: Any, alpha_value: Any) -> tuple[Any, Any]:
        rank = int(down_weight.shape[0])
        alpha = float(alpha_value.item())
        scale_down = alpha / rank
        scale_up = 1.0
        while scale_down * 2 < scale_up:
            scale_down *= 2
            scale_up /= 2
        return down_weight * scale_down, up_weight * scale_up

    @classmethod
    def _apply_diffusers_native_alpha_scale(
        cls,
        lora_id: str,
        canonical_state_dict: dict[str, Any],
    ) -> dict[str, Any]:
        working_state_dict = dict(canonical_state_dict)
        finalized: dict[str, Any] = {}

        for key in list(working_state_dict.keys()):
            if not key.endswith(".lora_A.weight"):
                continue

            up_key = key.replace(".lora_A.weight", ".lora_B.weight")
            alpha_key = key.replace(".lora_A.weight", ".alpha")
            if up_key not in working_state_dict:
                raise ValueError(
                    f"LoRA '{lora_id}' is missing the matching B weight for '{key}'."
                )

            down_weight = working_state_dict.pop(key)
            up_weight = working_state_dict.pop(up_key)
            if alpha_key in working_state_dict:
                alpha_value = working_state_dict.pop(alpha_key)
                down_weight, up_weight = cls._apply_zimage_alpha_scale(
                    down_weight,
                    up_weight,
                    alpha_value,
                )

            finalized[cls._ensure_transformer_lora_key(key)] = down_weight
            finalized[cls._ensure_transformer_lora_key(up_key)] = up_weight

        leftover_keys = [key for key in working_state_dict.keys() if not key.endswith(".alpha")]
        if leftover_keys:
            sample_key = leftover_keys[0]
            raise ValueError(
                f"LoRA '{lora_id}' uses an unsupported Z-Image key layout near '{sample_key}'."
            )
        return finalized

    @staticmethod
    def _ensure_transformer_lora_key(key: str) -> str:
        return key if key.startswith("transformer.") else f"transformer.{key}"

    @staticmethod
    def _split_qkv_lora_up_weight(up_weight: Any) -> tuple[Any, Any, Any]:
        return tuple(up_weight.chunk(3, dim=0))

    @classmethod
    def _convert_zimage_legacy_lora_state_dict_to_diffusers(
        cls,
        lora_id: str,
        state_dict: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        string_keys = [str(key) for key in state_dict.keys()]
        has_lora_unet = any("lora_unet_" in key for key in string_keys)
        has_diffusion_model = any(key.startswith("diffusion_model.") for key in string_keys)
        has_default = any("default." in key for key in string_keys)
        has_legacy_dotted = any(".lora.down.weight" in key or ".lora.up.weight" in key for key in string_keys)
        has_non_diffusers = any(".lora_down.weight" in key or ".lora_up.weight" in key for key in string_keys)
        has_alphas = any(key.endswith(".alpha") for key in string_keys)

        format_label = "diffusers-native"
        if has_lora_unet:
            format_label = "lora_unet"
        elif has_diffusion_model or has_default or has_legacy_dotted or has_non_diffusers or has_alphas:
            format_label = "legacy-zimage"

        canonical_state_dict: dict[str, Any] = {}
        for raw_key, value in state_dict.items():
            canonical_key = str(raw_key).replace("default.", "")
            if canonical_key.startswith("diffusion_model."):
                canonical_key = canonical_key.removeprefix("diffusion_model.")
            if canonical_key.startswith("lora_unet_"):
                canonical_key = cls._convert_lora_unet_key_to_legacy_zimage_key(canonical_key)
            canonical_key = cls._normalize_legacy_zimage_lora_key(canonical_key)
            canonical_state_dict[canonical_key] = value

        canonical_has_diffusers = any(".lora_A.weight" in key or ".lora_B.weight" in key for key in canonical_state_dict.keys())
        canonical_has_non_diffusers = any(
            ".lora_down.weight" in key or ".lora_up.weight" in key for key in canonical_state_dict.keys()
        )

        if canonical_has_diffusers and not canonical_has_non_diffusers:
            finalized = cls._apply_diffusers_native_alpha_scale(lora_id, canonical_state_dict)
            if not finalized:
                raise ValueError(f"LoRA '{lora_id}' did not contain any diffusers LoRA weights.")
            return finalized, format_label

        if not canonical_has_non_diffusers:
            sample_key = next(iter(canonical_state_dict.keys()), "<empty>")
            raise ValueError(
                f"LoRA '{lora_id}' uses an unsupported Z-Image key layout near '{sample_key}'."
            )

        working_state_dict = dict(canonical_state_dict)
        converted_state_dict: dict[str, Any] = {}
        for key in list(working_state_dict.keys()):
            if not key.endswith(".lora_down.weight"):
                continue

            up_key = key.replace(".lora_down.weight", ".lora_up.weight")
            alpha_key = key.replace(".lora_down.weight", ".alpha")
            if up_key not in working_state_dict:
                raise ValueError(
                    f"LoRA '{lora_id}' is missing the matching up weight for '{key}'."
                )
            if alpha_key not in working_state_dict:
                raise ValueError(
                    f"LoRA '{lora_id}' is missing the matching alpha for '{key}'."
                )

            down_weight = working_state_dict.pop(key)
            up_weight = working_state_dict.pop(up_key)
            alpha_value = working_state_dict.pop(alpha_key)
            scaled_down_weight, scaled_up_weight = cls._apply_zimage_alpha_scale(
                down_weight,
                up_weight,
                alpha_value,
            )
            if ".attention.qkv." in key:
                q_up_weight, k_up_weight, v_up_weight = cls._split_qkv_lora_up_weight(scaled_up_weight)
                for projection_name, projection_up_weight in (
                    ("to_q", q_up_weight),
                    ("to_k", k_up_weight),
                    ("to_v", v_up_weight),
                ):
                    projection_key = key.replace(".attention.qkv.", f".attention.{projection_name}.")
                    converted_state_dict[
                        cls._ensure_transformer_lora_key(projection_key.replace(".lora_down.weight", ".lora_A.weight"))
                    ] = scaled_down_weight
                    converted_state_dict[
                        cls._ensure_transformer_lora_key(projection_key.replace(".lora_down.weight", ".lora_B.weight"))
                    ] = projection_up_weight
                continue

            converted_state_dict[
                cls._ensure_transformer_lora_key(key.replace(".lora_down.weight", ".lora_A.weight"))
            ] = scaled_down_weight
            converted_state_dict[
                cls._ensure_transformer_lora_key(key.replace(".lora_down.weight", ".lora_B.weight"))
            ] = scaled_up_weight

        leftover_keys = list(working_state_dict.keys())
        if leftover_keys:
            sample_key = leftover_keys[0]
            raise ValueError(
                f"LoRA '{lora_id}' uses an unsupported Z-Image key layout near '{sample_key}'."
            )
        if not converted_state_dict:
            raise ValueError(f"LoRA '{lora_id}' did not contain any convertible Z-Image LoRA weights.")
        return converted_state_dict, format_label

    def _load_lora_adapters(self, pipe: Any, loras: tuple[LoraSelection, ...]) -> None:
        if not loras:
            return
        if not self._pipe_supports_lora(pipe):
            raise ValueError("Active pipeline does not support LoRA adapters.")

        adapter_names = [lora.id for lora in loras]
        adapter_weights = [float(lora.weight) for lora in loras]
        loaded_adapter_names = self._list_loaded_lora_adapters(pipe)
        conflicting_adapter_names = [name for name in adapter_names if name in loaded_adapter_names]
        if conflicting_adapter_names:
            LOGGER.info(
                "Clearing stale LoRA adapters before reload ids=%s",
                conflicting_adapter_names,
            )
            self._clear_lora_adapters(pipe, adapter_names=conflicting_adapter_names)
        adapter_formats: dict[str, str] = {}
        for lora in loras:
            raw_state_dict = load_safetensors_file(str(lora.path), device="cpu")
            prepared_state_dict, format_label = self._convert_zimage_legacy_lora_state_dict_to_diffusers(
                lora.id,
                raw_state_dict,
            )
            adapter_formats[lora.id] = format_label
            if format_label != "diffusers-native":
                LOGGER.info(
                    "Detected %s LoRA format for '%s'; converting locally to diffusers format.",
                    format_label,
                    lora.id,
                )
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"Already found a `peft_config` attribute in the model\. This will lead to having multiple adapters in the model\. Make sure to know what you are doing!",
                        category=UserWarning,
                    )
                    pipe.load_lora_weights(
                        prepared_state_dict,
                        adapter_name=lora.id,
                        hotswap=False,
                        use_safetensors=True,
                        local_files_only=True,
                    )
            except ValueError as exc:
                if "`state_dict` should be empty" in str(exc):
                    sample_key = next(iter(prepared_state_dict.keys()), "<empty>")
                    raise ValueError(
                        f"LoRA '{lora.id}' could not be converted to a diffusers-compatible transformer state dict. "
                        f"First converted key: '{sample_key}'."
                    ) from exc
                raise
        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
        if hasattr(pipe, "enable_lora"):
            try:
                pipe.enable_lora()
            except Exception:
                pass
        LOGGER.info(
            "Activating LoRAs ids=%s weights=%s formats=%s runtime_path=%s",
            adapter_names,
            adapter_weights,
            {name: adapter_formats.get(name, "unknown") for name in adapter_names},
            "unfused",
        )

    def _list_loaded_lora_adapters(self, pipe: Any) -> set[str]:
        try:
            if hasattr(pipe, "get_list_adapters"):
                adapters_by_component = pipe.get_list_adapters()
                loaded_adapter_names: set[str] = set()
                if isinstance(adapters_by_component, dict):
                    for names in adapters_by_component.values():
                        if isinstance(names, (list, tuple, set)):
                            loaded_adapter_names.update(str(name) for name in names)
                return loaded_adapter_names
        except Exception:
            pass

        try:
            transformer = getattr(pipe, "transformer", None)
            peft_config = getattr(transformer, "peft_config", None)
            if isinstance(peft_config, dict):
                return {str(name) for name in peft_config.keys()}
        except Exception:
            pass

        return set()

    def _clear_lora_adapters(self, pipe: Any, adapter_names: list[str] | None = None) -> None:
        if not self._pipe_supports_lora(pipe):
            return
        if hasattr(pipe, "disable_lora"):
            try:
                pipe.disable_lora()
            except Exception:
                pass
        try:
            pipe.unfuse_lora(components=["transformer"])
        except Exception:
            pass
        target_names = list(adapter_names or [])
        if target_names:
            try:
                pipe.delete_adapters(target_names)
            except Exception:
                transformer = getattr(pipe, "transformer", None)
                if transformer is not None and hasattr(transformer, "delete_adapters"):
                    try:
                        transformer.delete_adapters(target_names)
                    except Exception:
                        pass

    def drop_lora_adapters(self, lora_ids: list[str] | None = None) -> None:
        if self._loaded is None:
            return
        self._clear_lora_adapters(self._loaded.pipeline, adapter_names=lora_ids)

    def _resolve_refine_tiling(self, request: GenerationRequest, width: int, height: int) -> tuple[int, int]:
        overlap = max(8, int(request.refine_tile_overlap or 64))
        if request.refine_tile_size is not None:
            tile_size = max(0, int(request.refine_tile_size))
            return tile_size, overlap

        profile_name = self._resource_profile().name
        max_dim = max(width, height)
        if profile_name == "high" and max_dim <= self._REFINE_HIGH_FULL_FRAME_MAX_DIM:
            tile_size = 0
        else:
            grid_divisor = self._REFINE_GRID_DIVISOR_BY_PROFILE.get(profile_name, 3)
            tile_cap = self._REFINE_TILE_CAP_BY_PROFILE.get(profile_name, 1024)
            raw_tile = int(math.ceil(max_dim / max(1, grid_divisor)))
            snapped_tile = self._snap_up(raw_tile, self._REFINE_TILE_SNAP)
            tile_size = min(tile_cap, snapped_tile)
        return tile_size, overlap

    def _run_rewrite_attempt(
        self,
        *,
        tokenizer: Any,
        text_encoder: Any,
        encoded: dict[str, Any],
        prompt: str,
        torch_module: Any,
        generate_kwargs: dict[str, Any],
        enhancement_seed: int | None = None,
    ) -> tuple[str, str]:
        output_ids = None
        if hasattr(text_encoder, "generate"):
            try:
                with self._seeded_rng_context(torch_module, enhancement_seed):
                    with torch_module.inference_mode():
                        output_ids = text_encoder.generate(**encoded, **generate_kwargs)
            except Exception as exc:
                LOGGER.warning(
                    "Prompt enhancement generate() failed; falling back to base-model decode. %s",
                    exc,
                )

        if output_ids is None:
            try:
                with self._seeded_rng_context(torch_module, enhancement_seed):
                    output_ids = self._generate_with_base_model(
                        text_encoder=text_encoder,
                        encoded=encoded,
                        max_new_tokens=int(generate_kwargs.get("max_new_tokens", 72)),
                        eos_token_id=generate_kwargs.get("eos_token_id"),
                        torch_module=torch_module,
                        do_sample=bool(generate_kwargs.get("do_sample", False)),
                        temperature=float(generate_kwargs.get("temperature", 1.0)),
                        top_p=float(generate_kwargs.get("top_p", 1.0)),
                        repetition_penalty=float(generate_kwargs.get("repetition_penalty", 1.08)),
                    )
            except Exception as exc:
                LOGGER.warning("Prompt enhancement base-model decode failed; using original prompt. %s", exc)
                return prompt, "decode_failure"

        try:
            full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            input_text = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)
            rewritten = self._extract_rewritten_prompt(full_text, input_text)
            return rewritten, self._rewrite_rejection_reason(prompt, rewritten)
        except Exception as exc:
            LOGGER.warning("Prompt enhancement decode failed; using original prompt. %s", exc)
            return prompt, "decode_failure"

    @staticmethod
    def _generate_with_base_model(
        *,
        text_encoder: Any,
        encoded: dict[str, Any],
        max_new_tokens: int,
        eos_token_id: int | None,
        torch_module: Any,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> Any:
        if not hasattr(text_encoder, "get_input_embeddings"):
            raise ValueError("text_encoder does not expose input embeddings for greedy decode.")

        embed_layer = text_encoder.get_input_embeddings()
        if embed_layer is None or not hasattr(embed_layer, "weight"):
            raise ValueError("text_encoder input embedding weights are unavailable.")

        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        past_key_values = None
        generated = input_ids
        embed_weight = embed_layer.weight

        with torch_module.inference_mode():
            try:
                forward_params = inspect.signature(text_encoder.forward).parameters
            except Exception:
                forward_params = {}
            supports_cache_position = "cache_position" in forward_params
            supports_position_ids = "position_ids" in forward_params

            for _ in range(max_new_tokens):
                step_ids = generated if past_key_values is None else generated[:, -1:]
                past_length = generated.shape[1] - step_ids.shape[1]
                cache_position = torch_module.arange(
                    past_length,
                    past_length + step_ids.shape[1],
                    device=step_ids.device,
                    dtype=torch_module.long,
                )
                position_ids = cache_position.unsqueeze(0).expand(step_ids.shape[0], -1)
                model_inputs = {
                    "input_ids": step_ids,
                    "use_cache": True,
                }
                if supports_cache_position:
                    model_inputs["cache_position"] = cache_position
                if supports_position_ids:
                    model_inputs["position_ids"] = position_ids
                if attention_mask is not None:
                    model_inputs["attention_mask"] = attention_mask
                if past_key_values is not None:
                    model_inputs["past_key_values"] = past_key_values

                outputs = text_encoder(**model_inputs)
                past_key_values = getattr(outputs, "past_key_values", None)
                if past_key_values is None:
                    raise ValueError("text_encoder did not return past_key_values.")

                hidden = outputs.last_hidden_state[:, -1, :]
                logits = torch_module.nn.functional.linear(hidden, embed_weight)

                if repetition_penalty > 1.0:
                    for row in range(generated.shape[0]):
                        unique_ids = torch_module.unique(generated[row])
                        unique_ids = unique_ids.to(logits.device)
                        token_logits = logits[row, unique_ids]
                        adjusted = torch_module.where(
                            token_logits < 0,
                            token_logits * repetition_penalty,
                            token_logits / repetition_penalty,
                        )
                        logits[row, unique_ids] = adjusted

                if do_sample:
                    temp = max(float(temperature), 1e-5)
                    logits = logits / temp
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch_module.sort(logits, descending=True, dim=-1)
                        sorted_probs = torch_module.softmax(sorted_logits, dim=-1)
                        cumulative_probs = torch_module.cumsum(sorted_probs, dim=-1)
                        sorted_remove = cumulative_probs > top_p
                        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
                        sorted_remove[..., 0] = False
                        remove_mask = torch_module.zeros_like(sorted_remove, dtype=torch_module.bool)
                        remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_remove)
                        logits = logits.masked_fill(remove_mask, float("-inf"))
                    probs = torch_module.softmax(logits, dim=-1)
                    next_token = torch_module.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                if generated.device != next_token.device:
                    generated = generated.to(next_token.device)
                generated = torch_module.cat([generated, next_token], dim=-1)

                if attention_mask is not None:
                    if attention_mask.device != generated.device:
                        attention_mask = attention_mask.to(generated.device)
                    ones = torch_module.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    attention_mask = torch_module.cat([attention_mask, ones], dim=-1)

                if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                    break

        return generated

    def _run_img2img_once(
        self,
        *,
        pipe: Any,
        prompt: str,
        image: Image.Image,
        strength: float,
        steps: int,
        guidance_scale: float,
        generator: Any,
        torch_module: Any,
    ) -> Image.Image:
        with torch_module.inference_mode():
            output = pipe(
                prompt=prompt,
                image=image,
                strength=strength,
                width=image.width,
                height=image.height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        return output.images[0]

    def _run_img2img_tiled(
        self,
        *,
        pipe: Any,
        prompt: str,
        image: Image.Image,
        strength: float,
        steps: int,
        guidance_scale: float,
        seed: int | None,
        tile_size: int,
        tile_overlap: int,
        torch_module: Any,
    ) -> Image.Image:
        width, height = image.size
        if tile_size <= 0 or (width <= tile_size and height <= tile_size):
            generator = self._build_generator(torch_module, "cuda" if torch_module.cuda.is_available() else "cpu", seed)
            return self._run_img2img_once(
                pipe=pipe,
                prompt=prompt,
                image=image,
                strength=strength,
                steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                torch_module=torch_module,
            )

        canvas = Image.new("RGB", (width, height))
        tile_index = 0
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                tile_height = min(tile_size, height - y)
                tile_width = min(tile_size, width - x)

                in_y0 = max(y - tile_overlap, 0)
                in_x0 = max(x - tile_overlap, 0)
                in_y1 = min(y + tile_height + tile_overlap, height)
                in_x1 = min(x + tile_width + tile_overlap, width)

                tile_input = image.crop((in_x0, in_y0, in_x1, in_y1))
                tile_seed = (seed + tile_index) if seed is not None else None
                generator = self._build_generator(
                    torch_module,
                    "cuda" if torch_module.cuda.is_available() else "cpu",
                    tile_seed,
                )
                tile_output = self._run_img2img_once(
                    pipe=pipe,
                    prompt=prompt,
                    image=tile_input,
                    strength=strength,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    torch_module=torch_module,
                )

                crop_x0 = x - in_x0
                crop_y0 = y - in_y0
                crop_x1 = crop_x0 + tile_width
                crop_y1 = crop_y0 + tile_height
                core = tile_output.crop((crop_x0, crop_y0, crop_x1, crop_y1))
                canvas.paste(core, (x, y))
                tile_index += 1
        return canvas

    def _run_refine_with_oom_fallback(
        self,
        *,
        pipe: Any,
        prompt: str,
        image: Image.Image,
        strength: float,
        steps: int,
        guidance_scale: float,
        seed: int | None,
        tile_size: int,
        tile_overlap: int,
        torch_module: Any,
    ) -> tuple[Image.Image, int, int, int]:
        profile_name = self._resource_profile().name
        fallback_overlap = max(32, tile_overlap)
        min_tile = self._REFINE_FALLBACK_MIN_TILE_BY_PROFILE.get(profile_name, 512)

        attempt_tiles: list[int] = []
        if tile_size > 0:
            attempt_tiles.append(tile_size)
            attempt_tiles.extend(self._build_stepdown_tiles(tile_size, min_tile))
        else:
            attempt_tiles.append(0)
            max_dim = max(image.width, image.height)
            cap = self._REFINE_TILE_CAP_BY_PROFILE.get(profile_name, 1024)
            fallback_start = min(cap, max_dim)
            fallback_start = self._snap_up(fallback_start, self._REFINE_TILE_SNAP)
            fallback_start = max(min_tile, fallback_start)
            attempt_tiles.append(fallback_start)
            attempt_tiles.extend(self._build_stepdown_tiles(fallback_start, min_tile))

        normalized_attempt_tiles: list[int] = []
        for index, candidate in enumerate(attempt_tiles):
            if index > 0 and candidate > 0:
                candidate = self._snap_up(candidate, self._REFINE_TILE_SNAP)
                candidate = max(min_tile, candidate)
            if candidate not in normalized_attempt_tiles:
                normalized_attempt_tiles.append(candidate)
        attempt_tiles = normalized_attempt_tiles

        fallback_attempts = 0
        for idx, candidate_tile in enumerate(attempt_tiles):
            try:
                if candidate_tile > 0:
                    return (
                        self._run_img2img_tiled(
                            pipe=pipe,
                            prompt=prompt,
                            image=image,
                            strength=strength,
                            steps=steps,
                            guidance_scale=guidance_scale,
                            seed=seed,
                            tile_size=candidate_tile,
                            tile_overlap=fallback_overlap,
                            torch_module=torch_module,
                        ),
                        candidate_tile,
                        fallback_overlap,
                        fallback_attempts,
                    )
                generator = self._build_generator(
                    torch_module,
                    "cuda" if torch_module.cuda.is_available() else "cpu",
                    seed,
                )
                return (
                    self._run_img2img_once(
                        pipe=pipe,
                        prompt=prompt,
                        image=image,
                        strength=strength,
                        steps=steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        torch_module=torch_module,
                    ),
                    0,
                    tile_overlap,
                    fallback_attempts,
                )
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                if idx == len(attempt_tiles) - 1:
                    raise
                fallback_attempts += 1
                if candidate_tile == 0:
                    LOGGER.warning(
                        "Img2img refine OOM on full frame, retrying with tiled refine."
                    )
                else:
                    LOGGER.warning(
                        "Img2img refine OOM at tile size %s, retrying with smaller tile.",
                        candidate_tile,
                    )
                self._clear_cuda_cache(torch_module)
        raise RuntimeError("Unreachable OOM fallback state.")

    def generate(self, request: GenerationRequest) -> GenerationResult:
        loaded = self._ensure_loaded()
        pipe = loaded.pipeline
        effective_procedural_creativity = resolve_procedural_creativity(
            procedural_creativity=request.procedural_creativity
        )
        requested_scheduler_mode = self._normalize_scheduler_mode(request.scheduler_mode)
        effective_scheduler_mode, procedural_latent_scheduler_forced = self._resolve_generate_scheduler_mode(
            requested_mode=requested_scheduler_mode,
            procedural_creativity=effective_procedural_creativity,
        )
        scheduler_mode = self._apply_scheduler_mode(pipe, effective_scheduler_mode)

        import torch

        steps = request.steps or self._settings.runtime_profile.steps_default
        guidance_scale = (
            request.guidance_scale
            if request.guidance_scale is not None
            else self._settings.runtime_profile.guidance_scale_default
        )
        generator = self._build_generator(torch, "cuda" if loaded.device == "cuda" else "cpu", request.seed)
        procedural_latents = None
        procedural_latent_recipe = None
        procedural_latent_alpha = None
        procedural_latent_preprocess = None
        if effective_procedural_creativity > 0:
            (
                procedural_latents,
                procedural_latent_recipe,
                procedural_latent_alpha,
                procedural_latent_preprocess,
            ) = self._resolve_procedural_latents(
                pipe=pipe,
                request=request,
                torch_module=torch,
            )

        prompt_original, prompt_effective_base, prompt_enhanced = self._resolve_effective_prompt(
            pipe=pipe,
            prompt=request.prompt,
            enhance_prompt=request.enhance_prompt,
            seed=request.seed,
            torch_module=torch,
        )
        prompt_effective, lora_trigger_words = self._append_lora_triggers(
            prompt_effective_base,
            request.loras,
        )
        active_lora_ids = [lora.id for lora in request.loras]
        lora_payload = tuple(
            {
                "id": lora.id,
                "name": lora.name or lora.id,
                "weight": float(lora.weight),
                "trigger_words": list(lora.trigger_words),
            }
            for lora in request.loras
        )

        self._preflight_fallback_triggered = False
        generate_preflight = self._run_vram_preflight(torch)
        if (
            loaded.device == "cuda"
            and self._resource_profile().name == "high"
            and self._effective_execution_mode == self._HIGH_MODE_FULL_CUDA
            and generate_preflight.checked
            and not generate_preflight.passed
        ):
            applied_mode = self._apply_pipe_execution_mode(pipe, self._HIGH_MODE_MODEL_OFFLOAD)
            if applied_mode == self._HIGH_MODE_MODEL_OFFLOAD:
                self._effective_execution_mode = self._HIGH_MODE_MODEL_OFFLOAD
                self._high_runtime_fallback_latched = True
                self._preflight_fallback_triggered = True
                if self._img2img_pipe is not None:
                    self._apply_pipe_execution_mode(self._img2img_pipe, self._HIGH_MODE_MODEL_OFFLOAD)
                self._clear_cuda_cache(torch)
            else:
                LOGGER.warning(
                    "Pre-generate VRAM guard requested model_offload but pipeline remained in %s.",
                    applied_mode,
                )

        pre_mem = cuda_memory_snapshot(torch)
        pre_proc_mem = process_memory_snapshot()
        execution_mode_before_generate = self._effective_execution_mode
        cuda_free_before_generate, _ = self._cuda_free_total_snapshot(torch)
        started = now_perf()
        try:
            with torch.inference_mode():
                if request.loras:
                    self._load_lora_adapters(pipe, request.loras)
                try:
                    output = pipe(
                        prompt=prompt_effective,
                        width=request.width,
                        height=request.height,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        latents=procedural_latents,
                    )
                except (RuntimeError, ValueError, FloatingPointError) as exc:
                    retry_mode = self._resolve_scheduler_retry_mode(scheduler_mode, exc)
                    if retry_mode is None:
                        raise
                    LOGGER.warning(
                        "Scheduler mode %s failed for pack %s (%s). Retrying once with %s.",
                        scheduler_mode,
                        getattr(self._model_pack, "name", "<unknown-pack>"),
                        exc,
                        retry_mode,
                    )
                    self._clear_cuda_cache(torch)
                    scheduler_mode = self._apply_scheduler_mode(pipe, retry_mode)
                    output = pipe(
                        prompt=prompt_effective,
                        width=request.width,
                        height=request.height,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        latents=procedural_latents,
                    )
        finally:
            if active_lora_ids:
                self._clear_lora_adapters(pipe, adapter_names=active_lora_ids)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration_ms = int((now_perf() - started) * 1000)
        post_mem = cuda_memory_snapshot(torch)
        post_proc_mem = process_memory_snapshot()
        cuda_free_after_generate, _ = self._cuda_free_total_snapshot(torch)
        self._apply_high_runtime_fallback_if_needed(post_mem=post_mem, torch_module=torch)
        execution_mode_after_generate = self._effective_execution_mode
        image = output.images[0]
        selected_pack = getattr(self._model_pack, "base_name", None) or getattr(self._model_pack, "name", None)
        return GenerationResult(
            image=image,
            seed=request.seed,
            steps=steps,
            guidance_scale=guidance_scale,
            scheduler_mode=scheduler_mode,
            backend=self._backend_name,
            device=loaded.device,
            duration_ms=duration_ms,
            prompt_original=prompt_original,
            prompt_effective=prompt_effective,
            prompt_enhanced=prompt_enhanced,
            prompt_effective_base=prompt_effective_base,
            cuda_memory_before=pre_mem,
            cuda_memory_after=post_mem,
            process_memory_before=pre_proc_mem,
            process_memory_after=post_proc_mem,
            runtime_profile=self._settings.runtime_profile.name,
            resource_tier=self._resource_profile().name,
            execution_mode=self._effective_execution_mode,
            execution_mode_initial=self._initial_execution_mode,
            execution_mode_before_generate=execution_mode_before_generate,
            execution_mode_after_generate=execution_mode_after_generate,
            cuda_total_bytes=self._cuda_total_bytes,
            cuda_reserved_after_load_bytes=self._cuda_reserved_after_load_bytes,
            cuda_free_before_load_bytes=self._cuda_free_before_load_bytes,
            cuda_free_after_load_bytes=self._cuda_free_after_load_bytes,
            cuda_free_before_generate_bytes=cuda_free_before_generate,
            cuda_free_after_generate_bytes=cuda_free_after_generate,
            preflight_checked=generate_preflight.checked,
            preflight_cleanup_attempted=generate_preflight.cleanup_attempted,
            preflight_passed_before_cleanup=generate_preflight.passed_before_cleanup,
            preflight_passed_after_cleanup=generate_preflight.passed_after_cleanup,
            preflight_free_before_bytes=generate_preflight.free_before_bytes,
            preflight_free_after_cleanup_bytes=generate_preflight.free_after_cleanup_bytes,
            preflight_threshold_bytes=generate_preflight.threshold_bytes,
            preflight_fallback_triggered=self._preflight_fallback_triggered,
            procedural_latent_enabled=effective_procedural_creativity > 0,
            procedural_creativity=effective_procedural_creativity,
            procedural_latent_recipe=procedural_latent_recipe,
            procedural_latent_alpha=procedural_latent_alpha,
            procedural_latent_preprocess=procedural_latent_preprocess,
            procedural_latent_scheduler_forced=procedural_latent_scheduler_forced,
            selected_pack=selected_pack,
            effective_pack=getattr(self._model_pack, "name", None),
            fp8_checkpoint=self._fp8_checkpoint,
            fp8_fallback_used=self._fp8_fallback_used,
            fp8_fallback_reason=self._fp8_fallback_reason,
            fp8_runtime_mode=self._fp8_runtime_mode,
            fp8_normalized_tensor_count=self._fp8_normalized_tensor_count,
            fp8_storage_preserved_tensor_count=self._fp8_storage_preserved_tensor_count,
            fp8_promoted_tensor_count=self._fp8_promoted_tensor_count,
            fp8_normalized_tensor_names=self._fp8_normalized_tensor_names,
            loras=lora_payload,
            lora_count=len(lora_payload),
            lora_trigger_words=lora_trigger_words,
        )

    def upscale_and_refine(self, input_image: object, request: GenerationRequest) -> GenerationResult:
        if not isinstance(input_image, Image.Image):
            raise ValueError("input_image must be a PIL.Image.Image instance.")

        checkpoint_path = request.upscaler_checkpoint or (
            self._settings.paths.models_dir / "upscaler" / "2x_RealESRGAN_x2plus.pth"
        )
        if not checkpoint_path.exists():
            raise ValueError(f"Upscaler checkpoint not found: {checkpoint_path}")

        upscale_result = upscale_image(
            image=input_image,
            checkpoint_path=checkpoint_path,
            profile_name=self._settings.runtime_profile.name,
        )
        return self._refine_existing_image(
            input_image=input_image,
            refine_input_image=upscale_result.image,
            request=request,
            upscale_duration_ms=int(upscale_result.duration_ms),
            mode="upscale_then_img2img",
        )

    def refine_image(self, input_image: object, request: GenerationRequest) -> GenerationResult:
        if not isinstance(input_image, Image.Image):
            raise ValueError("input_image must be a PIL.Image.Image instance.")
        return self._refine_existing_image(
            input_image=input_image,
            refine_input_image=input_image,
            request=request,
            upscale_duration_ms=0,
            mode="img2img_refine",
        )

    def _refine_existing_image(
        self,
        *,
        input_image: Image.Image,
        refine_input_image: Image.Image,
        request: GenerationRequest,
        upscale_duration_ms: int,
        mode: str,
    ) -> GenerationResult:
        loaded = self._ensure_loaded()
        txt_pipe = loaded.pipeline
        img_pipe = self._ensure_img2img_pipe()
        scheduler_mode = self._normalize_scheduler_mode(request.scheduler_mode) or self._SCHEDULER_EULER
        scheduler_mode = self._apply_scheduler_mode(img_pipe, scheduler_mode)

        import torch

        refine_steps = request.refine_steps or 6
        refine_strength = request.refine_strength if request.refine_strength is not None else 0.20
        if refine_strength <= 0.0 or refine_strength >= 1.0:
            raise ValueError("refine_strength must be between 0 and 1.")

        guidance_scale = (
            request.guidance_scale
            if request.guidance_scale is not None
            else self._settings.runtime_profile.guidance_scale_default
        )
        prompt_original, prompt_effective, prompt_enhanced = self._resolve_effective_prompt(
            pipe=txt_pipe,
            prompt=request.prompt,
            enhance_prompt=request.enhance_prompt,
            seed=request.seed,
            torch_module=torch,
        )

        pre_mem = cuda_memory_snapshot(torch)
        pre_proc_mem = process_memory_snapshot()
        started = now_perf()

        tile_size_requested, tile_overlap_requested = self._resolve_refine_tiling(
            request,
            refine_input_image.width,
            refine_input_image.height,
        )
        refine_started = now_perf()
        (
            refined_image,
            effective_tile_size,
            effective_tile_overlap,
            fallback_attempt_count,
        ) = self._run_refine_with_oom_fallback(
            pipe=img_pipe,
            prompt=prompt_effective,
            image=refine_input_image,
            strength=refine_strength,
            steps=refine_steps,
            guidance_scale=guidance_scale,
            seed=request.seed,
            tile_size=tile_size_requested,
            tile_overlap=tile_overlap_requested,
            torch_module=torch,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        refine_duration_ms = int((now_perf() - refine_started) * 1000)
        duration_ms = int((now_perf() - started) * 1000)
        post_mem = cuda_memory_snapshot(torch)
        post_proc_mem = process_memory_snapshot()
        self._apply_high_runtime_fallback_if_needed(post_mem=post_mem, torch_module=torch)
        selected_pack = getattr(self._model_pack, "base_name", None) or getattr(self._model_pack, "name", None)

        return GenerationResult(
            image=refined_image,
            seed=request.seed,
            steps=refine_steps,
            guidance_scale=guidance_scale,
            scheduler_mode=scheduler_mode,
            backend=self._backend_name,
            device=loaded.device,
            duration_ms=duration_ms,
            prompt_original=prompt_original,
            prompt_effective=prompt_effective,
            prompt_enhanced=prompt_enhanced,
            mode=mode,
            upscale_duration_ms=int(upscale_duration_ms),
            refine_duration_ms=refine_duration_ms,
            refine_strength=refine_strength,
            refine_tile_size=effective_tile_size,
            refine_tile_overlap=effective_tile_overlap,
            refine_tile_size_requested=tile_size_requested,
            refine_tile_size_effective=effective_tile_size,
            refine_tile_overlap_effective=effective_tile_overlap,
            refine_fallback_used=fallback_attempt_count > 0,
            refine_fallback_attempt_count=fallback_attempt_count,
            input_image_width=input_image.width,
            input_image_height=input_image.height,
            cuda_memory_before=pre_mem,
            cuda_memory_after=post_mem,
            process_memory_before=pre_proc_mem,
            process_memory_after=post_proc_mem,
            runtime_profile=self._settings.runtime_profile.name,
            resource_tier=self._resource_profile().name,
            execution_mode=self._effective_execution_mode,
            cuda_total_bytes=self._cuda_total_bytes,
            cuda_reserved_after_load_bytes=self._cuda_reserved_after_load_bytes,
            selected_pack=selected_pack,
            effective_pack=getattr(self._model_pack, "name", None),
            fp8_checkpoint=self._fp8_checkpoint,
            fp8_fallback_used=self._fp8_fallback_used,
            fp8_fallback_reason=self._fp8_fallback_reason,
            fp8_runtime_mode=self._fp8_runtime_mode,
            fp8_normalized_tensor_count=self._fp8_normalized_tensor_count,
            fp8_storage_preserved_tensor_count=self._fp8_storage_preserved_tensor_count,
            fp8_promoted_tensor_count=self._fp8_promoted_tensor_count,
            fp8_normalized_tensor_names=self._fp8_normalized_tensor_names,
        )
