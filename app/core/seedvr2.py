from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import json
import logging
import multiprocessing
import os
import sys
import tempfile
import threading
import traceback
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image

from app.config.settings import AppSettings
from app.core.cancellation import GenerationCancelledError
from app.core.memory import now_perf
from app.core.platform_guidance import setup_repair_hint

SEEDVR2_MODEL_REPO = "themindstudio/SeedVR2-3B-FP8-e4m3fn"
SEEDVR2_MODEL_REVISION = "main"
SEEDVR2_DIT_FILENAME = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
SEEDVR2_VAE_FILENAME = "ema_vae_fp16.safetensors"
SEEDVR2_DEFAULT_TIMEOUT_SECONDS = 240
SEEDVR2_DEFAULT_ATTEMPT_TIMEOUT_SECONDS = 45
SEEDVR2_DEFAULT_COLOR_CORRECTION = "lab"
SEEDVR2_VERBOSE_RUNTIME_ENV = "JUSTRAYZIST_SEEDVR2_VERBOSE_RUNTIME"
SEEDVR2_ATTEMPT_TIMEOUT_ENV = "JUSTRAYZIST_SEEDVR2_ATTEMPT_TIMEOUT_S"
SEEDVR2_MAX_ATTEMPTS_ENV = "JUSTRAYZIST_SEEDVR2_MAX_ATTEMPTS"
SEEDVR2_POLICY_CACHE_FILENAME = "seedvr2_policy_cache.json"
SEEDVR2_EXECUTION_STRATEGY_CHAIN_X2 = "chain_x2"
SEEDVR2_EXECUTION_STRATEGY_DIRECT_TARGET = "direct_target"
SEEDVR2_EXECUTION_STRATEGIES = (
    SEEDVR2_EXECUTION_STRATEGY_CHAIN_X2,
    SEEDVR2_EXECUTION_STRATEGY_DIRECT_TARGET,
)
SEEDVR2_RUNTIME_PRESET_CURRENT_BASELINE = "current_baseline"
SEEDVR2_RUNTIME_PRESET_HIGHRES_AUTO = "highres_auto"
SEEDVR2_RUNTIME_PRESET_HIGHRES_TILED_1024 = "highres_tiled_1024"
SEEDVR2_RUNTIME_PRESET_HIGHRES_TILED_896 = "highres_tiled_896"
SEEDVR2_RUNTIME_PRESETS = (
    SEEDVR2_RUNTIME_PRESET_CURRENT_BASELINE,
    SEEDVR2_RUNTIME_PRESET_HIGHRES_AUTO,
    SEEDVR2_RUNTIME_PRESET_HIGHRES_TILED_1024,
    SEEDVR2_RUNTIME_PRESET_HIGHRES_TILED_896,
)

_RUNTIME_LOCK = threading.RLock()
_RUNTIME_MODULE: Any | None = None
_RUNTIME_SCRIPT: Path | None = None
_RUNNER_CACHE_BY_KEY: dict[str, dict[str, Any]] = {}
LOGGER = logging.getLogger(__name__)
_SEEDVR2_DECODE_TILE_OVERLAP_BY_SIZE = {
    1024: 256,
    896: 224,
    768: 192,
    640: 160,
}


@dataclass(frozen=True)
class SeedVR2Attempt:
    tier: int
    batch_size: int
    uniform_batch_size: bool
    dit_offload_device: str
    vae_offload_device: str
    tensor_offload_device: str
    blocks_to_swap: int
    swap_io_components: bool
    vae_encode_tiled: bool
    vae_encode_tile_size: int
    vae_encode_tile_overlap: int
    vae_decode_tiled: bool
    vae_decode_tile_size: int
    vae_decode_tile_overlap: int
    attention_mode: str
    color_correction: str

    @property
    def cache_key(self) -> str:
        return (
            f"tier{self.tier}"
            f":attn={self.attention_mode}"
            f":dit={self.dit_offload_device}"
            f":vae={self.vae_offload_device}"
            f":tensor={self.tensor_offload_device}"
            f":swap={self.blocks_to_swap}/{int(self.swap_io_components)}"
            f":tile={int(self.vae_encode_tiled)}/{int(self.vae_decode_tiled)}"
            f":ts={self.vae_encode_tile_size}/{self.vae_decode_tile_size}"
            f":ov={self.vae_encode_tile_overlap}/{self.vae_decode_tile_overlap}"
        )


@dataclass(frozen=True)
class SeedVR2StillImageConfig:
    input_noise_scale: float = 0.0
    latent_noise_scale: float = 0.0
    color_correction: str = SEEDVR2_DEFAULT_COLOR_CORRECTION
    vae_encode_tiled: bool | None = None
    vae_encode_tile_size: int | None = None
    vae_encode_tile_overlap: int | None = None
    vae_decode_tiled: bool | None = None
    vae_decode_tile_size: int | None = None
    vae_decode_tile_overlap: int | None = None

    @property
    def vae_tiling_policy(self) -> str:
        if any(
            value is not None
            for value in (
                self.vae_encode_tiled,
                self.vae_encode_tile_size,
                self.vae_encode_tile_overlap,
                self.vae_decode_tiled,
                self.vae_decode_tile_size,
                self.vae_decode_tile_overlap,
            )
        ):
            return "forced"
        return "auto"


@dataclass(frozen=True)
class SeedVR2TargetSpec:
    target_short_edge: int
    target_max_edge: int
    target_width: int | None = None
    target_height: int | None = None


@dataclass
class SeedVR2UpscaleResult:
    image: Image.Image
    duration_ms: int
    input_width: int
    input_height: int
    output_width: int
    output_height: int
    engine: str
    model_repo: str
    model_revision: str
    model_dit_filename: str
    model_vae_filename: str
    runtime_profile: str
    device: str
    dtype: str
    vram_peak_mb: int | None
    infer_ms: int
    load_ms: int | None
    total_ms: int
    fallback_tier: int
    runner_reused: bool
    execution_strategy: str
    runtime_preset: str
    target_short_edge: int
    target_max_edge: int
    target_width: int | None
    target_height: int | None
    offload_mode: str
    dit_offload_device: str
    vae_offload_device: str
    tensor_offload_device: str
    blocks_to_swap: int
    swap_io_components: bool
    batch_size: int
    attention_mode: str
    color_correction: str
    input_noise_scale: float
    latent_noise_scale: float
    vae_tiling_policy: str
    vae_encode_tiled: bool
    vae_encode_tile_size: int
    vae_encode_tile_overlap: int
    vae_decode_tiled: bool
    vae_decode_tile_size: int
    vae_decode_tile_overlap: int
    attempt_count: int
    attempts: list[dict[str, Any]]
    policy_source: str
    timeout_hit: bool

    def telemetry_dict(self) -> dict[str, Any]:
        return {
            "upscale_engine": self.engine,
            "upscale_model_repo": self.model_repo,
            "upscale_model_revision": self.model_revision,
            "upscale_model_dit_filename": self.model_dit_filename,
            "upscale_model_vae_filename": self.model_vae_filename,
            "upscale_dtype": self.dtype,
            "upscale_vram_peak_mb": self.vram_peak_mb,
            "upscale_infer_ms": self.infer_ms,
            "upscale_load_ms": self.load_ms,
            "upscale_total_ms": self.total_ms,
            "upscale_success": True,
            "upscale_fallback_tier": self.fallback_tier,
            "upscale_runner_reused": self.runner_reused,
            "upscale_execution_strategy": self.execution_strategy,
            "upscale_runtime_preset": self.runtime_preset,
            "upscale_target_short_edge": self.target_short_edge,
            "upscale_target_max_edge": self.target_max_edge,
            "upscale_target_width": self.target_width,
            "upscale_target_height": self.target_height,
            "upscale_offload_mode": self.offload_mode,
            "upscale_dit_offload_device": self.dit_offload_device,
            "upscale_vae_offload_device": self.vae_offload_device,
            "upscale_tensor_offload_device": self.tensor_offload_device,
            "upscale_blocks_to_swap": self.blocks_to_swap,
            "upscale_swap_io_components": self.swap_io_components,
            "upscale_batch_size": self.batch_size,
            "upscale_attention_mode": self.attention_mode,
            "upscale_color_correction": self.color_correction,
            "upscale_input_noise_scale": self.input_noise_scale,
            "upscale_latent_noise_scale": self.latent_noise_scale,
            "upscale_vae_tiling_policy": self.vae_tiling_policy,
            "upscale_vae_encode_tiled": self.vae_encode_tiled,
            "upscale_vae_encode_tile_size": self.vae_encode_tile_size,
            "upscale_vae_encode_tile_overlap": self.vae_encode_tile_overlap,
            "upscale_vae_decode_tiled": self.vae_decode_tiled,
            "upscale_vae_decode_tile_size": self.vae_decode_tile_size,
            "upscale_vae_decode_tile_overlap": self.vae_decode_tile_overlap,
            "upscale_seed_attempt_count": self.attempt_count,
            "upscale_seed_attempts": self.attempts,
            "upscale_seed_policy_source": self.policy_source,
            "upscale_seed_timeout_hit": self.timeout_hit,
            "input_image_width": self.input_width,
            "input_image_height": self.input_height,
            "output_image_width": self.output_width,
            "output_image_height": self.output_height,
            "runtime_profile": self.runtime_profile,
            "device": self.device,
            "duration_ms": self.duration_ms,
        }


def _runtime_script_path(settings: AppSettings) -> Path:
    override = os.getenv("JUSTRAYZIST_SEEDVR2_SCRIPT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (
        settings.paths.models_dir
        / "seedvr2"
        / "runtime"
        / "ComfyUI-SeedVR2_VideoUpscaler"
        / "inference_cli.py"
    )


def _ensure_runtime_allocator_env_compat(runtime_script: Path) -> None:
    try:
        original = runtime_script.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "SeedVR2 runtime compatibility check skipped (read failed): %s",
            exc,
        )
        return

    if "PYTORCH_CUDA_ALLOC_CONF" not in original:
        return

    updated = original.replace("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")
    if updated == original:
        return

    try:
        runtime_script.write_text(updated, encoding="utf-8")
        LOGGER.debug(
            "Patched SeedVR2 runtime allocator env var to PYTORCH_ALLOC_CONF: %s",
            runtime_script,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "Failed to patch SeedVR2 runtime allocator env var: %s",
            exc,
        )


def _seedvr2_model_dir(settings: AppSettings) -> Path:
    return settings.paths.models_dir / "seedvr2"


def _tail(text: str, max_lines: int = 40) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def _parse_bool_env(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _is_verbose_runtime_enabled() -> bool:
    return _parse_bool_env(SEEDVR2_VERBOSE_RUNTIME_ENV, default=False)


@contextlib.contextmanager
def _suppress_noncritical_logs():
    # Keep runtime errors visible while suppressing info/warnings noise.
    previous_disable = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        yield
    finally:
        logging.disable(previous_disable)


class _CapturedRuntimeStream:
    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self._cached_value: str | None = None

    def getvalue(self) -> str:
        if self._cached_value is not None:
            return self._cached_value
        self._handle.flush()
        position = self._handle.tell()
        self._handle.seek(0)
        data = self._handle.read()
        self._handle.seek(position)
        return str(data)

    def finalize(self) -> None:
        if self._cached_value is None:
            self._cached_value = self.getvalue()


@contextlib.contextmanager
def _runtime_output_context(*, verbose_runtime: bool):
    if verbose_runtime:
        yield None, None
        return
    stdout_tmp = tempfile.TemporaryFile(mode="w+t", encoding="utf-8", errors="replace")
    stderr_tmp = tempfile.TemporaryFile(mode="w+t", encoding="utf-8", errors="replace")
    captured_stdout = _CapturedRuntimeStream(stdout_tmp)
    captured_stderr = _CapturedRuntimeStream(stderr_tmp)
    try:
        with (
            contextlib.redirect_stdout(stdout_tmp),
            contextlib.redirect_stderr(stderr_tmp),
            _suppress_noncritical_logs(),
        ):
            yield captured_stdout, captured_stderr
    finally:
        captured_stdout.finalize()
        captured_stderr.finalize()
        stdout_tmp.close()
        stderr_tmp.close()


@contextlib.contextmanager
def _seedvr2_tmpdir(*, tmp_root: Path):
    import shutil

    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = tmp_root / f"tmp_{uuid4().hex}"
    tmp_dir.mkdir(parents=False, exist_ok=False)
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _resolution_targets(width: int, height: int) -> tuple[int, int]:
    short_side = min(width, height)
    long_side = max(width, height)
    target_short = max(64, short_side * 2)
    target_long = max(target_short, long_side * 2)
    return target_short, target_long


def build_seedvr2_target_spec_for_scale(
    width: int,
    height: int,
    scale: int,
) -> SeedVR2TargetSpec:
    source_width = max(1, int(width))
    source_height = max(1, int(height))
    normalized_scale = max(1, int(scale))
    target_width = max(64, source_width * normalized_scale)
    target_height = max(64, source_height * normalized_scale)
    return SeedVR2TargetSpec(
        target_short_edge=min(target_width, target_height),
        target_max_edge=max(target_width, target_height),
        target_width=target_width,
        target_height=target_height,
    )


def build_seedvr2_target_spec_for_long_edge(
    width: int,
    height: int,
    target_long_edge: int,
) -> SeedVR2TargetSpec:
    source_width = max(1, int(width))
    source_height = max(1, int(height))
    requested_long_edge = max(64, int(target_long_edge))
    source_long_edge = max(source_width, source_height)
    scale = requested_long_edge / float(source_long_edge)
    target_width = max(64, int(round(source_width * scale)))
    target_height = max(64, int(round(source_height * scale)))
    return SeedVR2TargetSpec(
        target_short_edge=min(target_width, target_height),
        target_max_edge=max(target_width, target_height),
        target_width=target_width,
        target_height=target_height,
    )


def _normalize_execution_strategy(raw_value: str | None) -> str:
    normalized = str(raw_value or SEEDVR2_EXECUTION_STRATEGY_CHAIN_X2).strip().lower()
    if normalized not in SEEDVR2_EXECUTION_STRATEGIES:
        raise ValueError(
            "Invalid SeedVR2 execution_strategy "
            f"'{raw_value}'. Allowed: {', '.join(SEEDVR2_EXECUTION_STRATEGIES)}."
        )
    return normalized


def _normalize_runtime_preset(raw_value: str | None) -> str:
    normalized = str(raw_value or SEEDVR2_RUNTIME_PRESET_CURRENT_BASELINE).strip().lower()
    if normalized not in SEEDVR2_RUNTIME_PRESETS:
        raise ValueError(
            "Invalid SeedVR2 runtime_preset "
            f"'{raw_value}'. Allowed: {', '.join(SEEDVR2_RUNTIME_PRESETS)}."
        )
    return normalized


def _resolve_target_spec(
    *,
    source_width: int,
    source_height: int,
    target_spec: SeedVR2TargetSpec | None,
) -> SeedVR2TargetSpec:
    if target_spec is None:
        return build_seedvr2_target_spec_for_scale(source_width, source_height, 2)

    target_short_edge = max(64, int(target_spec.target_short_edge))
    target_max_edge = max(target_short_edge, int(target_spec.target_max_edge))
    target_width = (
        max(64, int(target_spec.target_width))
        if target_spec.target_width is not None
        else None
    )
    target_height = (
        max(64, int(target_spec.target_height))
        if target_spec.target_height is not None
        else None
    )
    if (target_width is None) != (target_height is None):
        raise ValueError("SeedVR2TargetSpec target_width and target_height must both be provided or both omitted.")
    if target_width is not None and target_height is not None:
        derived_short = min(target_width, target_height)
        derived_long = max(target_width, target_height)
        target_short_edge = derived_short
        target_max_edge = derived_long

    return SeedVR2TargetSpec(
        target_short_edge=target_short_edge,
        target_max_edge=target_max_edge,
        target_width=target_width,
        target_height=target_height,
    )


def _resolve_still_image_config(
    config: SeedVR2StillImageConfig | None,
) -> SeedVR2StillImageConfig:
    resolved = config if isinstance(config, SeedVR2StillImageConfig) else SeedVR2StillImageConfig()
    input_noise = float(resolved.input_noise_scale)
    latent_noise = float(resolved.latent_noise_scale)
    if input_noise < 0.0 or input_noise > 1.0:
        raise ValueError("SeedVR2 input_noise_scale must be between 0.0 and 1.0.")
    if latent_noise < 0.0 or latent_noise > 1.0:
        raise ValueError("SeedVR2 latent_noise_scale must be between 0.0 and 1.0.")
    color_correction = str(resolved.color_correction or SEEDVR2_DEFAULT_COLOR_CORRECTION).strip()
    if not color_correction:
        raise ValueError("SeedVR2 color_correction must be a non-empty string.")

    normalized = replace(
        resolved,
        input_noise_scale=input_noise,
        latent_noise_scale=latent_noise,
        color_correction=color_correction,
    )
    if normalized.vae_encode_tile_size is not None and int(normalized.vae_encode_tile_size) <= 0:
        raise ValueError("SeedVR2 vae_encode_tile_size must be positive when provided.")
    if normalized.vae_decode_tile_size is not None and int(normalized.vae_decode_tile_size) <= 0:
        raise ValueError("SeedVR2 vae_decode_tile_size must be positive when provided.")
    if normalized.vae_encode_tile_overlap is not None and int(normalized.vae_encode_tile_overlap) < 0:
        raise ValueError("SeedVR2 vae_encode_tile_overlap cannot be negative.")
    if normalized.vae_decode_tile_overlap is not None and int(normalized.vae_decode_tile_overlap) < 0:
        raise ValueError("SeedVR2 vae_decode_tile_overlap cannot be negative.")
    if (
        normalized.vae_encode_tile_size is not None
        and normalized.vae_encode_tile_overlap is not None
        and int(normalized.vae_encode_tile_overlap) >= int(normalized.vae_encode_tile_size)
    ):
        raise ValueError("SeedVR2 vae_encode_tile_overlap must be smaller than vae_encode_tile_size.")
    if (
        normalized.vae_decode_tile_size is not None
        and normalized.vae_decode_tile_overlap is not None
        and int(normalized.vae_decode_tile_overlap) >= int(normalized.vae_decode_tile_size)
    ):
        raise ValueError("SeedVR2 vae_decode_tile_overlap must be smaller than vae_decode_tile_size.")
    return normalized


def _apply_still_image_config_to_attempt(
    attempt: SeedVR2Attempt,
    config: SeedVR2StillImageConfig,
) -> SeedVR2Attempt:
    return replace(
        attempt,
        color_correction=str(config.color_correction),
        vae_encode_tiled=attempt.vae_encode_tiled if config.vae_encode_tiled is None else bool(config.vae_encode_tiled),
        vae_encode_tile_size=(
            attempt.vae_encode_tile_size
            if config.vae_encode_tile_size is None
            else int(config.vae_encode_tile_size)
        ),
        vae_encode_tile_overlap=(
            attempt.vae_encode_tile_overlap
            if config.vae_encode_tile_overlap is None
            else int(config.vae_encode_tile_overlap)
        ),
        vae_decode_tiled=attempt.vae_decode_tiled if config.vae_decode_tiled is None else bool(config.vae_decode_tiled),
        vae_decode_tile_size=(
            attempt.vae_decode_tile_size
            if config.vae_decode_tile_size is None
            else int(config.vae_decode_tile_size)
        ),
        vae_decode_tile_overlap=(
            attempt.vae_decode_tile_overlap
            if config.vae_decode_tile_overlap is None
            else int(config.vae_decode_tile_overlap)
        ),
    )


def _detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        return "cpu"
    return "cpu"


def _is_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_attention_mode() -> str:
    forced = os.getenv("JUSTRAYZIST_SEEDVR2_ATTENTION", "").strip().lower()
    valid = {"sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"}
    if forced in valid:
        return forced
    if importlib.util.find_spec("flash_attn") is not None:
        return "flash_attn_2"
    return "sdpa"


def _silence_runtime_debug(runtime_module: Any) -> None:
    debug_obj = getattr(runtime_module, "debug", None)
    if debug_obj is None:
        return

    def _noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    debug_obj.enabled = False
    debug_obj.log = _noop
    debug_obj.print_header = _noop
    debug_obj.print_footer = _noop


def _ensure_runtime_module(runtime_script: Path) -> Any:
    global _RUNTIME_MODULE, _RUNTIME_SCRIPT
    with _RUNTIME_LOCK:
        if _RUNTIME_MODULE is not None and _RUNTIME_SCRIPT == runtime_script:
            return _RUNTIME_MODULE

        runtime_dir = str(runtime_script.parent.resolve())
        if runtime_dir not in sys.path:
            sys.path.insert(0, runtime_dir)

        if "inference_cli" in sys.modules:
            loaded = sys.modules["inference_cli"]
            loaded_file = str(Path(getattr(loaded, "__file__", "")).resolve()) if getattr(loaded, "__file__", None) else ""
            if loaded_file and loaded_file != str(runtime_script.resolve()):
                del sys.modules["inference_cli"]

        runtime_module = importlib.import_module("inference_cli")
        if not hasattr(runtime_module, "process_single_file"):
            raise RuntimeError("SeedVR2 runtime module is missing process_single_file().")
        _silence_runtime_debug(runtime_module)

        _RUNTIME_MODULE = runtime_module
        _RUNTIME_SCRIPT = runtime_script
        return _RUNTIME_MODULE


def _parse_positive_int_env(name: str) -> int | None:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value < 1:
        return None
    return value


def _resolve_attempt_timeout_seconds(timeout_seconds: int, max_dim: int) -> int:
    env_override = _parse_positive_int_env(SEEDVR2_ATTEMPT_TIMEOUT_ENV)
    if env_override is not None:
        return env_override

    requested = max(1, int(timeout_seconds))
    if max_dim >= 4096:
        adaptive_default = 180
    elif max_dim >= 3072:
        adaptive_default = 120
    else:
        adaptive_default = 75
    if requested >= SEEDVR2_DEFAULT_TIMEOUT_SECONDS:
        return adaptive_default
    return requested


def _resolve_max_attempts(profile_name: str, attempt_count: int) -> int:
    env_override = _parse_positive_int_env(SEEDVR2_MAX_ATTEMPTS_ENV)
    if env_override is not None:
        return max(1, min(env_override, attempt_count))
    normalized = profile_name.strip().lower()
    default_limit = 2 if normalized in {"high", "balanced"} else 3
    return max(1, min(default_limit, attempt_count))


def _gpu_identity() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            device = int(torch.cuda.current_device())
            name = str(torch.cuda.get_device_name(device)).strip().replace("|", "_")
            total = int(torch.cuda.get_device_properties(device).total_memory)
            return f"cuda:{name}:{total}"
    except Exception:
        return "unknown"
    return "cpu"


def _model_fingerprint(*, dit_path: Path, vae_path: Path) -> str:
    parts: list[str] = []
    for candidate in (dit_path, vae_path):
        stat = candidate.stat()
        parts.append(f"{candidate.name}:{stat.st_size}:{int(stat.st_mtime)}")
    return "|".join(parts)


def _still_image_config_fingerprint(config: SeedVR2StillImageConfig) -> str:
    return (
        f"input_noise={config.input_noise_scale:.4f}|"
        f"latent_noise={config.latent_noise_scale:.4f}|"
        f"color={config.color_correction}|"
        f"encode_tiled={config.vae_encode_tiled}|"
        f"encode_tile_size={config.vae_encode_tile_size}|"
        f"encode_tile_overlap={config.vae_encode_tile_overlap}|"
        f"decode_tiled={config.vae_decode_tiled}|"
        f"decode_tile_size={config.vae_decode_tile_size}|"
        f"decode_tile_overlap={config.vae_decode_tile_overlap}"
    )


def _policy_cache_path(settings: AppSettings) -> Path:
    return settings.paths.data_dir / SEEDVR2_POLICY_CACHE_FILENAME


def _load_policy_cache(settings: AppSettings) -> dict[str, dict[str, Any]]:
    path = _policy_cache_path(settings)
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(loaded, dict):
        return {}

    normalized: dict[str, dict[str, Any]] = {}
    for key, value in loaded.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        normalized[key] = value
    return normalized


def _save_policy_cache(settings: AppSettings, cache: dict[str, dict[str, Any]]) -> None:
    path = _policy_cache_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=True), encoding="utf-8")


def _policy_cache_key(
    *,
    runtime_profile: str,
    runtime_preset: str,
    target_spec: SeedVR2TargetSpec,
    dit_path: Path,
    vae_path: Path,
    still_image_config: SeedVR2StillImageConfig,
) -> str:
    return (
        f"{runtime_profile.strip().lower()}|"
        f"{runtime_preset.strip().lower()}|"
        f"target={target_spec.target_short_edge}x{target_spec.target_max_edge}|"
        f"{_gpu_identity()}|"
        f"{_model_fingerprint(dit_path=dit_path, vae_path=vae_path)}|"
        f"{_still_image_config_fingerprint(still_image_config)}"
    )


def _prioritize_cached_attempt(
    *,
    attempts: list[SeedVR2Attempt],
    settings: AppSettings,
    runtime_profile: str,
    runtime_preset: str,
    target_spec: SeedVR2TargetSpec,
    dit_path: Path,
    vae_path: Path,
    still_image_config: SeedVR2StillImageConfig,
) -> tuple[list[SeedVR2Attempt], str, str]:
    cache_key = _policy_cache_key(
        runtime_profile=runtime_profile,
        runtime_preset=runtime_preset,
        target_spec=target_spec,
        dit_path=dit_path,
        vae_path=vae_path,
        still_image_config=still_image_config,
    )
    cache = _load_policy_cache(settings)
    record = cache.get(cache_key, {})
    cached_attempt_key = str(record.get("attempt_cache_key") or "").strip()

    ordered = list(attempts)
    policy_source = "default"
    if cached_attempt_key:
        cached_index = next(
            (idx for idx, attempt in enumerate(ordered) if attempt.cache_key == cached_attempt_key),
            None,
        )
        if cached_index is not None:
            policy_source = "cached"
            if cached_index > 0:
                ordered.insert(0, ordered.pop(cached_index))
    return ordered, policy_source, cache_key


def _record_successful_attempt(
    *,
    settings: AppSettings,
    cache_key: str,
    attempt: SeedVR2Attempt,
) -> None:
    cache = _load_policy_cache(settings)
    cache[cache_key] = {
        "attempt_cache_key": attempt.cache_key,
        "attempt_tier": int(attempt.tier),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_policy_cache(settings, cache)


def _attempt_to_record(
    *,
    attempt: SeedVR2Attempt,
    still_image_config: SeedVR2StillImageConfig,
    status: str,
    duration_ms: int,
    timed_out: bool,
    error: str = "",
) -> dict[str, Any]:
    return {
        "tier": int(attempt.tier),
        "status": status,
        "duration_ms": int(duration_ms),
        "timed_out": bool(timed_out),
        "offload_mode": _offload_mode(attempt),
        "attention_mode": str(attempt.attention_mode),
        "color_correction": str(attempt.color_correction),
        "input_noise_scale": float(still_image_config.input_noise_scale),
        "latent_noise_scale": float(still_image_config.latent_noise_scale),
        "vae_tiling_policy": still_image_config.vae_tiling_policy,
        "vae_encode_tiled": bool(attempt.vae_encode_tiled),
        "vae_encode_tile_size": int(attempt.vae_encode_tile_size),
        "vae_encode_tile_overlap": int(attempt.vae_encode_tile_overlap),
        "vae_decode_tiled": bool(attempt.vae_decode_tiled),
        "vae_decode_tile_size": int(attempt.vae_decode_tile_size),
        "vae_decode_tile_overlap": int(attempt.vae_decode_tile_overlap),
        "dit_offload_device": str(attempt.dit_offload_device),
        "vae_offload_device": str(attempt.vae_offload_device),
        "tensor_offload_device": str(attempt.tensor_offload_device),
        "blocks_to_swap": int(attempt.blocks_to_swap),
        "swap_io_components": bool(attempt.swap_io_components),
        "error": str(error or ""),
    }


def _execute_seedvr2_attempt(payload: dict[str, Any]) -> dict[str, Any]:
    captured_stdout_text = ""
    captured_stderr_text = ""
    captured_stdout = None
    captured_stderr = None
    try:
        runtime_script = Path(str(payload["runtime_script"]))
        verbose_runtime = bool(payload.get("verbose_runtime", False))
        with _runtime_output_context(verbose_runtime=verbose_runtime) as (captured_stdout, captured_stderr):
            runtime_module = _ensure_runtime_module(runtime_script)
            args = argparse.Namespace(**dict(payload["args"]))
            runtime_module.process_single_file(
                str(payload["input_path"]),
                args,
                device_list=["0"],
                output_path=str(payload["output_path"]),
                format_auto_detected=False,
                runner_cache={},
            )
        if captured_stdout is not None:
            captured_stdout_text = captured_stdout.getvalue()
        if captured_stderr is not None:
            captured_stderr_text = captured_stderr.getvalue()
        return {
            "status": "success",
            "stdout_tail": _tail(captured_stdout_text, max_lines=6),
            "stderr_tail": _tail(captured_stderr_text, max_lines=6),
        }
    except Exception as exc:  # noqa: BLE001
        if captured_stdout is not None:
            captured_stdout_text = captured_stdout.getvalue()
        if captured_stderr is not None:
            captured_stderr_text = captured_stderr.getvalue()
        tb_text = traceback.format_exc()
        return {
            "status": "error",
            "error": str(exc),
            "traceback_tail": _tail(tb_text, max_lines=8),
            "stdout_tail": _tail(captured_stdout_text, max_lines=6),
            "stderr_tail": _tail(captured_stderr_text, max_lines=6),
        }


def _seedvr2_attempt_worker(payload: dict[str, Any], result_queue: Any) -> None:
    result_queue.put(_execute_seedvr2_attempt(payload))


def _run_attempt_with_hard_timeout(
    *,
    runtime_script: Path,
    args: argparse.Namespace,
    input_path: Path,
    output_path: Path,
    timeout_seconds: int,
    verbose_runtime: bool,
    is_cancel_requested: Any | None = None,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "runtime_script": str(runtime_script),
        "args": vars(args),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "verbose_runtime": bool(verbose_runtime),
    }
    try:
        context = multiprocessing.get_context("spawn")
        result_queue = context.Queue()
        process = context.Process(
            target=_seedvr2_attempt_worker,
            args=(payload, result_queue),
            daemon=True,
        )
        process.start()
    except (PermissionError, OSError) as exc:
        LOGGER.warning(
            "SeedVR2 hard-timeout subprocess unavailable; running inline without process timeout. Reason: %s",
            exc,
        )
        message = _execute_seedvr2_attempt(payload)
        return str(message.get("status") or "error"), message
    cancel_requested = is_cancel_requested if callable(is_cancel_requested) else lambda: False
    timeout_seconds = max(1, int(timeout_seconds))
    deadline = now_perf() + timeout_seconds
    while process.is_alive():
        if cancel_requested():
            process.terminate()
            process.join(timeout=5)
            message = {
                "status": "cancelled",
                "error": "SeedVR2 attempt cancelled by user request.",
            }
            result_queue.close()
            result_queue.join_thread()
            return "cancelled", message
        remaining = deadline - now_perf()
        if remaining <= 0:
            break
        process.join(timeout=min(0.2, max(0.01, remaining)))

    message: dict[str, Any] = {}
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        message = {
            "status": "timeout",
            "error": (
                f"SeedVR2 attempt timed out after {int(timeout_seconds)}s and was terminated."
            ),
        }
    else:
        try:
            maybe_message = result_queue.get_nowait()
            if isinstance(maybe_message, dict):
                message = maybe_message
        except Exception:
            message = {}
        if not message:
            if process.exitcode == 0:
                message = {"status": "success"}
            else:
                message = {
                    "status": "error",
                    "error": f"SeedVR2 runtime process exited with code {process.exitcode}.",
                }

    result_queue.close()
    result_queue.join_thread()
    return str(message.get("status") or "error"), message


def clear_seedvr2_runtime_cache(profile_name: str | None = None) -> None:
    with _RUNTIME_LOCK:
        if profile_name is None:
            _RUNNER_CACHE_BY_KEY.clear()
            return
        prefix = f"{profile_name.lower()}:"
        stale_keys = [key for key in _RUNNER_CACHE_BY_KEY if key.startswith(prefix)]
        for key in stale_keys:
            _RUNNER_CACHE_BY_KEY.pop(key, None)


def _attempts_for_profile(profile_name: str, max_dim: int, attention_mode: str) -> list[SeedVR2Attempt]:
    normalized = profile_name.strip().lower()
    cuda = _is_cuda_available()

    if normalized == "high":
        return [
            SeedVR2Attempt(
                tier=0,
                batch_size=1,
                uniform_batch_size=False,
                dit_offload_device="0" if cuda else "none",
                vae_offload_device="0" if cuda else "none",
                tensor_offload_device="none" if cuda else "cpu",
                blocks_to_swap=0,
                swap_io_components=False,
                vae_encode_tiled=False,
                vae_encode_tile_size=1024,
                vae_encode_tile_overlap=128,
                vae_decode_tiled=False,
                vae_decode_tile_size=1024,
                vae_decode_tile_overlap=128,
                attention_mode=attention_mode,
                color_correction=SEEDVR2_DEFAULT_COLOR_CORRECTION,
            ),
            SeedVR2Attempt(
                tier=1,
                batch_size=1,
                uniform_batch_size=False,
                dit_offload_device="cpu",
                vae_offload_device="cpu",
                tensor_offload_device="cpu",
                blocks_to_swap=0,
                swap_io_components=False,
                vae_encode_tiled=False,
                vae_encode_tile_size=1024,
                vae_encode_tile_overlap=128,
                vae_decode_tiled=False,
                vae_decode_tile_size=1024,
                vae_decode_tile_overlap=128,
                attention_mode=attention_mode,
                color_correction=SEEDVR2_DEFAULT_COLOR_CORRECTION,
            ),
            SeedVR2Attempt(
                tier=2,
                batch_size=1,
                uniform_batch_size=False,
                dit_offload_device="cpu",
                vae_offload_device="cpu",
                tensor_offload_device="cpu",
                blocks_to_swap=24 if cuda else 0,
                swap_io_components=cuda,
                vae_encode_tiled=max_dim > 1536,
                vae_encode_tile_size=1024,
                vae_encode_tile_overlap=128,
                vae_decode_tiled=max_dim > 1536,
                vae_decode_tile_size=1024,
                vae_decode_tile_overlap=_decode_tile_overlap_for_size(1024),
                attention_mode=attention_mode,
                color_correction=SEEDVR2_DEFAULT_COLOR_CORRECTION,
            ),
        ]

    if normalized == "balanced":
        return [
            SeedVR2Attempt(
                tier=0,
                batch_size=1,
                uniform_batch_size=False,
                dit_offload_device="0" if cuda else "none",
                vae_offload_device="0" if cuda else "none",
                tensor_offload_device="none" if cuda else "cpu",
                blocks_to_swap=0,
                swap_io_components=False,
                vae_encode_tiled=max_dim > 2048,
                vae_encode_tile_size=1024,
                vae_encode_tile_overlap=128,
                vae_decode_tiled=max_dim > 2048,
                vae_decode_tile_size=1024,
                vae_decode_tile_overlap=_decode_tile_overlap_for_size(1024),
                attention_mode=attention_mode,
                color_correction=SEEDVR2_DEFAULT_COLOR_CORRECTION,
            ),
            SeedVR2Attempt(
                tier=1,
                batch_size=1,
                uniform_batch_size=False,
                dit_offload_device="cpu",
                vae_offload_device="cpu",
                tensor_offload_device="cpu",
                blocks_to_swap=18 if cuda else 0,
                swap_io_components=cuda,
                vae_encode_tiled=max_dim > 1536,
                vae_encode_tile_size=1024,
                vae_encode_tile_overlap=128,
                vae_decode_tiled=max_dim > 1536,
                vae_decode_tile_size=1024,
                vae_decode_tile_overlap=_decode_tile_overlap_for_size(1024),
                attention_mode=attention_mode,
                color_correction=SEEDVR2_DEFAULT_COLOR_CORRECTION,
            ),
            SeedVR2Attempt(
                tier=2,
                batch_size=1,
                uniform_batch_size=False,
                dit_offload_device="cpu",
                vae_offload_device="cpu",
                tensor_offload_device="cpu",
                blocks_to_swap=24 if cuda else 0,
                swap_io_components=cuda,
                vae_encode_tiled=True,
                vae_encode_tile_size=896,
                vae_encode_tile_overlap=128,
                vae_decode_tiled=True,
                vae_decode_tile_size=896,
                vae_decode_tile_overlap=_decode_tile_overlap_for_size(896),
                attention_mode=attention_mode,
                color_correction=SEEDVR2_DEFAULT_COLOR_CORRECTION,
            ),
        ]

    return [
        SeedVR2Attempt(
            tier=0,
            batch_size=1,
            uniform_batch_size=False,
            dit_offload_device="cpu",
            vae_offload_device="cpu",
            tensor_offload_device="cpu",
            blocks_to_swap=24 if cuda else 0,
            swap_io_components=cuda,
            vae_encode_tiled=True,
            vae_encode_tile_size=896,
            vae_encode_tile_overlap=128,
            vae_decode_tiled=True,
            vae_decode_tile_size=896,
            vae_decode_tile_overlap=_decode_tile_overlap_for_size(896),
            attention_mode=attention_mode,
            color_correction=SEEDVR2_DEFAULT_COLOR_CORRECTION,
        ),
        SeedVR2Attempt(
            tier=1,
            batch_size=1,
            uniform_batch_size=False,
            dit_offload_device="cpu",
            vae_offload_device="cpu",
            tensor_offload_device="cpu",
            blocks_to_swap=32 if cuda else 0,
            swap_io_components=cuda,
            vae_encode_tiled=True,
            vae_encode_tile_size=768,
            vae_encode_tile_overlap=128,
            vae_decode_tiled=True,
            vae_decode_tile_size=768,
            vae_decode_tile_overlap=_decode_tile_overlap_for_size(768),
            attention_mode=attention_mode,
            color_correction=SEEDVR2_DEFAULT_COLOR_CORRECTION,
        ),
        SeedVR2Attempt(
            tier=2,
            batch_size=1,
            uniform_batch_size=False,
            dit_offload_device="cpu",
            vae_offload_device="cpu",
            tensor_offload_device="cpu",
            blocks_to_swap=32 if cuda else 0,
            swap_io_components=cuda,
            vae_encode_tiled=True,
            vae_encode_tile_size=640,
            vae_encode_tile_overlap=128,
            vae_decode_tiled=True,
            vae_decode_tile_size=640,
            vae_decode_tile_overlap=_decode_tile_overlap_for_size(640),
            attention_mode=attention_mode,
            color_correction=SEEDVR2_DEFAULT_COLOR_CORRECTION,
        ),
    ]


def _with_forced_tiling(
    attempt: SeedVR2Attempt,
    *,
    tile_size: int,
    tile_overlap: int,
    min_blockswap: int,
) -> SeedVR2Attempt:
    cuda = _is_cuda_available()
    can_enable_blockswap = str(attempt.dit_offload_device).strip().lower() == "cpu"
    return replace(
        attempt,
        blocks_to_swap=(
            max(int(attempt.blocks_to_swap), int(min_blockswap))
            if cuda and can_enable_blockswap
            else int(attempt.blocks_to_swap)
        ),
        swap_io_components=bool(
            attempt.swap_io_components
            or (cuda and can_enable_blockswap and int(min_blockswap) > 0)
        ),
        vae_encode_tiled=True,
        vae_encode_tile_size=int(tile_size),
        vae_encode_tile_overlap=int(tile_overlap),
        vae_decode_tiled=True,
        vae_decode_tile_size=int(tile_size),
        vae_decode_tile_overlap=_decode_tile_overlap_for_size(int(tile_size)),
    )


def _attempts_for_runtime_preset(
    profile_name: str,
    max_dim: int,
    attention_mode: str,
    runtime_preset: str,
) -> list[SeedVR2Attempt]:
    normalized_runtime_preset = _normalize_runtime_preset(runtime_preset)
    baseline_attempts = _attempts_for_profile(profile_name, max_dim, attention_mode)
    if normalized_runtime_preset == SEEDVR2_RUNTIME_PRESET_CURRENT_BASELINE:
        return baseline_attempts

    if normalized_runtime_preset == SEEDVR2_RUNTIME_PRESET_HIGHRES_TILED_1024:
        return [
            _with_forced_tiling(
                attempt,
                tile_size=1024,
                tile_overlap=128,
                min_blockswap=18 if idx == 0 else 24,
            )
            for idx, attempt in enumerate(baseline_attempts)
        ]

    if normalized_runtime_preset == SEEDVR2_RUNTIME_PRESET_HIGHRES_TILED_896:
        return [
            _with_forced_tiling(
                attempt,
                tile_size=896,
                tile_overlap=128,
                min_blockswap=24 if idx == 0 else 32,
            )
            for idx, attempt in enumerate(baseline_attempts)
        ]

    auto_tile_size = 896 if max_dim >= 8192 else 1024
    auto_min_blockswap = 24 if max_dim >= 6144 else 18
    return [
        _with_forced_tiling(
            attempt,
            tile_size=auto_tile_size,
            tile_overlap=128,
            min_blockswap=auto_min_blockswap if idx == 0 else max(auto_min_blockswap, 24),
        )
        for idx, attempt in enumerate(baseline_attempts)
    ]


def _make_runtime_args(
    *,
    input_path: Path,
    output_path: Path,
    model_dir: Path,
    target_spec: SeedVR2TargetSpec,
    seed: int,
    attempt: SeedVR2Attempt,
    still_image_config: SeedVR2StillImageConfig,
) -> argparse.Namespace:
    return argparse.Namespace(
        input=str(input_path),
        output=str(output_path),
        output_format="png",
        video_backend="opencv",
        use_10bit=False,
        model_dir=str(model_dir),
        dit_model=SEEDVR2_DIT_FILENAME,
        resolution=int(target_spec.target_short_edge),
        max_resolution=int(target_spec.target_max_edge),
        batch_size=int(attempt.batch_size),
        uniform_batch_size=bool(attempt.uniform_batch_size),
        seed=int(seed),
        skip_first_frames=0,
        load_cap=0,
        chunk_size=0,
        prepend_frames=0,
        temporal_overlap=0,
        color_correction=str(attempt.color_correction),
        input_noise_scale=float(still_image_config.input_noise_scale),
        latent_noise_scale=float(still_image_config.latent_noise_scale),
        cuda_device=None,
        dit_offload_device=str(attempt.dit_offload_device),
        vae_offload_device=str(attempt.vae_offload_device),
        tensor_offload_device=str(attempt.tensor_offload_device),
        blocks_to_swap=int(attempt.blocks_to_swap),
        swap_io_components=bool(attempt.swap_io_components),
        vae_encode_tiled=bool(attempt.vae_encode_tiled),
        vae_encode_tile_size=int(attempt.vae_encode_tile_size),
        vae_encode_tile_overlap=int(attempt.vae_encode_tile_overlap),
        vae_decode_tiled=bool(attempt.vae_decode_tiled),
        vae_decode_tile_size=int(attempt.vae_decode_tile_size),
        vae_decode_tile_overlap=int(attempt.vae_decode_tile_overlap),
        tile_debug="false",
        attention_mode=str(attempt.attention_mode),
        compile_dit=False,
        compile_vae=False,
        compile_backend="inductor",
        compile_mode="default",
        compile_fullgraph=False,
        compile_dynamic=False,
        compile_dynamo_cache_size_limit=64,
        compile_dynamo_recompile_limit=128,
        cache_dit=True,
        cache_vae=True,
        debug=False,
    )


def _offload_mode(attempt: SeedVR2Attempt) -> str:
    if attempt.dit_offload_device in {"0", "none"} and attempt.vae_offload_device in {"0", "none"}:
        return "full_cuda"
    if attempt.blocks_to_swap > 0:
        return "offload_blockswap"
    if attempt.dit_offload_device == "cpu" or attempt.vae_offload_device == "cpu":
        return "model_offload"
    return "custom"


def _ensure_seedvr2_runtime_dependencies() -> None:
    try:
        importlib.import_module("cv2")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SeedVR2 runtime dependency missing: cv2. "
            "Install OpenCV for the active Python environment or rerun the project bootstrap."
        ) from exc


def _decode_tile_overlap_for_size(tile_size: int) -> int:
    normalized_tile_size = int(tile_size)
    if normalized_tile_size in _SEEDVR2_DECODE_TILE_OVERLAP_BY_SIZE:
        return int(_SEEDVR2_DECODE_TILE_OVERLAP_BY_SIZE[normalized_tile_size])
    return max(128, normalized_tile_size // 4)


def _is_retryable_failure(exc: Exception) -> bool:
    text = str(exc).lower()
    retryable_fragments = (
        "out of memory",
        "oom",
        "cuda error",
        "cudnn",
        "allocation",
        "memory",
        "timed out",
    )
    return any(fragment in text for fragment in retryable_fragments)


def upscale_with_seedvr2(
    *,
    image: Image.Image,
    settings: AppSettings,
    runtime_profile: str,
    seed: int | None = None,
    timeout_seconds: int = SEEDVR2_DEFAULT_TIMEOUT_SECONDS,
    reuse_runner: bool = True,
    is_cancel_requested: Any | None = None,
    model_dir_override: Path | None = None,
    model_repo_override: str | None = None,
    model_revision_override: str | None = None,
    dit_filename: str | None = None,
    vae_filename: str | None = None,
    still_image_config: SeedVR2StillImageConfig | None = None,
    target_spec: SeedVR2TargetSpec | None = None,
    execution_strategy: str | None = None,
    runtime_preset: str | None = None,
) -> SeedVR2UpscaleResult:
    if not isinstance(image, Image.Image):
        raise ValueError("image must be a PIL.Image.Image instance.")

    runtime_script = _runtime_script_path(settings)
    if not runtime_script.exists():
        raise RuntimeError(
            "SeedVR2 runtime script not found. "
            + setup_repair_hint(purpose="fetch runtime components")
        )
    _ensure_runtime_allocator_env_compat(runtime_script)
    _ensure_seedvr2_runtime_dependencies()

    model_dir = Path(model_dir_override).expanduser().resolve() if model_dir_override else _seedvr2_model_dir(settings)
    effective_model_repo = str(model_repo_override or SEEDVR2_MODEL_REPO)
    effective_model_revision = str(model_revision_override or SEEDVR2_MODEL_REVISION)
    effective_dit_filename = str(dit_filename or SEEDVR2_DIT_FILENAME)
    effective_vae_filename = str(vae_filename or SEEDVR2_VAE_FILENAME)
    dit_path = model_dir / effective_dit_filename
    vae_path = model_dir / effective_vae_filename
    if not dit_path.exists() or not vae_path.exists():
        raise RuntimeError(
            "SeedVR2 model files are missing. "
            f"Expected dit='{dit_path.name}' and vae='{vae_path.name}' under {model_dir}."
        )

    source = image.convert("RGB")
    effective_target_spec = _resolve_target_spec(
        source_width=source.width,
        source_height=source.height,
        target_spec=target_spec,
    )
    verbose_runtime = _is_verbose_runtime_enabled()
    attention_mode = _resolve_attention_mode()
    effective_still_image_config = _resolve_still_image_config(still_image_config)
    effective_execution_strategy = _normalize_execution_strategy(execution_strategy)
    effective_runtime_preset = _normalize_runtime_preset(runtime_preset)
    attempts = _attempts_for_runtime_preset(
        runtime_profile,
        max_dim=max(
            int(effective_target_spec.target_short_edge),
            int(effective_target_spec.target_max_edge),
        ),
        attention_mode=attention_mode,
        runtime_preset=effective_runtime_preset,
    )
    attempts = [
        _apply_still_image_config_to_attempt(attempt, effective_still_image_config)
        for attempt in attempts
    ]
    attempts, policy_source, policy_cache_key = _prioritize_cached_attempt(
        attempts=attempts,
        settings=settings,
        runtime_profile=runtime_profile,
        runtime_preset=effective_runtime_preset,
        target_spec=effective_target_spec,
        dit_path=dit_path,
        vae_path=vae_path,
        still_image_config=effective_still_image_config,
    )
    attempts = attempts[: _resolve_max_attempts(runtime_profile, len(attempts))]
    attempt_timeout_seconds = _resolve_attempt_timeout_seconds(
        timeout_seconds,
        max(
            int(effective_target_spec.target_short_edge),
            int(effective_target_spec.target_max_edge),
        ),
    )
    effective_seed = int(seed if seed is not None else 42)
    cancel_requested = is_cancel_requested if callable(is_cancel_requested) else lambda: False

    tmp_root = settings.paths.root_dir / ".build" / "seedvr2_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    timeout_hit = False
    attempt_records: list[dict[str, Any]] = []
    errors: list[str] = []
    with _seedvr2_tmpdir(tmp_root=tmp_root) as tmp_dir:
        input_path = tmp_dir / "input.png"
        source.save(input_path, format="PNG")

        for attempt_idx, attempt in enumerate(attempts):
            if cancel_requested():
                raise GenerationCancelledError("Upscale cancelled.")
            output_path = tmp_dir / f"output_tier_{attempt.tier}.png"
            args = _make_runtime_args(
                input_path=input_path,
                output_path=output_path,
                model_dir=model_dir,
                target_spec=effective_target_spec,
                seed=effective_seed,
                attempt=attempt,
                still_image_config=effective_still_image_config,
            )

            started = now_perf()
            status: str = "error"
            message: dict[str, Any] = {}
            try:
                status, message = _run_attempt_with_hard_timeout(
                    runtime_script=runtime_script,
                    args=args,
                    input_path=input_path,
                    output_path=output_path,
                    timeout_seconds=attempt_timeout_seconds,
                    verbose_runtime=verbose_runtime,
                    is_cancel_requested=cancel_requested,
                )
                duration_ms = int((now_perf() - started) * 1000)
                if status == "cancelled":
                    raise GenerationCancelledError(str(message.get("error") or "Upscale cancelled."))
                if status == "timeout":
                    timeout_hit = True
                    raise TimeoutError(str(message.get("error") or "SeedVR2 attempt timed out."))
                if status != "success":
                    raise RuntimeError(str(message.get("error") or "SeedVR2 attempt failed."))
                if not output_path.exists() or output_path.stat().st_size <= 0:
                    raise RuntimeError("SeedVR2 finished without producing a valid output PNG.")
                with Image.open(output_path) as result_file:
                    output_image = result_file.convert("RGB").copy()
                attempt_records.append(
                    _attempt_to_record(
                        attempt=attempt,
                        still_image_config=effective_still_image_config,
                        status="success",
                        duration_ms=duration_ms,
                        timed_out=False,
                    )
                )
                _record_successful_attempt(
                    settings=settings,
                    cache_key=policy_cache_key,
                    attempt=attempt,
                )
                return SeedVR2UpscaleResult(
                    image=output_image,
                    duration_ms=duration_ms,
                    input_width=int(source.width),
                    input_height=int(source.height),
                    output_width=int(output_image.width),
                    output_height=int(output_image.height),
                    engine="seedvr2",
                    model_repo=effective_model_repo,
                    model_revision=effective_model_revision,
                    model_dit_filename=effective_dit_filename,
                    model_vae_filename=effective_vae_filename,
                    runtime_profile=runtime_profile,
                    device=_detect_device(),
                    dtype="fp8_e4m3fn",
                    vram_peak_mb=None,
                    infer_ms=duration_ms,
                    load_ms=None,
                    total_ms=duration_ms,
                    fallback_tier=attempt.tier,
                    runner_reused=False,
                    execution_strategy=effective_execution_strategy,
                    runtime_preset=effective_runtime_preset,
                    target_short_edge=int(effective_target_spec.target_short_edge),
                    target_max_edge=int(effective_target_spec.target_max_edge),
                    target_width=(
                        None
                        if effective_target_spec.target_width is None
                        else int(effective_target_spec.target_width)
                    ),
                    target_height=(
                        None
                        if effective_target_spec.target_height is None
                        else int(effective_target_spec.target_height)
                    ),
                    offload_mode=_offload_mode(attempt),
                    dit_offload_device=str(attempt.dit_offload_device),
                    vae_offload_device=str(attempt.vae_offload_device),
                    tensor_offload_device=str(attempt.tensor_offload_device),
                    blocks_to_swap=int(attempt.blocks_to_swap),
                    swap_io_components=bool(attempt.swap_io_components),
                    batch_size=attempt.batch_size,
                    attention_mode=attempt.attention_mode,
                    color_correction=attempt.color_correction,
                    input_noise_scale=effective_still_image_config.input_noise_scale,
                    latent_noise_scale=effective_still_image_config.latent_noise_scale,
                    vae_tiling_policy=effective_still_image_config.vae_tiling_policy,
                    vae_encode_tiled=attempt.vae_encode_tiled,
                    vae_encode_tile_size=attempt.vae_encode_tile_size,
                    vae_encode_tile_overlap=attempt.vae_encode_tile_overlap,
                    vae_decode_tiled=attempt.vae_decode_tiled,
                    vae_decode_tile_size=attempt.vae_decode_tile_size,
                    vae_decode_tile_overlap=attempt.vae_decode_tile_overlap,
                    attempt_count=len(attempt_records),
                    attempts=attempt_records,
                    policy_source=policy_source,
                    timeout_hit=timeout_hit,
                )
            except Exception as exc:  # noqa: BLE001
                duration_ms = int((now_perf() - started) * 1000)
                stdout_tail = _tail(str(message.get("stdout_tail") or ""), max_lines=6)
                stderr_tail = _tail(str(message.get("stderr_tail") or ""), max_lines=6)
                traceback_tail = _tail(str(message.get("traceback_tail") or ""), max_lines=8)
                attempt_records.append(
                    _attempt_to_record(
                        attempt=attempt,
                        still_image_config=effective_still_image_config,
                        status="timeout" if isinstance(exc, TimeoutError) else "error",
                        duration_ms=duration_ms,
                        timed_out=isinstance(exc, TimeoutError),
                        error=str(exc),
                    )
                )
                errors.append(
                    "tier="
                    f"{attempt.tier} duration_ms={duration_ms} "
                    f"error={_tail(str(exc), max_lines=8)} "
                    f"stdout_tail={stdout_tail} "
                    f"stderr_tail={stderr_tail} "
                    f"traceback_tail={traceback_tail}"
                )
                if attempt_idx >= len(attempts) - 1 or not _is_retryable_failure(exc):
                    raise RuntimeError(
                        "SeedVR2 upscale failed. "
                        f"Profile={runtime_profile} "
                        f"attempt={attempt.tier} "
                        f"error={exc}"
                    ) from exc

    summary = " | ".join(errors[-3:]) if errors else "unknown failure"
    raise RuntimeError(f"SeedVR2 upscale failed after retries: {summary}")


def upscale_with_seedvr2_direct_x2(
    *,
    image: Image.Image,
    settings: AppSettings,
    runtime_profile: str,
    seed: int | None = None,
    timeout_seconds: int = SEEDVR2_DEFAULT_TIMEOUT_SECONDS,
    reuse_runner: bool = True,
    is_cancel_requested: Any | None = None,
    model_dir_override: Path | None = None,
    model_repo_override: str | None = None,
    model_revision_override: str | None = None,
    dit_filename: str | None = None,
    vae_filename: str | None = None,
    still_image_config: SeedVR2StillImageConfig | None = None,
    runtime_preset: str = SEEDVR2_RUNTIME_PRESET_HIGHRES_AUTO,
) -> SeedVR2UpscaleResult:
    target_spec = build_seedvr2_target_spec_for_scale(image.width, image.height, 2)
    return upscale_with_seedvr2(
        image=image,
        settings=settings,
        runtime_profile=runtime_profile,
        seed=seed,
        timeout_seconds=timeout_seconds,
        reuse_runner=reuse_runner,
        is_cancel_requested=is_cancel_requested,
        model_dir_override=model_dir_override,
        model_repo_override=model_repo_override,
        model_revision_override=model_revision_override,
        dit_filename=dit_filename,
        vae_filename=vae_filename,
        still_image_config=still_image_config,
        target_spec=target_spec,
        execution_strategy=SEEDVR2_EXECUTION_STRATEGY_DIRECT_TARGET,
        runtime_preset=runtime_preset,
    )
