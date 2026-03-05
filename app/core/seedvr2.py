from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import io
import json
import logging
import multiprocessing
import os
import sys
import tempfile
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from app.config.settings import AppSettings
from app.core.memory import now_perf

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

_RUNTIME_LOCK = threading.RLock()
_RUNTIME_MODULE: Any | None = None
_RUNTIME_SCRIPT: Path | None = None
_RUNNER_CACHE_BY_KEY: dict[str, dict[str, Any]] = {}
LOGGER = logging.getLogger(__name__)


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
    offload_mode: str
    batch_size: int
    attention_mode: str
    color_correction: str
    vae_encode_tiled: bool
    vae_decode_tiled: bool
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
            "upscale_offload_mode": self.offload_mode,
            "upscale_batch_size": self.batch_size,
            "upscale_attention_mode": self.attention_mode,
            "upscale_color_correction": self.color_correction,
            "upscale_vae_encode_tiled": self.vae_encode_tiled,
            "upscale_vae_decode_tiled": self.vae_decode_tiled,
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
        LOGGER.info(
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


@contextlib.contextmanager
def _runtime_output_context(*, verbose_runtime: bool):
    if verbose_runtime:
        yield None, None
        return
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    with (
        contextlib.redirect_stdout(captured_stdout),
        contextlib.redirect_stderr(captured_stderr),
        _suppress_noncritical_logs(),
    ):
        yield captured_stdout, captured_stderr


def _resolution_targets(width: int, height: int) -> tuple[int, int]:
    short_side = min(width, height)
    long_side = max(width, height)
    target_short = max(64, short_side * 2)
    target_long = max(target_short, long_side * 2)
    return target_short, target_long


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
        adaptive_default = 90
    elif max_dim >= 3072:
        adaptive_default = 60
    else:
        adaptive_default = SEEDVR2_DEFAULT_ATTEMPT_TIMEOUT_SECONDS
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
    dit_path: Path,
    vae_path: Path,
) -> str:
    return (
        f"{runtime_profile.strip().lower()}|"
        f"{_gpu_identity()}|"
        f"{_model_fingerprint(dit_path=dit_path, vae_path=vae_path)}"
    )


def _prioritize_cached_attempt(
    *,
    attempts: list[SeedVR2Attempt],
    settings: AppSettings,
    runtime_profile: str,
    dit_path: Path,
    vae_path: Path,
) -> tuple[list[SeedVR2Attempt], str, str]:
    cache_key = _policy_cache_key(
        runtime_profile=runtime_profile,
        dit_path=dit_path,
        vae_path=vae_path,
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
        "vae_encode_tiled": bool(attempt.vae_encode_tiled),
        "vae_decode_tiled": bool(attempt.vae_decode_tiled),
        "error": str(error or ""),
    }


def _seedvr2_attempt_worker(payload: dict[str, Any], result_queue: Any) -> None:
    captured_stdout_text = ""
    captured_stderr_text = ""
    try:
        runtime_script = Path(str(payload["runtime_script"]))
        runtime_module = _ensure_runtime_module(runtime_script)
        args = argparse.Namespace(**dict(payload["args"]))
        verbose_runtime = bool(payload.get("verbose_runtime", False))
        with _runtime_output_context(verbose_runtime=verbose_runtime) as (captured_stdout, captured_stderr):
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
        result_queue.put(
            {
                "status": "success",
                "stdout_tail": _tail(captured_stdout_text, max_lines=6),
                "stderr_tail": _tail(captured_stderr_text, max_lines=6),
            }
        )
    except Exception as exc:  # noqa: BLE001
        tb_text = traceback.format_exc()
        result_queue.put(
            {
                "status": "error",
                "error": str(exc),
                "traceback_tail": _tail(tb_text, max_lines=8),
                "stdout_tail": _tail(captured_stdout_text, max_lines=6),
                "stderr_tail": _tail(captured_stderr_text, max_lines=6),
            }
        )


def _run_attempt_with_hard_timeout(
    *,
    runtime_script: Path,
    args: argparse.Namespace,
    input_path: Path,
    output_path: Path,
    timeout_seconds: int,
    verbose_runtime: bool,
) -> tuple[str, dict[str, Any]]:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    payload = {
        "runtime_script": str(runtime_script),
        "args": vars(args),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "verbose_runtime": bool(verbose_runtime),
    }
    process = context.Process(
        target=_seedvr2_attempt_worker,
        args=(payload, result_queue),
        daemon=True,
    )
    process.start()
    process.join(timeout=max(1, int(timeout_seconds)))

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
                vae_decode_tile_overlap=128,
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
                blocks_to_swap=18 if cuda else 0,
                swap_io_components=cuda,
                vae_encode_tiled=max_dim > 1536,
                vae_encode_tile_size=1024,
                vae_encode_tile_overlap=128,
                vae_decode_tiled=max_dim > 1536,
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
                vae_encode_tiled=True,
                vae_encode_tile_size=896,
                vae_encode_tile_overlap=128,
                vae_decode_tiled=True,
                vae_decode_tile_size=896,
                vae_decode_tile_overlap=128,
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
            blocks_to_swap=32 if cuda else 0,
            swap_io_components=cuda,
            vae_encode_tiled=True,
            vae_encode_tile_size=768,
            vae_encode_tile_overlap=128,
            vae_decode_tiled=True,
            vae_decode_tile_size=768,
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
            blocks_to_swap=32 if cuda else 0,
            swap_io_components=cuda,
            vae_encode_tiled=True,
            vae_encode_tile_size=640,
            vae_encode_tile_overlap=128,
            vae_decode_tiled=True,
            vae_decode_tile_size=640,
            vae_decode_tile_overlap=128,
            attention_mode=attention_mode,
            color_correction=SEEDVR2_DEFAULT_COLOR_CORRECTION,
        ),
    ]


def _make_runtime_args(
    *,
    input_path: Path,
    output_path: Path,
    model_dir: Path,
    target_short: int,
    target_long: int,
    seed: int,
    attempt: SeedVR2Attempt,
) -> argparse.Namespace:
    return argparse.Namespace(
        input=str(input_path),
        output=str(output_path),
        output_format="png",
        video_backend="opencv",
        use_10bit=False,
        model_dir=str(model_dir),
        dit_model=SEEDVR2_DIT_FILENAME,
        resolution=int(target_short),
        max_resolution=int(target_long),
        batch_size=int(attempt.batch_size),
        uniform_batch_size=bool(attempt.uniform_batch_size),
        seed=int(seed),
        skip_first_frames=0,
        load_cap=0,
        chunk_size=0,
        prepend_frames=0,
        temporal_overlap=0,
        color_correction=str(attempt.color_correction),
        input_noise_scale=0.0,
        latent_noise_scale=0.0,
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
) -> SeedVR2UpscaleResult:
    if not isinstance(image, Image.Image):
        raise ValueError("image must be a PIL.Image.Image instance.")

    runtime_script = _runtime_script_path(settings)
    if not runtime_script.exists():
        raise RuntimeError(
            "SeedVR2 runtime script not found. Run .\\RunMeFirst.bat to fetch runtime components."
        )
    _ensure_runtime_allocator_env_compat(runtime_script)

    model_dir = _seedvr2_model_dir(settings)
    dit_path = model_dir / SEEDVR2_DIT_FILENAME
    vae_path = model_dir / SEEDVR2_VAE_FILENAME
    if not dit_path.exists() or not vae_path.exists():
        raise RuntimeError(
            "SeedVR2 model files are missing. Run .\\RunMeFirst.bat to download SeedVR2 assets."
        )

    source = image.convert("RGB")
    target_short, target_long = _resolution_targets(source.width, source.height)
    verbose_runtime = _is_verbose_runtime_enabled()
    attention_mode = _resolve_attention_mode()
    attempts = _attempts_for_profile(
        runtime_profile,
        max_dim=max(target_short, target_long),
        attention_mode=attention_mode,
    )
    attempts, policy_source, policy_cache_key = _prioritize_cached_attempt(
        attempts=attempts,
        settings=settings,
        runtime_profile=runtime_profile,
        dit_path=dit_path,
        vae_path=vae_path,
    )
    attempts = attempts[: _resolve_max_attempts(runtime_profile, len(attempts))]
    attempt_timeout_seconds = _resolve_attempt_timeout_seconds(
        timeout_seconds,
        max(target_short, target_long),
    )
    effective_seed = int(seed if seed is not None else 42)

    tmp_root = settings.paths.root_dir / ".build" / "seedvr2_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    timeout_hit = False
    attempt_records: list[dict[str, Any]] = []
    errors: list[str] = []
    with tempfile.TemporaryDirectory(dir=str(tmp_root)) as tmp_dir_raw:
        tmp_dir = Path(tmp_dir_raw)
        input_path = tmp_dir / "input.png"
        source.save(input_path, format="PNG")

        for attempt_idx, attempt in enumerate(attempts):
            output_path = tmp_dir / f"output_tier_{attempt.tier}.png"
            args = _make_runtime_args(
                input_path=input_path,
                output_path=output_path,
                model_dir=model_dir,
                target_short=target_short,
                target_long=target_long,
                seed=effective_seed,
                attempt=attempt,
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
                )
                duration_ms = int((now_perf() - started) * 1000)
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
                    model_repo=SEEDVR2_MODEL_REPO,
                    model_revision=SEEDVR2_MODEL_REVISION,
                    model_dit_filename=SEEDVR2_DIT_FILENAME,
                    model_vae_filename=SEEDVR2_VAE_FILENAME,
                    runtime_profile=runtime_profile,
                    device=_detect_device(),
                    dtype="fp8_e4m3fn",
                    vram_peak_mb=None,
                    infer_ms=duration_ms,
                    load_ms=None,
                    total_ms=duration_ms,
                    fallback_tier=attempt.tier,
                    runner_reused=False,
                    offload_mode=_offload_mode(attempt),
                    batch_size=attempt.batch_size,
                    attention_mode=attempt.attention_mode,
                    color_correction=attempt.color_correction,
                    vae_encode_tiled=attempt.vae_encode_tiled,
                    vae_decode_tiled=attempt.vae_decode_tiled,
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
