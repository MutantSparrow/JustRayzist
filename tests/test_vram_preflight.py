from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from app.config.profiles import RUNTIME_PROFILES
from app.core.backends.diffusers_zimage import (
    DiffusersZImageBackend,
    VramPreflightResult,
)
from app.core.worker.types import GenerationRequest, LoraSelection


class _NamedContext:
    def __init__(self, probe, name: str) -> None:
        self._probe = probe
        self._name = name

    def __enter__(self):
        self._probe.active = self._name
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._probe.active = None


class _ContextProbe:
    def __init__(self) -> None:
        self.active: str | None = None

    def context(self, name: str) -> _NamedContext:
        return _NamedContext(self, name)


def _make_backend(profile_name: str = "high") -> DiffusersZImageBackend:
    settings = SimpleNamespace(runtime_profile=RUNTIME_PROFILES[profile_name])
    model_pack = SimpleNamespace(name="Rayzist_fp8_storage")
    return DiffusersZImageBackend(settings=settings, model_pack=model_pack)


def _torch_available() -> SimpleNamespace:
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))


def test_vram_preflight_passes_without_cleanup_when_free_memory_is_healthy() -> None:
    backend = _make_backend("high")
    torch_module = _torch_available()
    calls: list[str] = []

    def fake_snapshot(_torch):
        calls.append("mem_get_info")
        return 13 * 1024**3, 24 * 1024**3

    def fake_clear(_torch):
        calls.append("empty_cache")

    backend._cuda_free_total_snapshot = fake_snapshot
    backend._clear_cuda_cache = fake_clear

    result = backend._run_vram_preflight(torch_module)

    assert result.checked is True
    assert result.cleanup_attempted is False
    assert result.passed_before_cleanup is True
    assert result.passed_after_cleanup is True
    assert calls == ["mem_get_info"]


def test_vram_preflight_retries_once_after_cache_clear() -> None:
    backend = _make_backend("high")
    torch_module = _torch_available()
    snapshots = iter([
        (7 * 1024**3, 24 * 1024**3),
        (13 * 1024**3, 24 * 1024**3),
    ])
    calls: list[str] = []

    def fake_snapshot(_torch):
        calls.append("mem_get_info")
        return next(snapshots)

    def fake_clear(_torch):
        calls.append("empty_cache")

    backend._cuda_free_total_snapshot = fake_snapshot
    backend._clear_cuda_cache = fake_clear

    result = backend._run_vram_preflight(torch_module)

    assert result.checked is True
    assert result.cleanup_attempted is True
    assert result.passed_before_cleanup is False
    assert result.passed_after_cleanup is True
    assert calls == ["mem_get_info", "empty_cache", "mem_get_info"]


def test_vram_preflight_fails_when_cleanup_does_not_restore_free_memory() -> None:
    backend = _make_backend("high")
    torch_module = _torch_available()
    snapshots = iter([
        (7 * 1024**3, 24 * 1024**3),
        (7 * 1024**3, 24 * 1024**3),
    ])
    calls: list[str] = []

    def fake_snapshot(_torch):
        calls.append("mem_get_info")
        return next(snapshots)

    def fake_clear(_torch):
        calls.append("empty_cache")

    backend._cuda_free_total_snapshot = fake_snapshot
    backend._clear_cuda_cache = fake_clear

    result = backend._run_vram_preflight(torch_module)

    assert result.checked is True
    assert result.cleanup_attempted is True
    assert result.passed is False
    assert calls == ["mem_get_info", "empty_cache", "mem_get_info"]


def test_vram_preflight_skips_when_cuda_is_unavailable() -> None:
    backend = _make_backend("high")
    torch_module = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))

    result = backend._run_vram_preflight(torch_module)

    assert result.checked is False
    assert result.cleanup_attempted is False


def test_high_startup_mode_prefers_full_cuda_when_preflight_passes_and_reserved_ratio_is_healthy() -> None:
    backend = _make_backend("high")
    preflight = VramPreflightResult(True, False, True, True, 13 * 1024**3, 13 * 1024**3, 12 * 1024**3)

    mode = backend._resolve_high_startup_mode(
        total_bytes=24 * 1024**3,
        reserved_bytes=4 * 1024**3,
        preflight=preflight,
    )

    assert mode == "full_cuda"


def test_high_startup_mode_falls_back_when_preflight_fails() -> None:
    backend = _make_backend("high")
    preflight = VramPreflightResult(True, True, False, False, 7 * 1024**3, 7 * 1024**3, 12 * 1024**3)

    mode = backend._resolve_high_startup_mode(
        total_bytes=24 * 1024**3,
        reserved_bytes=1 * 1024**3,
        preflight=preflight,
    )

    assert mode == "model_offload"


def test_high_startup_mode_falls_back_when_reserved_ratio_is_too_high() -> None:
    backend = _make_backend("high")
    preflight = VramPreflightResult(True, False, True, True, 13 * 1024**3, 13 * 1024**3, 12 * 1024**3)

    mode = backend._resolve_high_startup_mode(
        total_bytes=24 * 1024**3,
        reserved_bytes=21 * 1024**3,
        preflight=preflight,
    )

    assert mode == "model_offload"


def test_generate_preflight_falls_back_before_pipe_when_full_cuda_is_not_safe(monkeypatch) -> None:
    backend = _make_backend("high")
    backend._effective_execution_mode = "full_cuda"
    backend._initial_execution_mode = "full_cuda"
    backend._cuda_total_bytes = 24 * 1024**3
    backend._cuda_reserved_after_load_bytes = 2 * 1024**3
    backend._cuda_free_before_load_bytes = 13 * 1024**3
    backend._cuda_free_after_load_bytes = 13 * 1024**3

    class FakePipe:
        def __call__(self, **kwargs):
            return SimpleNamespace(images=[Image.new("RGB", (32, 32), color=(10, 20, 30))])

    fake_pipe = FakePipe()
    fake_loaded = SimpleNamespace(pipeline=fake_pipe, device="cuda")
    monkeypatch.setattr(backend, "_ensure_loaded", lambda: fake_loaded)
    monkeypatch.setattr(backend, "_apply_scheduler_mode", lambda pipe, mode: mode)
    monkeypatch.setattr(
        backend,
        "_resolve_effective_prompt",
        lambda **kwargs: (kwargs["prompt"], kwargs["prompt"], False),
    )
    monkeypatch.setattr(backend, "_build_generator", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend, "_apply_high_runtime_fallback_if_needed", lambda **kwargs: None)
    monkeypatch.setattr(
        backend,
        "_run_vram_preflight",
        lambda _torch: VramPreflightResult(True, True, False, False, 7 * 1024**3, 7 * 1024**3, 12 * 1024**3),
    )
    free_snapshots = iter([
        (7 * 1024**3, 24 * 1024**3),
        (6 * 1024**3, 24 * 1024**3),
    ])
    monkeypatch.setattr(backend, "_cuda_free_total_snapshot", lambda _torch: next(free_snapshots))
    monkeypatch.setattr(backend, "_clear_cuda_cache", lambda _torch: None)
    applied_modes: list[str] = []

    def fake_apply_pipe_execution_mode(pipe, mode):
        applied_modes.append(mode)
        return mode

    monkeypatch.setattr(backend, "_apply_pipe_execution_mode", fake_apply_pipe_execution_mode)

    request = GenerationRequest(prompt="test", width=32, height=32, seed=1)
    result = backend.generate(request)

    assert applied_modes == ["model_offload"]
    assert result.execution_mode_initial == "full_cuda"
    assert result.execution_mode_before_generate == "model_offload"
    assert result.execution_mode_after_generate == "model_offload"
    assert result.preflight_checked is True
    assert result.preflight_cleanup_attempted is True
    assert result.preflight_passed_after_cleanup is False
    assert result.preflight_fallback_triggered is True


def test_generate_preflight_keeps_full_cuda_when_guard_passes(monkeypatch) -> None:
    backend = _make_backend("high")
    backend._effective_execution_mode = "full_cuda"
    backend._initial_execution_mode = "full_cuda"

    class FakePipe:
        def __call__(self, **kwargs):
            return SimpleNamespace(images=[Image.new("RGB", (32, 32), color=(40, 50, 60))])

    fake_pipe = FakePipe()
    fake_loaded = SimpleNamespace(pipeline=fake_pipe, device="cuda")
    monkeypatch.setattr(backend, "_ensure_loaded", lambda: fake_loaded)
    monkeypatch.setattr(backend, "_apply_scheduler_mode", lambda pipe, mode: mode)
    monkeypatch.setattr(
        backend,
        "_resolve_effective_prompt",
        lambda **kwargs: (kwargs["prompt"], kwargs["prompt"], False),
    )
    monkeypatch.setattr(backend, "_build_generator", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend, "_apply_high_runtime_fallback_if_needed", lambda **kwargs: None)
    monkeypatch.setattr(
        backend,
        "_run_vram_preflight",
        lambda _torch: VramPreflightResult(True, False, True, True, 13 * 1024**3, 13 * 1024**3, 12 * 1024**3),
    )
    free_snapshots = iter([
        (13 * 1024**3, 24 * 1024**3),
        (12 * 1024**3, 24 * 1024**3),
    ])
    monkeypatch.setattr(backend, "_cuda_free_total_snapshot", lambda _torch: next(free_snapshots))
    applied_modes: list[str] = []
    monkeypatch.setattr(backend, "_apply_pipe_execution_mode", lambda pipe, mode: applied_modes.append(mode) or mode)

    request = GenerationRequest(prompt="test", width=32, height=32, seed=1)
    result = backend.generate(request)

    assert applied_modes == []
    assert result.execution_mode_before_generate == "full_cuda"
    assert result.execution_mode_after_generate == "full_cuda"
    assert result.preflight_checked is True
    assert result.preflight_cleanup_attempted is False
    assert result.preflight_passed_after_cleanup is True
    assert result.preflight_fallback_triggered is False


def test_generate_loads_loras_before_forward_context(monkeypatch) -> None:
    backend = _make_backend("high")
    backend._effective_execution_mode = "full_cuda"
    backend._initial_execution_mode = "full_cuda"
    probe = _ContextProbe()
    events: list[str] = []

    class FakePipe:
        def __call__(self, **kwargs):
            assert probe.active == "forward"
            events.append("pipe")
            return SimpleNamespace(images=[Image.new("RGB", (32, 32), color=(40, 50, 60))])

    fake_pipe = FakePipe()
    fake_loaded = SimpleNamespace(pipeline=fake_pipe, device="cuda")
    monkeypatch.setattr(backend, "_ensure_loaded", lambda: fake_loaded)
    monkeypatch.setattr(backend, "_apply_scheduler_mode", lambda pipe, mode: mode)
    monkeypatch.setattr(
        backend,
        "_resolve_effective_prompt",
        lambda **kwargs: (kwargs["prompt"], kwargs["prompt"], False),
    )
    monkeypatch.setattr(backend, "_build_generator", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend, "_apply_high_runtime_fallback_if_needed", lambda **kwargs: None)
    monkeypatch.setattr(
        backend,
        "_run_vram_preflight",
        lambda _torch: VramPreflightResult(True, False, True, True, 13 * 1024**3, 13 * 1024**3, 12 * 1024**3),
    )
    monkeypatch.setattr(backend, "_cuda_free_total_snapshot", lambda _torch: (13 * 1024**3, 24 * 1024**3))
    monkeypatch.setattr(backend, "_pipeline_forward_context", lambda torch_module, pipe: probe.context("forward"))

    def fake_load_lora_adapters(pipe, loras):
        assert probe.active is None
        events.append("load_lora")

    monkeypatch.setattr(backend, "_load_lora_adapters", fake_load_lora_adapters)
    monkeypatch.setattr(backend, "_clear_lora_adapters", lambda pipe, adapter_names=None: events.append("clear_lora"))

    request = GenerationRequest(
        prompt="test",
        width=32,
        height=32,
        seed=1,
        loras=(LoraSelection(id="test-lora", path=Path("test-lora.safetensors")),),
    )
    backend.generate(request)

    assert events == ["load_lora", "pipe", "clear_lora"]
