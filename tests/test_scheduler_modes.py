from __future__ import annotations

from types import SimpleNamespace

import diffusers

from app.config.profiles import RUNTIME_PROFILES
from app.core.backends.diffusers_zimage import DiffusersZImageBackend, VramPreflightResult
from app.core.worker.types import GenerationRequest


def _make_backend(profile_name: str = "high") -> DiffusersZImageBackend:
    settings = SimpleNamespace(runtime_profile=RUNTIME_PROFILES[profile_name])
    model_pack = SimpleNamespace(name="Rayzist_bf16")
    return DiffusersZImageBackend(settings=settings, model_pack=model_pack)


def test_euler_scheduler_uses_base_config_after_dpm_switch(monkeypatch) -> None:
    backend = _make_backend()
    calls: list[tuple[str, dict[str, object], dict[str, object]]] = []

    class FakeEulerScheduler:
        @classmethod
        def from_config(cls, config, **kwargs):
            calls.append(("euler", dict(config), dict(kwargs)))
            return SimpleNamespace(config={**dict(config), **kwargs}, order=1)

    class FakeDPMScheduler:
        @classmethod
        def from_config(cls, config, **kwargs):
            calls.append(("dpm", dict(config), dict(kwargs)))
            merged = {**dict(config), **kwargs}
            return SimpleNamespace(config=merged, order=2, scale_noise=lambda *args, **kw: None)

    monkeypatch.setattr(diffusers, "FlowMatchEulerDiscreteScheduler", FakeEulerScheduler)
    monkeypatch.setattr(diffusers, "DPMSolverMultistepScheduler", FakeDPMScheduler)
    # Pin the DPM-sigmas capability to True so the apply path uses the requested DPM mode
    # rather than falling back to Euler at the probe. Real diffusers version-detection is
    # covered separately in tests/test_scheduler_dpm_fallback.py.
    monkeypatch.setattr(
        DiffusersZImageBackend,
        "_dpm_scheduler_accepts_sigmas",
        classmethod(lambda cls: True),
    )

    pipe = SimpleNamespace(scheduler=SimpleNamespace(config={"shift": 3.0, "use_dynamic_shifting": False}))

    assert backend._apply_scheduler_mode(pipe, backend._SCHEDULER_DPM_EXP_LIGHT) == backend._SCHEDULER_DPM_EXP_LIGHT
    assert backend._apply_scheduler_mode(pipe, backend._SCHEDULER_EULER) == backend._SCHEDULER_EULER

    assert calls[0][0] == "dpm"
    assert calls[1][0] == "euler"
    assert calls[1][1]["use_dynamic_shifting"] is False
    assert calls[1][2]["use_dynamic_shifting"] is False
    assert "time_shift_type" not in calls[1][1]


def test_generate_retries_once_with_dpm_when_euler_scheduler_fails(monkeypatch) -> None:
    backend = _make_backend()
    pipe_calls: list[str] = []
    scheduler_calls: list[str] = []

    class FakePipe:
        def __init__(self) -> None:
            self.scheduler = SimpleNamespace(config={"shift": 3.0, "use_dynamic_shifting": False})

        def __call__(self, **kwargs):
            pipe_calls.append(kwargs["prompt"])
            if len(pipe_calls) == 1:
                raise RuntimeError("scheduler timestep invalid value encountered")
            return SimpleNamespace(images=["ok"])

    fake_pipe = FakePipe()

    monkeypatch.setattr(
        backend,
        "_ensure_loaded",
        lambda: SimpleNamespace(pipeline=fake_pipe, device="cpu"),
    )
    monkeypatch.setattr(
        backend,
        "_apply_scheduler_mode",
        lambda pipe, mode: scheduler_calls.append(mode) or mode,
    )
    monkeypatch.setattr(
        backend,
        "_resolve_effective_prompt",
        lambda **kwargs: (kwargs["prompt"], kwargs["prompt"], False),
    )
    monkeypatch.setattr(backend, "_build_generator", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        backend,
        "_run_vram_preflight",
        lambda _torch: VramPreflightResult(False, False, None, None, None, None, None),
    )
    monkeypatch.setattr(backend, "_apply_high_runtime_fallback_if_needed", lambda **kwargs: None)
    monkeypatch.setattr(backend, "_cuda_free_total_snapshot", lambda _torch: (None, None))

    from app.core.backends import diffusers_zimage as backend_module

    monkeypatch.setattr(backend_module, "cuda_memory_snapshot", lambda _torch: None)
    monkeypatch.setattr(backend_module, "process_memory_snapshot", lambda: None)

    result = backend.generate(
        GenerationRequest(
            prompt="test prompt",
            width=512,
            height=512,
            seed=123,
        )
    )

    assert scheduler_calls == [backend._SCHEDULER_EULER, backend._SCHEDULER_DPM_EXP_LIGHT]
    assert pipe_calls == ["test prompt", "test prompt"]
    assert result.scheduler_mode == backend._SCHEDULER_DPM_EXP_LIGHT
    assert result.image == "ok"
