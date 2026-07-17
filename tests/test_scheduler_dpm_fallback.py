"""DPM scheduler fallback when the installed diffusers lacks `sigmas` support.

Regression cover for the "does not support custom sigmas schedules" crash that fires when
Creative Mode >= 2 is combined with Krea2 or Z-Image pipelines. Both pipelines call
``retrieve_timesteps(scheduler, ..., sigmas=sigmas)`` unconditionally; if the scheduler's
``set_timesteps`` signature lacks the kwarg, the pipeline raises before generation starts.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from app.core.backends.diffusers_zimage import DiffusersZImageBackend


def _reset_probe_cache() -> None:
    DiffusersZImageBackend._dpm_sigmas_capability_cache = None


def _make_backend() -> DiffusersZImageBackend:
    backend = DiffusersZImageBackend.__new__(DiffusersZImageBackend)
    backend._model_pack = SimpleNamespace(name="test-pack")  # type: ignore[attr-defined]
    backend._dpm_fallback_logged_pipes = set()  # type: ignore[attr-defined]
    backend._active_scheduler_mode_by_pipe = {}  # type: ignore[attr-defined]
    backend._base_scheduler_config_by_pipe = {}  # type: ignore[attr-defined]
    return backend


def test_probe_returns_false_when_signature_lacks_sigmas(monkeypatch) -> None:
    _reset_probe_cache()

    from diffusers import DPMSolverMultistepScheduler

    def _no_sigmas(self, num_inference_steps=None, device=None, timesteps=None):  # noqa: ARG001
        raise AssertionError("scheduler should not be called during signature probe")

    monkeypatch.setattr(DPMSolverMultistepScheduler, "set_timesteps", _no_sigmas)

    assert DiffusersZImageBackend._dpm_scheduler_accepts_sigmas() is False


def test_probe_returns_true_when_signature_has_sigmas(monkeypatch) -> None:
    _reset_probe_cache()

    from diffusers import DPMSolverMultistepScheduler

    def _with_sigmas(self, num_inference_steps=None, device=None, sigmas=None):  # noqa: ARG001
        raise AssertionError("scheduler should not be called during signature probe")

    monkeypatch.setattr(DPMSolverMultistepScheduler, "set_timesteps", _with_sigmas)

    assert DiffusersZImageBackend._dpm_scheduler_accepts_sigmas() is True


def test_apply_scheduler_mode_falls_back_to_euler(monkeypatch) -> None:
    """When DPM lacks sigma support, _apply_scheduler_mode should rewrite the mode to Euler
    before touching the pipeline's scheduler — no wasted pipe.__call__."""
    _reset_probe_cache()
    monkeypatch.setattr(
        DiffusersZImageBackend,
        "_dpm_scheduler_accepts_sigmas",
        classmethod(lambda cls: False),
    )
    backend = _make_backend()
    pipe = MagicMock()
    pipe.__class__.__name__ = "Krea2Pipeline"
    # Pre-populate the scheduler mode cache so the second call short-circuits on the same
    # normalized value (Euler) — confirms the mode was rewritten in place.
    backend._active_scheduler_mode_by_pipe[id(pipe)] = DiffusersZImageBackend._SCHEDULER_EULER

    result = backend._apply_scheduler_mode(pipe, DiffusersZImageBackend._SCHEDULER_DPM_EXP_LIGHT)

    assert result == DiffusersZImageBackend._SCHEDULER_EULER
    assert id(pipe) in backend._dpm_fallback_logged_pipes


def test_retry_mode_extends_to_dpm_fallback() -> None:
    backend = _make_backend()
    exc = RuntimeError("does not support custom sigmas schedules")

    # Euler → DPM (existing behavior, keep working).
    assert (
        backend._resolve_scheduler_retry_mode(DiffusersZImageBackend._SCHEDULER_EULER, exc)
        == DiffusersZImageBackend._SCHEDULER_DPM_EXP_LIGHT
    )
    # DPM_EXP_LIGHT → Euler (new fallback path).
    assert (
        backend._resolve_scheduler_retry_mode(DiffusersZImageBackend._SCHEDULER_DPM_EXP_LIGHT, exc)
        == DiffusersZImageBackend._SCHEDULER_EULER
    )
    # DPM_DDIM → Euler (new fallback path).
    assert (
        backend._resolve_scheduler_retry_mode(DiffusersZImageBackend._SCHEDULER_DPM_DDIM, exc)
        == DiffusersZImageBackend._SCHEDULER_EULER
    )
    # DPM_MANUAL → Euler (new fallback path).
    assert (
        backend._resolve_scheduler_retry_mode(DiffusersZImageBackend._SCHEDULER_DPM_MANUAL, exc)
        == DiffusersZImageBackend._SCHEDULER_EULER
    )


def test_retry_mode_ignores_non_scheduler_errors() -> None:
    backend = _make_backend()
    exc = RuntimeError("CUDA out of memory")

    assert (
        backend._resolve_scheduler_retry_mode(DiffusersZImageBackend._SCHEDULER_EULER, exc) is None
    )
    assert (
        backend._resolve_scheduler_retry_mode(DiffusersZImageBackend._SCHEDULER_DPM_EXP_LIGHT, exc)
        is None
    )
