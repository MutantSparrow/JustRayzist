"""Weightless tests for DiffusersKreaBackend chat / enhance / wildcard routing.

The Krea backend routes chat / prompt-enhance / wildcard-suggest through
``DiffusersQwen3VLInference`` (via ``_qwen_for_pipe``) rather than the base Qwen3 causal path.
This test locks that hoist point + confirms staging skips when the pipeline is on high tier
(no accelerate hooks) and fires when hooks are present.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from app.core.backends.diffusers_krea import DiffusersKreaBackend
from app.core.backends.diffusers_qwen import (
    DiffusersQwen3VLInference,
    DiffusersQwenInference,
)


def _make_backend() -> DiffusersKreaBackend:
    """Instantiate the Krea backend without triggering __init__ (no settings / pack / cuda)."""

    backend = DiffusersKreaBackend.__new__(DiffusersKreaBackend)
    backend._text_encoder_label = lambda: "text_encoder"  # type: ignore[attr-defined]
    return backend


def test_qwen_for_pipe_returns_qwen3vl_variant() -> None:
    backend = _make_backend()
    pipe = SimpleNamespace(tokenizer=object(), text_encoder=object())
    torch_stub = SimpleNamespace()

    result = backend._qwen_for_pipe(pipe, torch_stub)
    assert isinstance(result, DiffusersQwen3VLInference)
    # And still counts as a plain DiffusersQwenInference for callers that type-check the base.
    assert isinstance(result, DiffusersQwenInference)


def test_run_with_staged_text_encoder_skips_staging_without_hooks() -> None:
    """No accelerate hooks on the transformer means the pipeline is on high tier (no offload).
    The staging dance should be a no-op — just call the action."""

    backend = _make_backend()

    class _FakeMod:
        def modules(self):
            return iter([self])

    pipe = SimpleNamespace(
        text_encoder=SimpleNamespace(),
        transformer=_FakeMod(),
        _execution_device="cuda",
        remove_all_hooks=MagicMock(),
        enable_sequential_cpu_offload=MagicMock(),
    )
    torch_stub = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))

    def _action(inner_pipe):
        assert inner_pipe is pipe
        return "done"

    result = backend._run_with_staged_text_encoder(pipe, torch_stub, _action)
    assert result == "done"
    # No hooks touched.
    pipe.remove_all_hooks.assert_not_called()
    pipe.enable_sequential_cpu_offload.assert_not_called()


def test_run_with_staged_text_encoder_stages_and_restores_when_hooks_present(monkeypatch) -> None:
    """When the transformer carries accelerate hooks, the wrapper removes the encoder's hooks,
    moves it to the execution device, runs the action, then rebuilds the whole offload chain."""

    backend = _make_backend()

    class _Hooked:
        _hf_hook = object()

        def modules(self):
            return iter([self])

    class _FakeEncoder:
        def __init__(self) -> None:
            self.to_calls: list[str] = []

        def to(self, device):
            self.to_calls.append(str(device))
            return self

    encoder = _FakeEncoder()
    pipe = SimpleNamespace(
        text_encoder=encoder,
        transformer=_Hooked(),
        _execution_device="cuda:0",
        remove_all_hooks=MagicMock(),
        enable_sequential_cpu_offload=MagicMock(),
    )
    torch_stub = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))

    remove_hook_calls: list[object] = []

    def _fake_remove_hook(module, recurse: bool = False):
        remove_hook_calls.append(module)

    monkeypatch.setattr(
        "accelerate.hooks.remove_hook_from_module",
        _fake_remove_hook,
    )

    def _action(inner_pipe):
        return "ok"

    result = backend._run_with_staged_text_encoder(pipe, torch_stub, _action)
    assert result == "ok"
    assert remove_hook_calls == [encoder]
    assert encoder.to_calls == ["cuda:0"]
    pipe.remove_all_hooks.assert_called_once()
    pipe.enable_sequential_cpu_offload.assert_called_once()


def test_run_with_staged_text_encoder_restores_on_action_exception(monkeypatch) -> None:
    """An action exception should still leave the pipeline offload state restored."""

    backend = _make_backend()

    class _Hooked:
        _hf_hook = object()

        def modules(self):
            return iter([self])

    encoder = SimpleNamespace(to=lambda dev: encoder)
    pipe = SimpleNamespace(
        text_encoder=encoder,
        transformer=_Hooked(),
        _execution_device="cuda:0",
        remove_all_hooks=MagicMock(),
        enable_sequential_cpu_offload=MagicMock(),
    )
    torch_stub = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setattr(
        "accelerate.hooks.remove_hook_from_module",
        lambda module, recurse=False: None,
    )

    def _boom(inner_pipe):
        raise RuntimeError("simulated action failure")

    import pytest

    with pytest.raises(RuntimeError, match="simulated"):
        backend._run_with_staged_text_encoder(pipe, torch_stub, _boom)
    pipe.remove_all_hooks.assert_called_once()
    pipe.enable_sequential_cpu_offload.assert_called_once()
