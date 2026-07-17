"""Weightless tests for the Qwen3VL chat / rewrite decode routing.

Locks the contract: ``DiffusersQwen3VLInference._generate_with_base_model`` drives the
``.language_model`` submodule of a Qwen3VLModel rather than the top-level VL forward, so the
vision-tower / rope_deltas prefill path never fires on text-only chat turns.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

from app.core.backends.diffusers_qwen import (
    DiffusersQwen3VLInference,
    DiffusersQwenInference,
)


class _FakeLanguageModel:
    """Duck-types the subset of Qwen3VLTextModel the base decode loop touches."""

    def __init__(self) -> None:
        self.calls: list[dict] = []


def test_resolve_generation_module_returns_language_model_when_present() -> None:
    lang = _FakeLanguageModel()
    text_encoder = SimpleNamespace(language_model=lang)
    # Make it duck-type ``get_input_embeddings`` for the resolver's has-embeddings check.
    lang.get_input_embeddings = lambda: SimpleNamespace(weight=torch.zeros(2, 3))
    resolved = DiffusersQwen3VLInference._resolve_generation_module(text_encoder)
    assert resolved is lang


def test_resolve_generation_module_falls_back_to_encoder_when_no_language_model() -> None:
    text_encoder = SimpleNamespace(get_input_embeddings=lambda: None)
    resolved = DiffusersQwen3VLInference._resolve_generation_module(text_encoder)
    assert resolved is text_encoder


def test_resolve_generation_module_none_safe() -> None:
    assert DiffusersQwen3VLInference._resolve_generation_module(None) is None


def test_generate_delegates_to_language_model_submodule() -> None:
    """The Qwen3VL variant must call the base helper with ``.language_model`` as the module."""

    lang = _FakeLanguageModel()
    lang.get_input_embeddings = lambda: SimpleNamespace(weight=torch.zeros(2, 3))
    text_encoder = SimpleNamespace(language_model=lang)

    captured: dict = {}

    def _fake_base_generate(**kwargs):
        captured.update(kwargs)
        return torch.tensor([[1, 2, 3]])

    with patch.object(DiffusersQwenInference, "_generate_with_base_model", staticmethod(_fake_base_generate)):
        out = DiffusersQwen3VLInference._generate_with_base_model(
            text_encoder=text_encoder,
            encoded={"input_ids": torch.tensor([[1]])},
            max_new_tokens=16,
            eos_token_id=2,
            torch_module=torch,
            do_sample=False,
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=1.05,
        )
    assert torch.equal(out, torch.tensor([[1, 2, 3]]))
    assert captured["text_encoder"] is lang, (
        "DiffusersQwen3VLInference should route the base decode loop through the "
        "``.language_model`` submodule to avoid the top-level VL forward's rope_deltas path."
    )
    # Passthrough sanity for the rest of the kwargs.
    assert captured["max_new_tokens"] == 16
    assert captured["eos_token_id"] == 2
    assert captured["do_sample"] is False
    assert abs(captured["temperature"] - 0.9) < 1e-6
    assert abs(captured["top_p"] - 0.95) < 1e-6
    assert abs(captured["repetition_penalty"] - 1.05) < 1e-6


def test_qwen3vl_class_exposed_from_module() -> None:
    from app.core.backends import diffusers_qwen as mod

    assert "DiffusersQwen3VLInference" in mod.__all__
    assert issubclass(mod.DiffusersQwen3VLInference, mod.DiffusersQwenInference)
