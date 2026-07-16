"""Weightless plumbing tests for the WP-5 Krea2 VL image-conditioning path.

Verifies that ``DiffusersKreaBackend._build_generate_pipe_kwargs``:

* returns the inherited text-only kwargs when no ``context_image`` is set,
* substitutes ``prompt_embeds`` / ``prompt_embeds_mask`` (and drops ``prompt``) when it is,
* preserves the Z-Image path bit-for-bit (parent method still returns the exact prior kwarg set).

No torch, no weights, no diffusers Krea import — everything is mocked.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.core.backends.diffusers_krea import DiffusersKreaBackend
from app.core.backends.diffusers_zimage import DiffusersZImageBackend
from app.core.worker.types import GenerationRequest


class _StubPipe:
    tokenizer = None
    text_encoder = SimpleNamespace(device="cpu")
    text_encoder_select_layers = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
    prompt_template_encode_start_idx = 34


def _make_backend(cls, *, encode_return=None):
    """Instantiate a backend without running its __init__ (no settings, no pack)."""
    backend = cls.__new__(cls)
    if encode_return is not None:
        backend._encode_prompt_with_context_image = MagicMock(return_value=encode_return)
    return backend


def _base_kwargs():
    return dict(
        pipe=_StubPipe(),
        prompt="a red fox in fresh snow",
        request=GenerationRequest(prompt="a red fox in fresh snow", width=1024, height=1024),
        steps=8,
        guidance_scale=0.0,
        generator=object(),
        procedural_latents=None,
        torch_module=SimpleNamespace(),
    )


def _monkey_max_seq(monkeypatch, target_cls):
    monkeypatch.setattr(
        target_cls,
        "_resolve_pipeline_max_sequence_length",
        classmethod(lambda cls, tokenizer, prompt: 512),
    )


def test_zimage_kwargs_unchanged(monkeypatch) -> None:
    _monkey_max_seq(monkeypatch, DiffusersZImageBackend)
    backend = _make_backend(DiffusersZImageBackend)
    kwargs = backend._build_generate_pipe_kwargs(**_base_kwargs())

    assert kwargs["prompt"] == "a red fox in fresh snow"
    assert kwargs["width"] == 1024
    assert kwargs["height"] == 1024
    assert kwargs["num_inference_steps"] == 8
    assert kwargs["guidance_scale"] == 0.0
    assert kwargs["max_sequence_length"] == 512
    assert "prompt_embeds" not in kwargs
    assert "prompt_embeds_mask" not in kwargs


def test_krea_without_context_image_matches_zimage(monkeypatch) -> None:
    _monkey_max_seq(monkeypatch, DiffusersZImageBackend)
    backend = _make_backend(DiffusersKreaBackend)
    kwargs = backend._build_generate_pipe_kwargs(**_base_kwargs())

    assert kwargs["prompt"] == "a red fox in fresh snow"
    assert "prompt_embeds" not in kwargs
    assert "prompt_embeds_mask" not in kwargs


def test_krea_with_context_image_substitutes_prompt_embeds(monkeypatch, tmp_path: Path) -> None:
    _monkey_max_seq(monkeypatch, DiffusersZImageBackend)

    fake_embeds = object()
    fake_mask = object()
    backend = _make_backend(DiffusersKreaBackend, encode_return=(fake_embeds, fake_mask))

    ref_path = tmp_path / "ref.png"
    ref_path.write_bytes(b"placeholder")  # encode is mocked so the file is never opened

    base = _base_kwargs()
    base["request"] = GenerationRequest(
        prompt="a red fox in fresh snow",
        width=1024,
        height=1024,
        context_image=ref_path,
    )
    kwargs = backend._build_generate_pipe_kwargs(**base)

    # Prompt path swapped for embed path.
    assert kwargs["prompt"] is None
    assert kwargs["prompt_embeds"] is fake_embeds
    assert kwargs["prompt_embeds_mask"] is fake_mask
    # Non-conditioning kwargs still there.
    assert kwargs["num_inference_steps"] == 8
    assert kwargs["guidance_scale"] == 0.0
    assert kwargs["max_sequence_length"] == 512

    # Encode helper was called with the right context image + prompt.
    backend._encode_prompt_with_context_image.assert_called_once()
    call_kwargs = backend._encode_prompt_with_context_image.call_args.kwargs
    assert call_kwargs["prompt"] == "a red fox in fresh snow"
    assert call_kwargs["context_image"] == ref_path
    assert call_kwargs["max_sequence_length"] == 512
