from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from app.core.backends.diffusers_zimage import DiffusersZImageBackend
from app.core.worker.types import GenerationRequest


class _HelperTokenizer:
    def decode(self, _token_ids, skip_special_tokens: bool = True) -> str:
        return "decoded text"


def _make_backend(temp_app_paths, make_app_settings) -> DiffusersZImageBackend:
    settings = make_app_settings(paths=temp_app_paths)
    model_pack = SimpleNamespace(name="Rayzist_bf16", base_name="Rayzist_bf16")
    return DiffusersZImageBackend(settings=settings, model_pack=model_pack)


def test_prompt_enhancement_prefers_base_model_decode(monkeypatch, caplog, temp_app_paths, make_app_settings) -> None:
    backend = _make_backend(temp_app_paths, make_app_settings)
    tokenizer = _HelperTokenizer()
    encoded = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}
    calls: dict[str, object] = {}

    class _TextEncoder:
        def generate(self, **_kwargs):
            raise AssertionError("text_encoder.generate() should not be used on the normal prompt enhancement path.")

    def _fake_base_model(**kwargs):
        calls["kwargs"] = kwargs
        return torch.tensor([[1, 2, 3]])

    monkeypatch.setattr(backend, "_generate_with_base_model", _fake_base_model)
    monkeypatch.setattr(backend, "_extract_rewritten_prompt", lambda full_text, input_text: "enhanced prompt")
    monkeypatch.setattr(backend, "_rewrite_rejection_reason", lambda original, rewritten: "ok")
    caplog.set_level(logging.DEBUG, logger="app.core.backends.diffusers_zimage")

    rewritten, reason = backend._run_rewrite_attempt(
        tokenizer=tokenizer,
        text_encoder=_TextEncoder(),
        encoded=encoded,
        prompt="original prompt",
        torch_module=torch,
        generate_kwargs={
            "max_new_tokens": 24,
            "do_sample": True,
            "temperature": 0.72,
            "top_p": 0.92,
            "repetition_penalty": 1.08,
        },
        enhancement_seed=123,
    )

    assert rewritten == "enhanced prompt"
    assert reason == "ok"
    assert calls["kwargs"]["do_sample"] is True
    assert calls["kwargs"]["temperature"] == pytest.approx(0.72)
    assert calls["kwargs"]["top_p"] == pytest.approx(0.92)
    assert calls["kwargs"]["repetition_penalty"] == pytest.approx(1.08)
    assert not any("falling back to text_encoder.generate" in record.getMessage() for record in caplog.records)


def test_wildcard_suggestion_prefers_base_model_decode(monkeypatch, caplog, temp_app_paths, make_app_settings) -> None:
    backend = _make_backend(temp_app_paths, make_app_settings)
    tokenizer = _HelperTokenizer()
    encoded = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}
    calls: dict[str, object] = {}

    class _TextEncoder:
        def generate(self, **_kwargs):
            raise AssertionError("text_encoder.generate() should not be used on the normal wildcard suggestion path.")

    def _fake_base_model(**kwargs):
        calls["kwargs"] = kwargs
        return torch.tensor([[1, 2, 3]])

    monkeypatch.setattr(backend, "_generate_with_base_model", _fake_base_model)
    monkeypatch.setattr(backend, "_extract_generated_completion_text", lambda full_text, input_text: "mountain village")
    caplog.set_level(logging.DEBUG, logger="app.core.backends.diffusers_zimage")

    generated = backend._run_text_generation_attempt(
        tokenizer=tokenizer,
        text_encoder=_TextEncoder(),
        encoded=encoded,
        torch_module=torch,
        generate_kwargs={
            "max_new_tokens": 32,
            "do_sample": True,
            "temperature": 0.85,
            "top_p": 0.92,
            "repetition_penalty": 1.08,
        },
        generation_seed=456,
    )

    assert generated == "mountain village"
    assert calls["kwargs"]["do_sample"] is True
    assert calls["kwargs"]["temperature"] == pytest.approx(0.85)
    assert calls["kwargs"]["top_p"] == pytest.approx(0.92)
    assert calls["kwargs"]["repetition_penalty"] == pytest.approx(1.08)
    assert not any("falling back to text_encoder.generate" in record.getMessage() for record in caplog.records)


def test_helper_text_generation_falls_back_to_generate_when_base_decode_is_unavailable(
    monkeypatch,
    caplog,
    temp_app_paths,
    make_app_settings,
) -> None:
    backend = _make_backend(temp_app_paths, make_app_settings)
    tokenizer = _HelperTokenizer()
    encoded = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}

    class _TextEncoder:
        def __init__(self) -> None:
            self.generate_called = False

        def generate(self, **_kwargs):
            self.generate_called = True
            return torch.tensor([[1, 2, 3]])

    text_encoder = _TextEncoder()
    monkeypatch.setattr(
        backend,
        "_generate_with_base_model",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("missing base-model decode support")),
    )
    monkeypatch.setattr(backend, "_extract_generated_completion_text", lambda full_text, input_text: "fallback text")
    caplog.set_level(logging.DEBUG, logger="app.core.backends.diffusers_zimage")

    generated = backend._run_text_generation_attempt(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        encoded=encoded,
        torch_module=torch,
        generate_kwargs={"max_new_tokens": 16, "do_sample": True},
        generation_seed=789,
    )

    assert generated == "fallback text"
    assert text_encoder.generate_called is True
    assert any("falling back to text_encoder.generate" in record.getMessage() for record in caplog.records)


def test_refine_image_uses_effective_prompt_for_img2img(monkeypatch, temp_app_paths, make_app_settings) -> None:
    backend = _make_backend(temp_app_paths, make_app_settings)
    captured_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        backend,
        "_ensure_loaded",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(tokenizer=object(), text_encoder=object()), device="cpu"),
    )
    monkeypatch.setattr(backend, "_ensure_img2img_pipe", lambda: SimpleNamespace())
    monkeypatch.setattr(backend, "_apply_scheduler_mode", lambda pipe, mode: mode)
    monkeypatch.setattr(
        "app.core.backends.diffusers_zimage.expand_prompt_wildcards",
        lambda settings, prompt, seed: ("wildcard prompt", ()),
    )
    monkeypatch.setattr(
        backend,
        "_resolve_effective_prompt",
        lambda **kwargs: (kwargs["prompt"], "enhanced prompt", True),
    )
    monkeypatch.setattr(backend, "_append_lora_triggers", lambda prompt, loras: ("enhanced prompt", ()))
    monkeypatch.setattr(backend, "_resolve_refine_tiling", lambda request, width, height: (0, 64))

    def _fake_refine_with_oom_fallback(**kwargs):
        captured_calls.append(
            {
                "prompt": kwargs["prompt"],
                "seed": kwargs["seed"],
                "guidance_scale": kwargs["guidance_scale"],
                "strength": kwargs["strength"],
                "steps": kwargs["steps"],
                "image_size": kwargs["image"].size,
            }
        )
        return Image.new("RGB", kwargs["image"].size), 0, 64, 0

    monkeypatch.setattr(backend, "_run_refine_with_oom_fallback", _fake_refine_with_oom_fallback)
    monkeypatch.setattr("app.core.backends.diffusers_zimage.cuda_memory_snapshot", lambda _torch: None)
    monkeypatch.setattr("app.core.backends.diffusers_zimage.process_memory_snapshot", lambda: None)
    monkeypatch.setattr(backend, "_apply_high_runtime_fallback_if_needed", lambda **kwargs: None)

    result = backend.refine_image(
        Image.new("RGB", (512, 512), color=(12, 34, 56)),
        GenerationRequest(
            prompt="original prompt",
            width=512,
            height=512,
            seed=321,
            enhance_prompt=True,
            refine_strength=0.2,
        ),
    )

    assert len(captured_calls) == 2
    assert [call["prompt"] for call in captured_calls] == ["enhanced prompt", "enhanced prompt"]
    assert [call["seed"] for call in captured_calls] == [321, 321]
    assert [call["guidance_scale"] for call in captured_calls] == [
        backend._IMG2IMG_GUIDANCE_SCALE_DEFAULT,
        backend._IMG2IMG_GUIDANCE_SCALE_DEFAULT,
    ]
    assert captured_calls[0]["strength"] == pytest.approx(0.2)
    assert captured_calls[1]["strength"] == pytest.approx(0.10)
    assert [call["steps"] for call in captured_calls] == [
        backend._settings.runtime_profile.steps_default,
        backend._REFINE_POLISH_STEPS,
    ]
    assert captured_calls[1]["image_size"] == captured_calls[0]["image_size"]
    assert result.prompt_original == "original prompt"
    assert result.prompt_wildcard_resolved == "wildcard prompt"
    assert result.prompt_effective_base == "enhanced prompt"
    assert result.prompt_effective == "enhanced prompt"
    assert result.prompt_enhanced is True
    assert result.refine_pass_count == 2
    assert result.refine_pass1_steps == backend._settings.runtime_profile.steps_default
    assert result.refine_pass2_steps == backend._REFINE_POLISH_STEPS
    assert result.refine_pass2_strength == pytest.approx(0.10)


def test_refine_image_defaults_to_full_frame_and_higher_steps(monkeypatch, temp_app_paths, make_app_settings) -> None:
    backend = _make_backend(temp_app_paths, make_app_settings)
    captured_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        backend,
        "_ensure_loaded",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(tokenizer=object(), text_encoder=object()), device="cpu"),
    )
    monkeypatch.setattr(backend, "_ensure_img2img_pipe", lambda: SimpleNamespace())
    monkeypatch.setattr(backend, "_apply_scheduler_mode", lambda pipe, mode: mode)
    monkeypatch.setattr(
        "app.core.backends.diffusers_zimage.expand_prompt_wildcards",
        lambda settings, prompt, seed: (prompt, ()),
    )
    monkeypatch.setattr(
        backend,
        "_resolve_effective_prompt",
        lambda **kwargs: (kwargs["prompt"], kwargs["prompt"], False),
    )
    monkeypatch.setattr(backend, "_append_lora_triggers", lambda prompt, loras: (prompt, ()))

    def _fake_refine_with_oom_fallback(**kwargs):
        captured_calls.append(
            {
                "steps": kwargs["steps"],
                "tile_size": kwargs["tile_size"],
                "tile_overlap": kwargs["tile_overlap"],
                "guidance_scale": kwargs["guidance_scale"],
            }
        )
        return Image.new("RGB", kwargs["image"].size), 0, kwargs["tile_overlap"], 0

    monkeypatch.setattr(backend, "_run_refine_with_oom_fallback", _fake_refine_with_oom_fallback)
    monkeypatch.setattr("app.core.backends.diffusers_zimage.cuda_memory_snapshot", lambda _torch: None)
    monkeypatch.setattr("app.core.backends.diffusers_zimage.process_memory_snapshot", lambda: None)
    monkeypatch.setattr(backend, "_apply_high_runtime_fallback_if_needed", lambda **kwargs: None)

    backend.refine_image(
        Image.new("RGB", (1500, 1000), color=(12, 34, 56)),
        GenerationRequest(
            prompt="original prompt",
            width=1500,
            height=1000,
            seed=321,
            refine_strength=0.2,
        ),
    )

    assert len(captured_calls) == 2
    assert captured_calls[0]["steps"] == backend._settings.runtime_profile.steps_default
    assert captured_calls[1]["steps"] == backend._REFINE_POLISH_STEPS
    assert captured_calls[0]["guidance_scale"] == pytest.approx(backend._IMG2IMG_GUIDANCE_SCALE_DEFAULT)
    assert captured_calls[1]["guidance_scale"] == pytest.approx(backend._IMG2IMG_GUIDANCE_SCALE_DEFAULT)
    assert captured_calls[0]["tile_size"] == 0
    assert captured_calls[1]["tile_size"] == 0
    assert captured_calls[0]["tile_overlap"] == 64
    assert captured_calls[1]["tile_overlap"] == 64


def test_upscale_path_remains_single_pass(monkeypatch, temp_app_paths, make_app_settings) -> None:
    backend = _make_backend(temp_app_paths, make_app_settings)
    captured_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        backend,
        "_ensure_loaded",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(tokenizer=object(), text_encoder=object()), device="cpu"),
    )
    monkeypatch.setattr(backend, "_ensure_img2img_pipe", lambda: SimpleNamespace())
    monkeypatch.setattr(backend, "_apply_scheduler_mode", lambda pipe, mode: mode)
    monkeypatch.setattr(
        "app.core.backends.diffusers_zimage.expand_prompt_wildcards",
        lambda settings, prompt, seed: (prompt, ()),
    )
    monkeypatch.setattr(
        backend,
        "_resolve_effective_prompt",
        lambda **kwargs: (kwargs["prompt"], kwargs["prompt"], False),
    )
    monkeypatch.setattr(backend, "_append_lora_triggers", lambda prompt, loras: (prompt, ()))
    monkeypatch.setattr(backend, "_resolve_refine_tiling", lambda request, width, height: (0, 64))

    def _fake_refine_with_oom_fallback(**kwargs):
        captured_calls.append(
            {
                "steps": kwargs["steps"],
                "strength": kwargs["strength"],
                "guidance_scale": kwargs["guidance_scale"],
            }
        )
        return Image.new("RGB", kwargs["image"].size), 0, kwargs["tile_overlap"], 0

    monkeypatch.setattr(backend, "_run_refine_with_oom_fallback", _fake_refine_with_oom_fallback)
    monkeypatch.setattr("app.core.backends.diffusers_zimage.cuda_memory_snapshot", lambda _torch: None)
    monkeypatch.setattr("app.core.backends.diffusers_zimage.process_memory_snapshot", lambda: None)
    monkeypatch.setattr(backend, "_apply_high_runtime_fallback_if_needed", lambda **kwargs: None)

    result = backend._refine_existing_image(
        input_image=Image.new("RGB", (512, 512), color=(12, 34, 56)),
        refine_input_image=Image.new("RGB", (1024, 1024), color=(20, 40, 60)),
        request=GenerationRequest(
            prompt="original prompt",
            width=1024,
            height=1024,
            seed=321,
            refine_strength=0.2,
        ),
        upscale_duration_ms=10,
        mode="upscale_then_img2img",
    )

    assert len(captured_calls) == 1
    assert captured_calls[0]["steps"] == backend._settings.runtime_profile.steps_default
    assert captured_calls[0]["strength"] == pytest.approx(0.2)
    assert captured_calls[0]["guidance_scale"] == pytest.approx(
        backend._settings.runtime_profile.guidance_scale_default
    )
    assert result.refine_pass_count == 1
    assert result.refine_pass1_steps == backend._settings.runtime_profile.steps_default
    assert result.refine_pass2_steps is None
    assert result.refine_pass2_strength is None


def test_img2img_polish_strength_respects_min_floor(monkeypatch, temp_app_paths, make_app_settings) -> None:
    backend = _make_backend(temp_app_paths, make_app_settings)
    captured_strengths: list[float] = []

    monkeypatch.setattr(
        backend,
        "_ensure_loaded",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(tokenizer=object(), text_encoder=object()), device="cpu"),
    )
    monkeypatch.setattr(backend, "_ensure_img2img_pipe", lambda: SimpleNamespace())
    monkeypatch.setattr(backend, "_apply_scheduler_mode", lambda pipe, mode: mode)
    monkeypatch.setattr(
        "app.core.backends.diffusers_zimage.expand_prompt_wildcards",
        lambda settings, prompt, seed: (prompt, ()),
    )
    monkeypatch.setattr(
        backend,
        "_resolve_effective_prompt",
        lambda **kwargs: (kwargs["prompt"], kwargs["prompt"], False),
    )
    monkeypatch.setattr(backend, "_append_lora_triggers", lambda prompt, loras: (prompt, ()))
    monkeypatch.setattr(backend, "_resolve_refine_tiling", lambda request, width, height: (0, 64))

    def _fake_refine_with_oom_fallback(**kwargs):
        captured_strengths.append(float(kwargs["strength"]))
        return Image.new("RGB", kwargs["image"].size), 0, kwargs["tile_overlap"], 0

    monkeypatch.setattr(backend, "_run_refine_with_oom_fallback", _fake_refine_with_oom_fallback)
    monkeypatch.setattr("app.core.backends.diffusers_zimage.cuda_memory_snapshot", lambda _torch: None)
    monkeypatch.setattr("app.core.backends.diffusers_zimage.process_memory_snapshot", lambda: None)
    monkeypatch.setattr(backend, "_apply_high_runtime_fallback_if_needed", lambda **kwargs: None)

    result = backend.refine_image(
        Image.new("RGB", (512, 512), color=(12, 34, 56)),
        GenerationRequest(
            prompt="original prompt",
            width=512,
            height=512,
            seed=321,
            refine_strength=0.08,
        ),
    )

    assert captured_strengths == [pytest.approx(0.08), pytest.approx(backend._REFINE_MIN_STRENGTH)]
    assert result.refine_pass2_strength == pytest.approx(backend._REFINE_MIN_STRENGTH)


def test_run_img2img_tiled_reuses_same_seed_across_tiles(monkeypatch, temp_app_paths, make_app_settings) -> None:
    backend = _make_backend(temp_app_paths, make_app_settings)
    seeds: list[int | None] = []

    monkeypatch.setattr(
        backend,
        "_build_generator",
        lambda torch_module, device, seed: seeds.append(seed) or seed,
    )
    monkeypatch.setattr(
        backend,
        "_run_img2img_once",
        lambda **kwargs: kwargs["image"].copy(),
    )

    image = Image.new("RGB", (1600, 512), color=(20, 40, 60))
    result = backend._run_img2img_tiled(
        pipe=SimpleNamespace(),
        prompt="prompt",
        image=image,
        strength=0.2,
        steps=24,
        guidance_scale=0.0,
        seed=123,
        tile_size=896,
        tile_overlap=64,
        torch_module=torch,
    )

    assert result.size == image.size
    assert seeds == [123, 123]
