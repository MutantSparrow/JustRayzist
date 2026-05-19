from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from app.core.backends.diffusers_qwen import DiffusersQwenInference
from app.core.backends.diffusers_zimage import DiffusersZImageBackend
from app.core.worker.types import GenerationRequest


class _HelperTokenizer:
    def decode(self, _token_ids, skip_special_tokens: bool = True) -> str:
        return "decoded text"


class _NamedContext:
    def __init__(self, probe, name: str) -> None:
        self._probe = probe
        self._name = name

    def __enter__(self):
        self._probe.active = self._name
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._probe.active = None


class _TorchContextProbe:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.active: str | None = None

    def inference_mode(self):
        self.calls.append("inference_mode")
        return _NamedContext(self, "inference_mode")

    def no_grad(self):
        self.calls.append("no_grad")
        return _NamedContext(self, "no_grad")


def _make_backend(temp_app_paths, make_app_settings) -> DiffusersZImageBackend:
    settings = make_app_settings(paths=temp_app_paths)
    model_pack = SimpleNamespace(name="Rayzist_bf16", base_name="Rayzist_bf16")
    return DiffusersZImageBackend(settings=settings, model_pack=model_pack)


def _make_qwen() -> DiffusersQwenInference:
    return DiffusersQwenInference(
        tokenizer=object(),
        text_encoder=object(),
        torch_module=torch,
        encoder_label="text_encoder",
    )


def test_prompt_enhancement_prefers_base_model_decode(monkeypatch, caplog, temp_app_paths, make_app_settings) -> None:
    backend = _make_qwen()
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
    caplog.set_level(logging.DEBUG, logger="app.core.backends.diffusers_qwen")

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
    backend = _make_qwen()
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
    caplog.set_level(logging.DEBUG, logger="app.core.backends.diffusers_qwen")

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
    backend = _make_qwen()
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
    caplog.set_level(logging.DEBUG, logger="app.core.backends.diffusers_qwen")

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


def test_move_encoded_to_module_device_uses_embedding_weight_device(temp_app_paths, make_app_settings) -> None:
    backend = _make_qwen()

    class _FakeTensor:
        def __init__(self, label: str) -> None:
            self.label = label
            self.moves: list[object] = []

        def to(self, device: object) -> str:
            self.moves.append(device)
            return f"{self.label}@{device}"

    class _TextEncoder:
        def parameters(self):
            return iter(())

        def get_input_embeddings(self):
            return SimpleNamespace(weight=SimpleNamespace(device=torch.device("cuda")))

    input_ids = _FakeTensor("ids")
    attention_mask = _FakeTensor("mask")
    moved = backend._move_encoded_to_module_device(
        {"input_ids": input_ids, "attention_mask": attention_mask, "raw": "keep"},
        _TextEncoder(),
    )

    assert moved["input_ids"] == "ids@cuda"
    assert moved["attention_mask"] == "mask@cuda"
    assert moved["raw"] == "keep"
    assert input_ids.moves == [torch.device("cuda")]
    assert attention_mask.moves == [torch.device("cuda")]


def test_prompt_enhancement_fallback_sanitizes_greedy_generation_config(
    monkeypatch,
    caplog,
    temp_app_paths,
    make_app_settings,
) -> None:
    backend = _make_qwen()
    tokenizer = _HelperTokenizer()
    encoded = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}
    captured: dict[str, object] = {}

    class _TextEncoder:
        def __init__(self) -> None:
            self.generation_config = SimpleNamespace(
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
            )

        def generate(self, **kwargs):
            captured["kwargs"] = kwargs
            return torch.tensor([[1, 2, 3]])

    monkeypatch.setattr(
        backend,
        "_generate_with_base_model",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("missing base-model decode support")),
    )
    monkeypatch.setattr(backend, "_extract_rewritten_prompt", lambda full_text, input_text: "enhanced prompt")
    monkeypatch.setattr(backend, "_rewrite_rejection_reason", lambda original, rewritten: "ok")
    caplog.set_level(logging.DEBUG, logger="app.core.backends.diffusers_qwen")

    rewritten, reason = backend._run_rewrite_attempt(
        tokenizer=tokenizer,
        text_encoder=_TextEncoder(),
        encoded=encoded,
        prompt="original prompt",
        torch_module=torch,
        generate_kwargs={"max_new_tokens": 24, "do_sample": False},
        enhancement_seed=123,
    )

    kwargs = captured["kwargs"]
    generation_config = kwargs["generation_config"]
    assert rewritten == "enhanced prompt"
    assert reason == "ok"
    assert generation_config.do_sample is False
    assert generation_config.temperature == pytest.approx(1.0)
    assert generation_config.top_p == pytest.approx(1.0)
    assert generation_config.top_k == 50
    assert kwargs["use_model_defaults"] is False
    assert "do_sample" not in kwargs
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs
    assert any("falling back to text_encoder.generate" in record.getMessage() for record in caplog.records)


def test_helper_text_generation_fallback_builds_explicit_sampled_generation_config(
    monkeypatch,
    temp_app_paths,
    make_app_settings,
) -> None:
    backend = _make_qwen()
    tokenizer = _HelperTokenizer()
    encoded = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}
    captured: dict[str, object] = {}

    class _TextEncoder:
        def __init__(self) -> None:
            self.generation_config = SimpleNamespace(
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                repetition_penalty=1.08,
            )

        def generate(self, **kwargs):
            captured["kwargs"] = kwargs
            return torch.tensor([[1, 2, 3]])

    monkeypatch.setattr(
        backend,
        "_generate_with_base_model",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("missing base-model decode support")),
    )
    monkeypatch.setattr(backend, "_extract_generated_completion_text", lambda full_text, input_text: "fallback text")

    generated = backend._run_text_generation_attempt(
        tokenizer=tokenizer,
        text_encoder=_TextEncoder(),
        encoded=encoded,
        torch_module=torch,
        generate_kwargs={
            "max_new_tokens": 16,
            "do_sample": True,
            "temperature": 0.72,
            "top_p": 0.92,
            "repetition_penalty": 1.15,
        },
        generation_seed=789,
    )

    kwargs = captured["kwargs"]
    generation_config = kwargs["generation_config"]
    assert generated == "fallback text"
    assert generation_config.do_sample is True
    assert generation_config.temperature == pytest.approx(0.72)
    assert generation_config.top_p == pytest.approx(0.92)
    assert generation_config.top_k == 20
    assert generation_config.repetition_penalty == pytest.approx(1.15)
    assert kwargs["use_model_defaults"] is False
    assert "do_sample" not in kwargs
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs
    assert "repetition_penalty" not in kwargs


def test_helper_text_generation_falls_back_when_base_decode_is_blank(
    monkeypatch,
    caplog,
    temp_app_paths,
    make_app_settings,
) -> None:
    backend = _make_qwen()
    encoded = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}

    class _Tokenizer:
        def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
            values = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
            if values == [1, 2]:
                return "prompt"
            if values == [1, 2, 4, 5]:
                return "prompt fallback answer"
            if values == [4, 5]:
                return "fallback answer"
            return ""

    class _TextEncoder:
        def __init__(self) -> None:
            self.generate_called = False

        def generate(self, **_kwargs):
            self.generate_called = True
            return torch.tensor([[1, 2, 4, 5]])

    text_encoder = _TextEncoder()
    monkeypatch.setattr(backend, "_generate_with_base_model", lambda **_kwargs: torch.tensor([[1, 2]]))
    caplog.set_level(logging.DEBUG, logger="app.core.backends.diffusers_qwen")

    generated = backend._run_text_generation_attempt(
        tokenizer=_Tokenizer(),
        text_encoder=text_encoder,
        encoded=encoded,
        torch_module=torch,
        generate_kwargs={"max_new_tokens": 16, "do_sample": True},
        generation_seed=789,
    )

    assert generated == "fallback answer"
    assert text_encoder.generate_called is True
    assert any("returned no completion" in record.getMessage() for record in caplog.records)


def test_chat_prompt_uses_tokenizer_chat_template() -> None:
    class _ChatTokenizer:
        def __init__(self) -> None:
            self.messages: list[dict[str, str]] = []

        def apply_chat_template(self, messages, **_kwargs):
            self.messages = messages
            return "rendered chat"

    tokenizer = _ChatTokenizer()
    rendered = DiffusersQwenInference._build_chat_prompt(
        tokenizer,
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "help"},
        ],
    )

    assert rendered == "rendered chat"
    assert tokenizer.messages[0]["role"] == "system"
    assert "Visible state" in DiffusersQwenInference._build_chat_prompt(object(), [{"role": "user", "content": "hello"}], app_context="Visible state")
    assert tokenizer.messages[-1] == {"role": "user", "content": "help"}


def test_chat_response_extraction_removes_template_continuation() -> None:
    text = "<think>hidden</think>\nAssistant: Use stronger side light.\nUser: thanks"
    assert DiffusersQwenInference._extract_chat_response_text(text) == "Use stronger side light."


def test_chat_fallback_handles_blank_greeting() -> None:
    assert DiffusersQwenInference._extract_chat_response_parts("Assistant:")[0] == ""
    assert (
        DiffusersQwenInference._fallback_chat_response([{"role": "user", "content": "hi"}])
        == "Ask for prompt help, workflow help, or a prompt draft. I can also offer buttons to use a prompt or open /API."
    )


def test_chat_action_only_response_uses_prompt_as_visible_text() -> None:
    assert (
        DiffusersQwenInference._chat_response_from_actions(
            [{"type": "set_prompt", "label": "Use Prompt", "prompt": "rainy neon alley"}]
        )
        == "Prompt ready:\nrainy neon alley"
    )


def test_chat_response_extraction_returns_valid_actions_only() -> None:
    text = (
        "Assistant: Use this prompt.\n"
        "<rayzist-actions>{\"actions\":["
        "{\"type\":\"set_prompt\",\"label\":\"Use\",\"prompt\":\"rainy neon alley\"},"
        "{\"type\":\"open_route\",\"href\":\"/API\"},"
        "{\"type\":\"open_route\",\"href\":\"https://example.com\"}"
        "]}</rayzist-actions>"
    )

    response, actions = DiffusersQwenInference._extract_chat_response_parts(text)

    assert response == "Use this prompt."
    assert actions == [
        {"type": "set_prompt", "prompt": "rainy neon alley", "label": "Use"},
        {"type": "open_route", "href": "/API", "label": "Open API"},
    ]


def test_chat_response_extraction_strips_dangling_action_json() -> None:
    text = (
        "Clarity is an image refinement step.\n\n"
        "<rayzist-actions>{\"actions\":["
        "{\"type\":\"set_prompt\",\"label\":\"Clarity Definition (Final)\",\"prompt\":\"Clarity is post generation\"},"
        "{\"type\":\"set_prompt\",\"label\":\"Clarity Usage Notes\",\"prompt\":\"Use it on gallery images\"}"
        "]}"
    )

    response, actions = DiffusersQwenInference._extract_chat_response_parts(text)

    assert response == "Clarity is an image refinement step."
    assert actions == [
        {
            "type": "set_prompt",
            "prompt": "Clarity is post generation",
            "label": "Clarity Definition (Final)",
        },
        {
            "type": "set_prompt",
            "prompt": "Use it on gallery images",
            "label": "Clarity Usage Notes",
        },
    ]


def test_chat_action_filter_drops_prompt_buttons_for_help_answers() -> None:
    actions = [
        {
            "type": "set_prompt",
            "label": "Clarify Image Refinement",
            "prompt": "Clarity refines image detail on existing gallery images after generation.",
        },
        {"type": "open_route", "label": "Open API", "href": "/API"},
    ]

    assert DiffusersQwenInference._filter_chat_actions_for_turn(
        actions,
        messages=[{"role": "user", "content": "what does clarity do?"}],
    ) == [{"type": "open_route", "label": "Open API", "href": "/API"}]


def test_chat_action_filter_keeps_real_image_prompt_when_requested() -> None:
    actions = [
        {
            "type": "set_prompt",
            "label": "Use Prompt",
            "prompt": "fantasy babe with glowing eyes, mystical forest, moonlight, medium shot, cinematic lighting",
        }
    ]

    assert DiffusersQwenInference._filter_chat_actions_for_turn(
        actions,
        messages=[{"role": "user", "content": "write me a prompt about a fantasy babe"}],
    ) == actions


def test_zimage_chat_delegates_to_qwen_runtime(monkeypatch, temp_app_paths, make_app_settings) -> None:
    backend = _make_backend(temp_app_paths, make_app_settings)
    pipe = SimpleNamespace(tokenizer="tokenizer", text_encoder="encoder")
    captured: dict[str, object] = {}

    monkeypatch.setattr(backend, "_ensure_loaded", lambda: SimpleNamespace(pipeline=pipe, device="cpu"))

    def fake_chat(self, *, messages, max_new_tokens, seed, app_context, temperature):
        captured["tokenizer"] = self._tokenizer
        captured["text_encoder"] = self._text_encoder
        captured["messages"] = messages
        captured["max_new_tokens"] = max_new_tokens
        captured["seed"] = seed
        captured["app_context"] = app_context
        captured["temperature"] = temperature
        return {"text": "ok", "actions": [], "encoder": self.encoder_label}

    monkeypatch.setattr(DiffusersQwenInference, "chat", fake_chat)

    result = backend.chat(
        [{"role": "user", "content": "hello"}],
        max_new_tokens=17,
        seed=5,
        app_context="Visible state",
        temperature=0.25,
    )

    assert result == {"text": "ok", "actions": [], "encoder": "Rayzist_bf16"}
    assert captured == {
        "tokenizer": "tokenizer",
        "text_encoder": "encoder",
        "messages": [{"role": "user", "content": "hello"}],
        "max_new_tokens": 17,
        "seed": 5,
        "app_context": "Visible state",
        "temperature": 0.25,
    }


def test_zimage_prompt_enhancement_delegates_to_qwen_runtime(monkeypatch, temp_app_paths, make_app_settings) -> None:
    backend = _make_backend(temp_app_paths, make_app_settings)
    pipe = SimpleNamespace(tokenizer="tokenizer", text_encoder="encoder")
    captured: dict[str, object] = {}

    def fake_enhance_prompt(self, prompt, *, seed):
        captured["tokenizer"] = self._tokenizer
        captured["text_encoder"] = self._text_encoder
        captured["prompt"] = prompt
        captured["seed"] = seed
        return "enhanced"

    monkeypatch.setattr(DiffusersQwenInference, "enhance_prompt", fake_enhance_prompt)

    assert backend._enhance_prompt(pipe, "portrait", torch, seed=9) == "enhanced"
    assert captured == {
        "tokenizer": "tokenizer",
        "text_encoder": "encoder",
        "prompt": "portrait",
        "seed": 9,
    }


def test_pipeline_forward_context_uses_no_grad_for_accelerate_hooks() -> None:
    probe = _TorchContextProbe()
    pipe = SimpleNamespace(transformer=SimpleNamespace(_hf_hook=object()))

    with DiffusersZImageBackend._pipeline_forward_context(probe, pipe):
        assert probe.active == "no_grad"

    assert probe.calls == ["no_grad"]


def test_qwen_forward_context_uses_no_grad_for_accelerate_hooks() -> None:
    probe = _TorchContextProbe()
    text_encoder = SimpleNamespace(_hf_hook=object())

    with DiffusersQwenInference._module_forward_context(probe, text_encoder):
        assert probe.active == "no_grad"

    assert probe.calls == ["no_grad"]


def test_forward_context_uses_inference_mode_without_accelerate_hooks() -> None:
    probe = _TorchContextProbe()

    with DiffusersZImageBackend._pipeline_forward_context(probe, SimpleNamespace(transformer=SimpleNamespace())):
        assert probe.active == "inference_mode"

    assert probe.calls == ["inference_mode"]


def test_img2img_prepare_latents_encodes_with_vae_dtype_when_prompt_dtype_differs() -> None:
    records: dict[str, torch.dtype] = {}

    class _LatentDist:
        def __init__(self, tensor: torch.Tensor) -> None:
            self._tensor = tensor

        def sample(self, generator=None) -> torch.Tensor:
            return self._tensor

    class _FakeVae:
        dtype = torch.float32
        config = SimpleNamespace(shift_factor=0.0, scaling_factor=1.0)

        def encode(self, image: torch.Tensor) -> SimpleNamespace:
            records["encode_dtype"] = image.dtype
            latent = torch.zeros(
                (image.shape[0], 16, 4, 4),
                dtype=image.dtype,
                device=image.device,
            )
            return SimpleNamespace(latent_dist=_LatentDist(latent))

    class _FakeScheduler:
        def scale_noise(self, image_latents: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
            records["image_latents_dtype"] = image_latents.dtype
            records["noise_dtype"] = noise.dtype
            return image_latents + noise.to(dtype=image_latents.dtype)

    pipe = SimpleNamespace(
        prepare_latents=lambda *args, **kwargs: None,
        vae=_FakeVae(),
        scheduler=_FakeScheduler(),
        vae_scale_factor=8,
    )

    DiffusersZImageBackend._patch_img2img_prepare_latents_vae_dtype(pipe, torch)
    latents = pipe.prepare_latents(
        torch.zeros((1, 3, 32, 32), dtype=torch.float32),
        torch.tensor([1.0]),
        1,
        16,
        32,
        32,
        torch.bfloat16,
        "cpu",
        None,
    )

    assert records["encode_dtype"] == torch.float32
    assert records["image_latents_dtype"] == torch.float32
    assert records["noise_dtype"] == torch.bfloat16
    assert latents.dtype == torch.float32


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
