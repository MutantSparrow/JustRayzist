from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from app.core.backends.diffusers_zimage import DiffusersZImageBackend
from app.core.worker.types import LoraSelection


class _FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
    ) -> str:
        content = messages[0]["content"]
        return f"<user> {content}"

    def __call__(self, text, return_tensors: str = "pt", truncation: bool = False):
        if isinstance(text, list):
            text = " ".join(text)
        token_count = len(str(text).split())
        return {"input_ids": torch.zeros((1, token_count), dtype=torch.long)}


def test_build_rewrite_prompt_preserves_explicit_style_instructions() -> None:
    prompt = DiffusersZImageBackend._build_rewrite_prompt(_FakeTokenizer(), "anime portrait of a warrior")

    assert "preserve any explicit medium or style exactly" in prompt
    assert "If the user says anime, keep it anime." in prompt
    assert "under 125 tokens" not in prompt


def test_build_compression_prompt_requests_complete_sentence_output() -> None:
    prompt = DiffusersZImageBackend._build_compression_prompt(
        _FakeTokenizer(),
        "a long prompt",
        target_tokens=440,
    )

    assert "under 440 tokens" in prompt
    assert "Use complete sentences only." in prompt


def test_resolve_pipeline_max_sequence_length_expands_for_actual_prompt_tokens() -> None:
    prompt = " ".join(f"detail{i}" for i in range(530))

    max_sequence_length = DiffusersZImageBackend._resolve_pipeline_max_sequence_length(
        _FakeTokenizer(),
        prompt,
    )

    assert max_sequence_length > DiffusersZImageBackend._PROMPT_ENHANCEMENT_PIPELINE_MAX_SEQUENCE_LENGTH
    assert max_sequence_length == DiffusersZImageBackend._pipeline_prompt_token_length(_FakeTokenizer(), prompt)


def test_rewrite_rejection_allows_shorter_but_valid_rewrites() -> None:
    original = "anime portrait of a silver-haired swordswoman standing in moonlit rain with neon reflections and dramatic rim lighting"
    rewritten = "anime portrait of a silver-haired swordswoman in moonlit rain, neon reflections, dramatic rim lighting"

    assert DiffusersZImageBackend._rewrite_rejection_reason(original, rewritten) == "ok"


def test_fit_prompt_to_budget_compresses_and_preserves_style_constraints(monkeypatch) -> None:
    monkeypatch.setattr(
        DiffusersZImageBackend,
        "_PROMPT_ENHANCEMENT_PIPELINE_SAFE_TOKEN_BUDGET",
        14,
    )
    tokenizer = _FakeTokenizer()
    original = "anime portrait of a swordswoman, cel shading, dynamic composition"
    enhanced = (
        "battle-worn swordswoman, moonlit rooftop rain, neon puddle reflections, "
        "dramatic rim lighting, low-angle composition, cel-shaded wet fabric"
    )

    fitted, enhanced_used = DiffusersZImageBackend._fit_prompt_to_budget(
        tokenizer=tokenizer,
        original_prompt=original,
        enhanced_prompt=enhanced,
    )

    assert enhanced_used is True
    assert "anime" in fitted.lower()
    assert "cel" in fitted.lower()
    assert DiffusersZImageBackend._pipeline_prompt_token_length(tokenizer, fitted) <= 14


def test_fit_prompt_to_budget_falls_back_to_original_when_candidate_cannot_fit(monkeypatch) -> None:
    monkeypatch.setattr(
        DiffusersZImageBackend,
        "_PROMPT_ENHANCEMENT_PIPELINE_SAFE_TOKEN_BUDGET",
        6,
    )
    tokenizer = _FakeTokenizer()
    original = "anime portrait hero"
    enhanced = "highly detailed hyper elaborate cinematic anime portrait hero with layered atmospheric storytelling and intricate material rendering"

    fitted, enhanced_used = DiffusersZImageBackend._fit_prompt_to_budget(
        tokenizer=tokenizer,
        original_prompt=original,
        enhanced_prompt=enhanced,
    )

    assert enhanced_used is False
    assert fitted == original


def test_fit_prompt_to_budget_prioritizes_late_style_clauses(monkeypatch) -> None:
    monkeypatch.setattr(
        DiffusersZImageBackend,
        "_PROMPT_ENHANCEMENT_PIPELINE_SAFE_TOKEN_BUDGET",
        9,
    )
    tokenizer = _FakeTokenizer()
    original = (
        "hero portrait, extra detail one, extra detail two, "
        "extra detail three, anime illustration, cel shading"
    )
    enhanced = (
        "hero portrait, extra detail one, extra detail two, extra detail three, "
        "extra detail four, dramatic rim lighting, anime illustration, cel shading"
    )

    fitted, enhanced_used = DiffusersZImageBackend._fit_prompt_to_budget(
        tokenizer=tokenizer,
        original_prompt=original,
        enhanced_prompt=enhanced,
    )

    assert enhanced_used is True
    assert "anime" in fitted.lower()
    assert "cel" in fitted.lower()
    assert DiffusersZImageBackend._pipeline_prompt_token_length(tokenizer, fitted) <= 9


def test_fit_prompt_to_budget_uses_style_seed_when_compression_fails(monkeypatch) -> None:
    tokenizer = _FakeTokenizer()
    original = "hero portrait, rain detail, anime illustration, cel shading"
    enhanced = "hero portrait, atmospheric detail, cinematic framing, anime illustration, cel shading"
    reduced = "hero portrait, anime illustration, cel shading"
    calls: list[str] = []

    def _fake_compress(
        cls,
        *,
        tokenizer,
        original_prompt,
        candidate_prompt,
        max_tokens=None,
    ) -> str | None:
        calls.append(candidate_prompt)
        if candidate_prompt in {enhanced, original}:
            return None
        if candidate_prompt == reduced:
            return reduced
        raise AssertionError(f"Unexpected candidate prompt: {candidate_prompt}")

    monkeypatch.setattr(
        DiffusersZImageBackend,
        "_compress_prompt_to_token_budget",
        classmethod(_fake_compress),
    )

    fitted, enhanced_used = DiffusersZImageBackend._fit_prompt_to_budget(
        tokenizer=tokenizer,
        original_prompt=original,
        enhanced_prompt=enhanced,
    )

    assert enhanced_used is False
    assert fitted == reduced
    assert calls == [enhanced, original, reduced]


def test_fit_prompt_to_budget_preserves_whole_clause_when_over_budget(monkeypatch) -> None:
    monkeypatch.setattr(
        DiffusersZImageBackend,
        "_PROMPT_ENHANCEMENT_PIPELINE_SAFE_TOKEN_BUDGET",
        5,
    )
    tokenizer = _FakeTokenizer()
    original = "hero portrait. atmospheric fog and rain. neon reflections."
    enhanced = "hero portrait. atmospheric fog and rain. neon reflections. cinematic detail storybook framing."

    fitted, enhanced_used = DiffusersZImageBackend._fit_prompt_to_budget(
        tokenizer=tokenizer,
        original_prompt=original,
        enhanced_prompt=enhanced,
    )

    assert enhanced_used is True
    assert fitted == "hero portrait. neon reflections."
    assert fitted != original
    assert DiffusersZImageBackend._pipeline_prompt_token_length(tokenizer, fitted) <= 5


def test_resolve_effective_prompt_keeps_long_prompt_exact_when_enhancement_off() -> None:
    backend = object.__new__(DiffusersZImageBackend)
    prompt = " ".join(f"detail{i}" for i in range(80))

    original, effective, enhanced = backend._resolve_effective_prompt(
        pipe=SimpleNamespace(tokenizer=_FakeTokenizer()),
        prompt=prompt,
        enhance_prompt=False,
        seed=None,
        torch_module=torch,
    )

    assert original == prompt
    assert effective == prompt
    assert enhanced is False


def test_resolve_effective_prompt_uses_enhancement_for_short_prompt(monkeypatch) -> None:
    backend = object.__new__(DiffusersZImageBackend)
    monkeypatch.setattr(backend, "_ensure_loaded", lambda: SimpleNamespace(device="cpu"))
    monkeypatch.setattr(
        backend,
        "_enhance_prompt",
        lambda pipe, prompt, torch_module, seed=None: "A focused cinematic portrait.",
    )
    monkeypatch.setattr(
        backend,
        "_compress_long_prompt",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("compression should not run")),
    )

    _, effective, enhanced = backend._resolve_effective_prompt(
        pipe=SimpleNamespace(tokenizer=_FakeTokenizer(), text_encoder=object()),
        prompt="portrait",
        enhance_prompt=True,
        seed=1,
        torch_module=torch,
    )

    assert effective == "A focused cinematic portrait."
    assert enhanced is True


def test_resolve_effective_prompt_uses_compression_for_long_prompt(monkeypatch) -> None:
    monkeypatch.setattr(
        DiffusersZImageBackend,
        "_PROMPT_ENHANCEMENT_PIPELINE_SAFE_TOKEN_BUDGET",
        8,
    )
    backend = object.__new__(DiffusersZImageBackend)
    captured: dict[str, str] = {}
    monkeypatch.setattr(backend, "_ensure_loaded", lambda: SimpleNamespace(device="cpu"))
    monkeypatch.setattr(
        backend,
        "_enhance_prompt",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("enhancement should not run")),
    )

    def fake_compress(pipe, prompt, torch_module, seed=None):
        captured["prompt"] = prompt
        return "A compact prompt."

    monkeypatch.setattr(backend, "_compress_long_prompt", fake_compress)

    prompt = "one two three four five six seven eight nine"
    _, effective, enhanced = backend._resolve_effective_prompt(
        pipe=SimpleNamespace(tokenizer=_FakeTokenizer(), text_encoder=object()),
        prompt=prompt,
        enhance_prompt=True,
        seed=1,
        torch_module=torch,
    )

    assert captured["prompt"] == prompt
    assert effective == "A compact prompt."
    assert enhanced is True
    assert DiffusersZImageBackend._pipeline_prompt_token_length(_FakeTokenizer(), effective) <= 8


def test_trim_to_complete_sentences_removes_incomplete_clothing_fragment() -> None:
    text = (
        "A young woman stands in a bright bookstore with sale boxes and browsing customers. "
        "She wears a sleeveless"
    )

    assert (
        DiffusersZImageBackend._trim_to_complete_sentences(text)
        == "A young woman stands in a bright bookstore with sale boxes and browsing customers."
    )


def test_trim_to_complete_sentences_removes_incomplete_depth_fragment() -> None:
    text = (
        "A vast hangar laboratory contains machinery, cables, and a distant saucer on a raised platform. "
        "Further back,"
    )

    assert (
        DiffusersZImageBackend._trim_to_complete_sentences(text)
        == "A vast hangar laboratory contains machinery, cables, and a distant saucer on a raised platform."
    )


def test_fit_complete_clauses_to_budget_never_returns_partial_clause(monkeypatch) -> None:
    monkeypatch.setattr(
        DiffusersZImageBackend,
        "_PROMPT_ENHANCEMENT_PIPELINE_SAFE_TOKEN_BUDGET",
        5,
    )
    fitted = DiffusersZImageBackend._fit_complete_clauses_to_budget(
        tokenizer=_FakeTokenizer(),
        candidate_prompt="First complete sentence. Second complete sentence with many details.",
        max_tokens=5,
    )

    assert fitted == "First complete sentence."


def test_compress_long_prompt_retries_after_incomplete_sentence(monkeypatch) -> None:
    monkeypatch.setattr(
        DiffusersZImageBackend,
        "_PROMPT_ENHANCEMENT_PIPELINE_SAFE_TOKEN_BUDGET",
        12,
    )
    monkeypatch.setattr(
        DiffusersZImageBackend,
        "_PROMPT_ENHANCEMENT_COMPRESSION_TARGET_TOKENS",
        10,
    )
    backend = object.__new__(DiffusersZImageBackend)
    calls: list[int] = []

    def fake_run_rewrite_attempt(**kwargs):
        calls.append(int(kwargs["generate_kwargs"]["max_new_tokens"]))
        if len(calls) == 1:
            return "A woman stands in a bookstore. She wears a sleeveless", "ok"
        return "A woman stands in a bookstore.", "ok"

    monkeypatch.setattr(backend, "_run_rewrite_attempt", fake_run_rewrite_attempt)

    compressed = backend._compress_long_prompt(
        SimpleNamespace(tokenizer=_FakeTokenizer(), text_encoder=object()),
        " ".join(f"detail{i}" for i in range(20)),
        torch,
        seed=3,
    )

    assert compressed == "A woman stands in a bookstore."
    assert calls == [10, 8]


def test_lora_triggers_are_appended_after_compression_input(monkeypatch) -> None:
    monkeypatch.setattr(
        DiffusersZImageBackend,
        "_PROMPT_ENHANCEMENT_PIPELINE_SAFE_TOKEN_BUDGET",
        8,
    )
    backend = object.__new__(DiffusersZImageBackend)
    captured: dict[str, str] = {}
    monkeypatch.setattr(backend, "_ensure_loaded", lambda: SimpleNamespace(device="cpu"))

    def fake_compress(pipe, prompt, torch_module, seed=None):
        captured["prompt"] = prompt
        return "A compact portrait."

    monkeypatch.setattr(backend, "_compress_long_prompt", fake_compress)

    _, base_prompt, _ = backend._resolve_effective_prompt(
        pipe=SimpleNamespace(tokenizer=_FakeTokenizer(), text_encoder=object()),
        prompt="one two three four five six seven eight nine",
        enhance_prompt=True,
        seed=1,
        torch_module=torch,
    )
    final_prompt, triggers = DiffusersZImageBackend._append_lora_triggers(
        base_prompt,
        (
            LoraSelection(
                id="cinematic-style",
                path=Path("cinematic-style.safetensors"),
                trigger_words=("cinematic style",),
            ),
        ),
    )

    assert "cinematic style" not in captured["prompt"]
    assert final_prompt == "A compact portrait, cinematic style"
    assert triggers == ("cinematic style",)
