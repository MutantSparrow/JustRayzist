from __future__ import annotations

from types import SimpleNamespace

import torch

from app.core.backends.diffusers_zimage import DiffusersZImageBackend


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
        return SimpleNamespace(input_ids=torch.zeros((1, token_count), dtype=torch.long))


def test_build_rewrite_prompt_preserves_explicit_style_instructions() -> None:
    prompt = DiffusersZImageBackend._build_rewrite_prompt(_FakeTokenizer(), "anime portrait of a warrior")

    assert "preserve any explicit medium or style exactly" in prompt
    assert "If the user says anime, keep it anime." in prompt
    assert "under 125 tokens" not in prompt


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


def test_fit_prompt_to_budget_truncates_single_clause_fallback_to_budget(monkeypatch) -> None:
    monkeypatch.setattr(
        DiffusersZImageBackend,
        "_PROMPT_ENHANCEMENT_PIPELINE_SAFE_TOKEN_BUDGET",
        4,
    )
    tokenizer = _FakeTokenizer()
    original = "hero portrait atmospheric fog rain neon reflections"
    enhanced = "hero portrait atmospheric fog rain neon reflections cinematic detail storybook framing"

    fitted, enhanced_used = DiffusersZImageBackend._fit_prompt_to_budget(
        tokenizer=tokenizer,
        original_prompt=original,
        enhanced_prompt=enhanced,
    )

    assert enhanced_used is False
    assert fitted == "hero portrait atmospheric"
    assert fitted != original
    assert DiffusersZImageBackend._pipeline_prompt_token_length(tokenizer, fitted) <= 4
