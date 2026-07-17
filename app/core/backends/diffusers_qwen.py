from __future__ import annotations

import copy
import inspect
import json
import logging
import math
import re
import string
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

from app.core.chat_actions import normalize_chat_actions, strip_chat_action_markup
from app.storage.wildcard_library import normalize_wildcard_entry_value

LOGGER = logging.getLogger(__name__)


class DiffusersQwenInference:
    _PROMPT_ENHANCEMENT_PRIMARY_MAX_NEW_TOKENS = 120
    _PROMPT_ENHANCEMENT_RETRY_MAX_NEW_TOKENS = 160
    _PROMPT_ENHANCEMENT_PIPELINE_MAX_SEQUENCE_LENGTH = 512
    _PROMPT_ENHANCEMENT_PIPELINE_SAFE_TOKEN_BUDGET = 480
    _PROMPT_ENHANCEMENT_COMPRESSION_TARGET_TOKENS = 440
    _CHAT_DEFAULT_MAX_NEW_TOKENS = 256
    _CHAT_MAX_NEW_TOKENS = 512
    _CHAT_SYSTEM_PROMPT = (
        "You are Rayzist Chat inside Just Rayzist. Be direct, terse, and practical. No cheerleading. No flattery. "
        "No cute phrasing. No motivational language. Help with prompts, workflow, settings, and short creative "
        "iteration. If uncertain, say so. Do not claim access to files, generated images, or current UI state unless "
        "that context is explicitly provided. You may suggest actions only by adding one exact XML block after the "
        "visible answer: <rayzist-actions>{\"actions\":[...]}</rayzist-actions>. Allowed action types are "
        "set_prompt, append_prompt, start_generation, and open_route. For prompt actions include prompt. For "
        "open_route use href /API only. Example action block: "
        "<rayzist-actions>{\"actions\":[{\"type\":\"set_prompt\",\"label\":\"Use Prompt\",\"prompt\":\"cinematic portrait\"}]}</rayzist-actions>. "
        "Do not add prompt actions for definitions, explanations, app help, API help, Clarity help, or settings help. "
        "Prefer normal UI workflow instructions over API routes unless the user specifically asks about API usage. "
        "Never invent routes, file actions, shell actions, or hidden commands."
    )
    _PROMPT_STYLE_PATTERNS: tuple[str, ...] = (
        r"\banime\b",
        r"\bmanga\b",
        r"\bcartoon\b",
        r"\billustration\b",
        r"\bcomic(?: book)?\b",
        r"\bcel(?:-| )shad(?:e|ed|ing)\b",
        r"\bpixel art\b",
        r"\boil painting\b",
        r"\bwatercolou?r\b",
        r"\bgouache\b",
        r"\bpastel\b",
        r"\bconcept art\b",
        r"\bmatte painting\b",
        r"\bphotograph(?:y|ic)?\b",
        r"\bphoto(?:realistic)?\b",
        r"\bcinematic\b",
        r"\beditorial\b",
        r"\b3d render\b",
        r"\b3d\b",
        r"\bcgi\b",
        r"\bdigital painting\b",
        r"\bsketch\b",
        r"\bcharcoal\b",
        r"\bink(?: drawing)?\b",
    )
    _PROMPT_PRIORITY_KEYWORDS: tuple[str, ...] = (
        "lighting",
        "light",
        "rim light",
        "backlight",
        "composition",
        "framing",
        "camera",
        "lens",
        "shot on",
        "close-up",
        "portrait",
        "wide shot",
        "environment",
        "background",
        "interior",
        "exterior",
        "studio",
        "sunset",
        "night",
        "material",
        "texture",
        "color palette",
    )

    def __init__(
        self,
        *,
        tokenizer: Any,
        text_encoder: Any,
        torch_module: Any,
        encoder_label: str = "text_encoder",
    ) -> None:
        self._tokenizer = tokenizer
        self._text_encoder = text_encoder
        self._torch = torch_module
        self._encoder_label = encoder_label

    @classmethod
    def from_pipe(cls, pipe: Any, *, torch_module: Any, encoder_label: str = "text_encoder") -> "DiffusersQwenInference":
        return cls(
            tokenizer=getattr(pipe, "tokenizer", None),
            text_encoder=getattr(pipe, "text_encoder", None),
            torch_module=torch_module,
            encoder_label=encoder_label,
        )

    @property
    def encoder_label(self) -> str:
        return self._encoder_label

    @staticmethod
    def _module_has_accelerate_hook(module: Any) -> bool:
        if module is None:
            return False
        if hasattr(module, "_hf_hook"):
            return True
        iter_modules = getattr(module, "modules", None)
        if not callable(iter_modules):
            return False
        try:
            return any(hasattr(child, "_hf_hook") for child in iter_modules())
        except Exception:
            return False

    @classmethod
    def _module_forward_context(cls, torch_module: Any, module: Any) -> Any:
        if cls._module_has_accelerate_hook(module):
            return torch_module.no_grad()
        return torch_module.inference_mode()

    def _pipe_view(self) -> SimpleNamespace:
        return SimpleNamespace(tokenizer=self._tokenizer, text_encoder=self._text_encoder)

    def enhance_prompt(self, prompt: str, *, seed: int | None = None) -> str:
        return self._enhance_prompt(self._pipe_view(), prompt, self._torch, seed=seed)

    def compress_long_prompt(self, prompt: str, *, seed: int | None = None) -> str:
        return self._compress_long_prompt(self._pipe_view(), prompt, self._torch, seed=seed)

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        app_context: str | None = None,
        seed: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 0.75,
    ) -> dict[str, Any]:
        normalized_messages: list[dict[str, str]] = []
        for message in messages:
            role = str(message.get("role") or "").strip().lower()
            content = str(message.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            normalized_messages.append({"role": role, "content": content})
        if not any(message["role"] == "user" for message in normalized_messages):
            raise ValueError("Chat message is required.")

        tokenizer = self._tokenizer
        text_encoder = self._text_encoder
        if tokenizer is None or text_encoder is None:
            raise ValueError("Rayzist Chat is unavailable for the current runtime.")

        prompt_text = self._build_chat_prompt(tokenizer, normalized_messages, app_context=app_context)
        try:
            encoded = tokenizer(prompt_text, return_tensors="pt")
        except Exception as exc:
            raise ValueError(f"Chat tokenizer failed: {exc}") from exc

        encoded = self._move_encoded_to_module_device(encoded, text_encoder)
        token_budget = max(16, min(int(max_new_tokens or self._CHAT_DEFAULT_MAX_NEW_TOKENS), self._CHAT_MAX_NEW_TOKENS))
        sample_temperature = max(float(temperature), 0.0)
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": token_budget,
            "do_sample": sample_temperature > 0.0,
            "temperature": max(sample_temperature, 1e-5),
            "top_p": 0.92,
            "repetition_penalty": 1.08,
        }
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is not None:
            generate_kwargs["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            generate_kwargs["eos_token_id"] = eos_token_id

        try:
            generated_text = self._run_text_generation_attempt(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                encoded=encoded,
                torch_module=self._torch,
                generate_kwargs=generate_kwargs,
                generation_seed=seed,
            )
        except Exception as exc:
            raise ValueError(f"Chat decode failed: {exc}") from exc

        response_text, actions = self._extract_chat_response_parts(generated_text)
        actions = self._filter_chat_actions_for_turn(actions, messages=normalized_messages)
        if not response_text:
            response_text = self._chat_response_from_actions(actions) or self._fallback_chat_response(normalized_messages)
        return {
            "response": response_text,
            "actions": actions,
            "seed": int(seed) if seed is not None else None,
            "max_new_tokens": token_budget,
            "temperature": sample_temperature,
            "encoder": self._encoder_label,
        }

    def suggest_wildcard_entries(
        self,
        *,
        theme: str,
        format_example: str,
        seed: int | None = None,
        existing_entries: list[str] | tuple[str, ...] | None = None,
        target_count: int = 10,
    ) -> dict[str, Any]:
        normalized_theme = re.sub(r"\s+", " ", str(theme or "").strip())
        normalized_example = normalize_wildcard_entry_value(format_example)
        if not normalized_theme:
            raise ValueError("Wildcard suggestion theme is required.")
        if not normalized_example:
            raise ValueError("Wildcard suggestion format example is required.")

        example_word_count = self._count_whitespace_words(normalized_example)
        if example_word_count <= 0:
            raise ValueError("Wildcard suggestion format example is required.")
        word_delta = int(math.floor(example_word_count * 0.15))
        min_words = max(1, example_word_count - word_delta)
        max_words = max(min_words, example_word_count + word_delta)
        desired_count = max(1, int(target_count))

        tokenizer = self._tokenizer
        text_encoder = self._text_encoder
        if tokenizer is None or text_encoder is None:
            raise ValueError("Wildcard suggestions are unavailable for the current runtime.")

        seen_existing = {
            normalize_wildcard_entry_value(item).lower()
            for item in (existing_entries or [])
            if normalize_wildcard_entry_value(item)
        }
        accepted: list[str] = []
        accepted_keys: set[str] = set(seen_existing)
        partial_message: str | None = None
        effective_seed = int(seed) if seed is not None else None

        for attempt_index in range(4):
            needed = desired_count - len(accepted)
            if needed <= 0:
                break
            request_count = max(10, needed + 4)
            prompt_text = self._build_wildcard_suggestion_prompt(
                tokenizer,
                theme=normalized_theme,
                format_example=normalized_example,
                target_count=request_count,
                min_words=min_words,
                max_words=max_words,
            )
            try:
                encoded = tokenizer(prompt_text, return_tensors="pt")
            except Exception as exc:
                raise ValueError(f"Wildcard suggestion tokenizer failed: {exc}") from exc

            encoded = self._move_encoded_to_module_device(encoded, text_encoder)
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": 256,
                "do_sample": True,
                "temperature": 0.85,
                "top_p": 0.92,
                "repetition_penalty": 1.08,
            }
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            if pad_token_id is not None:
                generate_kwargs["pad_token_id"] = pad_token_id
            if eos_token_id is not None:
                generate_kwargs["eos_token_id"] = eos_token_id

            attempt_seed = None if effective_seed is None else effective_seed + (attempt_index * 9973)
            try:
                generated_text = self._run_text_generation_attempt(
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    encoded=encoded,
                    torch_module=self._torch,
                    generate_kwargs=generate_kwargs,
                    generation_seed=attempt_seed,
                )
            except Exception as exc:
                raise ValueError(f"Wildcard suggestion decode failed: {exc}") from exc

            for candidate in self._parse_wildcard_suggestion_candidates(generated_text):
                candidate_key = candidate.lower()
                if candidate_key in accepted_keys:
                    continue
                word_count = self._count_whitespace_words(candidate)
                if word_count < min_words or word_count > max_words:
                    continue
                accepted_keys.add(candidate_key)
                accepted.append(candidate)
                if len(accepted) >= desired_count:
                    break

        if len(accepted) < desired_count:
            partial_message = "The example format was restrictive, so only a partial suggestion set could be generated."

        return {
            "suggestions": accepted[:desired_count],
            "accepted_count": min(len(accepted), desired_count),
            "target_count": desired_count,
            "seed": effective_seed,
            "example_word_count": example_word_count,
            "min_words": min_words,
            "max_words": max_words,
            "partial": len(accepted) < desired_count,
            "message": partial_message,
        }

    def _resolve_module_device(module: Any) -> Any:
        if hasattr(module, "device"):
            device = module.device
            if str(getattr(device, "type", "")) != "meta":
                return device
        try:
            device = next(module.parameters()).device
            if str(getattr(device, "type", "")) != "meta":
                return device
        except Exception:
            pass
        try:
            embed_layer = module.get_input_embeddings()
            if embed_layer is not None and hasattr(embed_layer, "weight"):
                device = getattr(embed_layer.weight, "device", None)
                if str(getattr(device, "type", "")) != "meta":
                    return device
        except Exception:
            pass
        return None

    @staticmethod
    def _move_encoded_to_module_device(encoded: dict[str, Any], module: Any) -> dict[str, Any]:
        device = DiffusersQwenInference._resolve_module_device(module)
        if device is None:
            return dict(encoded)
        moved: dict[str, Any] = {}
        for key, value in encoded.items():
            if hasattr(value, "to"):
                try:
                    moved[key] = value.to(device)
                    continue
                except Exception:
                    pass
            moved[key] = value
        return moved

    @staticmethod
    def _build_generate_fallback_kwargs(text_encoder: Any, generate_kwargs: dict[str, Any]) -> dict[str, Any]:
        fallback_kwargs = dict(generate_kwargs)
        generation_config = getattr(text_encoder, "generation_config", None)
        if generation_config is None:
            return fallback_kwargs

        do_sample = bool(fallback_kwargs.pop("do_sample", getattr(generation_config, "do_sample", False)))
        temperature = float(fallback_kwargs.pop("temperature", getattr(generation_config, "temperature", 1.0)))
        top_p = float(fallback_kwargs.pop("top_p", getattr(generation_config, "top_p", 1.0)))
        raw_top_k = fallback_kwargs.pop("top_k", getattr(generation_config, "top_k", 50))
        repetition_penalty = float(
            fallback_kwargs.pop("repetition_penalty", getattr(generation_config, "repetition_penalty", 1.0))
        )
        fallback_kwargs["use_model_defaults"] = False
        top_k = 50
        try:
            if raw_top_k is not None:
                top_k = int(raw_top_k)
        except (TypeError, ValueError):
            top_k = 50

        try:
            from transformers import GenerationConfig
        except ImportError:
            try:
                fallback_config = copy.deepcopy(generation_config)
            except Exception:
                return fallback_kwargs
            setattr(fallback_config, "do_sample", do_sample)
            setattr(fallback_config, "temperature", max(temperature, 1e-5) if do_sample else 1.0)
            setattr(fallback_config, "top_p", top_p if do_sample else 1.0)
            setattr(fallback_config, "top_k", max(0, top_k) if do_sample else 50)
            setattr(fallback_config, "repetition_penalty", repetition_penalty)
            fallback_kwargs["generation_config"] = fallback_config
            return fallback_kwargs

        config_kwargs: dict[str, Any] = {
            "do_sample": do_sample,
            "temperature": max(temperature, 1e-5) if do_sample else 1.0,
            "top_p": top_p if do_sample else 1.0,
            "top_k": max(0, top_k) if do_sample else 50,
            "repetition_penalty": repetition_penalty,
        }
        for source in (generation_config, getattr(text_encoder, "config", None)):
            if source is None:
                continue
            for key in ("pad_token_id", "bos_token_id", "eos_token_id", "decoder_start_token_id"):
                value = getattr(source, key, None)
                if value is not None and key not in config_kwargs:
                    config_kwargs[key] = value

        fallback_kwargs["generation_config"] = GenerationConfig(**config_kwargs)
        return fallback_kwargs

    @staticmethod
    def _build_rewrite_prompt(tokenizer: Any, prompt: str) -> str:
        system = (
            "Rewrite the input as exactly one stronger image-generation prompt. Preserve the user's intent and "
            "preserve any explicit medium or style exactly. If the user says anime, keep it anime. If the user "
            "says oil painting, keep it painterly. If the user says photograph, cinematic, editorial, or 3D render, "
            "keep that visual mode and do not drift into a conflicting style. Expand only with concrete visible "
            "details that improve subject clarity, environment, lighting, composition, materials, mood, and camera "
            "when relevant. Prefer clear structured natural language over tag soup. Avoid filler adjectives unless "
            "they describe something visible. Output the rewritten prompt only, with no analysis or explanation."
        )
        user_message = (
            "Rewrite this image prompt for better visual fidelity and specificity.\n\n"
            f"Original prompt: {prompt}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                if isinstance(rendered, str) and rendered.strip():
                    return rendered
            except TypeError:
                try:
                    rendered = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    if isinstance(rendered, str) and rendered.strip():
                        return rendered
                except Exception:
                    pass
            except Exception:
                pass
        return f"{system}\n\n{user_message}\n\nRewritten prompt:"

    @staticmethod
    def _build_compression_prompt(
        tokenizer: Any,
        prompt: str,
        *,
        target_tokens: int | None = None,
    ) -> str:
        target = int(target_tokens or DiffusersQwenInference._PROMPT_ENHANCEMENT_COMPRESSION_TARGET_TOKENS)
        system = (
            "Compress the input into one compact image generation prompt under "
            f"{target} tokens. Keep subject, count, identity, style, camera, lighting, setting, spatial "
            "relations, named labels, and rare concrete details. Remove filler, repeated atmosphere, duplicate "
            "material detail, weak adjectives, and low priority background texture. Use complete sentences only. "
            "Output the compressed prompt only, with no analysis or explanation."
        )
        user_message = (
            "Compress this image prompt while preserving its visual intent.\n\n"
            f"Original prompt: {prompt}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                if isinstance(rendered, str) and rendered.strip():
                    return rendered
            except TypeError:
                try:
                    rendered = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    if isinstance(rendered, str) and rendered.strip():
                        return rendered
                except Exception:
                    pass
            except Exception:
                pass
        return f"{system}\n\n{user_message}\n\nCompressed prompt:"

    @staticmethod
    def _extract_rewritten_prompt(full_text: str, input_text: str) -> str:
        candidate = full_text[len(input_text) :].strip() if full_text.startswith(input_text) else full_text.strip()
        candidate = re.sub(r"<think>.*?</think>\s*", "", candidate, flags=re.DOTALL).strip()
        if "Rewritten prompt:" in candidate:
            candidate = candidate.split("Rewritten prompt:", 1)[-1].strip()
        if "Compressed prompt:" in candidate:
            candidate = candidate.split("Compressed prompt:", 1)[-1].strip()
        candidate = candidate.splitlines()[0].strip() if candidate else ""
        return candidate

    @staticmethod
    def _rewrite_quality_ok(original: str, rewritten: str) -> bool:
        return DiffusersQwenInference._rewrite_rejection_reason(original, rewritten) == "ok"

    @staticmethod
    def _rewrite_rejection_reason(original: str, rewritten: str) -> str:
        original_text = original.strip()
        text = rewritten.strip()
        if not text:
            return "empty"
        if len(text) < 8:
            return "too_short"
        if len(text) > 4000:
            return "too_long"
        if re.search(r"(.)\1{10,}", text):
            return "repeated_characters"

        letters = sum(1 for ch in text if ch.isalpha())
        if letters < max(3, int(len(text) * 0.15)):
            return "too_few_letters"

        punctuation = sum(1 for ch in text if ch in string.punctuation)
        if punctuation / max(1, len(text)) > 0.45:
            return "too_much_punctuation"

        words = re.findall(r"[A-Za-z0-9_'-]+", text.lower())
        if words:
            unique_ratio = len(set(words)) / len(words)
            if len(words) >= 6 and unique_ratio < 0.34:
                return "low_lexical_diversity"

        if text == original_text:
            return "unchanged"
        return "ok"

    @staticmethod
    def _render_pipeline_prompt(tokenizer: Any, prompt: str) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                if isinstance(rendered, str) and rendered.strip():
                    return rendered
            except TypeError:
                try:
                    rendered = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    if isinstance(rendered, str) and rendered.strip():
                        return rendered
                except Exception:
                    pass
            except Exception:
                pass
        return prompt

    @classmethod
    def _pipeline_prompt_token_length(cls, tokenizer: Any, prompt: str) -> int:
        rendered = cls._render_pipeline_prompt(tokenizer, prompt)
        encoded = tokenizer(rendered, return_tensors="pt", truncation=False)
        input_ids = getattr(encoded, "input_ids", None)
        if input_ids is None and isinstance(encoded, dict):
            input_ids = encoded.get("input_ids")
        if input_ids is None:
            raise ValueError("Tokenizer did not return input_ids while measuring prompt length.")
        return int(input_ids.shape[-1])

    @classmethod
    def _resolve_pipeline_max_sequence_length(cls, tokenizer: Any, prompt: str) -> int:
        baseline = int(cls._PROMPT_ENHANCEMENT_PIPELINE_MAX_SEQUENCE_LENGTH)
        if tokenizer is None:
            return baseline
        try:
            return max(baseline, cls._pipeline_prompt_token_length(tokenizer, prompt))
        except Exception:
            return baseline

    @staticmethod
    def _split_prompt_clauses(text: str) -> list[str]:
        if not text:
            return []
        clauses = re.split(r"(?<=[,;:.])\s+|\n+", text)
        cleaned: list[str] = []
        seen: set[str] = set()
        for clause in clauses:
            normalized = clause.strip(" ,;:.")
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(normalized)
        return cleaned

    @staticmethod
    def _trim_to_complete_sentences(text: str) -> str | None:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        normalized = normalized.strip("`\"' ")
        normalized = re.sub(
            r"^(?:rewritten|compressed)\s+prompt:\s*",
            "",
            normalized,
            flags=re.IGNORECASE,
        ).strip()
        if not normalized:
            return None

        boundary_matches = list(re.finditer(r"[.!?][\"')\]]*", normalized))
        if boundary_matches:
            last_boundary = boundary_matches[-1]
            trimmed = normalized[: last_boundary.end()].strip()
            return trimmed or None

        if normalized != normalized.rstrip(" ,;:"):
            return None

        words = re.findall(r"[A-Za-z0-9_'-]+", normalized)
        if "," in normalized and 3 <= len(words) <= 80:
            return normalized
        return None

    @classmethod
    def _extract_style_constraints(cls, text: str) -> tuple[str, ...]:
        matches: list[str] = []
        seen: set[str] = set()
        for pattern in cls._PROMPT_STYLE_PATTERNS:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                value = match.group(0).strip()
                key = value.lower()
                if key in seen:
                    continue
                seen.add(key)
                matches.append(value)
        return tuple(matches)

    @classmethod
    def _clause_contains_style_constraint(
        cls,
        clause: str,
        style_constraints: tuple[str, ...],
    ) -> bool:
        lower_clause = clause.lower()
        return any(style.lower() in lower_clause for style in style_constraints)

    @classmethod
    def _build_style_preserving_prompt_seed(cls, original_prompt: str) -> str | None:
        clauses = cls._split_prompt_clauses(original_prompt)
        if not clauses:
            return None

        style_constraints = cls._extract_style_constraints(original_prompt)
        style_clauses = [
            clause
            for clause in clauses
            if cls._clause_contains_style_constraint(clause, style_constraints)
        ]
        seeded: list[str] = []
        seen: set[str] = set()

        def push(part: str) -> None:
            normalized = part.strip(" ,;:.")
            if not normalized:
                return
            key = normalized.lower()
            if key in seen:
                return
            seen.add(key)
            seeded.append(normalized)

        push(clauses[0])
        for clause in style_clauses:
            push(clause)
        for style in style_constraints:
            if any(style.lower() in clause.lower() for clause in style_clauses):
                continue
            push(style)

        if not seeded:
            return None
        return ", ".join(seeded)

    @classmethod
    def _fit_complete_clauses_to_budget(
        cls,
        *,
        tokenizer: Any,
        candidate_prompt: str,
        original_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> str | None:
        budget = int(max_tokens or cls._PROMPT_ENHANCEMENT_PIPELINE_SAFE_TOKEN_BUDGET)
        normalized = re.sub(r"\s+", " ", candidate_prompt).strip()
        if not normalized:
            return None

        complete_candidate = cls._trim_to_complete_sentences(normalized)
        if complete_candidate is not None:
            try:
                if cls._pipeline_prompt_token_length(tokenizer, complete_candidate) <= budget:
                    return complete_candidate
            except Exception:
                return complete_candidate

        source = complete_candidate or normalized
        clauses = cls._split_prompt_clauses(source)
        if not clauses:
            return None

        sentence_style = bool(re.search(r"[.!?]", source))
        style_constraints = cls._extract_style_constraints(original_prompt or source)
        style_clauses = [
            clause
            for clause in clauses
            if cls._clause_contains_style_constraint(clause, style_constraints)
        ]
        prioritized: list[str] = []
        seen: set[str] = set()

        def push(part: str) -> None:
            normalized_part = part.strip(" ,;:.")
            if not normalized_part:
                return
            key = normalized_part.lower()
            if key in seen:
                return
            seen.add(key)
            prioritized.append(normalized_part)

        push(clauses[0])
        for clause in style_clauses:
            push(clause)
        for clause in clauses[1:]:
            clause_lower = clause.lower()
            if any(keyword in clause_lower for keyword in cls._PROMPT_PRIORITY_KEYWORDS):
                push(clause)
        for clause in clauses[1:]:
            push(clause)

        def join_parts(parts: list[str]) -> str:
            if sentence_style:
                return ". ".join(part.rstrip(".!?") for part in parts).strip() + "."
            return ", ".join(parts).strip()

        assembled: list[str] = []
        best: str | None = None
        for part in prioritized:
            trial = join_parts(assembled + [part])
            try:
                token_length = cls._pipeline_prompt_token_length(tokenizer, trial)
            except Exception:
                return trial
            if token_length > budget:
                continue
            assembled.append(part)
            best = trial
        return best

    @classmethod
    def _compress_prompt_to_token_budget(
        cls,
        *,
        tokenizer: Any,
        original_prompt: str,
        candidate_prompt: str,
        max_tokens: int | None = None,
    ) -> str | None:
        budget = int(max_tokens or cls._PROMPT_ENHANCEMENT_PIPELINE_SAFE_TOKEN_BUDGET)
        candidate = re.sub(r"\s+", " ", candidate_prompt).strip()
        if not candidate:
            return None
        candidate = cls._trim_to_complete_sentences(candidate) or candidate
        try:
            if cls._pipeline_prompt_token_length(tokenizer, candidate) <= budget:
                return candidate
        except Exception:
            return candidate

        style_constraints = cls._extract_style_constraints(original_prompt)
        prefix_parts: list[str] = []
        lower_candidate = candidate.lower()
        for style in style_constraints:
            if style.lower() not in lower_candidate:
                prefix_parts.append(style)
        prefix = ", ".join(prefix_parts).strip()

        clauses = cls._split_prompt_clauses(candidate)
        if not clauses:
            return cls._fit_complete_clauses_to_budget(
                tokenizer=tokenizer,
                candidate_prompt=candidate,
                original_prompt=original_prompt,
                max_tokens=budget,
            )
        style_clauses = [
            clause
            for clause in clauses
            if cls._clause_contains_style_constraint(clause, style_constraints)
        ]

        prioritized: list[str] = []
        seen: set[str] = set()

        def push(part: str) -> None:
            normalized = part.strip(" ,;:.")
            if not normalized:
                return
            key = normalized.lower()
            if key in seen:
                return
            seen.add(key)
            prioritized.append(normalized)

        if prefix:
            push(prefix)
        push(clauses[0])
        for clause in style_clauses:
            push(clause)
        for clause in clauses[1:]:
            clause_lower = clause.lower()
            if any(keyword in clause_lower for keyword in cls._PROMPT_PRIORITY_KEYWORDS):
                push(clause)
        for clause in clauses[1:]:
            push(clause)

        assembled: list[str] = []
        best: str | None = None
        sentence_style = bool(re.search(r"[.!?]", candidate))

        def join_parts(parts: list[str]) -> str:
            if sentence_style:
                return ". ".join(part.rstrip(".!?") for part in parts).strip() + "."
            return ", ".join(parts).strip()

        for part in prioritized:
            trial = join_parts(assembled + [part])
            try:
                token_length = cls._pipeline_prompt_token_length(tokenizer, trial)
            except Exception:
                token_length = 0
            if token_length <= budget:
                assembled.append(part)
                best = trial

        if best is None:
            return cls._fit_complete_clauses_to_budget(
                tokenizer=tokenizer,
                candidate_prompt=candidate,
                original_prompt=original_prompt,
                max_tokens=budget,
            )

        for style in style_constraints:
            if style.lower() not in best.lower():
                return None
        return best

    @classmethod
    def _fit_prompt_to_budget(
        cls,
        *,
        tokenizer: Any,
        original_prompt: str,
        enhanced_prompt: str,
    ) -> tuple[str, bool]:
        fitted = cls._compress_prompt_to_token_budget(
            tokenizer=tokenizer,
            original_prompt=original_prompt,
            candidate_prompt=enhanced_prompt,
        )
        if fitted is not None and cls._rewrite_quality_ok(original_prompt, fitted):
            return fitted, True

        fallback_original = cls._compress_prompt_to_token_budget(
            tokenizer=tokenizer,
            original_prompt=original_prompt,
            candidate_prompt=original_prompt,
        )
        if fallback_original is not None:
            return fallback_original, False

        seeded_original = cls._build_style_preserving_prompt_seed(original_prompt)
        if seeded_original is not None:
            fallback_seeded = cls._compress_prompt_to_token_budget(
                tokenizer=tokenizer,
                original_prompt=original_prompt,
                candidate_prompt=seeded_original,
            )
            if fallback_seeded is not None:
                return fallback_seeded, False
            fallback_seeded_clauses = cls._fit_complete_clauses_to_budget(
                tokenizer=tokenizer,
                candidate_prompt=seeded_original,
                original_prompt=original_prompt,
            )
            if fallback_seeded_clauses is not None:
                return fallback_seeded_clauses, False

        fitted_original_clauses = cls._fit_complete_clauses_to_budget(
            tokenizer=tokenizer,
            candidate_prompt=original_prompt,
            original_prompt=original_prompt,
        )
        if fitted_original_clauses is not None:
            return fitted_original_clauses, False

        normalized_original = re.sub(r"\s+", " ", original_prompt).strip()
        return normalized_original, False

    @staticmethod
    @contextmanager
    def _seeded_rng_context(torch_module: Any, seed: int | None):
        if seed is None:
            yield
            return

        cuda_devices: list[int] = []
        try:
            if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
                cuda_devices = [int(torch_module.cuda.current_device())]
        except Exception:
            cuda_devices = []

        with torch_module.random.fork_rng(devices=cuda_devices, enabled=True):
            torch_module.manual_seed(int(seed))
            if cuda_devices:
                torch_module.cuda.manual_seed_all(int(seed))
            yield

    @staticmethod
    def _count_whitespace_words(text: str) -> int:
        return len(re.findall(r"\S+", str(text or "").strip()))

    @staticmethod
    def _extract_generated_completion_text(full_text: str, input_text: str) -> str:
        candidate = full_text[len(input_text) :].strip() if full_text.startswith(input_text) else full_text.strip()
        candidate = re.sub(r"<think>.*?</think>\s*", "", candidate, flags=re.DOTALL).strip()
        if "Entries:" in candidate:
            candidate = candidate.split("Entries:", 1)[-1].strip()
        return candidate

    @staticmethod
    def _latest_user_chat_text(messages: list[dict[str, str]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content") or "").strip()
        return ""

    @staticmethod
    def _chat_user_wants_prompt_action(user_text: str) -> bool:
        text = str(user_text or "").lower()
        prompt_markers = (
            "prompt",
            "generate",
            "make an image",
            "make image",
            "create an image",
            "image of",
            "picture of",
            "write me",
            "draft",
            "rewrite",
            "use this",
            "put this",
            "paste",
            "append",
        )
        return any(marker in text for marker in prompt_markers)

    @staticmethod
    def _looks_like_image_prompt(text: str) -> bool:
        prompt = str(text or "").strip()
        lowered = prompt.lower()
        if not prompt or len(prompt) > 1200:
            return False
        if re.search(r"\b(is|are|does|do|means|refers to|works on|runs after|separate tool)\b", lowered) and "." in prompt:
            return False
        help_terms = (
            "clarity",
            "prompt enhancer",
            "api",
            "drawer",
            "button",
            "tool",
            "pipeline",
            "gallery",
            "setting",
            "control",
        )
        visual_terms = (
            "portrait",
            "scene",
            "shot",
            "lighting",
            "cinematic",
            "forest",
            "city",
            "camera",
            "style",
            "render",
            "photo",
            "illustration",
            "glowing",
            "moonlight",
            "ultra detailed",
            "medium shot",
            "wide shot",
            "close up",
        )
        if "," in prompt and any(term in lowered for term in visual_terms):
            return True
        if any(term in lowered for term in visual_terms) and not any(term in lowered for term in help_terms):
            return True
        return False

    @classmethod
    def _filter_chat_actions_for_turn(
        cls,
        actions: list[dict[str, Any]],
        *,
        messages: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        user_text = cls._latest_user_chat_text(messages)
        allow_prompt_actions = cls._chat_user_wants_prompt_action(user_text)
        filtered: list[dict[str, Any]] = []
        for action in actions:
            action_type = action.get("type")
            if action_type in {"set_prompt", "append_prompt", "start_generation"}:
                if allow_prompt_actions and cls._looks_like_image_prompt(str(action.get("prompt") or "")):
                    filtered.append(action)
                continue
            filtered.append(action)
        return filtered

    @classmethod
    def _build_chat_prompt(cls, tokenizer: Any, messages: list[dict[str, str]], app_context: str | None = None) -> str:
        system_parts = [cls._CHAT_SYSTEM_PROMPT]
        clean_context = str(app_context or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if clean_context:
            system_parts.append(f"Just Rayzist reference:\n{clean_context[:6000].rstrip()}")
        rendered_messages = [{"role": "system", "content": "\n\n".join(system_parts)}]
        for message in messages:
            role = str(message.get("role") or "").strip().lower()
            content = str(message.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            rendered_messages.append({"role": role, "content": content})
        if not any(message["role"] == "user" for message in rendered_messages):
            raise ValueError("Chat message is required.")
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                rendered = tokenizer.apply_chat_template(
                    rendered_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                if isinstance(rendered, str) and rendered.strip():
                    return rendered
            except TypeError:
                try:
                    rendered = tokenizer.apply_chat_template(
                        rendered_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    if isinstance(rendered, str) and rendered.strip():
                        return rendered
                except Exception:
                    pass
            except Exception:
                pass
        lines: list[str] = [f"System: {rendered_messages[0]['content']}"]
        for message in rendered_messages[1:]:
            role = "User" if message["role"] == "user" else "Assistant"
            lines.append(f"{role}: {message['content']}")
        lines.append("Assistant:")
        return "\n\n".join(lines)

    @staticmethod
    def _extract_chat_action_blocks(text: str) -> tuple[str, list[dict[str, Any]]]:
        actions: list[dict[str, Any]] = []
        candidate = str(text or "")

        marker_pattern = re.compile(r"<rayzist-actions\b[^>]*>\s*", flags=re.IGNORECASE)
        decoder = json.JSONDecoder()
        while True:
            marker = marker_pattern.search(candidate)
            if marker is None:
                break
            prefix = candidate[: marker.start()]
            suffix = candidate[marker.end():]
            leading = len(suffix) - len(suffix.lstrip())
            payload_text = suffix.lstrip()
            try:
                payload, end_index = decoder.raw_decode(payload_text)
                actions.extend(normalize_chat_actions(payload))
                after_payload = payload_text[end_index:]
                close_match = re.match(r"\s*</rayzist-actions>", after_payload, flags=re.IGNORECASE)
                consumed = leading + end_index + (close_match.end() if close_match else 0)
                candidate = f"{prefix}{suffix[consumed:]}"
            except Exception:
                close_match = re.search(r"</rayzist-actions>", suffix, flags=re.IGNORECASE)
                if close_match is None:
                    candidate = prefix.rstrip()
                    break
                candidate = f"{prefix}{suffix[close_match.end():]}"

        def strip_xml_block(match: re.Match[str]) -> str:
            try:
                payload = json.loads(match.group(1))
            except Exception:
                return ""
            actions.extend(normalize_chat_actions(payload))
            return ""

        candidate = re.sub(
            r"<rayzist-actions>\s*(\{.*?\})\s*</rayzist-actions>",
            strip_xml_block,
            candidate,
            flags=re.IGNORECASE | re.DOTALL,
        )

        def strip_fenced_block(match: re.Match[str]) -> str:
            try:
                payload = json.loads(match.group(1))
            except Exception:
                return match.group(0)
            parsed = normalize_chat_actions(payload)
            if not parsed:
                return match.group(0)
            actions.extend(parsed)
            return ""

        candidate = re.sub(
            r"```(?:json|rayzist-actions)?\s*(\{.*?\"actions\"\s*:\s*\[.*?\}\s*)```",
            strip_fenced_block,
            candidate,
            flags=re.IGNORECASE | re.DOTALL,
        )
        candidate = strip_chat_action_markup(candidate)
        return candidate, normalize_chat_actions(actions)

    @staticmethod
    def _extract_chat_response_parts(text: str) -> tuple[str, list[dict[str, Any]]]:
        candidate = re.sub(r"<think>.*?</think>\s*", "", str(text or ""), flags=re.DOTALL).strip()
        for marker in ("Assistant:", "assistant:"):
            if marker in candidate:
                candidate = candidate.rsplit(marker, 1)[-1].strip()
        for marker in ("\nUser:", "\nuser:", "\nSystem:", "\nsystem:"):
            if marker in candidate:
                candidate = candidate.split(marker, 1)[0].strip()
        candidate, actions = DiffusersQwenInference._extract_chat_action_blocks(candidate)
        candidate = candidate.strip(" \t\r\n\"'")
        return candidate, actions

    @staticmethod
    def _extract_chat_response_text(text: str) -> str:
        response, _actions = DiffusersQwenInference._extract_chat_response_parts(text)
        return response or "I could not generate a response."

    @staticmethod
    def _fallback_chat_response(messages: list[dict[str, str]]) -> str:
        last_user = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                last_user = str(message.get("content") or "").strip()
                break
        normalized = re.sub(r"[^\w\s]", "", last_user.lower()).strip()
        if normalized in {"hi", "hello", "hey", "yo"}:
            return "Ask for prompt help, workflow help, or a prompt draft. I can also offer buttons to use a prompt or open /API."
        return (
            "No usable text came back from the encoder. Try asking for a concrete prompt draft, a rewrite, "
            "or a specific Just Rayzist control."
        )

    @staticmethod
    def _chat_response_from_actions(actions: list[dict[str, Any]]) -> str:
        prompt_actions = [
            action
            for action in actions
            if action.get("type") in {"set_prompt", "append_prompt", "start_generation"}
            and str(action.get("prompt") or "").strip()
        ]
        if prompt_actions:
            prompt = str(prompt_actions[0].get("prompt") or "").strip()
            return f"Prompt ready:\n{prompt}"
        if any(action.get("type") == "open_route" and action.get("href") == "/API" for action in actions):
            return "Open the local API reference at /API."
        return ""

    @staticmethod
    def _pipe_supports_wildcard_suggestions(pipe: Any) -> bool:
        return getattr(pipe, "tokenizer", None) is not None and getattr(pipe, "text_encoder", None) is not None

    @staticmethod
    def _build_wildcard_suggestion_prompt(
        tokenizer: Any,
        *,
        theme: str,
        format_example: str,
        target_count: int,
        min_words: int,
        max_words: int,
    ) -> str:
        system = (
            "Generate wildcard entries for image prompts. Output newline-separated entries only. Do not number the "
            "lines. Do not use bullets, quotes, headings, or commentary. Match the requested theme and keep each "
            "entry close to the example's structure and length."
        )
        user_message = (
            f"Theme/topic: {theme}\n"
            f"Format example: {format_example}\n"
            f"Target count: {target_count}\n"
            f"Allowed word count per entry: {min_words} to {max_words}\n"
            "Return only the entries, one per line."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                if isinstance(rendered, str) and rendered.strip():
                    return rendered
            except TypeError:
                try:
                    rendered = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    if isinstance(rendered, str) and rendered.strip():
                        return rendered
                except Exception:
                    pass
            except Exception:
                pass
        return f"{system}\n\n{user_message}\n\nEntries:"

    def _run_text_generation_attempt(
        self,
        *,
        tokenizer: Any,
        text_encoder: Any,
        encoded: dict[str, Any],
        torch_module: Any,
        generate_kwargs: dict[str, Any],
        generation_seed: int | None = None,
    ) -> str:
        output_ids = None
        decode_exc: Exception | None = None
        try:
            with self._seeded_rng_context(torch_module, generation_seed):
                output_ids = self._generate_with_base_model(
                    text_encoder=text_encoder,
                    encoded=encoded,
                    max_new_tokens=int(generate_kwargs.get("max_new_tokens", 192)),
                    eos_token_id=generate_kwargs.get("eos_token_id"),
                    torch_module=torch_module,
                    do_sample=bool(generate_kwargs.get("do_sample", True)),
                    temperature=float(generate_kwargs.get("temperature", 0.85)),
                    top_p=float(generate_kwargs.get("top_p", 0.92)),
                    repetition_penalty=float(generate_kwargs.get("repetition_penalty", 1.08)),
                )
        except Exception as exc:
            decode_exc = exc

        if output_ids is not None:
            full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            input_text = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)
            completion_text = self._extract_generated_completion_text(full_text, input_text)
            if completion_text.strip():
                return completion_text
            try:
                prompt_token_count = int(encoded["input_ids"].shape[-1])
                generated_token_text = tokenizer.decode(output_ids[0][prompt_token_count:], skip_special_tokens=True).strip()
            except Exception:
                generated_token_text = ""
            if generated_token_text:
                return generated_token_text
            decode_exc = ValueError("base-model decode returned no completion")

        if decode_exc is not None:
            if not hasattr(text_encoder, "generate"):
                raise decode_exc
            LOGGER.debug(
                "Helper text generation base-model decode unavailable; falling back to text_encoder.generate(). %s",
                decode_exc,
            )
            try:
                fallback_encoded = self._move_encoded_to_module_device(encoded, text_encoder)
                fallback_kwargs = self._build_generate_fallback_kwargs(text_encoder, generate_kwargs)
                with self._seeded_rng_context(torch_module, generation_seed):
                    with self._module_forward_context(torch_module, text_encoder):
                        output_ids = text_encoder.generate(**fallback_encoded, **fallback_kwargs)
            except Exception as exc:
                LOGGER.warning(
                    "Helper text generation base-model decode failed and text_encoder.generate() fallback also failed. "
                    "base_model=%s generate=%s",
                    decode_exc,
                    exc,
                )
                raise
            full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            input_text = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)
            completion_text = self._extract_generated_completion_text(full_text, input_text)
            if completion_text.strip():
                return completion_text
            try:
                prompt_token_count = int(encoded["input_ids"].shape[-1])
                return tokenizer.decode(output_ids[0][prompt_token_count:], skip_special_tokens=True).strip()
            except Exception:
                return ""

        return ""

    @staticmethod
    def _parse_wildcard_suggestion_candidates(text: str) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()
        for raw_line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            line = re.sub(r"^\s*(?:[-*•]+|\d+[\.\)])\s*", "", raw_line).strip(" \t\"'")
            normalized = normalize_wildcard_entry_value(line)
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(normalized)
        return candidates

    def _enhance_prompt(
        self,
        pipe: Any,
        prompt: str,
        torch_module: Any,
        *,
        seed: int | None = None,
    ) -> str:
        tokenizer = getattr(pipe, "tokenizer", None)
        text_encoder = getattr(pipe, "text_encoder", None)
        if tokenizer is None or text_encoder is None:
            LOGGER.debug("Prompt enhancement skipped: text_encoder/tokenizer unavailable.")
            return prompt

        rewrite_input = self._build_rewrite_prompt(tokenizer, prompt)
        try:
            encoded = tokenizer(rewrite_input, return_tensors="pt")
        except Exception as exc:
            LOGGER.warning("Prompt enhancement tokenizer failed; using original prompt. %s", exc)
            return prompt

        encoded = self._move_encoded_to_module_device(encoded, text_encoder)

        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self._PROMPT_ENHANCEMENT_PRIMARY_MAX_NEW_TOKENS,
            "do_sample": False,
        }
        if pad_token_id is not None:
            generate_kwargs["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            generate_kwargs["eos_token_id"] = eos_token_id

        rewritten, rejection_reason = self._run_rewrite_attempt(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            encoded=encoded,
            prompt=prompt,
            torch_module=torch_module,
            generate_kwargs=generate_kwargs,
            enhancement_seed=seed,
        )
        if rejection_reason == "ok":
            return rewritten

        retryable_reasons = {
            "repeated_characters",
            "too_much_punctuation",
            "low_lexical_diversity",
            "too_few_letters",
        }
        if rejection_reason in retryable_reasons:
            LOGGER.debug(
                "Prompt enhancement retrying with sampled decode after %s.",
                rejection_reason,
            )
            retry_kwargs = dict(generate_kwargs)
            retry_kwargs["do_sample"] = True
            retry_kwargs["temperature"] = 0.72
            retry_kwargs["top_p"] = 0.92
            retry_kwargs["max_new_tokens"] = self._PROMPT_ENHANCEMENT_RETRY_MAX_NEW_TOKENS
            rewritten_retry, retry_reason = self._run_rewrite_attempt(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                encoded=encoded,
                prompt=prompt,
                torch_module=torch_module,
                generate_kwargs=retry_kwargs,
                enhancement_seed=seed,
            )
            if retry_reason == "ok":
                return rewritten_retry
            rejection_reason = retry_reason

        if rejection_reason in {"empty", "too_short", "unchanged"}:
            LOGGER.debug(
                "Prompt enhancement skipped (%s); using original prompt.",
                rejection_reason,
            )
        else:
            LOGGER.warning(
                "Prompt enhancement output rejected (%s); using original prompt.",
                rejection_reason,
            )
        return prompt

    def _compress_long_prompt(
        self,
        pipe: Any,
        prompt: str,
        torch_module: Any,
        *,
        seed: int | None = None,
    ) -> str:
        tokenizer = getattr(pipe, "tokenizer", None)
        text_encoder = getattr(pipe, "text_encoder", None)
        if tokenizer is None or text_encoder is None:
            LOGGER.debug("Prompt compression skipped: text_encoder/tokenizer unavailable.")
            return prompt

        safe_budget = int(self._PROMPT_ENHANCEMENT_PIPELINE_SAFE_TOKEN_BUDGET)
        primary_target = min(
            int(self._PROMPT_ENHANCEMENT_COMPRESSION_TARGET_TOKENS),
            safe_budget,
        )
        retry_target = max(1, min(int(primary_target * 0.82), safe_budget))
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        best_candidate: str | None = None

        for attempt_index, target_tokens in enumerate((primary_target, retry_target)):
            compression_input = self._build_compression_prompt(
                tokenizer,
                prompt,
                target_tokens=target_tokens,
            )
            try:
                encoded = tokenizer(compression_input, return_tensors="pt")
            except Exception as exc:
                LOGGER.warning("Prompt compression tokenizer failed; using fallback. %s", exc)
                break

            encoded = self._move_encoded_to_module_device(encoded, text_encoder)
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": target_tokens,
                "do_sample": attempt_index > 0,
            }
            if attempt_index > 0:
                generate_kwargs["temperature"] = 0.72
                generate_kwargs["top_p"] = 0.92
            if pad_token_id is not None:
                generate_kwargs["pad_token_id"] = pad_token_id
            if eos_token_id is not None:
                generate_kwargs["eos_token_id"] = eos_token_id

            rewritten, rejection_reason = self._run_rewrite_attempt(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                encoded=encoded,
                prompt=prompt,
                torch_module=torch_module,
                generate_kwargs=generate_kwargs,
                enhancement_seed=seed,
            )
            if rejection_reason not in {"ok", "too_long"}:
                continue

            trimmed = self._trim_to_complete_sentences(rewritten)
            if trimmed is None:
                continue

            fitted = self._compress_prompt_to_token_budget(
                tokenizer=tokenizer,
                original_prompt=prompt,
                candidate_prompt=trimmed,
                max_tokens=safe_budget,
            )
            if fitted is None or not self._rewrite_quality_ok(prompt, fitted):
                continue

            if best_candidate is None:
                best_candidate = fitted

            normalized_rewritten = re.sub(r"\s+", " ", rewritten).strip().strip("`\"' ")
            if attempt_index == 0 and fitted != normalized_rewritten:
                continue
            return fitted

        if best_candidate is not None:
            return best_candidate

        fallback = self._fit_complete_clauses_to_budget(
            tokenizer=tokenizer,
            candidate_prompt=prompt,
            original_prompt=prompt,
            max_tokens=safe_budget,
        )
        return fallback or prompt

    def _run_rewrite_attempt(
        self,
        *,
        tokenizer: Any,
        text_encoder: Any,
        encoded: dict[str, Any],
        prompt: str,
        torch_module: Any,
        generate_kwargs: dict[str, Any],
        enhancement_seed: int | None = None,
    ) -> tuple[str, str]:
        try:
            with self._seeded_rng_context(torch_module, enhancement_seed):
                output_ids = self._generate_with_base_model(
                    text_encoder=text_encoder,
                    encoded=encoded,
                    max_new_tokens=int(generate_kwargs.get("max_new_tokens", 72)),
                    eos_token_id=generate_kwargs.get("eos_token_id"),
                    torch_module=torch_module,
                    do_sample=bool(generate_kwargs.get("do_sample", False)),
                    temperature=float(generate_kwargs.get("temperature", 1.0)),
                    top_p=float(generate_kwargs.get("top_p", 1.0)),
                    repetition_penalty=float(generate_kwargs.get("repetition_penalty", 1.08)),
                )
        except Exception as decode_exc:
            if not hasattr(text_encoder, "generate"):
                LOGGER.warning("Prompt enhancement base-model decode failed; using original prompt. %s", decode_exc)
                return prompt, "decode_failure"
            LOGGER.debug(
                "Prompt enhancement base-model decode unavailable; falling back to text_encoder.generate(). %s",
                decode_exc,
            )
            try:
                fallback_encoded = self._move_encoded_to_module_device(encoded, text_encoder)
                fallback_kwargs = self._build_generate_fallback_kwargs(text_encoder, generate_kwargs)
                with self._seeded_rng_context(torch_module, enhancement_seed):
                    with self._module_forward_context(torch_module, text_encoder):
                        output_ids = text_encoder.generate(**fallback_encoded, **fallback_kwargs)
            except Exception as exc:
                LOGGER.warning(
                    "Prompt enhancement base-model decode failed and text_encoder.generate() fallback also failed. "
                    "base_model=%s generate=%s",
                    decode_exc,
                    exc,
                )
                return prompt, "decode_failure"

        try:
            full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            input_text = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)
            rewritten = self._extract_rewritten_prompt(full_text, input_text)
            return rewritten, self._rewrite_rejection_reason(prompt, rewritten)
        except Exception as exc:
            LOGGER.warning("Prompt enhancement decode failed; using original prompt. %s", exc)
            return prompt, "decode_failure"

    @staticmethod
    def _generate_with_base_model(
        *,
        text_encoder: Any,
        encoded: dict[str, Any],
        max_new_tokens: int,
        eos_token_id: int | None,
        torch_module: Any,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> Any:
        if not hasattr(text_encoder, "get_input_embeddings"):
            raise ValueError("text_encoder does not expose input embeddings for greedy decode.")

        embed_layer = text_encoder.get_input_embeddings()
        if embed_layer is None or not hasattr(embed_layer, "weight"):
            raise ValueError("text_encoder input embedding weights are unavailable.")

        encoded = DiffusersQwenInference._move_encoded_to_module_device(encoded, text_encoder)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        past_key_values = None
        generated = input_ids
        embed_weight = embed_layer.weight

        with DiffusersQwenInference._module_forward_context(torch_module, text_encoder):
            try:
                forward_params = inspect.signature(text_encoder.forward).parameters
            except Exception:
                forward_params = {}
            supports_cache_position = "cache_position" in forward_params
            supports_position_ids = "position_ids" in forward_params

            for _ in range(max_new_tokens):
                step_ids = generated if past_key_values is None else generated[:, -1:]
                past_length = generated.shape[1] - step_ids.shape[1]
                cache_position = torch_module.arange(
                    past_length,
                    past_length + step_ids.shape[1],
                    device=step_ids.device,
                    dtype=torch_module.long,
                )
                position_ids = cache_position.unsqueeze(0).expand(step_ids.shape[0], -1)
                model_inputs = {
                    "input_ids": step_ids,
                    "use_cache": True,
                }
                if supports_cache_position:
                    model_inputs["cache_position"] = cache_position
                if supports_position_ids:
                    model_inputs["position_ids"] = position_ids
                if attention_mask is not None:
                    model_inputs["attention_mask"] = attention_mask
                if past_key_values is not None:
                    model_inputs["past_key_values"] = past_key_values

                outputs = text_encoder(**model_inputs)
                past_key_values = getattr(outputs, "past_key_values", None)
                if past_key_values is None:
                    raise ValueError("text_encoder did not return past_key_values.")

                hidden = outputs.last_hidden_state[:, -1, :]
                logits = torch_module.nn.functional.linear(hidden, embed_weight)

                if repetition_penalty > 1.0:
                    for row in range(generated.shape[0]):
                        unique_ids = torch_module.unique(generated[row])
                        unique_ids = unique_ids.to(logits.device)
                        token_logits = logits[row, unique_ids]
                        adjusted = torch_module.where(
                            token_logits < 0,
                            token_logits * repetition_penalty,
                            token_logits / repetition_penalty,
                        )
                        logits[row, unique_ids] = adjusted

                if do_sample:
                    temp = max(float(temperature), 1e-5)
                    logits = logits / temp
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch_module.sort(logits, descending=True, dim=-1)
                        sorted_probs = torch_module.softmax(sorted_logits, dim=-1)
                        cumulative_probs = torch_module.cumsum(sorted_probs, dim=-1)
                        sorted_remove = cumulative_probs > top_p
                        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
                        sorted_remove[..., 0] = False
                        remove_mask = torch_module.zeros_like(sorted_remove, dtype=torch_module.bool)
                        remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_remove)
                        logits = logits.masked_fill(remove_mask, float("-inf"))
                    probs = torch_module.softmax(logits, dim=-1)
                    next_token = torch_module.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                if generated.device != next_token.device:
                    generated = generated.to(next_token.device)
                generated = torch_module.cat([generated, next_token], dim=-1)

                if attention_mask is not None:
                    if attention_mask.device != generated.device:
                        attention_mask = attention_mask.to(generated.device)
                    ones = torch_module.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    attention_mask = torch_module.cat([attention_mask, ones], dim=-1)

                if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                    break

        return generated



class DiffusersQwen3VLInference(DiffusersQwenInference):
    """Chat / prompt-enhancement path for packs whose text encoder is a ``Qwen3VLModel``.

    Why a subclass: the shared ``DiffusersQwenInference`` decode loop drives the pipeline's
    ``text_encoder`` directly with 2D position_ids and expects it to behave like a
    ``Qwen3ForCausalLM`` (tied lm_head via ``embed_tokens``, 1-axis RoPE, plain
    ``last_hidden_state``/``past_key_values`` output). Krea2 packs load ``Qwen3VLModel`` — a
    multimodal wrapper with a vision tower, a ``.language_model`` sub-model (``Qwen3VLTextModel``)
    and 3-axis mRoPE. Feeding the top-level ``Qwen3VLModel.forward`` for text generation trips
    the vision / rope_deltas prefill path even on pure-text inputs.

    Fix: bypass the top-level VL forward for chat / rewrite. Route directly to the
    ``.language_model`` submodule (a ``Qwen3VLTextModel``), which auto-broadcasts 2D
    position_ids to the 3-axis mRoPE layout its RoPE class expects and returns exactly the
    ``last_hidden_state`` + ``past_key_values`` the base decode loop needs. Tied weights still
    hold (config declares ``tie_word_embeddings=True``), so ``embed_tokens.weight`` doubles as
    the lm-head projection — the same trick the base class already uses.
    """

    @classmethod
    def _resolve_generation_module(cls, text_encoder: Any) -> Any:
        """Return the module the decode loop should actually drive.

        For ``Qwen3VLModel`` the language backbone lives at ``.language_model``. For anything
        else (Z-Image's Qwen3ForCausalLM.model, etc.) fall back to the encoder itself so the
        method stays safe to call from generic contexts.
        """
        if text_encoder is None:
            return None
        language_model = getattr(text_encoder, "language_model", None)
        if language_model is not None and hasattr(language_model, "get_input_embeddings"):
            return language_model
        return text_encoder

    @staticmethod
    def _generate_with_base_model(  # type: ignore[override]
        *,
        text_encoder: Any,
        encoded: dict[str, Any],
        max_new_tokens: int,
        eos_token_id: int | None,
        torch_module: Any,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> Any:
        # Reuse the base implementation but drive the ``.language_model`` submodule of a
        # Qwen3VLModel so the vision tower / rope_deltas prefill path never fires. The submodule
        # (``Qwen3VLTextModel``) accepts 2D position_ids and returns ``last_hidden_state`` +
        # ``past_key_values`` exactly like a plain Qwen3 causal LM backbone.
        language_module = DiffusersQwen3VLInference._resolve_generation_module(text_encoder)
        return DiffusersQwenInference._generate_with_base_model(
            text_encoder=language_module,
            encoded=encoded,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            torch_module=torch_module,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )


__all__ = ["DiffusersQwenInference", "DiffusersQwen3VLInference"]
