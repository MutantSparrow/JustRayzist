from __future__ import annotations

import logging
import math
import re
import string
import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from PIL import Image

from app.config.settings import AppSettings
from app.core.memory import (
    CudaMemorySnapshot,
    ProcessMemorySnapshot,
    cuda_memory_snapshot,
    now_perf,
    process_memory_snapshot,
)
from app.core.model_registry import ModelPack
from app.core.pipeline_factory import LoadedZImagePipeline, build_zimage_pipeline
from app.core.upscale import upscale_image
from app.core.worker.types import GenerationRequest

LOGGER = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    image: Any
    seed: int | None
    steps: int
    guidance_scale: float
    scheduler_mode: str
    backend: str
    device: str
    duration_ms: int
    prompt_original: str
    prompt_effective: str
    prompt_enhanced: bool
    mode: str = "text2img"
    upscale_duration_ms: int | None = None
    refine_duration_ms: int | None = None
    refine_strength: float | None = None
    refine_tile_size: int | None = None
    refine_tile_overlap: int | None = None
    refine_tile_size_requested: int | None = None
    refine_tile_size_effective: int | None = None
    refine_tile_overlap_effective: int | None = None
    refine_fallback_used: bool | None = None
    refine_fallback_attempt_count: int | None = None
    input_image_width: int | None = None
    input_image_height: int | None = None
    cuda_memory_before: CudaMemorySnapshot | None = None
    cuda_memory_after: CudaMemorySnapshot | None = None
    process_memory_before: ProcessMemorySnapshot | None = None
    process_memory_after: ProcessMemorySnapshot | None = None

    def telemetry_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "scheduler_mode": self.scheduler_mode,
            "backend": self.backend,
            "device": self.device,
            "duration_ms": self.duration_ms,
            "prompt_original": self.prompt_original,
            "prompt_effective": self.prompt_effective,
            "prompt_enhanced": self.prompt_enhanced,
            "mode": self.mode,
            "upscale_duration_ms": self.upscale_duration_ms,
            "refine_duration_ms": self.refine_duration_ms,
            "refine_strength": self.refine_strength,
            "refine_tile_size": self.refine_tile_size,
            "refine_tile_overlap": self.refine_tile_overlap,
            "refine_tile_size_requested": self.refine_tile_size_requested,
            "refine_tile_size_effective": self.refine_tile_size_effective,
            "refine_tile_overlap_effective": self.refine_tile_overlap_effective,
            "refine_fallback_used": self.refine_fallback_used,
            "refine_fallback_attempt_count": self.refine_fallback_attempt_count,
            "input_image_width": self.input_image_width,
            "input_image_height": self.input_image_height,
            "cuda_memory_before": (
                self.cuda_memory_before.to_dict() if self.cuda_memory_before else None
            ),
            "cuda_memory_after": (
                self.cuda_memory_after.to_dict() if self.cuda_memory_after else None
            ),
            "process_memory_before": (
                self.process_memory_before.to_dict() if self.process_memory_before else None
            ),
            "process_memory_after": (
                self.process_memory_after.to_dict() if self.process_memory_after else None
            ),
        }


class DiffusersZImageBackend:
    _REFINE_TILE_SNAP = 64
    _REFINE_GRID_DIVISOR_BY_PROFILE: dict[str, int] = {
        "high": 2,
        "balanced": 3,
        "constrained": 4,
    }
    _REFINE_TILE_CAP_BY_PROFILE: dict[str, int] = {
        "high": 896,
        "balanced": 1024,
        "constrained": 896,
    }
    _REFINE_HIGH_FULL_FRAME_MAX_DIM = 1024
    _REFINE_FALLBACK_MIN_TILE_BY_PROFILE: dict[str, int] = {
        "high": 512,
        "balanced": 640,
        "constrained": 512,
    }
    _REFINE_FALLBACK_STEP_FACTORS: tuple[float, ...] = (0.8, 0.64, 0.5)

    def __init__(self, settings: AppSettings, model_pack: ModelPack):
        self._settings = settings
        self._model_pack = model_pack
        self._loaded: LoadedZImagePipeline | None = None
        self._img2img_pipe: Any | None = None
        self._active_scheduler_mode_by_pipe: dict[int, str] = {}

    @staticmethod
    def _snap_up(value: int, multiple: int) -> int:
        if value <= 0:
            return 0
        return int(math.ceil(value / multiple) * multiple)

    @classmethod
    def _build_stepdown_tiles(cls, start_tile: int, min_tile: int) -> list[int]:
        if start_tile <= 0:
            return []
        tiles: list[int] = []
        current = start_tile
        for factor in cls._REFINE_FALLBACK_STEP_FACTORS:
            candidate = cls._snap_up(int(current * factor), cls._REFINE_TILE_SNAP)
            candidate = max(min_tile, candidate)
            if candidate < current and candidate not in tiles:
                tiles.append(candidate)
                current = candidate
        return tiles

    @staticmethod
    def _normalize_scheduler_mode(mode: str | None) -> str:
        if mode is None:
            return "euler"
        normalized = str(mode).strip().lower()
        if normalized not in {"euler", "dpm"}:
            raise ValueError("Unsupported scheduler_mode. Use 'euler' or 'dpm'.")
        return normalized

    def _apply_scheduler_mode(self, pipe: Any, mode: str) -> str:
        pipe_id = id(pipe)
        if self._active_scheduler_mode_by_pipe.get(pipe_id) == mode:
            return mode

        from diffusers import DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler

        current_config = dict(pipe.scheduler.config)
        requested_mode = mode

        def _build_euler_scheduler() -> Any:
            shift = current_config.get("shift")
            if shift is None:
                shift = current_config.get("flow_shift", 3.0)
            return FlowMatchEulerDiscreteScheduler.from_config(
                current_config,
                shift=shift,
                use_dynamic_shifting=current_config.get("use_dynamic_shifting", False),
            )

        if mode == "dpm":
            flow_shift = current_config.get("flow_shift")
            if flow_shift is None:
                flow_shift = current_config.get("shift", 3.0)
            scheduler = DPMSolverMultistepScheduler.from_config(
                current_config,
                algorithm_type="sde-dpmsolver++",
                solver_order=2,
                prediction_type="flow_prediction",
                use_flow_sigmas=True,
                flow_shift=flow_shift,
                use_dynamic_shifting=True,
                time_shift_type="exponential",
                timestep_spacing=current_config.get("timestep_spacing", "linspace"),
            )
            is_img2img_pipe = "img2img" in pipe.__class__.__name__.lower()
            if is_img2img_pipe and not hasattr(scheduler, "scale_noise"):
                LOGGER.warning(
                    "Requested DPM scheduler is incompatible with %s (missing scale_noise). Falling back to Euler.",
                    pipe.__class__.__name__,
                )
                scheduler = _build_euler_scheduler()
                mode = "euler"
        else:
            scheduler = _build_euler_scheduler()

        pipe.scheduler = scheduler
        self._active_scheduler_mode_by_pipe[pipe_id] = mode
        if mode != requested_mode:
            LOGGER.info("Scheduler mode requested=%s applied=%s", requested_mode, mode)
        else:
            LOGGER.info("Scheduler mode set to %s", mode)
        return mode

    def _ensure_loaded(self) -> LoadedZImagePipeline:
        if self._loaded is None:
            LOGGER.info("Loading pipeline for model pack '%s'", self._model_pack.name)
            self._loaded = build_zimage_pipeline(
                self._model_pack,
                self._settings.runtime_profile,
            )
        return self._loaded

    def _ensure_img2img_pipe(self) -> Any:
        if self._img2img_pipe is not None:
            return self._img2img_pipe

        try:
            from diffusers import ZImageImg2ImgPipeline
        except ImportError as exc:
            raise ImportError(
                "Installed diffusers build is missing ZImageImg2ImgPipeline. "
                "Run RunMeFirst.bat to repair the environment."
            ) from exc

        loaded = self._ensure_loaded()
        base_pipe = loaded.pipeline
        pipe = ZImageImg2ImgPipeline(
            scheduler=base_pipe.scheduler,
            vae=base_pipe.vae,
            text_encoder=base_pipe.text_encoder,
            tokenizer=base_pipe.tokenizer,
            transformer=base_pipe.transformer,
        )
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)

        if loaded.device == "cuda":
            profile = self._settings.runtime_profile
            if profile.enable_sequential_offload and hasattr(pipe, "enable_sequential_cpu_offload"):
                pipe.enable_sequential_cpu_offload()
            elif profile.enable_cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
        self._img2img_pipe = pipe
        return self._img2img_pipe

    @staticmethod
    def _clear_cuda_cache(torch_module: Any) -> None:
        try:
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _build_generator(torch_module: Any, device: str, seed: int | None) -> Any:
        if seed is None:
            return None
        return torch_module.Generator(device=device).manual_seed(int(seed))

    @staticmethod
    def _resolve_module_device(module: Any) -> Any:
        if hasattr(module, "device"):
            return module.device
        try:
            return next(module.parameters()).device
        except Exception:
            return None

    @staticmethod
    def _build_rewrite_prompt(tokenizer: Any, prompt: str) -> str:
        system = (
            "Rewrite the input. Preserve intent and return exactly one rewritten prompt with stronger visual "
            "detail. Do not include analysis. Use concrete visual details only. Prefer natural language over "
            "tag lists. Keep under 125 tokens. Include: subject, environment, lighting, composition/camera "
            "if relevant, style/medium."
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
    def _extract_rewritten_prompt(full_text: str, input_text: str) -> str:
        candidate = full_text[len(input_text) :].strip() if full_text.startswith(input_text) else full_text.strip()
        candidate = re.sub(r"<think>.*?</think>\s*", "", candidate, flags=re.DOTALL).strip()
        if "Rewritten prompt:" in candidate:
            candidate = candidate.split("Rewritten prompt:", 1)[-1].strip()
        candidate = candidate.splitlines()[0].strip() if candidate else ""
        return candidate

    @staticmethod
    def _rewrite_quality_ok(original: str, rewritten: str) -> bool:
        return DiffusersZImageBackend._rewrite_rejection_reason(original, rewritten) == "ok"

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

        if original_text and len(text) < len(original_text):
            return "shorter_than_input"

        if text == original_text:
            return "unchanged"
        return "ok"

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
            LOGGER.info("Prompt enhancement skipped: text_encoder/tokenizer unavailable.")
            return prompt

        rewrite_input = self._build_rewrite_prompt(tokenizer, prompt)
        try:
            encoded = tokenizer(rewrite_input, return_tensors="pt")
        except Exception as exc:
            LOGGER.warning("Prompt enhancement tokenizer failed; using original prompt. %s", exc)
            return prompt

        model_device = self._resolve_module_device(text_encoder)
        model_device_type = str(getattr(model_device, "type", ""))
        if model_device is not None and model_device_type != "meta":
            encoded = {key: value.to(model_device) for key, value in encoded.items()}

        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": 72,
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
            return rewritten[:4000]

        retryable_reasons = {
            "repeated_characters",
            "too_much_punctuation",
            "low_lexical_diversity",
            "too_few_letters",
        }
        if rejection_reason in retryable_reasons:
            LOGGER.info(
                "Prompt enhancement retrying with sampled decode after %s.",
                rejection_reason,
            )
            retry_kwargs = dict(generate_kwargs)
            retry_kwargs["do_sample"] = True
            retry_kwargs["temperature"] = 0.72
            retry_kwargs["top_p"] = 0.92
            retry_kwargs["max_new_tokens"] = 96
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
                return rewritten_retry[:4000]
            rejection_reason = retry_reason

        if rejection_reason in {"empty", "too_short", "shorter_than_input", "unchanged"}:
            LOGGER.info(
                "Prompt enhancement skipped (%s); using original prompt.",
                rejection_reason,
            )
        else:
            LOGGER.warning(
                "Prompt enhancement output rejected (%s); using original prompt.",
                rejection_reason,
            )
        return prompt

    def _prepare_pipe_for_prompt_enhancement(self, pipe: Any) -> bool:
        profile = self._settings.runtime_profile
        if (
            not profile.enable_sequential_offload
            or not profile.enable_cpu_offload
            or not hasattr(pipe, "enable_model_cpu_offload")
            or not hasattr(pipe, "enable_sequential_cpu_offload")
        ):
            return False
        try:
            pipe.enable_model_cpu_offload()
            LOGGER.info(
                "Switched to model CPU offload for prompt enhancement (sequential offload is restored before image generation)."
            )
            return True
        except Exception as exc:
            LOGGER.warning(
                "Failed to switch offload mode for prompt enhancement; keeping sequential offload. %s",
                exc,
            )
            return False

    @staticmethod
    def _restore_pipe_after_prompt_enhancement(pipe: Any) -> None:
        if not hasattr(pipe, "enable_sequential_cpu_offload"):
            return
        try:
            pipe.enable_sequential_cpu_offload()
        except Exception as exc:
            LOGGER.warning("Failed to restore sequential offload after prompt enhancement. %s", exc)

    def _resolve_effective_prompt(
        self,
        *,
        pipe: Any,
        prompt: str,
        enhance_prompt: bool,
        seed: int | None,
        torch_module: Any,
    ) -> tuple[str, str, bool]:
        prompt_original = prompt
        prompt_effective = prompt_original
        prompt_enhanced = False
        if not enhance_prompt:
            return prompt_original, prompt_effective, prompt_enhanced

        restore_sequential = False
        loaded = self._ensure_loaded()
        if loaded.device == "cuda":
            restore_sequential = self._prepare_pipe_for_prompt_enhancement(pipe)
        try:
            enhanced_candidate = self._enhance_prompt(
                pipe,
                prompt_original,
                torch_module,
                seed=seed,
            )
        finally:
            if restore_sequential:
                self._restore_pipe_after_prompt_enhancement(pipe)

        if self._rewrite_quality_ok(prompt_original, enhanced_candidate):
            prompt_effective = enhanced_candidate
            prompt_enhanced = True
        else:
            if enhanced_candidate.strip() != prompt_original.strip():
                LOGGER.warning(
                    "Prompt enhancement candidate rejected by final guard; using original prompt."
                )
            prompt_effective = prompt_original
            prompt_enhanced = False
        return prompt_original, prompt_effective, prompt_enhanced

    def _resolve_refine_tiling(self, request: GenerationRequest, width: int, height: int) -> tuple[int, int]:
        overlap = max(8, int(request.refine_tile_overlap or 64))
        if request.refine_tile_size is not None:
            tile_size = max(0, int(request.refine_tile_size))
            return tile_size, overlap

        profile_name = self._settings.runtime_profile.name
        max_dim = max(width, height)
        if profile_name == "high" and max_dim <= self._REFINE_HIGH_FULL_FRAME_MAX_DIM:
            tile_size = 0
        else:
            grid_divisor = self._REFINE_GRID_DIVISOR_BY_PROFILE.get(profile_name, 3)
            tile_cap = self._REFINE_TILE_CAP_BY_PROFILE.get(profile_name, 1024)
            raw_tile = int(math.ceil(max_dim / max(1, grid_divisor)))
            snapped_tile = self._snap_up(raw_tile, self._REFINE_TILE_SNAP)
            tile_size = min(tile_cap, snapped_tile)
        return tile_size, overlap

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
        output_ids = None
        if hasattr(text_encoder, "generate"):
            try:
                with self._seeded_rng_context(torch_module, enhancement_seed):
                    with torch_module.inference_mode():
                        output_ids = text_encoder.generate(**encoded, **generate_kwargs)
            except Exception as exc:
                LOGGER.warning(
                    "Prompt enhancement generate() failed; falling back to base-model decode. %s",
                    exc,
                )

        if output_ids is None:
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
            except Exception as exc:
                LOGGER.warning("Prompt enhancement base-model decode failed; using original prompt. %s", exc)
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

        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        past_key_values = None
        generated = input_ids
        embed_weight = embed_layer.weight

        with torch_module.inference_mode():
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

    def _run_img2img_once(
        self,
        *,
        pipe: Any,
        prompt: str,
        image: Image.Image,
        strength: float,
        steps: int,
        guidance_scale: float,
        generator: Any,
        torch_module: Any,
    ) -> Image.Image:
        with torch_module.inference_mode():
            output = pipe(
                prompt=prompt,
                image=image,
                strength=strength,
                width=image.width,
                height=image.height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        return output.images[0]

    def _run_img2img_tiled(
        self,
        *,
        pipe: Any,
        prompt: str,
        image: Image.Image,
        strength: float,
        steps: int,
        guidance_scale: float,
        seed: int | None,
        tile_size: int,
        tile_overlap: int,
        torch_module: Any,
    ) -> Image.Image:
        width, height = image.size
        if tile_size <= 0 or (width <= tile_size and height <= tile_size):
            generator = self._build_generator(torch_module, "cuda" if torch_module.cuda.is_available() else "cpu", seed)
            return self._run_img2img_once(
                pipe=pipe,
                prompt=prompt,
                image=image,
                strength=strength,
                steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                torch_module=torch_module,
            )

        canvas = Image.new("RGB", (width, height))
        tile_index = 0
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                tile_height = min(tile_size, height - y)
                tile_width = min(tile_size, width - x)

                in_y0 = max(y - tile_overlap, 0)
                in_x0 = max(x - tile_overlap, 0)
                in_y1 = min(y + tile_height + tile_overlap, height)
                in_x1 = min(x + tile_width + tile_overlap, width)

                tile_input = image.crop((in_x0, in_y0, in_x1, in_y1))
                tile_seed = (seed + tile_index) if seed is not None else None
                generator = self._build_generator(
                    torch_module,
                    "cuda" if torch_module.cuda.is_available() else "cpu",
                    tile_seed,
                )
                tile_output = self._run_img2img_once(
                    pipe=pipe,
                    prompt=prompt,
                    image=tile_input,
                    strength=strength,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    torch_module=torch_module,
                )

                crop_x0 = x - in_x0
                crop_y0 = y - in_y0
                crop_x1 = crop_x0 + tile_width
                crop_y1 = crop_y0 + tile_height
                core = tile_output.crop((crop_x0, crop_y0, crop_x1, crop_y1))
                canvas.paste(core, (x, y))
                tile_index += 1
        return canvas

    def _run_refine_with_oom_fallback(
        self,
        *,
        pipe: Any,
        prompt: str,
        image: Image.Image,
        strength: float,
        steps: int,
        guidance_scale: float,
        seed: int | None,
        tile_size: int,
        tile_overlap: int,
        torch_module: Any,
    ) -> tuple[Image.Image, int, int, int]:
        profile_name = self._settings.runtime_profile.name
        fallback_overlap = max(32, tile_overlap)
        min_tile = self._REFINE_FALLBACK_MIN_TILE_BY_PROFILE.get(profile_name, 512)

        attempt_tiles: list[int] = []
        if tile_size > 0:
            attempt_tiles.append(tile_size)
            attempt_tiles.extend(self._build_stepdown_tiles(tile_size, min_tile))
        else:
            attempt_tiles.append(0)
            max_dim = max(image.width, image.height)
            cap = self._REFINE_TILE_CAP_BY_PROFILE.get(profile_name, 1024)
            fallback_start = min(cap, max_dim)
            fallback_start = self._snap_up(fallback_start, self._REFINE_TILE_SNAP)
            fallback_start = max(min_tile, fallback_start)
            attempt_tiles.append(fallback_start)
            attempt_tiles.extend(self._build_stepdown_tiles(fallback_start, min_tile))

        normalized_attempt_tiles: list[int] = []
        for index, candidate in enumerate(attempt_tiles):
            if index > 0 and candidate > 0:
                candidate = self._snap_up(candidate, self._REFINE_TILE_SNAP)
                candidate = max(min_tile, candidate)
            if candidate not in normalized_attempt_tiles:
                normalized_attempt_tiles.append(candidate)
        attempt_tiles = normalized_attempt_tiles

        fallback_attempts = 0
        for idx, candidate_tile in enumerate(attempt_tiles):
            try:
                if candidate_tile > 0:
                    return (
                        self._run_img2img_tiled(
                            pipe=pipe,
                            prompt=prompt,
                            image=image,
                            strength=strength,
                            steps=steps,
                            guidance_scale=guidance_scale,
                            seed=seed,
                            tile_size=candidate_tile,
                            tile_overlap=fallback_overlap,
                            torch_module=torch_module,
                        ),
                        candidate_tile,
                        fallback_overlap,
                        fallback_attempts,
                    )
                generator = self._build_generator(
                    torch_module,
                    "cuda" if torch_module.cuda.is_available() else "cpu",
                    seed,
                )
                return (
                    self._run_img2img_once(
                        pipe=pipe,
                        prompt=prompt,
                        image=image,
                        strength=strength,
                        steps=steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        torch_module=torch_module,
                    ),
                    0,
                    tile_overlap,
                    fallback_attempts,
                )
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                if idx == len(attempt_tiles) - 1:
                    raise
                fallback_attempts += 1
                if candidate_tile == 0:
                    LOGGER.warning(
                        "Img2img refine OOM on full frame, retrying with tiled refine."
                    )
                else:
                    LOGGER.warning(
                        "Img2img refine OOM at tile size %s, retrying with smaller tile.",
                        candidate_tile,
                    )
                self._clear_cuda_cache(torch_module)
        raise RuntimeError("Unreachable OOM fallback state.")

    def generate(self, request: GenerationRequest) -> GenerationResult:
        loaded = self._ensure_loaded()
        pipe = loaded.pipeline
        scheduler_mode = self._normalize_scheduler_mode(request.scheduler_mode)
        scheduler_mode = self._apply_scheduler_mode(pipe, scheduler_mode)

        import torch

        steps = request.steps or self._settings.runtime_profile.steps_default
        guidance_scale = (
            request.guidance_scale
            if request.guidance_scale is not None
            else self._settings.runtime_profile.guidance_scale_default
        )
        generator = self._build_generator(torch, "cuda" if loaded.device == "cuda" else "cpu", request.seed)

        prompt_original, prompt_effective, prompt_enhanced = self._resolve_effective_prompt(
            pipe=pipe,
            prompt=request.prompt,
            enhance_prompt=request.enhance_prompt,
            seed=request.seed,
            torch_module=torch,
        )

        pre_mem = cuda_memory_snapshot(torch)
        pre_proc_mem = process_memory_snapshot()
        started = now_perf()
        with torch.inference_mode():
            output = pipe(
                prompt=prompt_effective,
                width=request.width,
                height=request.height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration_ms = int((now_perf() - started) * 1000)
        post_mem = cuda_memory_snapshot(torch)
        post_proc_mem = process_memory_snapshot()
        image = output.images[0]
        return GenerationResult(
            image=image,
            seed=request.seed,
            steps=steps,
            guidance_scale=guidance_scale,
            scheduler_mode=scheduler_mode,
            backend="diffusers_zimage",
            device=loaded.device,
            duration_ms=duration_ms,
            prompt_original=prompt_original,
            prompt_effective=prompt_effective,
            prompt_enhanced=prompt_enhanced,
            cuda_memory_before=pre_mem,
            cuda_memory_after=post_mem,
            process_memory_before=pre_proc_mem,
            process_memory_after=post_proc_mem,
        )

    def upscale_and_refine(self, input_image: object, request: GenerationRequest) -> GenerationResult:
        if not isinstance(input_image, Image.Image):
            raise ValueError("input_image must be a PIL.Image.Image instance.")

        loaded = self._ensure_loaded()
        txt_pipe = loaded.pipeline
        img_pipe = self._ensure_img2img_pipe()
        scheduler_mode = self._normalize_scheduler_mode(request.scheduler_mode)
        scheduler_mode = self._apply_scheduler_mode(img_pipe, scheduler_mode)

        import torch

        refine_steps = request.refine_steps or 6
        refine_strength = request.refine_strength if request.refine_strength is not None else 0.20
        if refine_strength <= 0.0 or refine_strength >= 1.0:
            raise ValueError("refine_strength must be between 0 and 1.")

        guidance_scale = (
            request.guidance_scale
            if request.guidance_scale is not None
            else self._settings.runtime_profile.guidance_scale_default
        )
        prompt_original, prompt_effective, prompt_enhanced = self._resolve_effective_prompt(
            pipe=txt_pipe,
            prompt=request.prompt,
            enhance_prompt=request.enhance_prompt,
            seed=request.seed,
            torch_module=torch,
        )

        checkpoint_path = request.upscaler_checkpoint or (
            self._settings.paths.models_dir / "upscaler" / "2x_RealESRGAN_x2plus.pth"
        )
        if not checkpoint_path.exists():
            raise ValueError(f"Upscaler checkpoint not found: {checkpoint_path}")

        pre_mem = cuda_memory_snapshot(torch)
        pre_proc_mem = process_memory_snapshot()
        started = now_perf()

        upscale_result = upscale_image(
            image=input_image,
            checkpoint_path=checkpoint_path,
            profile_name=self._settings.runtime_profile.name,
        )
        refine_input_image = upscale_result.image

        tile_size_requested, tile_overlap_requested = self._resolve_refine_tiling(
            request,
            refine_input_image.width,
            refine_input_image.height,
        )
        refine_started = now_perf()
        (
            refined_image,
            effective_tile_size,
            effective_tile_overlap,
            fallback_attempt_count,
        ) = self._run_refine_with_oom_fallback(
            pipe=img_pipe,
            prompt=prompt_effective,
            image=refine_input_image,
            strength=refine_strength,
            steps=refine_steps,
            guidance_scale=guidance_scale,
            seed=request.seed,
            tile_size=tile_size_requested,
            tile_overlap=tile_overlap_requested,
            torch_module=torch,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        refine_duration_ms = int((now_perf() - refine_started) * 1000)
        duration_ms = int((now_perf() - started) * 1000)
        post_mem = cuda_memory_snapshot(torch)
        post_proc_mem = process_memory_snapshot()

        return GenerationResult(
            image=refined_image,
            seed=request.seed,
            steps=refine_steps,
            guidance_scale=guidance_scale,
            scheduler_mode=scheduler_mode,
            backend="diffusers_zimage",
            device=loaded.device,
            duration_ms=duration_ms,
            prompt_original=prompt_original,
            prompt_effective=prompt_effective,
            prompt_enhanced=prompt_enhanced,
            mode="upscale_then_img2img",
            upscale_duration_ms=upscale_result.duration_ms,
            refine_duration_ms=refine_duration_ms,
            refine_strength=refine_strength,
            refine_tile_size=effective_tile_size,
            refine_tile_overlap=effective_tile_overlap,
            refine_tile_size_requested=tile_size_requested,
            refine_tile_size_effective=effective_tile_size,
            refine_tile_overlap_effective=effective_tile_overlap,
            refine_fallback_used=fallback_attempt_count > 0,
            refine_fallback_attempt_count=fallback_attempt_count,
            input_image_width=input_image.width,
            input_image_height=input_image.height,
            cuda_memory_before=pre_mem,
            cuda_memory_after=post_mem,
            process_memory_before=pre_proc_mem,
            process_memory_after=post_proc_mem,
        )
