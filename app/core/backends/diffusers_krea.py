"""Krea2-Turbo image backends.

Tier-A design (see ``docs/KREA2_IMPLEMENTATION_STATUS.md``): subclass
``DiffusersZImageBackend`` and override only the pipeline-construction seam plus the turbo
defaults. Krea2 and Z-Image are near-siblings (flow-matching, Qwen-conditioned, Qwen-VAE DiT
turbo), so every generate / img2img / upscale / refine / scheduler / tiering path is inherited.

Two backends are provided, mirroring the Z-Image ``diffusers_zimage`` / ``fp8_zimage`` pair:

* ``DiffusersKreaBackend``  -> ``build_krea_pipeline``      (bf16)
* ``Fp8KreaBackend``        -> ``build_fp8_krea_pipeline``  (fp8 storage; primary path on <=24GB)

Tier-B escalation (extracting a shared ``_QwenFlowMatchBackend``) is intentionally NOT done here.
Only helpers that actually diverge get narrow per-method overrides — see
``_rplus_prepare_prompt_embeds``, ``_build_generate_pipe_kwargs``, and
``_encode_prompt_with_context_image``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.core.backends.diffusers_qwen import DiffusersQwen3VLInference, DiffusersQwenInference
from app.core.backends.diffusers_zimage import DiffusersZImageBackend
from app.core.pipeline_factory import (
    LoadedKreaPipeline,
    build_fp8_krea_pipeline,
    build_krea_pipeline,
)
from app.core.worker.types import GenerationRequest

LOGGER = logging.getLogger(__name__)


class DiffusersKreaBackend(DiffusersZImageBackend):
    """bf16 Krea2-Turbo backend.

    Overrides only ``_build_pipeline`` and the turbo-default seams. All other behavior is inherited
    from :class:`DiffusersZImageBackend`.
    """

    BACKEND_NAME = "diffusers_krea"

    # Krea2-Turbo is distilled and runs CFG-free at 8 steps (HF model card). Z-Image profiles
    # already default guidance_scale to 0.0 (see app/config/profiles.py), so only the step count
    # differs from the Z-Image turbo default; guidance is overridden defensively for clarity.
    _KREA_DEFAULT_STEPS = 8
    _KREA_DEFAULT_GUIDANCE_SCALE = 0.0
    # A CFG-free model must not apply classifier-free guidance on the img2img/refine path either.
    _IMG2IMG_GUIDANCE_SCALE_DEFAULT = 0.0

    def _build_pipeline(self) -> LoadedKreaPipeline:
        return build_krea_pipeline(
            self._model_pack,
            self._resource_profile(),
        )

    # --- Turbo-default seams (see DiffusersZImageBackend._default_steps / _default_guidance_scale) ---

    def _default_steps(self) -> int:
        return self._KREA_DEFAULT_STEPS

    def _default_guidance_scale(self) -> float:
        return self._KREA_DEFAULT_GUIDANCE_SCALE

    # --- Tier-B override: R+ prompt embeds ---
    # Krea2Pipeline.encode_prompt has a different signature than Z-Image's: it takes no
    # negative_prompt/do_classifier_free_guidance and returns (prompt_embeds, prompt_embeds_mask)
    # rather than (prompt_embeds, negative_prompt_embeds). The inherited R+ helper would raise a
    # TypeError on Krea, so it is overridden here to call encode_prompt with Krea's kwargs.
    @classmethod
    def _rplus_prepare_prompt_embeds(cls, pipe: Any, prompt: str, device: Any) -> list[Any]:
        prompt_embeds, _prompt_embeds_mask = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            max_sequence_length=cls._resolve_pipeline_max_sequence_length(
                getattr(pipe, "tokenizer", None),
                prompt,
            ),
        )
        return prompt_embeds

    # R+ is Z-Image-only. Krea2's ``Krea2Pipeline.prepare_latents`` returns packed 3D latents
    # (batch, image_seq_len, in_channels) rather than the 4D (B, C, H, W) shape the R+ schedule
    # estimator expects — see ``_rplus_scheduler_mu`` reading ``latents.shape[3]``. Rather than
    # port the whole R+ denoise loop to packed latents (deep Z-Image internals), block R+ at
    # entry with a clean error so the UI can grey out the toggle. Bench 2026-07-17 confirms this
    # path currently crashes with IndexError deep in ``_rplus_estimate_initial_noise_features``.
    def _run_rplus_generate(self, *, pipe: Any, request: Any, prompt_effective: str,
                            procedural_latents: Any, torch_module: Any) -> tuple[Any, dict]:
        raise NotImplementedError(
            "R+ inference is not supported on the Krea2-Turbo backend. R+ was designed against "
            "Z-Image Turbo's 4D (B, C, H, W) latent layout; Krea2Pipeline uses packed 3D latents. "
            "Turn off the R+ toggle before generating with a Krea2 pack."
        )

    # --- WP-5: Qwen3VL image conditioning ---
    # Krea2Pipeline itself is text2img; it has no image argument on ``__call__`` or ``encode_prompt``.
    # But its text encoder is a Qwen3VL vision-language model, and the Krea2 transformer was trained
    # to consume the same 12 hidden-state taps the pipeline builds from text. ComfyUI's reference
    # "style image" workflow works by encoding the chat template with vision tokens + pixel_values
    # through Qwen3VL, tapping those same layers, and feeding the result to Krea2 as prompt_embeds.
    # We mirror that here — same encoder call surface, same layer indices from
    # ``pipe.text_encoder_select_layers``, then pass the embeds via ``prompt_embeds``.

    def _build_generate_pipe_kwargs(
        self,
        *,
        pipe: Any,
        prompt: str,
        request: GenerationRequest,
        steps: int,
        guidance_scale: float,
        generator: Any,
        procedural_latents: Any,
        torch_module: Any,
    ) -> dict[str, Any]:
        base = super()._build_generate_pipe_kwargs(
            pipe=pipe,
            prompt=prompt,
            request=request,
            steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            procedural_latents=procedural_latents,
            torch_module=torch_module,
        )
        context_image = getattr(request, "context_image", None)
        if context_image is None:
            return base

        embeds, embeds_mask = self._encode_prompt_with_context_image(
            pipe=pipe,
            prompt=prompt,
            context_image=context_image,
            max_sequence_length=base["max_sequence_length"],
            torch_module=torch_module,
        )
        # Krea2Pipeline validates: pass prompt_embeds+mask ⇒ prompt must be None.
        base["prompt"] = None
        base["prompt_embeds"] = embeds
        base["prompt_embeds_mask"] = embeds_mask
        return base

    def _encode_prompt_with_context_image(
        self,
        *,
        pipe: Any,
        prompt: str,
        context_image: Path,
        max_sequence_length: int,
        torch_module: Any,
    ) -> tuple[Any, Any]:
        """Encode ``prompt`` jointly with a reference image through Qwen3VL.

        Returns ``(prompt_embeds, prompt_embeds_mask)`` shaped to match Krea2's ``prompt_embeds``
        contract: ``(1, text_seq_len, num_text_layers, text_hidden_dim)`` and
        ``(1, text_seq_len)``. Follows the same chat-template layout that Krea2Pipeline uses
        text-only (``get_text_hidden_states``) with vision tokens injected so the encoder attends
        to the image; the pipeline's ``text_encoder_select_layers`` are then tapped the same way.
        """
        from PIL import Image

        text_encoder = pipe.text_encoder
        tokenizer = pipe.tokenizer
        del tokenizer, max_sequence_length  # processor drives tokenization/length for the VL path
        # We are called BEFORE the pipeline enters its ``__call__`` (from ``_build_generate_pipe_kwargs``).
        # Under sequential CPU offload, accelerate has hooked every submodule and swapped params for
        # meta placeholders whose real storage lives in a ``weights_map`` staged on-demand per
        # module. The pipeline's text-only ``get_text_hidden_states`` works because it's called
        # from inside ``__call__`` where accelerate is already routing correctly; but calling
        # ``text_encoder(...)`` directly from here trips the hook on submodules the text-only path
        # never touches (the vision tower's ``pos_embed`` / ``patch_embed``).
        #
        # Strategy: strip the pipeline's accelerate hooks, move the encoder onto the execution
        # device, run the VL encode, then re-enable sequential offload before returning. The
        # encoder is small next to the 12B transformer, so this brief residency is safe on <=24GB.
        device = pipe._execution_device
        from accelerate.hooks import remove_hook_from_module

        remove_hook_from_module(text_encoder, recurse=True)
        text_encoder.to(device)
        select_layers = tuple(pipe.text_encoder_select_layers)
        prefix_idx = pipe.prompt_template_encode_start_idx

        image = Image.open(str(context_image)).convert("RGB")

        # Build the Qwen3VL chat message with an image turn. This matches the ComfyUI style-ref
        # workflow: the assistant sees a user turn containing both an image and the description
        # prompt, so the Qwen3VL hidden states carry visual features fused with the text.
        processor = self._resolve_qwen3vl_processor(pipe)
        messages = [
            {
                "role": "system",
                "content": (
                    "Describe the image by detailing the color, shape, size, texture, quantity, "
                    "text, spatial relationships of the objects and background:"
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # Do NOT truncate: an image expands to ~1000 vision tokens after patch merge, so a
        # fixed ``max_length`` from the text-only path would clip image tokens and desync the
        # processor's image/text token counts. The Krea2 transformer's text-side reads whatever
        # ``text_seq_len`` the encoder produces (position ids are built from the mask below), so
        # a variable length is fine. Padding is disabled for the same reason — a single-image
        # batch does not need it.
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        # Match Krea2Pipeline.get_text_hidden_states position-id handling: positions count only
        # real tokens (padding does not consume a position) and are broadcast across the 3 mRoPE
        # axes. Padding position sits in the middle so image + suffix tokens don't get a shifted
        # mRoPE phase.
        attention_mask = inputs["attention_mask"].bool()
        position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        inputs["position_ids"] = position_ids
        inputs["attention_mask"] = attention_mask

        with torch_module.no_grad():
            outputs = text_encoder(**inputs, output_hidden_states=True)

        hidden = torch_module.stack(
            [outputs.hidden_states[i] for i in select_layers], dim=2
        )
        # Drop the system-prefix tokens exactly like the text-only path.
        hidden = hidden[:, prefix_idx:]
        mask = attention_mask[:, prefix_idx:]

        # Restore the pipeline's offload strategy so the downstream denoise loop keeps its VRAM
        # budget. remove_all_hooks() clears every module (including the ones we didn't touch);
        # enable_sequential_cpu_offload() re-attaches the full chain that was originally built by
        # the pipeline factory.
        pipe.remove_all_hooks()
        if hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
        return hidden, mask

    def _resolve_qwen3vl_processor(self, pipe: Any) -> Any:
        """Return the Qwen3VL processor (tokenizer + image processor).

        Krea2Pipeline stores only a tokenizer, so build the multimodal processor lazily against the
        same local text-encoder config directory used by the pipeline builder. Cached on the
        pipeline instance so repeated context_image generations don't rebuild it.
        """
        cached = getattr(pipe, "_rayzist_qwen3vl_processor", None)
        if cached is not None:
            return cached

        from transformers import AutoProcessor

        config_dir = self._model_pack.pipeline_config_dir / "text_encoder"
        processor = AutoProcessor.from_pretrained(str(config_dir), local_files_only=True)
        setattr(pipe, "_rayzist_qwen3vl_processor", processor)
        return processor

    # --- Chat / prompt-enhancement / wildcard-suggest ---
    # The shared DiffusersQwenInference decode loop targets Qwen3ForCausalLM (1D RoPE, plain
    # last_hidden_state / past_key_values). Krea2's text encoder is Qwen3VLModel — a wrapper
    # with a Qwen3VLTextModel at ``.language_model`` and a vision tower at ``.visual``. Driving
    # the top-level VL forward triggers the rope_deltas / image-prefill path even for text-only
    # inputs. Route through the language sub-model via ``DiffusersQwen3VLInference``.
    #
    # Under sequential CPU offload, the pipeline's accelerate hooks keep parameters on meta
    # placeholders — same problem WP-5's ``_encode_prompt_with_context_image`` handles by
    # ``remove_hook_from_module`` + move to device + run + ``pipe.remove_all_hooks()`` +
    # re-enable offload. ``_run_with_staged_text_encoder`` wraps that dance around every chat /
    # rewrite / wildcard-suggest call so the caller-facing methods stay clean.

    def _qwen_for_pipe(self, pipe: Any, torch_module: Any) -> DiffusersQwenInference:
        return DiffusersQwen3VLInference.from_pipe(
            pipe,
            torch_module=torch_module,
            encoder_label=self._text_encoder_label(),
        )

    def _run_with_staged_text_encoder(self, pipe: Any, torch_module: Any, action):
        """Stage the pipeline's text encoder on the execution device, run ``action(pipe)``, and
        restore the pipeline's offload chain.

        ``action`` is a callable that receives the (staged) pipeline and returns whatever the
        caller needs. Any exception the action raises is propagated after the restore step so
        the pipeline is always left in a consistent state.
        """
        text_encoder = getattr(pipe, "text_encoder", None)
        # Nothing to stage if the pipeline doesn't hold a text encoder or CUDA isn't around.
        # Fall straight through to the action.
        if text_encoder is None or not torch_module.cuda.is_available():
            return action(pipe)

        # No accelerate hooks on the transformer's submodules => pipeline was placed with
        # ``pipe.to("cuda")`` (i.e. ``high`` tier). No staging dance needed; caller uses the
        # encoder as-is.
        transformer = getattr(pipe, "transformer", None)
        has_hooks = False
        if transformer is not None:
            for module in transformer.modules():
                if hasattr(module, "_hf_hook"):
                    has_hooks = True
                    break
        if not has_hooks:
            return action(pipe)

        from accelerate.hooks import remove_hook_from_module

        device = pipe._execution_device
        remove_hook_from_module(text_encoder, recurse=True)
        text_encoder.to(device)
        try:
            return action(pipe)
        finally:
            # remove_all_hooks() also clears our text_encoder hooks (which we already dropped
            # above), then enable_sequential_cpu_offload() rebuilds the whole offload chain
            # exactly as the pipeline builder did originally.
            try:
                pipe.remove_all_hooks()
                if hasattr(pipe, "enable_sequential_cpu_offload"):
                    pipe.enable_sequential_cpu_offload()
            except Exception:  # pragma: no cover - defensive teardown
                LOGGER.warning("Failed to restore sequential CPU offload after chat.", exc_info=True)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 256,
        seed: int | None = None,
        app_context: str | None = None,
        temperature: float = 0.75,
    ) -> dict[str, Any]:
        loaded = self._ensure_loaded()
        import torch

        def _run(pipe: Any) -> dict[str, Any]:
            return self._qwen_for_pipe(pipe, torch).chat(
                messages=messages,
                max_new_tokens=max_new_tokens,
                seed=seed,
                app_context=app_context,
                temperature=temperature,
            )

        return self._run_with_staged_text_encoder(loaded.pipeline, torch, _run)

    def suggest_wildcard_entries(
        self,
        *,
        theme: str,
        format_example: str,
        existing_entries: list[str] | None = None,
        target_count: int = 6,
        min_words: int = 1,
        max_words: int = 32,
        seed: int | None = None,
    ) -> list[str]:
        loaded = self._ensure_loaded()
        import torch

        def _run(pipe: Any) -> list[str]:
            return self._qwen_for_pipe(pipe, torch).suggest_wildcard_entries(
                theme=theme,
                format_example=format_example,
                existing_entries=existing_entries,
                target_count=target_count,
                min_words=min_words,
                max_words=max_words,
                seed=seed,
            )

        return self._run_with_staged_text_encoder(loaded.pipeline, torch, _run)

    def _enhance_prompt(
        self,
        pipe: Any,
        prompt: str,
        torch_module: Any,
        *,
        seed: int | None = None,
    ) -> str:
        # Called from the generate() forward — pipe is already the loaded pipeline. Stage the
        # encoder just for the rewrite call so the base generate() flow can keep running.
        def _run(inner_pipe: Any) -> str:
            return self._qwen_for_pipe(inner_pipe, torch_module).enhance_prompt(prompt, seed=seed)

        return self._run_with_staged_text_encoder(pipe, torch_module, _run)

    def _compress_long_prompt(
        self,
        pipe: Any,
        prompt: str,
        torch_module: Any,
        *,
        seed: int | None = None,
    ) -> str:
        def _run(inner_pipe: Any) -> str:
            return self._qwen_for_pipe(inner_pipe, torch_module).compress_long_prompt(prompt, seed=seed)

        return self._run_with_staged_text_encoder(pipe, torch_module, _run)


class Fp8KreaBackend(DiffusersKreaBackend):
    """fp8-storage Krea2-Turbo backend.

    Mirrors :class:`app.core.backends.fp8_zimage.Fp8ZImageBackend`. For a 12B DiT this is the
    primary path on <=24GB cards.
    """

    BACKEND_NAME = "fp8_krea"

    def _build_pipeline(self) -> LoadedKreaPipeline:
        return build_fp8_krea_pipeline(
            self._model_pack,
            self._resource_profile(),
        )
