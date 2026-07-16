"""Krea2-Turbo image backends.

Implements the plan's **Tier A** design (JustRayzist-Krea.md §4): subclass
``DiffusersZImageBackend`` and override only the pipeline-construction seam plus the turbo
defaults. Krea2 and Z-Image are near-siblings (flow-matching, Qwen-conditioned, Qwen-VAE DiT
turbo), so every generate / img2img / upscale / refine / scheduler / tiering path is inherited.

Two backends are provided, mirroring the Z-Image ``diffusers_zimage`` / ``fp8_zimage`` pair:

* ``DiffusersKreaBackend``  -> ``build_krea_pipeline``      (bf16)
* ``Fp8KreaBackend``        -> ``build_fp8_krea_pipeline``  (fp8 storage; primary path on <=24GB)

Tier-B escalation (extracting a shared ``_QwenFlowMatchBackend``) is intentionally NOT done here.
Per the plan, that only happens if a real-hardware spike (WP-0) proves >=3 core helpers diverge.
Divergences discovered during WP-5 (e.g. the Qwen3VL image-conditioning path) are added as narrow
per-method overrides in this file — see ``_build_generate_pipe_kwargs`` and
``_encode_prompt_with_context_image``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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
        device = text_encoder.device
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
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length + prefix_idx,
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


class Fp8KreaBackend(DiffusersKreaBackend):
    """fp8-storage Krea2-Turbo backend.

    Mirrors :class:`app.core.backends.fp8_zimage.Fp8ZImageBackend`. For a 12B DiT this is the
    primary path on <=24GB cards (JustRayzist-Krea.md §4).
    """

    BACKEND_NAME = "fp8_krea"

    def _build_pipeline(self) -> LoadedKreaPipeline:
        return build_fp8_krea_pipeline(
            self._model_pack,
            self._resource_profile(),
        )
