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
Divergences discovered during WP-5 (e.g. the Qwen3VL image-conditioning path for img2img) are
added as narrow per-method overrides in this file — see ``_prepare_krea_image_conditioning``.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.backends.diffusers_zimage import DiffusersZImageBackend
from app.core.pipeline_factory import (
    LoadedKreaPipeline,
    build_fp8_krea_pipeline,
    build_krea_pipeline,
)

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
    # TypeError on Krea, so it is overridden here to call encode_prompt with Krea's kwargs. Verified
    # against diffusers 0.39.0 signatures (WP-0 compatibility matrix); the R+ denoise math itself is
    # still GPU-gated for validation (WP-5 rplus-procedural parity agent).
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

    # --- WP-5 hook: Qwen3VL image conditioning for img2img ---

    def _prepare_krea_image_conditioning(self, pipe: Any, context_image: Any) -> dict[str, Any]:
        """Return pipeline kwargs that inject an optional reference image into Krea2 conditioning.

        Krea2's text encoder is a vision-language model (``Qwen3VLModel``), so a reference image can
        be jointly encoded with the prompt (image-edit style). The exact ``Krea2Pipeline`` argument
        name for this is a WP-0 open question (JustRayzist-Krea.md §12); until confirmed on real
        hardware, this hook is a documented, isolated seam that returns no extra kwargs when no
        context image is supplied. Wiring it into the generate/img2img call and validating the
        argument name is GPU-gated work (WP-5 exit criteria).
        """

        if context_image is None:
            return {}
        raise NotImplementedError(
            "Qwen3VL image conditioning (context_image) requires the Krea2Pipeline "
            "image-encode argument confirmed on real hardware (WP-0/WP-5). "
            "See JustRayzist-Krea.md §12."
        )


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
