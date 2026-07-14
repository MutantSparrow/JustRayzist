# Krea2-Turbo Integration — Implementation Status

Tracks what landed on `feat/krea2-support`. The full design is in `JustRayzist-Krea.md` (repo root).

**End-to-end verified on an RTX 4080 (16GB):** the `Krea2_Turbo` fp8 pack generates images through
the real app backend (`create_backend` → `Fp8KreaBackend` → `build_fp8_krea_pipeline` →
`backend.generate()`). A 1024×1024, 8-step, cfg-0.0 render of "a red fox sitting in fresh snow"
completed in ~230s and produced a correct photorealistic image. See "ComfyUI checkpoint support"
below.

## ComfyUI checkpoint support (AlperKTS/Krea2_FP8)

The `Krea2_Turbo` pack uses the ComfyUI-native fp8 weights from
[`AlperKTS/Krea2_FP8`](https://huggingface.co/AlperKTS/Krea2_FP8) (transformer, VAE, Qwen3VL
encoder). ComfyUI checkpoints use different key layouts than diffusers, so
`app/core/pipeline_factory/krea_comfy_convert.py` converts each component — the same pattern as the
existing Z-Image `_convert_prefixed_fused_zimage_state_dict`:

- **Transformer** (`blocks.*` → `transformer_blocks.*`, `first/last/tmlp/tproj/txtmlp` globals,
  `mod.lin (6*H,)` → `scale_shift_table (6,H)`; drops the 2 extra `last.up/last.down` tensors some
  fp8 repacks carry). Mapping derived by aligning against the official
  `krea/Krea-2-Turbo/turbo.safetensors` (same native layout, 430 keys) and the diffusers model;
  validated by an exact key+shape match and a full 12.82B-param CPU load.
- **VAE** — reuses diffusers' own `convert_wan_vae_to_diffusers` (the Qwen-image VAE shares the Wan
  VAE layout); exact 194/194 match, strict load.
- **Qwen3VL encoder** — ComfyUI scaled-fp8 (`.weight` fp8 × scalar `.weight_scale`, plus a
  `.comfy_quant` JSON-metadata tensor that is dropped) with a `model.* → language_model.*` /
  `model.visual.* → visual.*` prefix remap; exact 713/713 match.

The pack config dirs (`config/{scheduler,tokenizer,transformer,vae,text_encoder}`) are the official
`krea/Krea-2-Turbo` diffusers configs. Weights are gitignored; fetch them with
`scripts/fetch_krea2_assets.ps1` (or the `AlperKTS/Krea2_FP8` files) into
`models/packs/Krea2_Turbo/weights/`.

On a 16GB card the builder sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (unless already
set) before CUDA init to avoid fragmentation OOMs; the `constrained` profile's sequential CPU
offload keeps the fp8 transformer + bf16 encoder + VAE within budget. A manual smoke test lives at
`scripts/dev_krea2_smoke.py`.

## Coexistence

Implemented against `diffusers==0.39.0` (a normal PyPI release), which ships both the `Krea2*` and
`ZImage*` classes — the plan's §6.1 option (a). The full Z-Image regression suite passes on 0.39.0
(342 passed / 1 skipped), so Krea2 support does not disturb the Z-Image path.

## Landed (code-complete, CPU-verifiable)

| WP | Area | Files |
|----|------|-------|
| WP-1 | Registry + dispatch | `app/core/model_registry/model_pack.py` (`krea2_turbo` architecture), `app/core/backends/__init__.py` (`diffusers_krea`/`fp8_krea` dispatch, lazy import, `SUPPORTED_BACKENDS`) |
| WP-2 | Pipeline builders | `app/core/pipeline_factory/krea.py` (`build_krea_pipeline`, `build_fp8_krea_pipeline`), exported from `pipeline_factory/__init__.py` |
| WP-3 | Backends | `app/core/backends/diffusers_krea.py` (`DiffusersKreaBackend`, `Fp8KreaBackend`); turbo-default seam (`_default_steps`/`_default_guidance_scale`) added to `diffusers_zimage.py` |
| WP-5 | VL img2img field | `app/core/worker/types.py` (`GenerationRequest.context_image`); encode hook `_prepare_krea_image_conditioning` in the Krea backend |
| WP-7 | Runtime model switch | `app/core/worker/session.py` (`GenerationSession.switch_model_pack`, tier-adaptive keep-resident/unload, `recycle` releases resident cache) |
| WP-4 | Pack scaffold | `models/packs/Krea2_Turbo/` (template, `config/model_index.json`, READMEs), `scripts/fetch_krea2_assets.ps1` |
| WP-8 | Weightless tests | `tests/test_krea_registry_dispatch.py`, `tests/test_runtime_model_switch.py` |
| WP-9 | Docs | `models/packs/README.md` (architectures section), this file |

### Behavior-preservation note (Z-Image regression)

The only change to the shared Z-Image backend is the extraction of two default-resolving methods
(`_default_steps`, `_default_guidance_scale`) that return exactly the previous inline values
(`self._settings.runtime_profile.steps_default` / `.guidance_scale_default`). Z-Image behavior is
therefore unchanged. The Krea backend overrides these to 8 steps / 0.0 guidance.

Note also that **all runtime profiles already default `guidance_scale_default=0.0`**
(`app/config/profiles.py`), so the CFG-free path is already exercised by Z-Image — this de-risks
the plan's cfg=0.0 concern (§6.4). The remaining cfg=0.0 verification is confirming the diffusers
`Krea2Pipeline.__call__` handles `guidance_scale=0.0` without a degenerate branch (WP-0).

## WP-0 findings gathered here (static, no GPU)

Partial WP-0 spike results obtained by introspecting a pip-installed `diffusers==0.39.0` on CPU.
These de-risk the plan but do **not** replace the on-GPU generate spike:

- **Coexistence is option (a).** `diffusers==0.39.0` is a normal PyPI **release** (not source-only)
  and exports **both** the Krea2 classes (`Krea2Pipeline`, `Krea2Transformer2DModel`,
  `AutoencoderKLQwenImage`) **and** the existing `ZImage*` classes. Both import in one process.
  The plan's §3 note "requires diffusers >=0.39.0.dev0 (from source)" is now **stale** — update the
  pin to `diffusers>=0.39.0` and treat coexistence risk (§6.1) as largely resolved, pending a Z-Image
  regression run on 0.39.0 (the repo currently pins `>=0.36.0`).
- **Basic generate() is Tier-A compatible.** `Krea2Pipeline.__call__` accepts every kwarg the
  inherited `generate()` passes (`prompt`, `width`, `height`, `num_inference_steps`, `guidance_scale`,
  `generator`, `latents`, `max_sequence_length`). `_interrupt`/`interrupt` exist, so the inherited
  `cancel_active()` works.
- **R+ needs a Tier-B override (done).** `Krea2Pipeline.encode_prompt` has a different signature
  (no `negative_prompt`/`do_classifier_free_guidance`; returns `(prompt_embeds, prompt_embeds_mask)`),
  so the inherited `_rplus_prepare_prompt_embeds` would raise `TypeError`. It is overridden in
  `DiffusersKreaBackend`. Only 1 core helper diverges so far — below the Tier-B "≥3 helpers"
  escalation threshold (Checkpoint ②), so the minimal-subclass approach holds.
- **VL img2img has no `__call__`/`encode_prompt` image argument.** There is no `Krea2Img2ImgPipeline`,
  but `QwenImageImg2ImgPipeline` / `QwenImageEditPipeline` exist and are the likely route for
  Krea2 image conditioning (shared Qwen-image VAE/encoder). WP-5's img2img agent should target the
  Qwen-image edit pipeline; the `context_image` field + `_prepare_krea_image_conditioning` hook are
  in place awaiting that wiring.

## Remaining — GPU-gated (WP-0 gate + downstream)

These require a CUDA box, the real Krea2 weights/config, and a diffusers build with the Krea2
classes (`>=0.39.0.dev0`). They are explicitly out of reach in this environment:

1. **WP-0 spike (blocks all runtime claims):**
   - Resolve diffusers coexistence: confirm one build serves both Z-Image and Krea2 (option a), or
     vendor Krea2 modules (option b), or process-isolate (option c). The repo currently pins
     `diffusers>=0.36.0`; the Krea classes are imported lazily with a clear error until this lands.
   - Confirm the duck-typed surface (`Krea2Pipeline.transformer(...)` signature, `encode_prompt`,
     `vae_scale_factor`, img2img variant, scheduler flow-shift).
   - Confirm the `Qwen3VLModel` image-conditioning argument name (linchpin for WP-5 wiring).
   - Confirm an fp8 load path for `Krea2Transformer2DModel`.
   - Fill in the real `config/` files (scheduler/tokenizer/transformer/vae/text_encoder) — only
     `model_index.json` is scaffolded.
2. **WP-3/WP-5 runtime:** wire `_prepare_krea_image_conditioning` into the generate/img2img call
   once the encode argument is known; end-to-end `generate()` on a real pack (bf16 + fp8).
3. **WP-6 tiering:** recalibrate offload/backend selection for the 12B model; bf16-vs-fp8 choice
   per tier; decide whether `constrained` runs Krea2 or fails gracefully.
4. **WP-10 regression:** run the full Z-Image suite on the (possibly bumped) diffusers env and
   confirm functional/perceptual equivalence at a fixed seed.

## How to verify what landed (no GPU)

```bash
# From repo root, in an env with torch + runtime deps installed:
pytest tests/test_krea_registry_dispatch.py tests/test_runtime_model_switch.py -q
# Registry-only check needs just pyyaml:
python -c "from app.core.model_registry.model_pack import ALLOWED_ARCHITECTURES; assert 'krea2_turbo' in ALLOWED_ARCHITECTURES"
```
