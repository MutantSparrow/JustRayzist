# Krea2-Turbo Integration — Implementation Status

Tracks what landed on `feat/krea2-support`.

**End-to-end verified on RTX 4080 (16GB) + RTX 4090 (24GB), through the full web app.** The
`Krea2_Turbo` fp8 pack generates correct images both directly through the backend and through the
running web server:

- **Backend path:** `create_backend` → `Fp8KreaBackend` → `build_fp8_krea_pipeline` →
  `backend.generate()`. 1024×1024, 8 steps, cfg 0.0, "a red fox in fresh snow" → correct image
  (~230 s on 16 GB constrained tier; ~20 s on 24 GB high tier with the optimization stack).
  Smoke script: `scripts/dev_krea2_smoke.py`.
- **Web/API path:** `StartWeb` → select `Krea2_Turbo` → `POST /generate` → image saved to the
  gallery. Response confirmed `model_pack=Krea2_Turbo`, `backend=fp8_krea`, `device=cuda`,
  `execution_mode=sequential_offload`, 8 steps, cfg 0.0.

Weights are gitignored and provisioned on demand — see "Provisioning" below. The operator supplies
their own finetuned Krea2-Turbo checkpoint (licensing handled off-repo); the app only knows the
disk layout it expects.

## Native/ComfyUI fp8 checkpoint support

The `Krea2_Turbo` pack loads ComfyUI-native fp8 weights and converts their key layout to the
`diffusers` expectations. `app/core/pipeline_factory/krea_comfy_convert.py` handles each component
— same pattern as the existing Z-Image `_convert_prefixed_fused_zimage_state_dict`:

- **Transformer** (`blocks.*` → `transformer_blocks.*`, `first/last/tmlp/tproj/txtmlp` globals,
  `mod.lin (6*H,)` → `scale_shift_table (6,H)`; drops the 2 extra `last.up/last.down` tensors some
  fp8 repacks carry). Mapping derived by aligning against the official Krea 2 Turbo layout (native
  format, 430 keys) and the diffusers model; validated by an exact key+shape match and a full
  12.82B-param CPU load.
- **VAE** — reuses diffusers' own `convert_wan_vae_to_diffusers` (the Qwen-image VAE shares the Wan
  VAE layout); exact 194/194 match, strict load.
- **Qwen3VL encoder** — ComfyUI scaled-fp8 (`.weight` fp8 × scalar `.weight_scale`, plus a
  `.comfy_quant` JSON-metadata tensor that is dropped) with a `model.* → language_model.*` /
  `model.visual.* → visual.*` prefix remap; exact 713/713 match.

The pack config dirs (`config/{scheduler,tokenizer,transformer,vae,text_encoder}`) hold the
official diffusers configs for the Krea 2 Turbo architecture and are committed. See "Provisioning"
for weight fetching.

On a 16GB card the builder sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (unless already
set) before CUDA init to avoid fragmentation OOMs; the `constrained` profile's sequential CPU
offload keeps the fp8 transformer + bf16 encoder + VAE within budget. A manual smoke test lives at
`scripts/dev_krea2_smoke.py`.

## Provisioning (installers / launchers)

Only the ~18 GB weights are fetched (config dirs are committed). Krea is wired into the same
asset machinery as the optional `Rayzist_qwen3_4b_fp8` pack, opt-in and drop-in:

- `scripts/portable/fetch_model_assets.py` — `OPTIONAL_KREA2_ASSETS` lists three weight files
  keyed by a single `_KREA2_FINETUNE_REPO` constant (currently the placeholder
  `MutantSparrow/krea2-placeholder`; operators replace this with their own finetune repo id and
  fill in the per-file SHA256s — see the `TODO(krea2 finetune)` comment).
- `scripts/fetch_model_assets.ps1` — `-IncludeKrea2` switch forwards to the portable helper.
- `scripts/fetch_krea2_assets.ps1` — two-liner wrapper (`fetch_model_assets.ps1 -IncludeKrea2`).
- `StartWeb.bat` (`:ensure_krea2_pack_assets`) and `scripts/portable/start_web.py`
  (`ensure_pack_assets`, used by `StartWeb.sh`) — on selecting `Krea2_Turbo`, prompt to confirm
  the ~18 GB download (size-based safety valve, no licensing framing); non-interactive launches
  print the manual fetch command instead of silently downloading 18 GB.

Fetch manually with:
```
scripts/fetch_model_assets.ps1 -IncludeKrea2                    # Windows
python scripts/portable/fetch_model_assets.py --include-krea2   # any platform
```

## Coexistence

Implemented against `diffusers==0.39.0` (a normal PyPI release), which ships both the `Krea2*` and
`ZImage*` classes — the plan's §6.1 option (a). The full Z-Image regression suite passes on 0.39.0
(**344 passed / 1 skipped**, including the Krea unit tests), so Krea2 support does not disturb the
Z-Image path.

## Landed

| WP | Area | Files |
|----|------|-------|
| WP-1 | Registry + dispatch | `app/core/model_registry/model_pack.py` (`krea2_turbo` architecture), `app/core/backends/__init__.py` (`diffusers_krea`/`fp8_krea` dispatch, lazy import, `SUPPORTED_BACKENDS`) |
| WP-2 | Pipeline builders | `app/core/pipeline_factory/krea.py` (`build_krea_pipeline`, `build_fp8_krea_pipeline`) + `krea_comfy_convert.py` (ComfyUI→diffusers converters), exported from `pipeline_factory/__init__.py` |
| WP-3 | Backends | `app/core/backends/diffusers_krea.py` (`DiffusersKreaBackend`, `Fp8KreaBackend`, R+ `encode_prompt` override); turbo-default seam (`_default_steps`/`_default_guidance_scale`) added to `diffusers_zimage.py` |
| WP-5 | VL image conditioning (style ref) | `app/core/worker/types.py` (`GenerationRequest.context_image`); `_build_generate_pipe_kwargs` seam in Z-Image + Krea override that runs Qwen3VL with vision tokens + `pixel_values`, taps `pipe.text_encoder_select_layers`, and passes the result as `prompt_embeds`/`prompt_embeds_mask` to `Krea2Pipeline`. Mirrors the ComfyUI style-ref workflow. `tests/test_krea_vl_context_image.py` covers the plumbing weightlessly (kwarg substitution, prompt dropped when embeds set, Z-Image path unchanged). |
| WP-7 | Runtime model switch | `app/core/worker/session.py` (`GenerationSession.switch_model_pack`, tier-adaptive keep-resident/unload, `recycle` releases resident cache) |
| WP-4 | Real pack | `models/packs/Krea2_Turbo/` — `modelpack.yaml` (fp8 component paths) + official diffusers configs |
| WP-6 | Tiering (fp8 path + per-pack thresholds) | fp8 backend runs on 16GB via `constrained` sequential offload; on ≥24GB Krea2 auto-selects `high` (no offload). Per-pack `resource_tier_thresholds` in `modelpack.yaml` override the global `RUNTIME_PROFILES[tier].min_free_vram_gb` bar per model (Krea2 sets high=22 / balanced=14 / constrained=4 based on the RTX 4090 bench: ~5.8× faster no-offload vs sequential-offload for the 12B fp8 transformer). Wired through `ResourceTierController.current_for(pack)` and `GenerationSession`'s pack-switch path. |
| — | Runtime optimization framework | `app/core/pipeline_factory/optimizations.py` — post-load applier for `torch.compile`, `torchao` fp8 dynamic quant (Ada+ only), SageAttention (Turing+), TF32 (Ampere+), VAE tiling, and persistent Inductor cache. Each is capability-gated at apply time so a pack requesting an option the local GPU cannot support degrades gracefully with a log line. Krea2 pack enables compile+sage+tf32+tiling; Z-Image pack enables sage+tf32+tiling (compile off because its `_cast` seam recompile-storms on first gen). fp8 quant kept OFF pending an aten `abs_cuda(Float8_e4m3fn)` kernel — bumping to torch 2.11 / torchao 0.17 was not enough. Bench (RTX 4090 warm): Krea2 high tier 32.7 s → 21.9 s (~1.5× on top of the tier win). |
| WP-8 | Tests | `tests/test_krea_registry_dispatch.py`, `test_runtime_model_switch.py`, `test_krea_comfy_convert.py`, Krea cases in `test_fetch_model_assets.py` |
| WP-9 | Docs | `models/packs/README.md`, pack READMEs, this file |
| — | Provisioning | installers/launchers fetch Krea weights (opt-in, size-gated) — see "Provisioning" |

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
- **VL image conditioning is done via prompt_embeds, not a pipeline image arg.** `Krea2Pipeline`
  itself is text2img — no `image=` on `__call__` or `encode_prompt`, and no `Krea2Img2ImgPipeline`
  exists. `QwenImageEditPipeline` is a *different* model (Qwen2.5-VL + `QwenImageTransformer2DModel`)
  and is not a route for Krea2 weights. The Krea backend now encodes a vision-token chat template
  through Qwen3VL, taps the pipeline's `text_encoder_select_layers`, and feeds the result to
  `Krea2Pipeline` as `prompt_embeds`+`prompt_embeds_mask` — the same pattern ComfyUI's Krea2
  style-ref workflow uses.

## Remaining

WP-0 is resolved, both backend and web-app generation are proven on RTX 4080 + 4090, the runtime
optimization framework is landed, and the pyproject deps are bumped. What's left:

1. **WP-5 VL image conditioning — UI wiring.** Field + encode path are landed and GPU-verified
   (fox → wolf style-ref test 2026-07-16). The existing "reference image" UI drawer currently maps
   to the img2img flow, not to `GenerationRequest.context_image`. Route it — Krea2 packs should
   pass the reference through the VL encode; Z-Image packs continue img2img.
2. **Parity on real weights.** LoRA compose, upscale/refine, R+/procedural, and prompt-enhance have
   not been exercised on Krea (R+ has a code override but is unrun).
3. **bf16 (`diffusers_krea`) backend.** Implemented but unrun — 12B bf16 needs ≥24GB card in `high`
   tier.
4. **Native fp8 compute.** `torchao.Float8DynamicActivationFloat8WeightConfig` still fails on
   Windows/torch 2.11 with `"abs_cuda" not implemented for 'Float8_e4m3fn'`. Either wait for the
   aten kernel or route through a bf16→fp8 conversion at load. Unblocks another ~1.5× on Krea2.
5. **WP-10 release.** Branch merge to `main` / final review.

## How to verify

```bash
# Weightless (torch + runtime deps, no GPU):
pytest tests/test_krea_registry_dispatch.py tests/test_runtime_model_switch.py \
       tests/test_krea_comfy_convert.py tests/test_fetch_model_assets.py -q

# GPU (with weights fetched):
python scripts/dev_krea2_smoke.py                 # backend path
# or: StartWeb → select Krea2_Turbo → generate    # web/API path
```
