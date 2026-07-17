# Model Packs

Each model pack lives in its own folder under `models/packs/<pack_name>/` and must include `modelpack.yaml`.

Example layout:

```text
models/packs/my_z_turbo_pack/
  modelpack.yaml
  weights/
    transformer.safetensors
    vae.safetensors
    text_encoder.gguf
  config/
    model_index.json
```

Example `modelpack.yaml`:

```yaml
name: my_z_turbo_pack
user_visible: true
enabled: true
architecture: z_image_turbo
backend_preference:
  - diffusers
pipeline_config_dir: ./config
components:
  transformer:
    path: ./weights/transformer.safetensors
    format: safetensors
  vae:
    path: ./weights/vae.safetensors
    format: safetensors
  text_encoder:
    path: ./weights/text_encoder.gguf
    format: gguf
required_configs:
  - ./config/model_index.json
```

## Required fields

- `name`
- `architecture`
- `backend_preference`
- `pipeline_config_dir`
- `components`
- `required_configs`

## Public vs hidden packs

- `user_visible` is optional and defaults to `true`.
- `enabled` is optional and defaults to `true`.
- Normal launcher and API pack lists only show packs where `user_visible: true` and `enabled: true`.
- Packs can stay installed but unavailable to normal users by setting `enabled: false`.
- Hidden or disabled packs are still valid for engineering and benchmark workflows when explicitly named.
- The bundled app ships with `Rayzist_bf16` as the default public enabled pack.
- Setup can optionally create `Rayzist_qwen3_4b_fp8` from [MutantSparrow/qwen3_4b_Rayzist_v1.0_fp8](https://huggingface.co/MutantSparrow/qwen3_4b_Rayzist_v1.0_fp8); it reuses `Rayzist_bf16` transformer and VAE weights and swaps only the text encoder.
- Scaled FP8 text-encoder tensors are converted to BF16 at runtime before loading.
- Derived FP8 storage is an internal runtime strategy; it does not provide native FP8 inference.
- `GET /model-packs`, `StartWeb.bat`, and `StartWeb.sh` show only public enabled packs.
- Install-time asset fetch provisions the bundled `Rayzist_bf16` pack and can optionally fetch the `Rayzist_qwen3_4b_fp8` encoder pack.

## Supported architectures

Two model families are supported. `architecture` in a pack selects which:

- `z_image_turbo` — the default Rayzist Z-Image Turbo family (`diffusers` / `fp8_zimage` backends).
- `krea2_turbo` — Krea2-Turbo, a 12B distilled flow-matching DiT (`diffusers_krea` / `fp8_krea`
  backends). Near-sibling of Z-Image: same `FlowMatchEulerDiscreteScheduler`, Qwen-family text
  encoder + VAE. Differs in transformer class (`Krea2Transformer2DModel`) and pipeline
  (`Krea2Pipeline`), which require diffusers `>=0.39.0.dev0`. See
  `models/packs/Krea2_Turbo/` for the pack scaffold and `JustRayzist-Krea.md` for the full plan.

Krea2 notes:

- Its text encoder is `Qwen3VLModel` (vision-language), enabling optional style-reference
  conditioning via `GenerationRequest.context_image`: a reference image is jointly encoded with
  the prompt through Qwen3VL, and the resulting hidden states are passed to `Krea2Pipeline` as
  `prompt_embeds` (the same pattern ComfyUI's Krea2 style-ref workflow uses).
- The 12B bf16 transformer is ~24GB; list `fp8_krea` first in `backend_preference` so limited-VRAM
  cards use the fp8 path and fall back to bf16.
- The runtime can switch between families without a restart (see the model switch in the worker
  session); on high-VRAM tiers the previous family is kept resident for instant switch-back, and on
  tighter tiers it is released before the next loads so two large models are never resident at once.
- Krea2 weights are governed by the **Krea 2 Community License** (distinct from Z-Image); fetching
  is opt-in and license-gated. The pack's config dirs are committed; only the weights are fetched:
  `scripts/fetch_model_assets.ps1 -IncludeKrea2 -AcceptKrea2License` (Windows) or
  `python scripts/portable/fetch_model_assets.py --include-krea2 --accept-krea2-license` (any
  platform). `StartWeb` prompts to fetch on first selection of the `Krea2_Turbo` pack.

## Supported component formats

- `safetensors`
- `gguf`

GGUF component loading is supported for `transformer`, `vae`, and `text_encoder`.

## Per-pack runtime optimizations

Each pack can opt into post-load runtime optimizations via a top-level `optimizations` block.
Every option is capability-gated at apply time (see
`app/core/pipeline_factory/optimizations.py`), so a manifest requesting an option the local GPU
cannot support degrades gracefully with a log line rather than crashing.

```yaml
optimizations:
  torch_compile:
    enabled: true          # universal CUDA (Turing+)
    mode: default          # or "reduce-overhead" / "max-autotune"
  fp8_quantization:
    enabled: false         # Ada+ only (sm_89), currently blocked on aten fp8 kernel
    scope: transformer     # or "transformer+text_encoder"
  sage_attention:
    enabled: true          # Turing+ (sm_75+) via a shim on F.scaled_dot_product_attention
  tf32:
    enabled: true          # Ampere+ (sm_80+), no-op on Turing/Volta
  vae_tiling:
    enabled: true          # universal — vae.enable_tiling() + enable_slicing()
```

A boolean shortcut is accepted (`torch_compile: true` == `{enabled: true}` with defaults).
Unknown keys are rejected at pack-load time.

The framework also sets `TORCHINDUCTOR_CACHE_DIR` to `.build/inductor` (override with
`JUSTRAYZIST_INDUCTOR_CACHE_DIR`) so torch.compile artifacts persist across sessions — the second
run of the same pack + shape re-uses the cache and skips most of the JIT warmup.

An escape hatch `JUSTRAYZIST_DISABLE_OPTIMIZATIONS=1` turns every option off at runtime without
editing manifests — useful when debugging suspected optimization regressions.

## Per-pack resource tier thresholds

By default the runtime tier (`high` / `balanced` / `constrained`) is picked by comparing free VRAM
against `RUNTIME_PROFILES[tier].min_free_vram_gb` in `app/config/profiles.py`. A pack can override
these thresholds for its own tier selection via the top-level `resource_tier_thresholds` block:

```yaml
resource_tier_thresholds:
  high: 22        # min free-VRAM (GB) to pick `high` for this pack
  balanced: 14
  constrained: 4
```

Absent keys fall back to the global defaults. This lets a large model (e.g. Krea2's 12B fp8 DiT +
Qwen3VL encoder) demand more free VRAM to skip sequential CPU offload than a smaller Z-Image pack
would need — the RTX 4090 bench measured ~5.8× faster generation for Krea2 when `high` (no
offload) is selected. Any user/env tier override still wins.

## Optional runtime storage hints

Component entries may include optional advanced runtime hints:

- `storage_mode`
- `storage_dtype`
- `compute_dtype`

These are engineering/runtime controls, not normal end-user pack requirements.
The main current use is transformer-only layerwise FP8 storage with BF16 compute, for example:

```yaml
components:
  transformer:
    path: ./weights/transformer.safetensors
    format: safetensors
    storage_mode: layerwise
    storage_dtype: fp8_e4m3fn
    compute_dtype: bfloat16
```

## Derived FP8 storage

`fp8_storage` is now a derived runtime strategy, not a normal on-disk user pack.

- Users should not create duplicate packs such as `Rayzist_fp8_storage`.
- In constrained conditions, the runtime may derive an internal variant such as `<base>__auto_fp8_storage`.
- That derived variant reuses the same base weights and only changes runtime storage behavior.
- Derived aliases are intended for engineering telemetry and benchmark workflows.

## Minimal compatible packs and donor completion

Compatible minimal packs can omit some shared components and rely on `Rayzist_bf16` as the donor pack for completion.

Current donor completion behavior is intended for compatible `z_image_turbo` packs where the user supplies the model-specific transformer and reuses the canonical config / VAE / text encoder from `Rayzist_bf16`.

Practical guidance:

- include your own transformer path
- keep local config files when you need pack-specific config
- if a compatible pack is missing VAE, text encoder, or shared config, the runtime may resolve those from `Rayzist_bf16`
- constrained mode may still derive internal FP8 storage from that resolved runtime pack when the transformer format is compatible

## Validation rules

- Local files only. Remote URLs are rejected.
- Declared `format` must match file extension.
- Missing files fail validation.
- `pipeline_config_dir` and every `required_configs` path must exist.
- `user_visible`, when present, must be a boolean.
- `enabled`, when present, must be a boolean.

## Templates

- Generic template: `models/packs/modelpack.yaml.example`
- Pack-specific starter: `models/packs/Rayzist_bf16/modelpack.yaml.template`

Typical activation steps:

1. Copy `models/packs/modelpack.yaml.example` into a new pack folder.
2. Point the component paths at your local weights.
3. Add the required local config files.
4. Run `python -m app.cli.main validate-models`.
5. Use `python -m app.cli.main validate-models --all` when you also want to validate disabled packs.
