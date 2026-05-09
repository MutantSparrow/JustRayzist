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

## Supported component formats

- `safetensors`
- `gguf`

GGUF component loading is supported for `transformer`, `vae`, and `text_encoder`.

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
