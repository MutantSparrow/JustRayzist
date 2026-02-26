# Model Packs

Each model pack must be in its own folder and include `modelpack.yaml`.

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

Notes:
- Local files only. Remote URLs are rejected.
- Declared `format` must match file extension.
- Missing files fail validation.
- GGUF component loading is supported for `transformer`, `vae`, and `text_encoder`.
- For GGUF `text_encoder`, the pack must include local `pipeline_config_dir/text_encoder/*` config files.
- A ready-to-copy template is available at `models/packs/modelpack.yaml.example`.
- A pack-specific starter template is available at `models/packs/Rayzist_bf16/modelpack.yaml.template`.

Activation steps:
1. Copy `models/packs/Rayzist_bf16/modelpack.yaml.template` to `models/packs/Rayzist_bf16/modelpack.yaml`.
2. Put your real weights in `models/packs/Rayzist_bf16/weights/`.
3. Put local diffusers config files in `models/packs/Rayzist_bf16/config/`.
4. Run `python -m app.cli.main validate-models`.
