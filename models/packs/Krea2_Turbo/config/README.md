# Krea2_Turbo local config directory

This directory holds the local diffusers-style config used by
`Krea2Pipeline.from_pretrained("./config", local_files_only=True)`. Only `model_index.json` is
committed as a scaffold; the remaining config files are fetched from the Krea2-Turbo model repo
(they are not redistributed here — see the Krea 2 Community License note in `../weights/README.md`).

Required structure (mirrors the Z-Image `Rayzist_bf16/config` layout):

```
config/
├── model_index.json              # committed scaffold (Krea2Pipeline component map)
├── scheduler/
│   └── scheduler_config.json     # FlowMatchEulerDiscreteScheduler config
├── tokenizer/
│   ├── tokenizer.json            # Qwen2Tokenizer
│   ├── tokenizer_config.json
│   ├── vocab.json
│   └── merges.txt
├── transformer/
│   └── config.json               # Krea2Transformer2DModel config (patch size 2, 12B)
├── vae/
│   └── config.json               # AutoencoderKLQwenImage config
└── text_encoder/
    ├── config.json               # Qwen3VLModel config (vision-language)
    ├── generation_config.json
    └── model.safetensors         # staged by the pipeline builder from the pack component
```

Fetch these from the Krea2-Turbo repo before first run — see
`scripts/fetch_krea2_assets.ps1` (project root) and `../weights/README.md`.

> The text encoder is `Qwen3VLModel` (vision-language), not the plain `Qwen3Model` used by Z-Image.
> This is what enables the optional image+text joint conditioning for img2img
> (`GenerationRequest.context_image`; see JustRayzist-Krea.md §4 / WP-5).
