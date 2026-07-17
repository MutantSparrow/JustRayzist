# Krea2_Turbo local config directory

This directory holds the local diffusers-style config used by
`Krea2Pipeline.from_pretrained("./config", local_files_only=True)`. Only `model_index.json` and
the small per-component config files are committed as scaffolds; larger sidecars (tokenizer +
Qwen3VL processor configs) are fetched at setup — see `../weights/README.md`.

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
    ├── config.json                     # Qwen3VLModel config (vision-language)
    ├── generation_config.json
    ├── preprocessor_config.json        # fetched: Qwen3VL image processor sidecar (WP-5)
    ├── chat_template.json              # fetched: Qwen3VL chat template with vision tokens (WP-5)
    ├── video_preprocessor_config.json  # fetched: Qwen3VL video processor sidecar (WP-5)
    └── model.safetensors               # staged by the pipeline builder from the pack component
```

The three `*_preprocessor_config.json` / `chat_template.json` files come from
`Qwen/Qwen3-VL-4B-Instruct` (Apache-2.0) and are fetched by the same asset script as the weights.
They are required for the WP-5 style-reference path so `AutoProcessor.from_pretrained(...)` can
build a multimodal processor that preprocesses a `context_image` before Qwen3VL encoding.

Fetch these before first run — see `scripts/fetch_krea2_assets.ps1` (project root) and
`../weights/README.md`.

> The text encoder is `Qwen3VLModel` (vision-language), not the plain `Qwen3Model` used by Z-Image.
> This is what enables the optional style-reference conditioning
> (`GenerationRequest.context_image`; see `docs/KREA2_IMPLEMENTATION_STATUS.md`) — a reference
> image is jointly encoded with the prompt through Qwen3VL and passed to `Krea2Pipeline` as
> `prompt_embeds`.
