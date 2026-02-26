# JustRayzist - Development Roadmap and Persistent TODO

## Purpose
This file is the authoritative roadmap and working TODO for building the local/offline image generation app.
If development is paused, resumed, or transferred, continue from this document first.

Repository layout note (`2026-02-26`):
- App workspace root: `./JustRayzist`
- Packaging output root: `./dist`
- Run app/scripts commands from `./JustRayzist` unless explicitly documented otherwise.

This app must:
- Run local inference on NVIDIA GPUs (12GB to 24GB VRAM) with memory-efficient behavior.
- Load models from local files only (no runtime internet).
- Support local model components in `.safetensors` and `.gguf` formats (Z-Image Turbo target architecture).
- Provide CLI (headless) and minimal web UI.
- Save generated images as PNG with metadata.
- Be packageable as a portable offline application.

---

## Ground Rules (Non-Negotiable Requirements)
1. Offline runtime:
- No model/network downloads during runtime.
- Enforce offline mode using environment/runtime flags.
- All required artifacts must be local before running generation.

2. Stability and memory behavior:
- Support repeated generations without VRAM creep over long sessions.
- Detect and mitigate allocator fragmentation and memory drift.
- Provide robust fallback behavior for 12GB cards via CPU/RAM offload.

3. Local model loading:
- Accept and validate local paths for model components.
- Handle mixed component formats when possible (`.safetensors` + `.gguf`).
- Fail fast with clear validation errors if a component is missing/incompatible.

4. User-facing generation controls:
- Prompt and resolution are user-configurable.
- Steps/CFG are fixed defaults in v1.
- Scheduler/sampler supports a simple toggle: `Euler` (default) or `DPM`.

5. Output requirements:
- Save PNG locally.
- Include metadata: `timestamp`, `prompt`, `application_name`, `application_version`.

---

## Technical Direction
### Core stack
- Python 3.11
- PyTorch (CUDA)
- Diffusers + Transformers + Accelerate
- `safetensors`, `gguf`
- FastAPI (backend + local web server)
- Typer (CLI)
- SQLite (metadata index)
- Pillow (PNG metadata writing)

### Inference strategy
- Primary backend: Diffusers Z-Image pipeline.
- GGUF support handled via model-level loading/injection.
- Keep a backend abstraction to allow fallback adapters if a specific GGUF layout is not compatible.

### Runtime profiles
- `high` (24GB): minimal offload, best speed.
- `balanced` (16GB): moderate offload, stable defaults.
- `constrained` (12GB): aggressive offload + memory safeguards.

---

## Deliverables
1. Working local CLI generation command.
2. Working local web UI with:
- Prompt + resolution input.
- Generate action.
- Masonry-style gallery.
- Basic metadata/prompt filtering.
3. Local PNG outputs with metadata.
4. Stability-tested memory behavior over repeated generations.
5. Portable offline package and launch scripts.

---

## Proposed Project Structure
```text
JustZit4/
  JustRayzist.md
  app/
    api/
    cli/
    ui/
    core/
      backends/
      memory/
      model_registry/
      pipeline_factory/
      worker/
    storage/
    config/
  models/
    packs/
  outputs/
  data/
    gallery.db
  tests/
  scripts/
  launch/
```

---

## Phase Plan
## Phase 0 - Bootstrapping and Constraints Lock-In
Goal: establish repo skeleton, pinned dependencies, runtime policy.

Tasks:
- Create directory structure.
- Define dependency lock strategy and versions.
- Add app versioning strategy (`app_name`, `app_version` constants).
- Add global config file(s): paths, defaults, runtime profile definitions.
- Add logging baseline with per-job IDs.

Exit criteria:
- Project starts cleanly.
- Config loads and validates.
- Runtime reports active profile and paths.

---

## Phase 1 - Inference Core (CLI First)
Goal: generate a local PNG from local model files only.

Tasks:
- Implement model registry and model pack schema.
- Implement local component validation for safetensors/GGUF paths.
- Implement pipeline factory for Z-Image Turbo.
- Implement CLI command:
  - Input: prompt, width, height, optional output dir.
  - Fixed defaults for sampler/scheduler/steps/CFG in v1.
- Implement PNG writer with metadata fields.
- Add runtime offline guards and local-only flags.

Exit criteria:
- CLI generates 1024x1024 image successfully from local files.
- Output PNG has required metadata.
- No runtime network access occurs.

---

## Phase 2 - Memory Management and Stability Hardening
Goal: reliable repeated generation without memory creep.

Tasks:
- Add memory telemetry per generation:
  - `allocated`, `reserved`, peak metrics.
- Add profile-specific offload configuration (12/16/24GB).
- Add warmup routine for stable allocator behavior.
- Add drift detection thresholds and worker recycle strategy.
- Add long-run soak test command (`N` repeated generations).

Exit criteria:
- Multiple sequential generations remain stable under each profile target.
- Memory creep remains inside defined thresholds.
- Recovery/recycle path works when threshold is exceeded.

---

## Phase 3 - Web UI (Minimal and Functional)
Goal: local web interface with generation + gallery.

Tasks:
- Add FastAPI endpoints:
  - `POST /generate`
  - `GET /images`
  - `GET /images/{id or filename}`
- Build minimal UI:
  - Prompt input.
  - Resolution selection.
  - Generate button.
  - Masonry gallery view.
  - Prompt keyword filter.
- Hook gallery indexing to SQLite metadata store.

Exit criteria:
- User can generate and browse results in browser.
- Prompt filtering works.
- CLI and web share same inference engine.

## Web UI Design Contract (Pinned for Current Build)
Use these requirements as the source of truth for UI refactors and regression checks.

1. Top bar (always visible/pinned):
- Left-to-right order: app logo (`/img/logo.png`), prompt area, generate icon button.
- Prompt box is primary generation input.
- Prompt box must be resizable by the user.
- Prompt area includes an inline settings button inside the prompt shell.
- Generate uses `/img/generate.svg` and matches prompt-input height.
- Settings button opens a popup with:
  - Resolution dropdown grouped and sorted by ratio:
    - `2:3`: `832x1248`, `1024x1536`
    - `3:4`: `864x1152`, `1104x1472`
    - `9:16`: `720x1280`, `864x1536`
    - `1:1`: `1024x1024 (default)`, `1280x1280`, `1536x1536`
  - Orientation toggle: `PORTRAIT` / `LANDSCAPE`
  - `FREEZE SEED` toggle:
    - seed is randomized for generation and saved to PNG metadata
    - when freeze is ON, seed remains constant until freeze is toggled OFF
  - `SCHEDULER/SAMPLER` toggle:
    - `OFF` => Euler
    - `ON` => DPM
  - `PROMPT ENHANCER` toggle:
    - `OFF` => use raw prompt
    - `ON` => rewrite prompt via already-loaded text encoder before image generation
  - `DELETE GALLERY` button requiring manual typed confirmation `DELETE`

2. Gallery area (below pinned top bar):
- Top controls: prompt keyword filter input + reverse-order toggle.
- Masonry layout with 4 columns on desktop, date-sorted with reversible order.
- Image tiles are sharp-edged rectangles with small inter-tile gaps.
- On hover tile shows: date, prompt prefix, resolution, `Download`, and magenta `Upscale` button.
- Upscale jobs must use the same gallery placeholder behavior as generation jobs while queued/running.

3. Full-view image mode:
- Clicking a tile opens fullscreen viewer.
- Top summary bar actions order: `Use Prompt`, `Copy Prompt`, `Upscale`, `Download`, `Close`.
- For upscaled items, metadata line is:
  - `Upscaled from <source filename> | HOLD TO SEE ORIGINAL | <upscaled resolution>`
- Holding `HOLD TO SEE ORIGINAL` temporarily swaps the main preview to the source image for side-by-side mental compare.
- Main area shows image with wheel zoom support.
- Zoom range: `x0.01` to `x10`.
- Clicking and dragging in fullscreen moves the image viewport.
- `Delete` or `Backspace` in fullscreen opens a confirmation modal (`Yes`/`No`) for image deletion.
- Show a subtle single-line keyboard hint below the top bar separator.

4. Visual style:
- Very dark gray background and black panels.
- Flat material-like controls with saturated lime/magenta highlights.
- No rounded corners.

---

## Phase 4 - Packaging and Offline Portability
Goal: zero-install portable distribution.

Tasks:
- Bundle runtime + dependencies + app code + launchers.
- Bundle model pack conventions and example config.
- Add startup checks to validate offline completeness.
- Test in clean/offline environment.

Exit criteria:
- App runs without internet.
- App runs without pip install on target machine.
- CLI and web launchers work in packaged form.

---

## Model Pack Specification (v1)
Each local model bundle should include a descriptor (example: `models/packs/<pack_name>/modelpack.yaml`) with:
- `architecture`: `z_image_turbo`
- `backend_preference`: `diffusers` (or fallback list)
- `components`:
  - transformer/checkpoint path + format
  - VAE path + format
  - encoder path + format
- `required_configs`: local config files needed for loader resolution
- `precision_profile_overrides` (optional)

Rules:
- Paths must resolve locally.
- No remote identifiers in production runtime path.
- Validate presence, extension, and readable permissions at startup.

---

## Performance Targets (Initial)
- Reference target: ~10s for one 1024x1024 generation on development hardware (ComfyUI-class expectation).
- Stable memory behavior over repeated runs (no unbounded growth).
- 12GB profile should prioritize reliability over speed.

Note:
- Establish explicit benchmark scripts and record results in `/tests` or `/scripts`.

---

## Quality Gates
Before moving phases forward:
1. Functional gate:
- Core functionality works end-to-end for that phase.

2. Stability gate:
- No major memory leak/drift beyond accepted limits.

3. Regression gate:
- Existing CLI flow still works after new changes.

4. Offline gate:
- No hidden network dependency introduced.

---

## Testing Strategy
1. Unit tests:
- Config validation.
- Model pack parser.
- Metadata writer.
- Path/local-only guards.

2. Integration tests:
- End-to-end generation from local pack.
- CLI + API generation path.
- DB/gallery index operations.

3. Soak tests:
- Repeated generation loops by profile.
- Monitor generation time and memory metrics.

4. Packaging tests:
- Clean-machine run.
- Offline run.

---

## Suspend/Resume Protocol
When pausing development:
1. Update this file:
- Current phase and task status.
- Open issues/blockers.
- Next exact action.

2. Record operational notes:
- Last known working command(s).
- Model pack used for validation.
- Hardware profile used for tests.

3. Leave repo in coherent state:
- No ambiguous partial refactors without notes.
- Mark TODOs with clear ownership and intent.

When resuming development:
1. Read this file first.
2. Confirm current phase checkpoint status.
3. Run quick smoke test (CLI generation).
4. Continue from `Next Action`.

---

## Current Status
- Phase: `Phase 3 - Web UI (Minimal and Functional)`
- Execution status: `In progress`
- Blockers: `None active`
- Next Action:
  1. Run one full end-to-end web generation smoke (`POST /generate`) on GPU profile `balanced` and verify gallery insertion in SQLite.
  2. Validate refreshed web layout in browser against the pinned design contract (top bar, settings popup, filter/reverse, hover overlay, fullscreen zoom + drag, freeze-seed behavior, delete-gallery behavior).
  3. Start Phase 4 packaging design: portable folder layout + launch scripts + offline completeness preflight.

---

## Active TODO Checklist
## Setup
- [x] Create folder skeleton.
- [x] Initialize Python project and dependency files.
- [x] Add config system and environment loading.
- [x] Add structured logging.

## Core Inference
- [x] Implement model pack schema and parser.
- [x] Implement local file validation and offline guards.
- [x] Implement Z-Image Turbo pipeline loader.
- [x] Implement GGUF component loader path.
- [x] Validate GGUF component loading with real local GGUF artifacts.
- [x] Implement first successful CLI image generation.

## Output + Metadata
- [x] Implement PNG writer.
- [x] Embed required metadata fields.
- [x] Implement filename/date conventions.

## Memory + Stability
- [x] Add runtime memory telemetry.
- [x] Add profile-based offload policies.
- [x] Add warmup and drift detection.
- [x] Add worker recycle policy.
- [x] Add soak test script.

## Web App
- [x] Implement FastAPI routes.
- [x] Implement minimal frontend.
- [x] Implement masonry gallery.
- [x] Implement prompt filter.
- [x] Connect SQLite metadata index.

## Packaging
- [ ] Define portable packaging format.
- [x] Build launcher scripts.
- [ ] Validate clean offline run.

## Documentation
- [x] Write usage docs (CLI + web).
- [x] Write model pack authoring guide.
- [x] Write troubleshooting notes (VRAM/offload/errors).

---

## Latest Validation Notes
- Date: `2026-02-22`
- Current default upscale checkpoint: `models/upscaler/2x_RealESRGAN_x2plus.pth` (compact checkpoint removed).
- Upscale baseline validation (`2026-02-25`, compact checkpoint removed from current workspace):
  - Baseline used prior compact checkpoint and remains for historical perf/reference notes only.
  - Fixed Real-ESRGAN Compact forward path to include nearest-neighbor base-image skip connection (`out += base`).
  - `high`: `1024x1024 -> 2048x2048`, `132ms`, `cuda/fp16`, tile `0`
  - `balanced`: `1024x1024 -> 2048x2048`, `53ms`, `cuda/fp16`, tile `640`
  - `constrained`: `1024x1024 -> 2048x2048`, `83ms`, `cuda/fp32`, tile `384`
  - Test outputs/artifacts from this run were cleaned from workspace.
- Upscaler architecture extension + benchmark (`2026-02-25`):
  - Added auto-detection/loading for Real-ESRGAN `RRDB` checkpoints (supports `2x_RealESRGAN_x2plus.pth`) in addition to compact `SRVGG`-style checkpoints.
  - Verified visual output for `2x_RealESRGAN_x2plus.pth` in all profiles (no residual-edge artifact, correct `2048x2048` output).
  - Benchmark (`5` measured runs per model/profile after warmup, source `outputs/_Upscale_test.png`):
    - `compact_500k`: high `47.6ms`, balanced `41.0ms`, constrained `64.6ms`
    - `realesrgan_x2plus`: high `266.2ms`, balanced `274.6ms`, constrained `450.4ms`
    - Peak VRAM (nvidia-smi): compact `3.63-3.92GB`, x2plus `4.98-6.68GB`
  - Raw benchmark artifact from this run was cleaned from workspace.
- Upscaler policy tuning (`2026-02-25`):
  - Added architecture-aware upscale policy so RRDB/x2plus uses tighter tiling in constrained profile:
    - RRDB high: tile `768`, balanced: `512`, constrained: `256` (overlap `16` in constrained).
  - Switched RRDB constrained precision to fp16-first path.
  - Sustained x2plus benchmark (`20` measured runs/profile) now shows constrained VRAM below balanced:
    - Peak VRAM (nvidia-smi): high `5.17GB`, balanced `4.17GB`, constrained `3.81GB`.
    - Avg latency: high `288.6ms`, balanced `218.35ms`, constrained `587.05ms`.
  - Raw benchmark artifact from this run was cleaned from workspace.
- Runtime bootstrap hardening (`2026-02-25`):
  - Root-cause found for launcher failure: `.venv` existed but was incomplete (no `pip` and no installed deps), while launcher selected `.venv` by path existence alone.
  - Launchers now resolve the first dependency-ready interpreter (runtime bundle, then `.venv`, then PATH `python`) instead of trusting path existence only.
  - Added self-healing in `scripts/bootstrap_env.ps1` (`ensurepip`, then `venv --clear` rebuild fallback) to prevent stale/incomplete `.venv` regression.
- Upscale + img2img refine (`2026-02-25`):
  - Added two-step generation path: `RealESRGAN x2plus` upscaling first, then Z-Image Turbo `img2img` refinement on the x2 frame.
  - New CLI command: `python -m app.cli.main upscale-refine ...` with defaults `strength=0.20`, `refine_steps=6`, and optional prompt enhancer.
  - Added profile-aware refine tiling defaults and OOM fallback to tiled refine:
    - high `tile=0` (full frame), balanced `tile=1280`, constrained `tile=896` (overlap default `64`).
  - Full-size smoke (`outputs/_Upscale_test.png -> 2048x2048`) verified visually in all profiles with correct non-edge output:
    - high `122059ms` (upscale `436ms`, refine `121302ms`, tile `0`)
    - balanced `25465ms` (upscale `382ms`, refine `24612ms`, tile `1280`)
    - constrained `47918ms` (upscale `649ms`, refine `46941ms`, tile `896`)
- Active pack: `models/packs/Rayzist_bf16/modelpack.yaml`
- Working validation commands:
  1. `python -m app.cli.main validate-models`
  2. `python -m pytest -q tests/test_settings_smoke.py tests/test_model_pack_validation.py tests/test_soak_report.py tests/test_gallery_index.py tests/test_api_routes.py -p no:cacheprovider`
  3. `python -m app.cli.main serve --host 127.0.0.1 --port 37717 --profile balanced`
- Cleanup status:
  - Removed deprecated FP8 conversion utility and related FP8 report artifacts.
  - Removed deprecated ASCII preview generator/test assets; kept launcher splash asset used by `StartWeb.bat`.
  - Consolidated code comments to only runtime-relevant notes.
- CUDA verification:
  - PyTorch runtime: `2.10.0+cu128`
  - CUDA available: `True`
  - Device: `NVIDIA GeForce RTX 4090`
- Profile soak summary (`1024x1024`, 12 iterations each, warmup enabled):
  - `high` (`soak_20260221_220122_b6a7de9d`): p50 `22.11s`, p95 `45.00s`, drift `+2MB` flat, CUDA reserved `~24.49GB`.
  - `balanced` (`soak_20260221_220740_7f88f631`): p50 `9.00s`, p95 `9.36s`, drift `0..+2MB`, CUDA reserved `~0.029-0.031GB`.
  - `constrained` (`soak_20260221_221012_713d827d`): p50 `23.25s`, p95 `23.72s`, drift `-2MB` flat, CUDA reserved `~0.795GB`.
- Recycle threshold validation (`balanced`, drift threshold `1MB`):
  - Session `soak_20260221_221737_6d30e903` completed `4/4` iterations with `0` errors and `2` automatic recycle events.
  - Summary: avg `10.21s`, p50 `10.16s`, p95 `11.56s`, drift range `0..4MB`.
- Tuned soak default policy (profile-driven):
  - `high`: drift threshold `256MB`, recycle every `0`.
  - `balanced`: drift threshold `128MB`, recycle every `0`.
  - `constrained`: drift threshold `64MB`, recycle every `24`.
  - Verified by runtime output and metrics summary fields (`drift_threshold_mb`, `recycle_every`) in session `soak_20260221_231845_e39154e5`.
- Performance note:
  - `balanced` currently meets the stated ~10s/1024 reference target on this hardware.
- Verified PNG metadata keys:
  - `timestamp`, `prompt`, `application_name`, `application_version`
  - plus runtime metadata: `backend`, `device`, `model_pack`, `steps`, `guidance_scale`
- Verified metrics payload keys:
  - `duration_ms`, `cuda_memory_before`, `cuda_memory_after`, prompt/size/output path
  - `process_memory_before`, `process_memory_after`, `mode`, `iteration`, `memory_drift_mb`, `recycle_reason`
- Implementation note:
  - Added transformer conversion path for prefixed/fused checkpoints (`model.diffusion_model.*`, fused `attention.qkv.weight`) before Diffusers load.
  - Added GGUF loading path for VAE and text encoder components (text encoder uses local config + gguf file with AutoModel fallback loaders).
  - Added soak runner with warmup toggle, drift monitoring, and recycle policy (`app.cli.main soak`, `app.core.worker.session`).
  - Drift baseline now resets after each recycle event to keep drift measurements session-local.
  - Added soak report analyzer and CLI command (`app.storage.soak_report`, `app.cli.main soak-report`) with latency percentiles, drift trend, recycle counts, and error counts.
  - Added profile-aware soak policy defaults and CLI auto-resolution for drift threshold/recycle cadence (`app.config.profiles`, `app.cli.main soak`).
  - Added web inference service and API endpoints (`/generate`, `/images`, `/images/{filename}`, `/model-packs`) with shared core generation pipeline (`app.api.inference_service`, `app.api.main`).
  - Hardened gallery delete API (`DELETE /gallery`) to accept confirmation via body/query and clear indexed files + output image files.
  - Added SQLite gallery index with output-file sync and prompt filtering (`app.storage.gallery_index`, `data/gallery.db`).
  - Reworked web UI to pinned top bar + prompt resizable input + icon-based generate button + settings popup + filter/reverse toolbar + hover metadata overlays + fullscreen viewer zoom/drag (`x0.01..x10`) + freeze-seed control (`app/ui/index.html`, `app/ui/styles.css`, `app/ui/app.js`).
  - Addressed operator feedback: robust gallery wipe behavior, deterministic reverse-order rendering, and seed metadata persistence for web generations.
  - Added generation placeholder tile flow (`GENERATING...`) with black tile + spinner and requested aspect ratio during in-flight generation.
  - Expanded API error rendering in frontend so validation failures show readable field/message text instead of `[object Object]`.
  - Updated generation request validation to accept required portrait presets (dimension multiple-of-16 check).
  - Added web settings toggle `SCHEDULER/SAMPLER` (`OFF=Euler`, `ON=DPM`) and wired it through API payload into runtime scheduler selection before inference.
  - Added prompt-enhancer prepass option using loaded pipeline text encoder/tokenizer (`AutoTokenizer`/`AutoModelForCausalLM` pattern) before prompt encoding, with safe fallback to original prompt on failure.
  - Replaced topbar text brand with `/img/logo.png`, served via static `/img` mount, with CSS `max-height: 200px`.
  - Removed redundant local model artifacts and stale cache copies; kept only manifest-referenced weights and runtime-staged hardlinks.
  - Simplified text-encoder loading to single-source files at `config/text_encoder/model.safetensors`.
  - Added `StartWeb.bat` launcher to prompt runtime profile selection (`constrained`, `balanced`, `high`) before starting web server.
  - Performed development-artifact cleanup (`__pycache__`, `.pyc`, temporary `data/test_*` dirs). Remaining blocked folders are ACL-protected pytest temp/cache directories requiring manual admin cleanup.
  - Added API/gallery tests including delete route and gallery order/delete behavior (`tests/test_api_routes.py`, `tests/test_gallery_index.py`).
  - Added operator docs: `docs/USAGE.md` and `docs/TROUBLESHOOTING.md`.

---

## Development Process Rules for Future Sessions
1. Do not bypass local-only runtime requirement.
2. Prefer reliability and repeatability over peak benchmark speed.
3. Every major change must include:
- rationale,
- impact on memory/performance,
- validation steps.
4. Keep defaults simple for end users (prompt + resolution first).
5. Avoid feature sprawl before memory stability is proven.

---

## Definition of Done (v1)
The app is v1-complete when:
1. A user can run CLI or web mode on a local machine with no internet.
2. Models load from local files only (safetensors/GGUF paths as supported by selected backend).
3. User can input prompt and resolution and generate images.
4. PNG outputs are saved with required metadata.
5. Gallery is viewable/filterable in web UI.
6. Long-run generation sessions are memory-stable on target VRAM classes.
7. Portable package runs without additional installation.
