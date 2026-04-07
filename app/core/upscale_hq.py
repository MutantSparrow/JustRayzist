from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from app.config.settings import AppSettings
from app.core.seedvr2 import (
    SEEDVR2_DIT_FILENAME,
    SEEDVR2_MODEL_REPO,
    SEEDVR2_MODEL_REVISION,
    SEEDVR2_RUNTIME_PRESET_HIGHRES_AUTO,
    SEEDVR2_VAE_FILENAME,
    SeedVR2StillImageConfig,
    upscale_with_seedvr2_direct_x2,
)

FAST_UPSCALE_ENGINE_NAME = "seedvr2_direct_x2_faithful"
UPSCALE_MODE_FAST = "fast"
UPSCALE_SCALES = (2,)


@dataclass
class UpscalePriorResult:
    image: Image.Image
    duration_ms: int
    output_width: int
    output_height: int
    scale_factor: int
    engine: str
    telemetry: dict[str, Any]
    checkpoint_path: str | None = None
    model_repo: str | None = None
    model_revision: str | None = None


@dataclass(frozen=True)
class SeedVR2ModelSpec:
    name: str
    model_repo: str
    model_revision: str
    model_dir_candidates: tuple[Path, ...]
    dit_filename_candidates: tuple[str, ...]
    vae_filename: str = SEEDVR2_VAE_FILENAME


@dataclass(frozen=True)
class ResolvedSeedVR2ModelSpec:
    spec: SeedVR2ModelSpec
    model_dir: Path
    dit_filename: str
    vae_filename: str
    dit_path: Path
    vae_path: Path


DEFAULT_SEEDVR2_3B_MODEL_SPEC = SeedVR2ModelSpec(
    name="seedvr2_3b",
    model_repo=SEEDVR2_MODEL_REPO,
    model_revision=SEEDVR2_MODEL_REVISION,
    model_dir_candidates=(Path("models/seedvr2"),),
    dit_filename_candidates=(SEEDVR2_DIT_FILENAME,),
    vae_filename=SEEDVR2_VAE_FILENAME,
)


def resolve_seedvr2_model_spec(
    settings: AppSettings,
    *,
    spec: SeedVR2ModelSpec,
    explicit_model_dir: Path | None = None,
    preferred_dit_filename: str | None = None,
    preferred_vae_filename: str | None = None,
) -> ResolvedSeedVR2ModelSpec | None:
    model_dirs: list[Path] = []
    if explicit_model_dir is not None:
        model_dirs.append(Path(explicit_model_dir).expanduser().resolve())
    model_dirs.extend(
        (settings.paths.root_dir / candidate).expanduser().resolve()
        for candidate in spec.model_dir_candidates
    )

    dit_filenames: list[str] = []
    if preferred_dit_filename:
        dit_filenames.append(str(preferred_dit_filename).strip())
    dit_filenames.extend(str(candidate).strip() for candidate in spec.dit_filename_candidates)

    vae_filenames: list[str] = []
    if preferred_vae_filename:
        vae_filenames.append(str(preferred_vae_filename).strip())
    vae_filenames.append(str(spec.vae_filename).strip())

    unique_model_dirs: list[Path] = []
    seen_dirs: set[Path] = set()
    for candidate in model_dirs:
        if candidate in seen_dirs:
            continue
        seen_dirs.add(candidate)
        unique_model_dirs.append(candidate)

    unique_dit_filenames: list[str] = []
    seen_dits: set[str] = set()
    for candidate in dit_filenames:
        normalized = candidate.strip()
        if not normalized or normalized in seen_dits:
            continue
        seen_dits.add(normalized)
        unique_dit_filenames.append(normalized)

    unique_vae_filenames: list[str] = []
    seen_vaes: set[str] = set()
    for candidate in vae_filenames:
        normalized = candidate.strip()
        if not normalized or normalized in seen_vaes:
            continue
        seen_vaes.add(normalized)
        unique_vae_filenames.append(normalized)

    for model_dir in unique_model_dirs:
        if not model_dir.exists() or not model_dir.is_dir():
            continue
        for dit_filename in unique_dit_filenames:
            dit_path = (model_dir / dit_filename).resolve()
            if not dit_path.exists() or not dit_path.is_file():
                continue
            for vae_filename in unique_vae_filenames:
                vae_path = (model_dir / vae_filename).resolve()
                if not vae_path.exists() or not vae_path.is_file():
                    continue
                return ResolvedSeedVR2ModelSpec(
                    spec=spec,
                    model_dir=model_dir,
                    dit_filename=dit_filename,
                    vae_filename=vae_filename,
                    dit_path=dit_path,
                    vae_path=vae_path,
                )
    return None


def target_dimensions(width: int, height: int, scale: int = 2) -> tuple[int, int]:
    normalized_scale = int(scale)
    if normalized_scale not in UPSCALE_SCALES:
        raise ValueError("Only x2 upscale is supported.")
    return max(64, int(width) * normalized_scale), max(64, int(height) * normalized_scale)


def build_seedvr2_3b_prior(
    *,
    image: Image.Image,
    settings: AppSettings,
    runtime_profile: str,
    seed: int | None = None,
    timeout_seconds: int = 240,
    model_dir: Path | None = None,
    dit_filename: str = SEEDVR2_DIT_FILENAME,
    vae_filename: str = SEEDVR2_VAE_FILENAME,
    model_repo: str = SEEDVR2_MODEL_REPO,
    model_revision: str = SEEDVR2_MODEL_REVISION,
    still_image_config: SeedVR2StillImageConfig | None = None,
    runtime_preset: str = SEEDVR2_RUNTIME_PRESET_HIGHRES_AUTO,
    is_cancel_requested: Any | None = None,
) -> UpscalePriorResult:
    result = upscale_with_seedvr2_direct_x2(
        image=image,
        settings=settings,
        runtime_profile=runtime_profile,
        seed=seed,
        timeout_seconds=timeout_seconds,
        is_cancel_requested=is_cancel_requested,
        model_dir_override=model_dir,
        model_repo_override=model_repo,
        model_revision_override=model_revision,
        dit_filename=dit_filename,
        vae_filename=vae_filename,
        still_image_config=still_image_config,
        runtime_preset=runtime_preset,
    )
    return UpscalePriorResult(
        image=result.image,
        duration_ms=int(result.duration_ms),
        output_width=int(result.output_width),
        output_height=int(result.output_height),
        scale_factor=2,
        engine=FAST_UPSCALE_ENGINE_NAME,
        telemetry=result.telemetry_dict(),
        checkpoint_path=str((Path(model_dir).expanduser().resolve() if model_dir else settings.paths.models_dir / "seedvr2") / dit_filename),
        model_repo=model_repo,
        model_revision=model_revision,
    )


def build_fast_x2_prior(
    *,
    image: Image.Image,
    settings: AppSettings,
    runtime_profile: str,
    seed: int | None = None,
    is_cancel_requested: Any | None = None,
) -> UpscalePriorResult:
    return build_seedvr2_3b_prior(
        image=image,
        settings=settings,
        runtime_profile=runtime_profile,
        seed=seed,
        still_image_config=SeedVR2StillImageConfig(
            input_noise_scale=0.0,
            latent_noise_scale=0.0,
            color_correction="lab",
        ),
        is_cancel_requested=is_cancel_requested,
    )
