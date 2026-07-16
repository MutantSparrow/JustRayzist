from __future__ import annotations

import argparse
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.portable.common import (  # noqa: E402
    normalize_platform_name,
    resolve_project_root,
    run_checked,
    run_probe,
    sha256_file,
    shell_setup_command,
    temp_env,
    venv_hf_cli_path,
)


@dataclass(frozen=True)
class AssetSpec:
    name: str
    repo_id: str
    repo_file: str
    relative_output_path: str
    sha256: str = ""
    revision: str = "main"


def staged_download_path(stage_dir: Path, repo_file: str) -> Path:
    segments = [segment for segment in repo_file.split("/") if segment]
    return stage_dir.joinpath(*segments)


def resolve_hf_cli_executable(project_root: Path, *, platform_name: str) -> str | None:
    candidates = [
        str(venv_hf_cli_path(project_root, platform_name=platform_name)),
        shutil.which("hf") or "",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        ok, _ = run_probe([candidate, "version"])
        if ok:
            return candidate
    return None


def ensure_hf_cli_prerequisites(hf_exe: str) -> None:
    run_checked([hf_exe, "version"])
    run_checked([hf_exe, "download", "--help"])


def should_skip_download(output_path: Path, expected_sha256: str, *, overwrite: bool) -> bool:
    if overwrite or not output_path.exists():
        return False
    normalized_hash = expected_sha256.strip().lower()
    if not normalized_hash:
        return True
    return sha256_file(output_path) == normalized_hash


def download_asset(
    *,
    project_root: Path,
    hf_exe: str,
    asset: AssetSpec,
    overwrite: bool,
    env: dict[str, str],
) -> None:
    output_path = project_root / asset.relative_output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage_root = project_root / ".build" / "hf_downloads"
    stage_root.mkdir(parents=True, exist_ok=True)
    expected_hash = asset.sha256.strip().lower()

    if should_skip_download(output_path, expected_hash, overwrite=overwrite):
        if expected_hash:
            size_mb = round(output_path.stat().st_size / (1024 * 1024), 2)
            print(f"[skip] {asset.name} already exists and passed SHA256 check ({size_mb} MB): {output_path}")
        else:
            size_mb = round(output_path.stat().st_size / (1024 * 1024), 2)
            print(f"[skip] {asset.name} already exists ({size_mb} MB): {output_path}")
        return

    if output_path.exists() and expected_hash:
        actual_hash = sha256_file(output_path)
        if actual_hash != expected_hash:
            print(f"[warn] {asset.name} exists but SHA256 did not match expected value. Re-downloading...")
            print(f"  expected: {expected_hash}")
            print(f"  actual:   {actual_hash}")

    stage_dir = stage_root / uuid.uuid4().hex
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    print(f"[download] {asset.name}")
    print(f"  repo: {asset.repo_id}")
    print(f"  file: {asset.repo_file}")
    print(f"  to:   {output_path}")

    stage_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_checked(
            [
                hf_exe,
                "download",
                asset.repo_id,
                asset.repo_file,
                "--repo-type",
                "model",
                "--revision",
                asset.revision,
                "--local-dir",
                str(stage_dir),
                "--max-workers",
                "8",
            ],
            cwd=project_root,
            env=env,
        )
        downloaded_path = staged_download_path(stage_dir, asset.repo_file)
        if not downloaded_path.exists():
            raise RuntimeError(f"HF CLI download completed but file is missing: {downloaded_path}")
        shutil.move(str(downloaded_path), str(tmp_path))
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)

    if expected_hash:
        actual_hash = sha256_file(tmp_path)
        if actual_hash != expected_hash:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA256 mismatch for {asset.name}. Expected {expected_hash}, got {actual_hash}."
            )
        print(f"  sha256: {actual_hash}")

    shutil.move(str(tmp_path), str(output_path))
    size_mb = round(output_path.stat().st_size / (1024 * 1024), 2)
    print(f"[ok] {asset.name} saved ({size_mb} MB)")


ASSETS = [
    AssetSpec(
        name="Transformer checkpoint (BF16)",
        repo_id="MutantSparrow/Ray",
        repo_file="Z-IMAGE-TURBO/Rayzist.v1.0.safetensors",
        relative_output_path="models/packs/Rayzist_bf16/weights/Rayzist.v1.0.safetensors",
        sha256="e1d396329a3d5ebde6d81df5d4753367a61fa9f0cb45ed6fa78336f69bd975a1",
    ),
    AssetSpec(
        name="VAE checkpoint",
        repo_id="Tongyi-MAI/Z-Image-Turbo",
        repo_file="vae/diffusion_pytorch_model.safetensors",
        relative_output_path="models/packs/Rayzist_bf16/weights/diffusion_pytorch_model.safetensors",
        sha256="f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3",
    ),
    AssetSpec(
        name="Text encoder checkpoint",
        repo_id="Comfy-Org/z_image_turbo",
        repo_file="split_files/text_encoders/qwen_3_4b.safetensors",
        relative_output_path="models/packs/Rayzist_bf16/config/text_encoder/model.safetensors",
        sha256="6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a",
    ),
    AssetSpec(
        name="SeedVR2 3B FP8 DiT checkpoint",
        repo_id="themindstudio/SeedVR2-3B-FP8-e4m3fn",
        repo_file="seedvr2_ema_3b_fp8_e4m3fn.safetensors",
        relative_output_path="models/seedvr2/seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    ),
    AssetSpec(
        name="SeedVR2 VAE checkpoint",
        repo_id="themindstudio/SeedVR2-3B-FP8-e4m3fn",
        repo_file="ema_vae_fp16.safetensors",
        relative_output_path="models/seedvr2/ema_vae_fp16.safetensors",
        sha256="20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1",
    ),
]


QWEN3_4B_FP8_PACK_NAME = "Rayzist_qwen3_4b_fp8"
QWEN3_4B_FP8_ENCODER_RELATIVE_PATH = (
    "models/packs/Rayzist_qwen3_4b_fp8/config/text_encoder/model.safetensors"
)

OPTIONAL_QWEN3_4B_FP8_ENCODER_ASSETS = [
    AssetSpec(
        name="Qwen3 4B FP8 Rayzist text encoder",
        repo_id="MutantSparrow/qwen3_4b_Rayzist_v1.0_fp8",
        repo_file="qwen3_4b_Rayzist_v1.0_fp8.safetensors",
        relative_output_path=QWEN3_4B_FP8_ENCODER_RELATIVE_PATH,
        sha256="61fdc05e9ce80e82397f41f5f0cb80e4eda402629bb892d42c4b51ec74e80c1c",
    ),
]

OPTIONAL_ASSETS = OPTIONAL_QWEN3_4B_FP8_ENCODER_ASSETS


KREA2_PACK_NAME = "Krea2_Turbo"

# Krea2-Turbo is opt-in and governed by the Krea 2 Community License (distinct from the Z-Image
# assets). These are the ComfyUI-native fp8 weights from AlperKTS/Krea2_FP8; the app converts their
# key layout at load time (app/core/pipeline_factory/krea_comfy_convert.py). The pack's diffusers
# config dirs are committed, so only the 3 weight files are fetched.
OPTIONAL_KREA2_ASSETS = [
    AssetSpec(
        name="Krea2-Turbo transformer (ComfyUI fp8)",
        repo_id="AlperKTS/Krea2_FP8",
        repo_file="krea2_turbo_fp8.safetensors",
        relative_output_path="models/packs/Krea2_Turbo/weights/krea2_turbo_fp8.safetensors",
        sha256="2d3523507c59df965e5d4ec9a1b9b4591297a50c058915c36fb29d124d30f64e",
    ),
    AssetSpec(
        name="Krea2-Turbo Qwen3-VL text encoder (ComfyUI fp8)",
        repo_id="AlperKTS/Krea2_FP8",
        repo_file="qwen3vl_4b_fp8_scaled.safetensors",
        relative_output_path="models/packs/Krea2_Turbo/weights/qwen3vl_4b_fp8_scaled.safetensors",
        sha256="54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094",
    ),
    AssetSpec(
        name="Krea2-Turbo VAE (Qwen-image)",
        repo_id="AlperKTS/Krea2_FP8",
        repo_file="qwen_image_vae.safetensors",
        relative_output_path="models/packs/Krea2_Turbo/weights/qwen_image_vae.safetensors",
        sha256="a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f",
    ),
    # Qwen3VL processor sidecar files (from the source Qwen/Qwen3-VL-4B-Instruct repo) required to
    # build a multimodal AutoProcessor for the WP-5 style-reference conditioning path. The Krea
    # backend uses these to preprocess a context image before Qwen3VL encodes it alongside the
    # prompt. Small (<10 KB total); no license issue distinct from the base Qwen release.
    AssetSpec(
        name="Krea2-Turbo Qwen3VL image preprocessor config",
        repo_id="Qwen/Qwen3-VL-4B-Instruct",
        repo_file="preprocessor_config.json",
        relative_output_path="models/packs/Krea2_Turbo/config/text_encoder/preprocessor_config.json",
        sha256="27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    ),
    AssetSpec(
        name="Krea2-Turbo Qwen3VL chat template",
        repo_id="Qwen/Qwen3-VL-4B-Instruct",
        repo_file="chat_template.json",
        relative_output_path="models/packs/Krea2_Turbo/config/text_encoder/chat_template.json",
        sha256="6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4",
    ),
    AssetSpec(
        name="Krea2-Turbo Qwen3VL video preprocessor config",
        repo_id="Qwen/Qwen3-VL-4B-Instruct",
        repo_file="video_preprocessor_config.json",
        relative_output_path="models/packs/Krea2_Turbo/config/text_encoder/video_preprocessor_config.json",
        sha256="7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
    ),
]

KREA2_LICENSE_NOTICE = (
    "Krea2-Turbo weights are governed by the Krea 2 Community License, which is distinct from the "
    "Z-Image assets. They are downloaded for local use only; redistribution or bundling requires "
    "reviewing that license. Re-run with --accept-krea2-license to proceed."
)


def selected_assets(
    *,
    include_qwen3_4b_fp8_encoder: bool = False,
    include_krea2: bool = False,
) -> list[AssetSpec]:
    assets = list(ASSETS)
    if include_qwen3_4b_fp8_encoder:
        assets.extend(OPTIONAL_QWEN3_4B_FP8_ENCODER_ASSETS)
    if include_krea2:
        assets.extend(OPTIONAL_KREA2_ASSETS)
    return assets


def ensure_qwen3_4b_fp8_pack(project_root: Path) -> None:
    source_config_dir = project_root / "models" / "packs" / "Rayzist_bf16" / "config"
    target_pack_dir = project_root / "models" / "packs" / QWEN3_4B_FP8_PACK_NAME
    target_config_dir = target_pack_dir / "config"
    if not source_config_dir.exists():
        raise RuntimeError(f"Base Rayzist_bf16 config directory not found: {source_config_dir}")

    for source_path in source_config_dir.rglob("*"):
        if source_path.is_dir():
            continue
        relative_path = source_path.relative_to(source_config_dir)
        if relative_path.as_posix() == "text_encoder/model.safetensors":
            continue
        target_path = target_config_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

    manifest_path = target_pack_dir / "modelpack.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "name: Rayzist_qwen3_4b_fp8",
                "user_visible: true",
                "enabled: true",
                "architecture: z_image_turbo",
                "backend_preference:",
                "  - diffusers",
                "pipeline_config_dir: ./config",
                "components:",
                "  transformer:",
                "    path: ../Rayzist_bf16/weights/Rayzist.v1.0.safetensors",
                "    format: safetensors",
                "  vae:",
                "    path: ../Rayzist_bf16/weights/diffusion_pytorch_model.safetensors",
                "    format: safetensors",
                "  text_encoder:",
                "    path: ./config/text_encoder/model.safetensors",
                "    format: safetensors",
                "required_configs:",
                "  - ./config/model_index.json",
                "  - ./config/scheduler/scheduler_config.json",
                "  - ./config/tokenizer/tokenizer.json",
                "  - ./config/transformer/config.json",
                "  - ./config/vae/config.json",
                "  - ./config/text_encoder/config.json",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch bundled JustRayzist model assets.")
    parser.add_argument("--project-root", default=str(REPO_ROOT))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--platform", default="")
    parser.add_argument(
        "--include-qwen3-4b-fp8-encoder",
        action="store_true",
        help="Also fetch the optional Rayzist_qwen3_4b_fp8 text encoder pack asset.",
    )
    parser.add_argument(
        "--include-krea2",
        action="store_true",
        help="Also fetch the optional Krea2_Turbo pack weights (requires --accept-krea2-license).",
    )
    parser.add_argument(
        "--accept-krea2-license",
        action="store_true",
        help="Acknowledge the Krea 2 Community License, required to fetch Krea2_Turbo weights.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = resolve_project_root(args.project_root)
    platform_name = normalize_platform_name(args.platform)

    if args.include_krea2 and not args.accept_krea2_license:
        print(KREA2_LICENSE_NOTICE, file=sys.stderr)
        return 2

    hf_exe = resolve_hf_cli_executable(project_root, platform_name=platform_name)
    if not hf_exe:
        raise RuntimeError(
            "Hugging Face CLI executable (hf) not found. "
            f"Run {shell_setup_command(platform_name)} to install or repair the environment."
        )
    if not str(hf_exe).startswith(str(project_root / ".venv")):
        print(
            "Using 'hf' from PATH. For strict reproducibility, prefer the .venv copy by running the setup script first.",
            file=sys.stderr,
        )

    env = temp_env(project_root)
    env["HF_XET_HIGH_PERFORMANCE"] = "1"
    env.pop("HF_HUB_DISABLE_XET", None)

    if args.include_krea2:
        print(KREA2_LICENSE_NOTICE.replace(" Re-run with --accept-krea2-license to proceed.", ""))

    ensure_hf_cli_prerequisites(hf_exe)
    for asset in selected_assets(
        include_qwen3_4b_fp8_encoder=args.include_qwen3_4b_fp8_encoder,
        include_krea2=args.include_krea2,
    ):
        download_asset(
            project_root=project_root,
            hf_exe=hf_exe,
            asset=asset,
            overwrite=args.force,
            env=env,
        )
    if args.include_qwen3_4b_fp8_encoder:
        ensure_qwen3_4b_fp8_pack(project_root)
        print(f"[ok] Optional model pack ready: {QWEN3_4B_FP8_PACK_NAME}")
    if args.include_krea2:
        missing = [
            asset.relative_output_path
            for asset in OPTIONAL_KREA2_ASSETS
            if not (project_root / asset.relative_output_path).exists()
        ]
        if missing:
            raise RuntimeError(
                "Krea2_Turbo weights missing after fetch: " + ", ".join(missing)
            )
        print(f"[ok] Optional model pack ready: {KREA2_PACK_NAME}")

    deprecated_vae_path = project_root / "models/packs/Rayzist_bf16/weights/ultrafluxVAEImproved_v10.safetensors"
    if deprecated_vae_path.exists():
        deprecated_vae_path.unlink()
        print(f"[cleanup] Removed deprecated VAE file: {deprecated_vae_path}")

    print("")
    print("Model asset fetch complete (HF CLI + XET).")
    print("Selected model pack: Rayzist_bf16")
    if args.include_qwen3_4b_fp8_encoder:
        print(f"Optional model pack installed: {QWEN3_4B_FP8_PACK_NAME}")
    if args.include_krea2:
        print(f"Optional model pack installed: {KREA2_PACK_NAME}")
    print("Next step:")
    print("  python -m app.cli.main validate-models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
