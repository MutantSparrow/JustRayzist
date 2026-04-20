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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch bundled JustRayzist model assets.")
    parser.add_argument("--project-root", default=str(REPO_ROOT))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--platform", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = resolve_project_root(args.project_root)
    platform_name = normalize_platform_name(args.platform)
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

    ensure_hf_cli_prerequisites(hf_exe)
    for asset in ASSETS:
        download_asset(
            project_root=project_root,
            hf_exe=hf_exe,
            asset=asset,
            overwrite=args.force,
            env=env,
        )

    deprecated_vae_path = project_root / "models/packs/Rayzist_bf16/weights/ultrafluxVAEImproved_v10.safetensors"
    if deprecated_vae_path.exists():
        deprecated_vae_path.unlink()
        print(f"[cleanup] Removed deprecated VAE file: {deprecated_vae_path}")

    print("")
    print("Model asset fetch complete (HF CLI + XET).")
    print("Selected model pack: Rayzist_bf16")
    print("Next step:")
    print("  python -m app.cli.main validate-models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
