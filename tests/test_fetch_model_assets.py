from __future__ import annotations

from pathlib import Path

from scripts.portable.fetch_model_assets import (
    ASSETS,
    OPTIONAL_KREA2_ASSETS,
    OPTIONAL_QWEN3_4B_FP8_ENCODER_ASSETS,
    ensure_qwen3_4b_fp8_pack,
    selected_assets,
)


def test_fetch_model_assets_excludes_deprecated_clarity_checkpoint() -> None:
    assert all(asset.repo_id != "Phips/1xDeJPG_realplksr_otf" for asset in ASSETS)


def test_optional_qwen3_fp8_encoder_asset_is_opt_in() -> None:
    optional_asset = OPTIONAL_QWEN3_4B_FP8_ENCODER_ASSETS[0]

    assert optional_asset not in ASSETS
    assert optional_asset not in selected_assets()
    assert optional_asset in selected_assets(include_qwen3_4b_fp8_encoder=True)
    assert optional_asset.repo_id == "MutantSparrow/qwen3_4b_Rayzist_v1.0_fp8"
    assert optional_asset.repo_file == "qwen3_4b_Rayzist_v1.0_fp8.safetensors"
    assert (
        optional_asset.relative_output_path
        == "models/packs/Rayzist_qwen3_4b_fp8/config/text_encoder/model.safetensors"
    )


def test_krea2_assets_are_opt_in_and_map_to_pack_weights() -> None:
    # Not part of the default set.
    for asset in OPTIONAL_KREA2_ASSETS:
        assert asset not in ASSETS
        assert asset not in selected_assets()
        assert asset not in selected_assets(include_qwen3_4b_fp8_encoder=True)

    # Included only when explicitly requested.
    krea_selected = selected_assets(include_krea2=True)
    for asset in OPTIONAL_KREA2_ASSETS:
        assert asset in krea_selected

    # All three come from the AlperKTS ComfyUI fp8 repo and land in the pack weights dir with hashes.
    repo_files = {asset.repo_file for asset in OPTIONAL_KREA2_ASSETS}
    assert repo_files == {
        "krea2_turbo_fp8.safetensors",
        "qwen3vl_4b_fp8_scaled.safetensors",
        "qwen_image_vae.safetensors",
    }
    for asset in OPTIONAL_KREA2_ASSETS:
        assert asset.repo_id == "AlperKTS/Krea2_FP8"
        assert asset.relative_output_path.startswith("models/packs/Krea2_Turbo/weights/")
        assert len(asset.sha256) == 64


def test_krea2_license_gate_blocks_without_acceptance() -> None:
    # main() must refuse --include-krea2 unless --accept-krea2-license is also passed.
    import subprocess
    import sys
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "portable" / "fetch_model_assets.py"),
            "--include-krea2",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Krea 2 Community License" in result.stderr


def test_ensure_qwen3_fp8_pack_copies_config_without_base_encoder(temp_app_root: Path) -> None:
    base_config = temp_app_root / "models" / "packs" / "Rayzist_bf16" / "config"
    (base_config / "text_encoder").mkdir(parents=True)
    (base_config / "tokenizer").mkdir(parents=True)
    (base_config / "model_index.json").write_text("{}", encoding="utf-8")
    (base_config / "text_encoder" / "config.json").write_text("{}", encoding="utf-8")
    (base_config / "text_encoder" / "model.safetensors").write_bytes(b"base encoder")
    (base_config / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")

    ensure_qwen3_4b_fp8_pack(temp_app_root)

    pack_dir = temp_app_root / "models" / "packs" / "Rayzist_qwen3_4b_fp8"
    manifest = (pack_dir / "modelpack.yaml").read_text(encoding="utf-8")
    assert "name: Rayzist_qwen3_4b_fp8" in manifest
    assert "enabled: true" in manifest
    assert "../Rayzist_bf16/weights/Rayzist.v1.0.safetensors" in manifest
    assert (pack_dir / "config" / "text_encoder" / "config.json").exists()
    assert not (pack_dir / "config" / "text_encoder" / "model.safetensors").exists()
