from __future__ import annotations

from pathlib import Path

from scripts.portable.fetch_model_assets import (
    ASSETS,
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
