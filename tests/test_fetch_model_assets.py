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

    # TODO(krea2 finetune): the weight repo_id + sha256s are placeholders until the operator
    # uploads their finetuned Krea2-Turbo checkpoint. Keep the layout assertions (three weight
    # files landing in ``models/packs/Krea2_Turbo/weights/`` with a single shared repo_id) so a
    # future edit only has to update the placeholder to point at the real repo.
    weight_assets = [
        a for a in OPTIONAL_KREA2_ASSETS
        if a.relative_output_path.startswith("models/packs/Krea2_Turbo/weights/")
    ]
    assert {a.repo_file for a in weight_assets} == {
        "krea2_turbo_fp8.safetensors",
        "qwen3vl_4b_fp8_scaled.safetensors",
        "qwen_image_vae.safetensors",
    }
    weight_repo_ids = {a.repo_id for a in weight_assets}
    assert len(weight_repo_ids) == 1, (
        f"Krea2 weights should share a single source repo, got {sorted(weight_repo_ids)}"
    )

    # Qwen3VL processor sidecar files come from the source Qwen repo and land next to the
    # text_encoder config, so AutoProcessor can build a multimodal processor for the WP-5
    # style-reference conditioning path.
    processor_assets = [
        a for a in OPTIONAL_KREA2_ASSETS if a.repo_id == "Qwen/Qwen3-VL-4B-Instruct"
    ]
    assert {a.repo_file for a in processor_assets} == {
        "preprocessor_config.json",
        "chat_template.json",
        "video_preprocessor_config.json",
        "vocab.json",
        "merges.txt",
    }
    for asset in processor_assets:
        assert asset.relative_output_path.startswith(
            "models/packs/Krea2_Turbo/config/text_encoder/"
        )

    # Qwen3VL sidecar SHA256s stay pinned (Apache-2.0 upstream, stable content). Weight SHA256s
    # are blank placeholders that the finetune-provisioning TODO above will fill in.
    for asset in processor_assets:
        assert len(asset.sha256) == 64


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
