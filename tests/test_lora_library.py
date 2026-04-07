from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file

from app.storage.lora_library import (
    PLACEHOLDER_PREVIEW_SIZE,
    create_lora_draft,
    delete_lora,
    detect_lora_draft_triggers,
    finalize_lora_draft,
    get_lora,
    get_lora_draft,
    list_loras,
    preview_path_for_lora,
    update_lora,
)


def _build_lora_bytes(path: Path, metadata: dict[str, str] | None = None) -> bytes:
    save_file({"lora": torch.zeros((1, 1), dtype=torch.float32)}, str(path), metadata=metadata or {})
    return path.read_bytes()


def _build_preview_bytes(size: tuple[int, int] = (1400, 900), color: tuple[int, int, int] = (32, 180, 96)) -> bytes:
    image = Image.new("RGB", size, color=color)
    target = Path.cwd() / ".tmp_lora_preview.png"
    try:
        image.save(target, format="PNG")
        return target.read_bytes()
    finally:
        target.unlink(missing_ok=True)


def test_create_lora_draft_stages_metadata_without_exposing_live_library(
    temp_app_paths,
    make_app_settings,
) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    payload_path = temp_app_paths.data_dir / "cinematic-style.safetensors"
    content = _build_lora_bytes(
        payload_path,
        metadata={
            "ss_output_name": "cinematic-style",
            "trained_words": "cinematic style, moody light",
        },
    )

    draft = create_lora_draft(settings, filename="Cinematic Style.safetensors", content=content)

    assert draft["draft_id"] == "cinematic-style"
    assert draft["display_name"] == "Cinematic Style"
    assert draft["source_filename"] == "Cinematic Style.safetensors"
    assert draft["detected_trigger_words"] == ["cinematic style", "moody light"]
    assert list_loras(settings) == []
    assert get_lora(settings, "cinematic-style") is None

    full_draft = get_lora_draft(settings, draft["draft_id"])
    assert full_draft is not None
    assert full_draft["metadata_summary"]["ss_output_name"] == "cinematic-style"
    assert (temp_app_paths.data_dir / "lora_drafts" / "cinematic-style.safetensors").exists()
    assert (temp_app_paths.data_dir / "lora_drafts" / "cinematic-style.json").exists()


def test_detect_lora_draft_triggers_refreshes_suggestions_without_live_exposure(
    temp_app_paths,
    make_app_settings,
) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    payload_path = temp_app_paths.data_dir / "portrait-helper.safetensors"
    content = _build_lora_bytes(payload_path, metadata={"activation text": "portrait helper, neon wash"})

    draft = create_lora_draft(settings, filename="portrait-helper.safetensors", content=content)
    refreshed = detect_lora_draft_triggers(settings, draft["draft_id"])

    assert refreshed["draft_id"] == "portrait-helper"
    assert refreshed["detected_trigger_words"] == ["portrait helper", "neon wash"]
    assert list_loras(settings) == []


def test_explicit_trigger_metadata_wins_over_tag_frequency_fallback(
    temp_app_paths,
    make_app_settings,
) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    payload_path = temp_app_paths.data_dir / "explicit-style.safetensors"
    content = _build_lora_bytes(
        payload_path,
        metadata={
            "trained_words": "cinematic style, moody light",
            "tag_frequency": '{"set":{"soft bloom": 20, "studio haze": 12, "cinematic style": 40}}',
        },
    )

    draft = create_lora_draft(settings, filename="explicit-style.safetensors", content=content)

    assert draft["detected_trigger_words"] == ["cinematic style", "moody light"]


def test_tag_frequency_fallback_returns_top_three_cleaned_triggers(
    temp_app_paths,
    make_app_settings,
) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    payload_path = temp_app_paths.data_dir / "scored-style.safetensors"
    content = _build_lora_bytes(
        payload_path,
        metadata={
            "ss_tag_frequency": (
                '{"dataset":{"cinematic style": 18, "moody light": 11, "soft bloom": 7, '
                '"layers.0.attention.to_out.0": 999, "true": 50}}'
            )
        },
    )

    draft = create_lora_draft(settings, filename="scored-style.safetensors", content=content)

    assert draft["detected_trigger_words"] == ["cinematic style", "moody light", "soft bloom"]


def test_malformed_tag_frequency_metadata_does_not_break_draft_creation(
    temp_app_paths,
    make_app_settings,
) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    payload_path = temp_app_paths.data_dir / "broken-tags.safetensors"
    content = _build_lora_bytes(
        payload_path,
        metadata={
            "tag_frequency": "{not valid json",
            "trained_words": "dream haze",
        },
    )

    draft = create_lora_draft(settings, filename="broken-tags.safetensors", content=content)

    assert draft["draft_id"] == "broken-tags"
    assert draft["detected_trigger_words"] == ["dream haze"]


def test_finalize_lora_draft_commits_live_record_and_custom_preview(
    temp_app_paths,
    make_app_settings,
) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    payload_path = temp_app_paths.data_dir / "cinematic-style.safetensors"
    content = _build_lora_bytes(payload_path, metadata={"trained_words": "cinematic style, moody light"})
    draft = create_lora_draft(settings, filename="Cinematic Style.safetensors", content=content)

    item = finalize_lora_draft(
        settings,
        draft_id=draft["draft_id"],
        display_name="Cinematic Style Deluxe",
        trigger_words=["cinematic style", "moody light", "soft bloom"],
        preview_content=_build_preview_bytes(),
    )

    assert item["id"] == "cinematic-style"
    assert item["display_name"] == "Cinematic Style Deluxe"
    assert item["trigger_words"] == ["cinematic style", "moody light", "soft bloom"]
    assert item["preview_is_custom"] is True
    assert get_lora_draft(settings, draft["draft_id"]) is None

    full_record = get_lora(settings, item["id"])
    assert full_record is not None
    assert full_record["path"].endswith("cinematic-style.safetensors")
    assert full_record["preview_is_custom"] is True

    preview_path = preview_path_for_lora(settings, item["id"])
    with Image.open(preview_path) as preview_image:
        assert preview_image.size == (PLACEHOLDER_PREVIEW_SIZE, PLACEHOLDER_PREVIEW_SIZE)


def test_finalize_lora_draft_without_thumbnail_keeps_placeholder_preview(
    temp_app_paths,
    make_app_settings,
) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    payload_path = temp_app_paths.data_dir / "helper.safetensors"
    content = _build_lora_bytes(payload_path, metadata={"trained_words": "helper style"})
    draft = create_lora_draft(settings, filename="helper.safetensors", content=content)

    item = finalize_lora_draft(
        settings,
        draft_id=draft["draft_id"],
        display_name="Helper",
        trigger_words=None,
    )

    assert item["preview_is_custom"] is False
    preview_path = preview_path_for_lora(settings, item["id"])
    with Image.open(preview_path) as preview_image:
        assert preview_image.size == (PLACEHOLDER_PREVIEW_SIZE, PLACEHOLDER_PREVIEW_SIZE)


def test_update_lora_updates_name_triggers_and_preview_without_renaming_id(
    temp_app_paths,
    make_app_settings,
) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    payload_path = temp_app_paths.data_dir / "cinematic-style.safetensors"
    content = _build_lora_bytes(payload_path, metadata={"trained_words": "cinematic style"})
    draft = create_lora_draft(settings, filename="cinematic-style.safetensors", content=content)
    created = finalize_lora_draft(
        settings,
        draft_id=draft["draft_id"],
        display_name="Cinematic Style",
        trigger_words=["cinematic style"],
    )

    updated = update_lora(
        settings,
        lora_id=created["id"],
        display_name="Cinematic Style Reloaded",
        trigger_words=json.dumps(["cinematic style", "soft bloom"]),
        preview_content=_build_preview_bytes(size=(800, 1400), color=(120, 32, 210)),
    )

    assert updated["id"] == created["id"]
    assert updated["display_name"] == "Cinematic Style Reloaded"
    assert updated["trigger_words"] == ["cinematic style", "soft bloom"]
    assert updated["preview_is_custom"] is True

    full_record = get_lora(settings, created["id"])
    assert full_record is not None
    assert full_record["path"].endswith("cinematic-style.safetensors")
    assert full_record["source_filename"] == "cinematic-style.safetensors"


def test_list_loras_exposes_preview_cache_key_that_changes_with_preview_file(
    temp_app_paths,
    make_app_settings,
) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    payload_path = temp_app_paths.data_dir / "preview-refresh.safetensors"
    content = _build_lora_bytes(payload_path, metadata={"trained_words": "preview refresh"})
    draft = create_lora_draft(settings, filename="preview-refresh.safetensors", content=content)
    created = finalize_lora_draft(
        settings,
        draft_id=draft["draft_id"],
        display_name="Preview Refresh",
        trigger_words=["preview refresh"],
        preview_content=_build_preview_bytes(color=(32, 180, 96)),
    )

    initial_entry = next((entry for entry in list_loras(settings) if entry["id"] == created["id"]), None)
    assert initial_entry is not None
    initial_key = str(initial_entry["preview_cache_key"])
    assert initial_key

    update_lora(
        settings,
        lora_id=created["id"],
        display_name="Preview Refresh",
        trigger_words=["preview refresh"],
        preview_content=_build_preview_bytes(color=(180, 32, 96)),
    )

    updated_entry = next((entry for entry in list_loras(settings) if entry["id"] == created["id"]), None)
    assert updated_entry is not None
    assert str(updated_entry["preview_cache_key"])
    assert updated_entry["preview_cache_key"] != initial_key


def test_list_loras_is_sorted_by_display_name(temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)

    alpha_payload = temp_app_paths.data_dir / "alpha-style.safetensors"
    beta_payload = temp_app_paths.data_dir / "beta-style.safetensors"
    alpha_draft = create_lora_draft(settings, filename="alpha-style.safetensors", content=_build_lora_bytes(alpha_payload))
    beta_draft = create_lora_draft(settings, filename="beta-style.safetensors", content=_build_lora_bytes(beta_payload))
    finalize_lora_draft(settings, draft_id=beta_draft["draft_id"], display_name="Zulu Style", trigger_words=[])
    finalize_lora_draft(settings, draft_id=alpha_draft["draft_id"], display_name="Aurora Style", trigger_words=[])

    assert [entry["display_name"] for entry in list_loras(settings)] == ["Aurora Style", "Zulu Style"]


def test_delete_lora_removes_weights_sidecar_and_preview(temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    payload_path = temp_app_paths.data_dir / "portrait-helper.safetensors"
    content = _build_lora_bytes(payload_path, metadata={"activation text": "portrait helper"})

    draft = create_lora_draft(settings, filename="portrait-helper.safetensors", content=content)
    item = finalize_lora_draft(settings, draft_id=draft["draft_id"], display_name="Portrait Helper", trigger_words=[])
    result = delete_lora(settings, item["id"])

    assert result == {"id": "portrait-helper", "deleted_files": 3}
    assert get_lora(settings, item["id"]) is None
    assert not (temp_app_paths.models_dir / "loras" / "portrait-helper.safetensors").exists()
    assert not (temp_app_paths.models_dir / "loras" / "portrait-helper.json").exists()
    assert not (temp_app_paths.models_dir / "loras" / "portrait-helper.png").exists()


def test_create_lora_draft_rejects_invalid_safetensors_payload(temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)

    try:
        create_lora_draft(settings, filename="broken-lora.safetensors", content=b"not a safetensors file")
    except ValueError as exc:
        assert "Invalid LoRA safetensors file" in str(exc)
    else:
        raise AssertionError("Expected invalid safetensors payload to raise ValueError.")
