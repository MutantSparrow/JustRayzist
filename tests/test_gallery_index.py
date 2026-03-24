from __future__ import annotations

from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from app.config import load_settings
from app.storage.gallery_index import delete_gallery, get_image, list_images, sync_outputs_to_gallery


def _save_test_png(path: Path, prompt: str, timestamp: str = "2026-02-22T00:00:00+00:00") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (64, 64), color=(120, 140, 180))
    metadata = PngInfo()
    metadata.add_text("timestamp", timestamp)
    metadata.add_text("prompt", prompt)
    metadata.add_text("application_name", "JustRayzist")
    metadata.add_text("application_version", "0.1.0")
    metadata.add_text("width", "64")
    metadata.add_text("height", "64")
    metadata.add_text("model_pack", "Rayzist_bf16")
    image.save(path, format="PNG", pnginfo=metadata)


def test_gallery_sync_orders_and_filters_images(monkeypatch, workspace_tmp_path: Path) -> None:
    root = workspace_tmp_path / "gallery-sync"
    monkeypatch.setenv("JUSTRAYZIST_ROOT", str(root))
    settings = load_settings()

    output_path = settings.paths.outputs_dir / "sample.png"
    _save_test_png(output_path, "A red mountain under stars")
    newer_path = settings.paths.outputs_dir / "sample_new.png"
    _save_test_png(
        newer_path,
        "A green valley under moonlight",
        timestamp="2026-02-22T02:00:00+00:00",
    )

    indexed = sync_outputs_to_gallery(settings)
    assert indexed == 2

    rows = list_images(settings, limit=50)
    assert [row["filename"] for row in rows] == ["sample_new.png", "sample.png"]

    oldest_first = list_images(settings, newest_first=False, limit=50)
    assert [row["filename"] for row in oldest_first] == ["sample.png", "sample_new.png"]

    filtered = list_images(settings, prompt_query="mountain", limit=50)
    assert [row["filename"] for row in filtered] == ["sample.png"]

    row = get_image(settings, "sample.png")
    assert row is not None
    assert row["width"] == 64
    assert row["height"] == 64


def test_delete_gallery_removes_unindexed_owner_files(monkeypatch, workspace_tmp_path: Path) -> None:
    root = workspace_tmp_path / "gallery-cleanup"
    monkeypatch.setenv("JUSTRAYZIST_ROOT", str(root))
    settings = load_settings()

    owner_dir = settings.paths.outputs_dir / "example-client"
    indexed_image = owner_dir / "indexed.png"
    stray_image = owner_dir / "stray.png"
    _save_test_png(indexed_image, "Indexed")
    _save_test_png(stray_image, "Stray")

    assert sync_outputs_to_gallery(settings) == 2
    stray_image.unlink()
    replacement_stray = owner_dir / "stray_unindexed.png"
    _save_test_png(replacement_stray, "Unindexed")

    rows = list_images(settings, owner_id="example-client", limit=50)
    assert len(rows) == 1
    assert rows[0]["filename"] == "indexed.png"

    deletion = delete_gallery(settings, owner_id="example-client")
    assert deletion["deleted_rows"] == 1
    assert deletion["deleted_files"] == 2
    assert deletion["remaining_rows"] == 0
    assert list_images(settings, owner_id="example-client", limit=10) == []
    assert not owner_dir.exists()

