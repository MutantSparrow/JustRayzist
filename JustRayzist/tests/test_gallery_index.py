from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from app.config import load_settings
from app.storage.gallery_index import delete_gallery, get_image, list_images, sync_outputs_to_gallery


def _save_test_png(path: Path, prompt: str, timestamp: str = "2026-02-22T00:00:00+00:00") -> None:
    image = Image.new("RGB", (64, 64), color=(120, 140, 180))
    metadata = PngInfo()
    metadata.add_text("timestamp", timestamp)
    metadata.add_text("prompt", prompt)
    metadata.add_text("application_name", "JustRayzist")
    metadata.add_text("application_version", "0.1.0")
    metadata.add_text("width", "64")
    metadata.add_text("height", "64")
    metadata.add_text("model_pack", "zimage_turbo_local")
    image.save(path, format="PNG", pnginfo=metadata)


def test_gallery_sync_and_filter(monkeypatch) -> None:
    root = Path.cwd() / "data" / f"test_gallery_{uuid4().hex}"
    try:
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
        assert len(rows) == 2
        assert rows[0]["filename"] == "sample_new.png"
        assert rows[1]["filename"] == "sample.png"

        oldest_first = list_images(settings, newest_first=False, limit=50)
        assert oldest_first[0]["filename"] == "sample.png"

        filtered = list_images(settings, prompt_query="mountain", limit=50)
        assert len(filtered) == 1
        filtered_empty = list_images(settings, prompt_query="desert", limit=50)
        assert filtered_empty == []

        row = get_image(settings, "sample.png")
        assert row is not None
        assert row["width"] == 64
        assert row["height"] == 64
        assert "mode" in row
        assert "source_filename" in row

        deletion = delete_gallery(settings)
        assert deletion["deleted_files"] == 2
        assert deletion["deleted_rows"] == 2
        assert list_images(settings, limit=10) == []
    finally:
        shutil.rmtree(root, ignore_errors=True)
