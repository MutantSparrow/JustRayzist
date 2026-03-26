from __future__ import annotations

from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from app.config import load_settings
from app.storage.gallery_index import list_images, set_image_favorite, sync_outputs_to_gallery


def _save_test_png(
    path: Path,
    prompt: str,
    timestamp: str = "2026-02-22T00:00:00+00:00",
    *,
    color: tuple[int, int, int] = (120, 140, 180),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (64, 64), color=color)
    metadata = PngInfo()
    metadata.add_text("timestamp", timestamp)
    metadata.add_text("prompt", prompt)
    metadata.add_text("application_name", "JustRayzist")
    metadata.add_text("application_version", "0.1.0")
    metadata.add_text("width", "64")
    metadata.add_text("height", "64")
    metadata.add_text("model_pack", "Rayzist_bf16")
    image.save(path, format="PNG", pnginfo=metadata)


def test_gallery_favorite_mutation_and_filtering(monkeypatch, workspace_tmp_path: Path) -> None:
    root = workspace_tmp_path / "gallery-favorites"
    monkeypatch.setenv("JUSTRAYZIST_ROOT", str(root))
    settings = load_settings()

    favorite_path = settings.paths.outputs_dir / "favorite.png"
    regular_path = settings.paths.outputs_dir / "regular.png"
    _save_test_png(favorite_path, "favorite prompt")
    _save_test_png(
        regular_path,
        "regular prompt",
        timestamp="2026-02-22T01:00:00+00:00",
    )

    sync_outputs_to_gallery(settings)

    updated = set_image_favorite(settings, "favorite.png", True)
    assert updated["favorite"] == 1

    favorite_rows = list_images(settings, favorites_only=True, limit=20)
    assert [row["filename"] for row in favorite_rows] == ["favorite.png"]

    cleared = set_image_favorite(settings, "favorite.png", False)
    assert cleared["favorite"] == 0
    assert list_images(settings, favorites_only=True, limit=20) == []