from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from app.config import load_settings
from app.storage.gallery_index import (
    COLOR_CACHE_VERSION,
    COLOR_FLAG_BLACK,
    COLOR_FLAG_BLUE,
    COLOR_FLAG_GREEN,
    COLOR_FLAG_RED,
    COLOR_FLAG_WHITE,
    COLOR_FLAG_YELLOW,
    _classify_image_color_flags,
    _gallery_db_path,
    delete_gallery,
    gallery_color_cache_needs_rebuild,
    gallery_color_cache_version,
    get_image,
    list_images,
    rebuild_gallery_color_cache,
    sync_outputs_to_gallery,
)


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


def test_gallery_sync_orders_and_filters_images(monkeypatch, workspace_tmp_path: Path) -> None:
    root = workspace_tmp_path / "gallery-sync"
    monkeypatch.setenv("JUSTRAYZIST_ROOT", str(root))
    settings = load_settings()

    output_path = settings.paths.outputs_dir / "sample.png"
    _save_test_png(output_path, "A red mountain under stars", color=(220, 40, 90))
    newer_path = settings.paths.outputs_dir / "sample_new.png"
    _save_test_png(
        newer_path,
        "A green valley under moonlight",
        timestamp="2026-02-22T02:00:00+00:00",
        color=(30, 190, 70),
    )

    indexed = sync_outputs_to_gallery(settings)
    assert indexed == 2

    rows = list_images(settings, limit=50)
    assert [row["filename"] for row in rows] == ["sample_new.png", "sample.png"]

    oldest_first = list_images(settings, newest_first=False, limit=50)
    assert [row["filename"] for row in oldest_first] == ["sample.png", "sample_new.png"]

    filtered = list_images(settings, prompt_query="mountain", limit=50)
    assert [row["filename"] for row in filtered] == ["sample.png"]

    red_filtered = list_images(settings, color_filter="red", limit=50)
    assert [row["filename"] for row in red_filtered] == ["sample.png"]

    row = get_image(settings, "sample.png")
    assert row is not None
    assert row["width"] == 64
    assert row["height"] == 64


def test_gallery_color_classification_flags_primary_buckets(monkeypatch, workspace_tmp_path: Path) -> None:
    root = workspace_tmp_path / "gallery-colors"
    monkeypatch.setenv("JUSTRAYZIST_ROOT", str(root))
    settings = load_settings()
    outputs_dir = settings.paths.outputs_dir

    color_cases = {
        "black.png": ((10, 10, 10), COLOR_FLAG_BLACK),
        "white.png": ((245, 245, 245), COLOR_FLAG_WHITE),
        "red.png": ((235, 45, 99), COLOR_FLAG_RED),
        "yellow.png": ((245, 210, 60), COLOR_FLAG_YELLOW),
        "green.png": ((40, 210, 70), COLOR_FLAG_GREEN),
        "blue.png": ((35, 110, 235), COLOR_FLAG_BLUE),
    }
    for filename, (color, _flag) in color_cases.items():
        _save_test_png(outputs_dir / filename, filename, color=color)

    sync_outputs_to_gallery(settings)

    for filename, (_color, flag) in color_cases.items():
        image_path = outputs_dir / filename
        assert _classify_image_color_flags(image_path) & flag

    assert [row["filename"] for row in list_images(settings, color_filter="black", limit=20)] == ["black.png"]
    assert [row["filename"] for row in list_images(settings, color_filter="white", limit=20)] == ["white.png"]
    assert [row["filename"] for row in list_images(settings, color_filter="red", limit=20)] == ["red.png"]

    assert [row["filename"] for row in list_images(settings, color_filter="yellow", limit=20)] == ["yellow.png"]
    assert [row["filename"] for row in list_images(settings, color_filter="green", limit=20)] == ["green.png"]
    assert [row["filename"] for row in list_images(settings, color_filter="blue", limit=20)] == ["blue.png"]
def test_gallery_color_classification_black_matches_charcoal_images(monkeypatch, workspace_tmp_path: Path) -> None:
    root = workspace_tmp_path / "gallery-charcoal-black"
    monkeypatch.setenv("JUSTRAYZIST_ROOT", str(root))
    settings = load_settings()

    charcoal_path = settings.paths.outputs_dir / "charcoal.png"
    _save_test_png(charcoal_path, "Charcoal study", color=(34, 34, 36))
    flags = _classify_image_color_flags(charcoal_path)
    assert flags & COLOR_FLAG_BLACK

    sync_outputs_to_gallery(settings)


def test_gallery_color_classification_uses_single_dominant_bucket(monkeypatch, workspace_tmp_path: Path) -> None:
    root = workspace_tmp_path / "gallery-dominant-color"
    monkeypatch.setenv("JUSTRAYZIST_ROOT", str(root))
    settings = load_settings()

    image_path = settings.paths.outputs_dir / "dominant_green.png"
    image = Image.new("RGB", (64, 64), color=(240, 90, 170))
    dominant_green = Image.new("RGB", (48, 64), color=(140, 255, 40))
    image.paste(dominant_green, (16, 0))
    metadata = PngInfo()
    metadata.add_text("timestamp", "2026-02-22T00:00:00+00:00")
    metadata.add_text("prompt", "green-dominant with pink accent")
    metadata.add_text("width", "64")
    metadata.add_text("height", "64")
    image.save(image_path, format="PNG", pnginfo=metadata)

    flags = _classify_image_color_flags(image_path)
    assert flags & COLOR_FLAG_GREEN
    assert flags & COLOR_FLAG_RED == 0

    sync_outputs_to_gallery(settings)
    assert list_images(settings, color_filter="red", limit=20) == []
    assert [row["filename"] for row in list_images(settings, color_filter="green", limit=20)] == [
        "dominant_green.png"
    ]



def test_gallery_color_filter_uses_cached_flags_without_rebuild_on_read(
    monkeypatch, workspace_tmp_path: Path
) -> None:
    root = workspace_tmp_path / "gallery-color-cached-read"
    monkeypatch.setenv("JUSTRAYZIST_ROOT", str(root))
    settings = load_settings()

    red_path = settings.paths.outputs_dir / "backfill_red.png"
    _save_test_png(red_path, "Backfill red", color=(225, 40, 90))
    sync_outputs_to_gallery(settings)

    db_path = _gallery_db_path(settings)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("UPDATE images SET color_flags = ?", (COLOR_FLAG_BLACK,))
        conn.commit()

    assert list_images(settings, color_filter="red", limit=20) == []
    assert [row["filename"] for row in list_images(settings, color_filter="black", limit=20)] == [
        "backfill_red.png"
    ]

    with closing(sqlite3.connect(db_path)) as conn:
        refreshed = conn.execute(
            "SELECT color_flags FROM images WHERE filename = ?",
            ("backfill_red.png",),
        ).fetchone()
        assert refreshed is not None
        assert int(refreshed[0]) & COLOR_FLAG_BLACK


def test_gallery_color_cache_rebuild_updates_stale_flags_and_marks_version(
    monkeypatch, workspace_tmp_path: Path
) -> None:
    root = workspace_tmp_path / "gallery-color-rebuild"
    monkeypatch.setenv("JUSTRAYZIST_ROOT", str(root))
    settings = load_settings()

    red_path = settings.paths.outputs_dir / "rebuild_red.png"
    _save_test_png(red_path, "Rebuild red", color=(225, 40, 90))
    sync_outputs_to_gallery(settings)

    db_path = _gallery_db_path(settings)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("UPDATE images SET color_flags = ?", (COLOR_FLAG_BLACK,))
        conn.commit()

    assert gallery_color_cache_needs_rebuild(settings) is True

    updated = rebuild_gallery_color_cache(settings, batch_size=1)
    assert updated >= 1
    assert gallery_color_cache_version(settings) == COLOR_CACHE_VERSION
    assert gallery_color_cache_needs_rebuild(settings) is False
    assert [row["filename"] for row in list_images(settings, color_filter="red", limit=20)] == [
        "rebuild_red.png"
    ]
    assert list_images(settings, color_filter="black", limit=20) == []




def test_gallery_color_classification_ignores_low_saturation_noise(monkeypatch, workspace_tmp_path: Path) -> None:
    root = workspace_tmp_path / "gallery-low-sat"
    monkeypatch.setenv("JUSTRAYZIST_ROOT", str(root))
    settings = load_settings()

    neutral_path = settings.paths.outputs_dir / "neutral.png"
    _save_test_png(neutral_path, "Neutral gray", color=(176, 172, 170))
    flags = _classify_image_color_flags(neutral_path)
    assert flags & COLOR_FLAG_BLACK == 0
    assert flags & COLOR_FLAG_YELLOW == 0
    assert flags & COLOR_FLAG_GREEN == 0
    assert flags & COLOR_FLAG_BLUE == 0


def test_gallery_color_classification_black_requires_dark_neutral_pixels(
    monkeypatch, workspace_tmp_path: Path
) -> None:
    root = workspace_tmp_path / "gallery-dark-chromatic"
    monkeypatch.setenv("JUSTRAYZIST_ROOT", str(root))
    settings = load_settings()

    dark_red_path = settings.paths.outputs_dir / "dark_red.png"
    _save_test_png(dark_red_path, "Dark red tone", color=(40, 18, 18))
    flags = _classify_image_color_flags(dark_red_path)
    assert flags & COLOR_FLAG_BLACK == 0


def test_gallery_color_classification_black_requires_dominance(monkeypatch, workspace_tmp_path: Path) -> None:
    root = workspace_tmp_path / "gallery-black-dominance"
    monkeypatch.setenv("JUSTRAYZIST_ROOT", str(root))
    settings = load_settings()

    image_path = settings.paths.outputs_dir / "green_with_black_frame.png"
    image = Image.new("RGB", (64, 64), color=(8, 8, 8))
    center = Image.new("RGB", (48, 48), color=(40, 210, 70))
    image.paste(center, (8, 8))
    metadata = PngInfo()
    metadata.add_text("timestamp", "2026-02-22T00:00:00+00:00")
    metadata.add_text("prompt", "Green image with black frame")
    metadata.add_text("width", "64")
    metadata.add_text("height", "64")
    image.save(image_path, format="PNG", pnginfo=metadata)

    flags = _classify_image_color_flags(image_path)
    assert flags & COLOR_FLAG_BLACK == 0
    assert flags & COLOR_FLAG_GREEN



def test_gallery_color_classification_requires_stronger_chroma_for_warm_tones(
    monkeypatch, workspace_tmp_path: Path
) -> None:
    root = workspace_tmp_path / "gallery-muted-warm"
    monkeypatch.setenv("JUSTRAYZIST_ROOT", str(root))
    settings = load_settings()

    muted_warm_path = settings.paths.outputs_dir / "muted_warm.png"
    _save_test_png(muted_warm_path, "Muted warm tone", color=(200, 165, 135))
    flags = _classify_image_color_flags(muted_warm_path)
    assert flags & COLOR_FLAG_YELLOW == 0


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



