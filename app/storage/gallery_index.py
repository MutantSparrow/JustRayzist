from __future__ import annotations

import colorsys
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from app.config.settings import AppSettings

LEGACY_OWNER_ID = "legacy"
LEGACY_IMPORT_SOURCE_ID = "__legacy_root__"
MANAGED_IMAGE_SUFFIXES = {".png"}
_WINDOWS_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    "com1",
    "com2",
    "com3",
    "com4",
    "com5",
    "com6",
    "com7",
    "com8",
    "com9",
    "lpt1",
    "lpt2",
    "lpt3",
    "lpt4",
    "lpt5",
    "lpt6",
    "lpt7",
    "lpt8",
    "lpt9",
}
COLOR_FLAG_BLACK = 1
COLOR_FLAG_WHITE = 2
COLOR_FLAG_RED = 4
COLOR_FLAG_YELLOW = 8
COLOR_FLAG_BLUE = 16
COLOR_FLAG_GREEN = 32
COLOR_FLAG_BY_NAME = {
    "black": COLOR_FLAG_BLACK,
    "white": COLOR_FLAG_WHITE,
    "red": COLOR_FLAG_RED,
    "yellow": COLOR_FLAG_YELLOW,
    "blue": COLOR_FLAG_BLUE,
    "green": COLOR_FLAG_GREEN,
}
_COLOR_ANALYSIS_SIZE = 64
_COLOR_ALPHA_MIN = 32
_COLOR_BLACK_VALUE_MAX = 0.19
_COLOR_WHITE_VALUE_MIN = 0.82
_COLOR_WHITE_SATURATION_MAX = 0.20
_COLOR_BLACK_SATURATION_MAX = 0.22
_COLOR_BLACK_DOMINANT_RATIO_MIN = 0.32
_COLOR_WHITE_DOMINANT_RATIO_MIN = 0.26
_COLOR_CHROMA_VALUE_MIN = 0.30
_COLOR_RED_SATURATION_MIN = 0.50
_COLOR_YELLOW_SATURATION_MIN = 0.40
_COLOR_GREEN_SATURATION_MIN = 0.32
_COLOR_BLUE_SATURATION_MIN = 0.42
_COLOR_RED_DOMINANT_RATIO_MIN = 0.18
_COLOR_YELLOW_DOMINANT_RATIO_MIN = 0.13
_COLOR_GREEN_DOMINANT_RATIO_MIN = 0.11
_COLOR_BLUE_DOMINANT_RATIO_MIN = 0.15
_COLOR_RED_HUE_MIN = 325.0
_COLOR_RED_HUE_MAX = 14.0
_COLOR_YELLOW_HUE_MIN = 36.0
_COLOR_YELLOW_HUE_MAX = 68.0
_COLOR_GREEN_HUE_MIN = 82.0
_COLOR_GREEN_HUE_MAX = 160.0
_COLOR_BLUE_HUE_MIN = 198.0
_COLOR_BLUE_HUE_MAX = 252.0
COLOR_CACHE_VERSION = "dominant_v6"
_GALLERY_META_COLOR_CACHE_VERSION_KEY = "color_cache_version"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _gallery_db_path(settings: AppSettings) -> Path:
    return settings.paths.data_dir / "gallery.db"


def normalize_owner_id(raw_owner_id: str) -> str:
    raw = str(raw_owner_id or "").strip().lower()
    if not raw:
        raise ValueError("Owner id is required.")
    pieces: list[str] = []
    for ch in raw:
        if ch.isalnum() or ch in {"_", "-"}:
            pieces.append(ch)
        elif ch in {".", " ", ":"}:
            pieces.append("_")
    normalized = "".join(pieces).strip("._-")
    if not normalized:
        raise ValueError("Owner id is invalid.")
    normalized = normalized[:64]
    if normalized in _WINDOWS_RESERVED_NAMES:
        normalized = f"user_{normalized}"
    return normalized


def _resolve_managed_output_path(settings: AppSettings, raw_path: Any) -> Path:
    output_path = Path(str(raw_path)).expanduser().resolve()
    outputs_dir = settings.paths.outputs_dir.resolve()
    try:
        output_path.relative_to(outputs_dir)
    except ValueError as exc:
        raise ValueError("Image path is outside managed outputs directory.") from exc
    return output_path


def _owner_id_from_output_path(settings: AppSettings, raw_path: Any) -> str:
    output_path = _resolve_managed_output_path(settings, raw_path)
    relative = output_path.relative_to(settings.paths.outputs_dir.resolve())
    if len(relative.parts) <= 1:
        return LEGACY_OWNER_ID
    first_segment = str(relative.parts[0]).strip()
    try:
        return normalize_owner_id(first_segment)
    except ValueError:
        return LEGACY_OWNER_ID


def _owner_output_dir(settings: AppSettings, owner_id: str) -> Path:
    outputs_dir = settings.paths.outputs_dir.resolve()
    owner_dir = (outputs_dir / normalize_owner_id(owner_id)).resolve()
    try:
        owner_dir.relative_to(outputs_dir)
    except ValueError as exc:
        raise ValueError("Owner output directory is outside managed outputs directory.") from exc
    return owner_dir


def _iter_managed_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in MANAGED_IMAGE_SUFFIXES
    ]


def _remove_empty_parents(path: Path, stop_at: Path) -> None:
    current = path.resolve()
    boundary = stop_at.resolve()
    while current != boundary:
        if not current.exists() or not current.is_dir():
            current = current.parent
            continue
        try:
            next(current.iterdir())
        except StopIteration:
            current.rmdir()
            current = current.parent
            continue
        break


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def _open_connection(db_path: Path):
    connection = _connect(db_path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _create_images_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            output_path TEXT NOT NULL,
            prompt TEXT NOT NULL,
            prompt_wildcard_resolved TEXT,
            timestamp TEXT,
            application_name TEXT,
            application_version TEXT,
            width INTEGER,
            height INTEGER,
            model_pack TEXT,
            backend TEXT,
            device TEXT,
            steps INTEGER,
            guidance_scale REAL,
            duration_ms INTEGER,
            inference_process TEXT,
            procedural_creativity INTEGER,
            mode TEXT,
            source_image TEXT,
            source_filename TEXT,
            source_width INTEGER,
            source_height INTEGER,
            similarity REAL,
            wildcards_json TEXT,
            wildcard_count INTEGER,
            loras_json TEXT,
            lora_count INTEGER,
            color_flags INTEGER,
            favorite INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_images_owner_filename ON images(owner_id, filename);"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_images_owner_timestamp ON images(owner_id, timestamp DESC);")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_images_owner_favorite_timestamp ON images(owner_id, favorite, timestamp DESC);"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_images_timestamp ON images(timestamp DESC);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_images_prompt ON images(prompt);")


def _create_gallery_meta_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS gallery_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )


def _images_has_legacy_unique_filename(conn: sqlite3.Connection) -> bool:
    rows = conn.execute("PRAGMA index_list(images)").fetchall()
    for row in rows:
        if int(row["unique"]) != 1:
            continue
        index_name = str(row["name"])
        columns = conn.execute(f"PRAGMA index_info('{index_name}')").fetchall()
        names = [str(col["name"]) for col in columns]
        if names == ["filename"]:
            return True
    return False


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _read_png_metadata(image_path: Path) -> dict[str, Any]:
    with Image.open(image_path) as image:
        info = image.info or {}
    return {str(key): value for key, value in info.items()}


def _resolve_color_filter_flag(color_filter: str | None) -> int | None:
    if color_filter is None:
        return None
    normalized = str(color_filter or "").strip().lower()
    if not normalized:
        return None
    return COLOR_FLAG_BY_NAME.get(normalized)


def _classify_image_color_flags(image_path: Path) -> int:
    with Image.open(image_path) as original:
        rgba = original.convert("RGBA")
        resampling = getattr(getattr(Image, "Resampling", Image), "BOX", Image.BILINEAR)
        sample = rgba.resize((_COLOR_ANALYSIS_SIZE, _COLOR_ANALYSIS_SIZE), resample=resampling)

    classified_pixels = 0
    black_pixels = 0
    white_pixels = 0
    red_pixels = 0
    yellow_pixels = 0
    green_pixels = 0
    blue_pixels = 0

    for red, green, blue, alpha in sample.getdata():
        if int(alpha) < _COLOR_ALPHA_MIN:
            continue
        classified_pixels += 1
        red_f = float(red) / 255.0
        green_f = float(green) / 255.0
        blue_f = float(blue) / 255.0
        hue, saturation, value = colorsys.rgb_to_hsv(red_f, green_f, blue_f)
        hue_deg = hue * 360.0

        if value <= _COLOR_BLACK_VALUE_MAX and saturation <= _COLOR_BLACK_SATURATION_MAX:
            black_pixels += 1
            continue

        if value >= _COLOR_WHITE_VALUE_MIN and saturation <= _COLOR_WHITE_SATURATION_MAX:
            white_pixels += 1

        if value < _COLOR_CHROMA_VALUE_MIN:
            continue
        if (
            saturation >= _COLOR_RED_SATURATION_MIN
            and (hue_deg >= _COLOR_RED_HUE_MIN or hue_deg < _COLOR_RED_HUE_MAX)
        ):
            red_pixels += 1
        elif (
            saturation >= _COLOR_YELLOW_SATURATION_MIN
            and _COLOR_YELLOW_HUE_MIN <= hue_deg < _COLOR_YELLOW_HUE_MAX
        ):
            yellow_pixels += 1
        elif (
            saturation >= _COLOR_GREEN_SATURATION_MIN
            and _COLOR_GREEN_HUE_MIN <= hue_deg < _COLOR_GREEN_HUE_MAX
        ):
            green_pixels += 1
        elif (
            saturation >= _COLOR_BLUE_SATURATION_MIN
            and _COLOR_BLUE_HUE_MIN <= hue_deg < _COLOR_BLUE_HUE_MAX
        ):
            blue_pixels += 1

    if classified_pixels <= 0:
        return 0

    black_ratio = black_pixels / classified_pixels
    white_ratio = white_pixels / classified_pixels
    red_ratio = red_pixels / classified_pixels
    yellow_ratio = yellow_pixels / classified_pixels
    green_ratio = green_pixels / classified_pixels
    blue_ratio = blue_pixels / classified_pixels

    candidates: list[tuple[int, float]] = []
    if black_ratio >= _COLOR_BLACK_DOMINANT_RATIO_MIN:
        candidates.append((COLOR_FLAG_BLACK, black_ratio))
    if white_ratio >= _COLOR_WHITE_DOMINANT_RATIO_MIN:
        candidates.append((COLOR_FLAG_WHITE, white_ratio))
    if red_ratio >= _COLOR_RED_DOMINANT_RATIO_MIN:
        candidates.append((COLOR_FLAG_RED, red_ratio))
    if yellow_ratio >= _COLOR_YELLOW_DOMINANT_RATIO_MIN:
        candidates.append((COLOR_FLAG_YELLOW, yellow_ratio))
    if green_ratio >= _COLOR_GREEN_DOMINANT_RATIO_MIN:
        candidates.append((COLOR_FLAG_GREEN, green_ratio))
    if blue_ratio >= _COLOR_BLUE_DOMINANT_RATIO_MIN:
        candidates.append((COLOR_FLAG_BLUE, blue_ratio))

    if not candidates:
        return 0
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0][0]


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _get_gallery_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM gallery_meta WHERE key = ?", (key,)).fetchone()
    if row is None:
        return None
    value = row["value"]
    return str(value) if value is not None else None


def _set_gallery_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO gallery_meta (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value=excluded.value,
            updated_at=excluded.updated_at
        ;
        """,
        (str(key), str(value), _utc_timestamp()),
    )


def _ensure_optional_columns(conn: sqlite3.Connection) -> None:
    existing_rows = conn.execute("PRAGMA table_info(images)").fetchall()
    existing = {str(row["name"]) for row in existing_rows}
    required_columns = {
        "owner_id": f"TEXT NOT NULL DEFAULT '{LEGACY_OWNER_ID}'",
        "prompt_wildcard_resolved": "TEXT",
        "mode": "TEXT",
        "source_image": "TEXT",
        "source_filename": "TEXT",
        "source_width": "INTEGER",
        "source_height": "INTEGER",
        "similarity": "REAL",
        "wildcards_json": "TEXT",
        "wildcard_count": "INTEGER",
        "loras_json": "TEXT",
        "lora_count": "INTEGER",
        "color_flags": "INTEGER",
        "favorite": "INTEGER NOT NULL DEFAULT 0",
        "inference_process": "TEXT",
        "procedural_creativity": "INTEGER",
    }
    for name, sql_type in required_columns.items():
        if name in existing:
            continue
        conn.execute(f"ALTER TABLE images ADD COLUMN {name} {sql_type}")


def _migrate_images_schema(conn: sqlite3.Connection, settings: AppSettings) -> None:
    _create_gallery_meta_table(conn)
    if not _table_exists(conn, "images"):
        _create_images_table(conn)
        return

    _ensure_optional_columns(conn)
    needs_rebuild = _images_has_legacy_unique_filename(conn)
    if not needs_rebuild:
        _create_images_table(conn)
        return

    legacy_table = "images_legacy_migration"
    conn.execute(f"ALTER TABLE images RENAME TO {legacy_table}")
    legacy_indexes = conn.execute(f"PRAGMA index_list('{legacy_table}')").fetchall()
    for legacy_index in legacy_indexes:
        index_name = str(legacy_index["name"] or "").strip()
        if not index_name or index_name.startswith("sqlite_autoindex_"):
            continue
        conn.execute(f"DROP INDEX IF EXISTS {index_name}")
    _create_images_table(conn)

    rows = conn.execute(f"SELECT * FROM {legacy_table}").fetchall()
    for row in rows:
        record = _row_to_dict(row)
        raw_owner = str(record.get("owner_id") or "").strip()
        if raw_owner:
            try:
                owner_id = normalize_owner_id(raw_owner)
            except ValueError:
                owner_id = LEGACY_OWNER_ID
        else:
            try:
                owner_id = _owner_id_from_output_path(settings, record.get("output_path", ""))
            except ValueError:
                owner_id = LEGACY_OWNER_ID

        filename = str(record.get("filename") or "").strip() or Path(str(record.get("output_path") or "")).name
        if not filename:
            continue
        conn.execute(
            """
            INSERT INTO images (
                owner_id, filename, output_path, prompt, timestamp, application_name, application_version,
                prompt_wildcard_resolved, width, height, model_pack, backend, device, steps, guidance_scale,
                duration_ms, inference_process, procedural_creativity, mode, source_image, source_filename, source_width, source_height, similarity,
                wildcards_json, wildcard_count, loras_json, lora_count, color_flags, favorite, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner_id, filename) DO UPDATE SET
                output_path=excluded.output_path,
                prompt=excluded.prompt,
                timestamp=excluded.timestamp,
                application_name=excluded.application_name,
                application_version=excluded.application_version,
                prompt_wildcard_resolved=excluded.prompt_wildcard_resolved,
                width=excluded.width,
                height=excluded.height,
                model_pack=excluded.model_pack,
                backend=excluded.backend,
                device=excluded.device,
                steps=excluded.steps,
                guidance_scale=excluded.guidance_scale,
                duration_ms=excluded.duration_ms,
                inference_process=excluded.inference_process,
                procedural_creativity=excluded.procedural_creativity,
                mode=excluded.mode,
                source_image=excluded.source_image,
                source_filename=excluded.source_filename,
                source_width=excluded.source_width,
                source_height=excluded.source_height,
                similarity=excluded.similarity,
                wildcards_json=excluded.wildcards_json,
                wildcard_count=excluded.wildcard_count,
                loras_json=excluded.loras_json,
                lora_count=excluded.lora_count,
                color_flags=excluded.color_flags,
                favorite=CASE WHEN images.favorite != 0 THEN images.favorite ELSE excluded.favorite END
            ;
            """,
            (
                owner_id,
                filename,
                str(record.get("output_path") or ""),
                str(record.get("prompt") or "(missing prompt metadata)"),
                str(record.get("timestamp") or _utc_timestamp()),
                record.get("application_name"),
                record.get("application_version"),
                record.get("prompt_wildcard_resolved"),
                _to_int(record.get("width")),
                _to_int(record.get("height")),
                record.get("model_pack"),
                record.get("backend"),
                record.get("device"),
                _to_int(record.get("steps")),
                _to_float(record.get("guidance_scale")),
                _to_int(record.get("duration_ms")),
                record.get("inference_process"),
                _to_int(record.get("procedural_creativity")),
                record.get("mode"),
                record.get("source_image"),
                record.get("source_filename"),
                _to_int(record.get("source_width")),
                _to_int(record.get("source_height")),
                _to_float(record.get("similarity")),
                record.get("wildcards_json"),
                _to_int(record.get("wildcard_count")),
                record.get("loras_json"),
                _to_int(record.get("lora_count")),
                _to_int(record.get("color_flags")),
                1 if int(record.get("favorite") or 0) != 0 else 0,
                str(record.get("created_at") or _utc_timestamp()),
            ),
        )

    conn.execute(f"DROP TABLE IF EXISTS {legacy_table}")


def ensure_gallery_schema(settings: AppSettings) -> Path:
    db_path = _gallery_db_path(settings)
    with _open_connection(db_path) as conn:
        _migrate_images_schema(conn, settings)
        _create_gallery_meta_table(conn)
        conn.commit()
    return db_path


def _prune_missing_rows(
    conn: sqlite3.Connection,
    settings: AppSettings,
    owner_id: str | None = None,
) -> int:
    if owner_id:
        rows = conn.execute("SELECT id, output_path FROM images WHERE owner_id = ?", (owner_id,)).fetchall()
    else:
        rows = conn.execute("SELECT id, output_path FROM images").fetchall()
    missing_ids: list[int] = []
    for row in rows:
        try:
            output_path = _resolve_managed_output_path(settings, row["output_path"])
        except ValueError:
            missing_ids.append(int(row["id"]))
            continue
        if not output_path.exists():
            missing_ids.append(int(row["id"]))

    if not missing_ids:
        return 0

    placeholders = ",".join("?" for _ in missing_ids)
    conn.execute(f"DELETE FROM images WHERE id IN ({placeholders})", tuple(missing_ids))
    return len(missing_ids)


def _upsert_image(
    conn: sqlite3.Connection,
    settings: AppSettings,
    image_path: Path,
    metadata: dict[str, Any],
    owner_id: str | None = None,
) -> dict[str, Any]:
    prompt = str(metadata.get("prompt") or "").strip() or "(missing prompt metadata)"
    timestamp = str(metadata.get("timestamp") or _utc_timestamp())
    created_at = _utc_timestamp()
    filename = image_path.name
    resolved_owner = normalize_owner_id(owner_id) if owner_id else _owner_id_from_output_path(settings, image_path)
    color_flags = _classify_image_color_flags(image_path)

    conn.execute(
        """
        INSERT INTO images (
            owner_id, filename, output_path, prompt, timestamp, application_name, application_version,
            prompt_wildcard_resolved, width, height, model_pack, backend, device, steps, guidance_scale,
            duration_ms, inference_process, procedural_creativity, mode, source_image, source_filename, source_width, source_height, similarity,
            wildcards_json, wildcard_count, loras_json, lora_count, color_flags, favorite, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(owner_id, filename) DO UPDATE SET
            output_path=excluded.output_path,
            prompt=excluded.prompt,
            timestamp=excluded.timestamp,
            application_name=excluded.application_name,
            application_version=excluded.application_version,
            prompt_wildcard_resolved=excluded.prompt_wildcard_resolved,
            width=excluded.width,
            height=excluded.height,
            model_pack=excluded.model_pack,
            backend=excluded.backend,
            device=excluded.device,
            steps=excluded.steps,
            guidance_scale=excluded.guidance_scale,
            duration_ms=excluded.duration_ms,
            inference_process=excluded.inference_process,
            procedural_creativity=excluded.procedural_creativity,
            mode=excluded.mode,
            source_image=excluded.source_image,
            source_filename=excluded.source_filename,
            source_width=excluded.source_width,
            source_height=excluded.source_height,
            similarity=excluded.similarity,
            wildcards_json=excluded.wildcards_json,
            wildcard_count=excluded.wildcard_count,
            loras_json=excluded.loras_json,
            lora_count=excluded.lora_count,
            color_flags=excluded.color_flags,
            favorite=images.favorite
        ;
        """,
        (
            resolved_owner,
            filename,
            str(image_path.resolve()),
            prompt,
            timestamp,
            metadata.get("application_name"),
            metadata.get("application_version"),
            metadata.get("prompt_wildcard_resolved"),
            _to_int(metadata.get("width")),
            _to_int(metadata.get("height")),
            metadata.get("model_pack"),
            metadata.get("backend"),
            metadata.get("device"),
            _to_int(metadata.get("steps")),
            _to_float(metadata.get("guidance_scale")),
            _to_int(metadata.get("duration_ms")),
            metadata.get("inference_process"),
            _to_int(metadata.get("procedural_creativity")),
            metadata.get("mode"),
            metadata.get("source_image"),
            metadata.get("source_filename"),
            _to_int(metadata.get("source_width")),
            _to_int(metadata.get("source_height")),
            _to_float(metadata.get("similarity")),
            metadata.get("wildcards_json"),
            _to_int(metadata.get("wildcard_count")),
            metadata.get("loras_json"),
            _to_int(metadata.get("lora_count")),
            color_flags,
            0,
            created_at,
        ),
    )
    row = conn.execute(
        "SELECT * FROM images WHERE owner_id = ? AND filename = ?",
        (resolved_owner, filename),
    ).fetchone()
    if row is None:
        raise ValueError(f"Failed to index image metadata for '{filename}'.")
    return _row_to_dict(row)


def index_image(settings: AppSettings, image_path: Path, owner_id: str | None = None) -> dict[str, Any]:
    db_path = ensure_gallery_schema(settings)
    resolved = _resolve_managed_output_path(settings, image_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Image file not found: {resolved}")
    metadata = _read_png_metadata(resolved)
    with _open_connection(db_path) as conn:
        record = _upsert_image(conn, settings, resolved, metadata, owner_id=owner_id)
        conn.commit()
    return record


def sync_outputs_to_gallery(settings: AppSettings) -> int:
    db_path = ensure_gallery_schema(settings)
    outputs_dir = settings.paths.outputs_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_files = sorted(_iter_managed_images(outputs_dir.resolve()))

    indexed = 0
    with _open_connection(db_path) as conn:
        removed_missing = _prune_missing_rows(conn, settings)
        existing_rows = conn.execute("SELECT owner_id, filename FROM images").fetchall()
        existing = {(str(row["owner_id"]), str(row["filename"])) for row in existing_rows}
        for output_path in output_files:
            owner = _owner_id_from_output_path(settings, output_path)
            key = (owner, output_path.name)
            if key in existing:
                continue
            metadata = _read_png_metadata(output_path)
            _upsert_image(conn, settings, output_path, metadata, owner_id=owner)
            indexed += 1
        conn.commit()
    return indexed + removed_missing


def rebuild_gallery(settings: AppSettings, owner_id: str) -> dict[str, int | str]:
    db_path = ensure_gallery_schema(settings)
    safe_owner = normalize_owner_id(owner_id)
    outputs_dir = settings.paths.outputs_dir.resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    if safe_owner == LEGACY_OWNER_ID:
        output_files = sorted(path for path in outputs_dir.glob("*.png") if path.is_file())
    else:
        owner_dir = _owner_output_dir(settings, safe_owner)
        owner_dir.mkdir(parents=True, exist_ok=True)
        output_files = sorted(_iter_managed_images(owner_dir))

    indexed = 0
    updated = 0
    with _open_connection(db_path) as conn:
        removed_missing = _prune_missing_rows(conn, settings, owner_id=safe_owner)
        existing_rows = conn.execute(
            "SELECT filename FROM images WHERE owner_id = ?",
            (safe_owner,),
        ).fetchall()
        existing = {str(row["filename"]) for row in existing_rows}
        for output_path in output_files:
            metadata = _read_png_metadata(output_path)
            _upsert_image(conn, settings, output_path, metadata, owner_id=safe_owner)
            if output_path.name in existing:
                updated += 1
            else:
                indexed += 1
        total_row = conn.execute(
            "SELECT COUNT(*) AS total FROM images WHERE owner_id = ?",
            (safe_owner,),
        ).fetchone()
        conn.commit()

    return {
        "owner_id": safe_owner,
        "scanned_files": len(output_files),
        "indexed": indexed,
        "updated": updated,
        "removed_missing": removed_missing,
        "total_items": int(total_row["total"]) if total_row is not None else 0,
    }


def _rebuild_color_flags(
    conn: sqlite3.Connection,
    settings: AppSettings,
    *,
    owner_id: str | None = None,
    batch_size: int = 100,
) -> int:
    clauses: list[str] = []
    params: list[Any] = []
    if owner_id:
        clauses.append("owner_id = ?")
        params.append(owner_id)
    rows = conn.execute(
        "SELECT id, output_path, color_flags FROM images"
        + (f" WHERE {' AND '.join(clauses)}" if clauses else ""),
        tuple(params),
    ).fetchall()
    updated = 0
    processed = 0
    for row in rows:
        try:
            image_path = _resolve_managed_output_path(settings, row["output_path"])
            color_flags = _classify_image_color_flags(image_path) if image_path.exists() else 0
        except Exception:
            color_flags = 0
        if int(row["color_flags"] or 0) == int(color_flags):
            processed += 1
        else:
            conn.execute(
                "UPDATE images SET color_flags = ? WHERE id = ?",
                (int(color_flags), int(row["id"])),
            )
            updated += 1
            processed += 1
        if processed % max(1, batch_size) == 0:
            conn.commit()
    return updated


def gallery_color_cache_version(settings: AppSettings) -> str | None:
    db_path = ensure_gallery_schema(settings)
    with _open_connection(db_path) as conn:
        return _get_gallery_meta(conn, _GALLERY_META_COLOR_CACHE_VERSION_KEY)


def gallery_color_cache_needs_rebuild(settings: AppSettings) -> bool:
    version = gallery_color_cache_version(settings)
    return version != COLOR_CACHE_VERSION


def rebuild_gallery_color_cache(settings: AppSettings, *, batch_size: int = 100) -> int:
    db_path = ensure_gallery_schema(settings)
    with _open_connection(db_path) as conn:
        updated = _rebuild_color_flags(conn, settings, batch_size=batch_size)
        _set_gallery_meta(conn, _GALLERY_META_COLOR_CACHE_VERSION_KEY, COLOR_CACHE_VERSION)
        conn.commit()
    return updated


def list_images(
    settings: AppSettings,
    owner_id: str | None = None,
    prompt_query: str | None = None,
    color_filter: str | None = None,
    favorites_only: bool = False,
    limit: int = 100,
    offset: int = 0,
    newest_first: bool = True,
) -> list[dict[str, Any]]:
    db_path = ensure_gallery_schema(settings)
    safe_limit = max(1, min(limit, 500))
    safe_offset = max(0, offset)
    order_keyword = "DESC" if newest_first else "ASC"
    scoped_owner = normalize_owner_id(owner_id) if owner_id else None
    color_flag = _resolve_color_filter_flag(color_filter)
    with _open_connection(db_path) as conn:
        removed_missing = _prune_missing_rows(conn, settings, owner_id=scoped_owner)
        if removed_missing:
            conn.commit()

        clauses: list[str] = []
        params: list[Any] = []
        if scoped_owner:
            clauses.append("owner_id = ?")
            params.append(scoped_owner)
        if prompt_query:
            clauses.append("prompt LIKE ?")
            params.append(f"%{prompt_query}%")
        if color_flag is not None:
            clauses.append("(COALESCE(color_flags, 0) & ?) != 0")
            params.append(color_flag)
        if favorites_only:
            clauses.append("COALESCE(favorite, 0) = 1")
        where_clause = "WHERE " + " AND ".join(clauses) if clauses else ""

        rows = conn.execute(
            f"""
            SELECT * FROM images
            {where_clause}
            ORDER BY COALESCE(timestamp, created_at) {order_keyword}, id {order_keyword}
            LIMIT ? OFFSET ?
            """,
            (*params, safe_limit, safe_offset),
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def get_image(settings: AppSettings, filename: str, owner_id: str | None = None) -> dict[str, Any] | None:
    db_path = ensure_gallery_schema(settings)
    scoped_owner = normalize_owner_id(owner_id) if owner_id else None
    with _open_connection(db_path) as conn:
        if scoped_owner:
            row = conn.execute(
                "SELECT * FROM images WHERE owner_id = ? AND filename = ?",
                (scoped_owner, filename),
            ).fetchone()
        else:
            row = conn.execute("SELECT * FROM images WHERE filename = ?", (filename,)).fetchone()
        if row is not None:
            try:
                output_path = _resolve_managed_output_path(settings, row["output_path"])
            except ValueError:
                conn.execute("DELETE FROM images WHERE id = ?", (int(row["id"]),))
                conn.commit()
                row = None
            else:
                if not output_path.exists():
                    conn.execute("DELETE FROM images WHERE id = ?", (int(row["id"]),))
                    conn.commit()
                    row = None
    if row is None:
        return None
    return _row_to_dict(row)


def set_image_favorite(
    settings: AppSettings,
    filename: str,
    favorite: bool,
    owner_id: str | None = None,
) -> dict[str, Any]:
    db_path = ensure_gallery_schema(settings)
    scoped_owner = normalize_owner_id(owner_id) if owner_id else None
    favorite_value = 1 if favorite else 0
    with _open_connection(db_path) as conn:
        if scoped_owner:
            row = conn.execute(
                "SELECT id FROM images WHERE owner_id = ? AND filename = ?",
                (scoped_owner, filename),
            ).fetchone()
        else:
            row = conn.execute("SELECT id FROM images WHERE filename = ?", (filename,)).fetchone()
        if row is None:
            raise ValueError("Image not found.")
        conn.execute(
            "UPDATE images SET favorite = ? WHERE id = ?",
            (favorite_value, int(row["id"])),
        )
        conn.commit()
    updated = get_image(settings, filename, owner_id=scoped_owner)
    if updated is None:
        raise ValueError("Image not found.")
    return updated


def delete_gallery(settings: AppSettings, owner_id: str | None = None) -> dict[str, int]:
    db_path = ensure_gallery_schema(settings)
    deleted_files = 0
    deleted_rows = 0
    remaining_rows = 0
    scoped_owner = normalize_owner_id(owner_id) if owner_id else None

    outputs_dir = settings.paths.outputs_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)
    with _open_connection(db_path) as conn:
        if scoped_owner:
            rows = conn.execute(
                "SELECT id, output_path FROM images WHERE owner_id = ?",
                (scoped_owner,),
            ).fetchall()
            for record in rows:
                try:
                    output_path = _resolve_managed_output_path(settings, record["output_path"])
                except ValueError:
                    continue
                try:
                    if output_path.exists():
                        output_path.unlink()
                        deleted_files += 1
                except OSError:
                    continue
            deleted_rows = len(rows)
            conn.execute("DELETE FROM images WHERE owner_id = ?", (scoped_owner,))
            verify_row = conn.execute(
                "SELECT COUNT(*) AS total FROM images WHERE owner_id = ?",
                (scoped_owner,),
            ).fetchone()
            if verify_row is not None:
                remaining_rows = int(verify_row["total"])
        else:
            rows = conn.execute("SELECT id, output_path FROM images").fetchall()
            deleted_rows = len(rows)
            for record in rows:
                try:
                    output_path = _resolve_managed_output_path(settings, record["output_path"])
                except ValueError:
                    continue
                try:
                    if output_path.exists():
                        output_path.unlink()
                        deleted_files += 1
                except OSError:
                    continue
            conn.execute("DELETE FROM images")
            conn.execute("DELETE FROM sqlite_sequence WHERE name = 'images'")
            verify_row = conn.execute("SELECT COUNT(*) AS total FROM images").fetchone()
            if verify_row is not None:
                remaining_rows = int(verify_row["total"])
        conn.commit()

    cleanup_root = _owner_output_dir(settings, scoped_owner) if scoped_owner else outputs_dir.resolve()
    for image_path in _iter_managed_images(cleanup_root):
        try:
            image_path.unlink()
            deleted_files += 1
        except OSError:
            continue

    if scoped_owner:
        _remove_empty_parents(cleanup_root, outputs_dir.resolve())

    return {
        "deleted_files": deleted_files,
        "deleted_rows": deleted_rows,
        "remaining_rows": remaining_rows,
    }


def delete_image(settings: AppSettings, filename: str, owner_id: str | None = None) -> dict[str, int]:
    db_path = ensure_gallery_schema(settings)
    deleted_files = 0
    deleted_rows = 0
    remaining_rows = 0
    scoped_owner = normalize_owner_id(owner_id) if owner_id else None

    with _open_connection(db_path) as conn:
        if scoped_owner:
            row = conn.execute(
                "SELECT id, output_path FROM images WHERE owner_id = ? AND filename = ?",
                (scoped_owner, filename),
            ).fetchone()
        else:
            row = conn.execute("SELECT id, output_path FROM images WHERE filename = ?", (filename,)).fetchone()
        if row is None:
            raise ValueError("Image not found.")

        output_path = _resolve_managed_output_path(settings, row["output_path"])
        conn.execute("DELETE FROM images WHERE id = ?", (int(row["id"]),))
        deleted_rows = conn.total_changes
        if scoped_owner:
            remaining = conn.execute(
                "SELECT COUNT(*) AS total FROM images WHERE owner_id = ?",
                (scoped_owner,),
            ).fetchone()
        else:
            remaining = conn.execute("SELECT COUNT(*) AS total FROM images").fetchone()
        if remaining is not None:
            remaining_rows = int(remaining["total"])
        conn.commit()

    try:
        if output_path.exists():
            output_path.unlink()
            deleted_files = 1
    except OSError:
        deleted_files = 0

    return {
        "deleted_files": deleted_files,
        "deleted_rows": deleted_rows,
        "remaining_rows": remaining_rows,
    }


def list_import_sources(settings: AppSettings, target_owner_id: str) -> list[dict[str, Any]]:
    target_owner = normalize_owner_id(target_owner_id)
    outputs_dir = settings.paths.outputs_dir.resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    sources: list[dict[str, Any]] = []

    legacy_files = sorted(outputs_dir.glob("*.png"))
    if legacy_files and target_owner != LEGACY_OWNER_ID:
        sources.append(
            {
                "source_id": LEGACY_IMPORT_SOURCE_ID,
                "label": "Legacy root outputs",
                "path": str(outputs_dir),
                "image_count": len(legacy_files),
                "legacy_root": True,
            }
        )

    for child in sorted(outputs_dir.iterdir()):
        if not child.is_dir():
            continue
        child_owner = str(child.name).strip()
        try:
            normalized_owner = normalize_owner_id(child_owner)
        except ValueError:
            continue
        if normalized_owner == target_owner:
            continue
        count = len(list(child.rglob("*.png")))
        if count <= 0:
            continue
        sources.append(
            {
                "source_id": normalized_owner,
                "label": f"User space: {normalized_owner}",
                "path": str(child.resolve()),
                "image_count": count,
                "legacy_root": False,
            }
        )
    return sources


def _resolve_import_source(settings: AppSettings, source_id: str) -> tuple[Path, bool]:
    outputs_dir = settings.paths.outputs_dir.resolve()
    source = str(source_id or "").strip()
    if source == LEGACY_IMPORT_SOURCE_ID:
        return outputs_dir, False

    normalized = normalize_owner_id(source)
    source_dir = (outputs_dir / normalized).resolve()
    try:
        source_dir.relative_to(outputs_dir)
    except ValueError as exc:
        raise ValueError("Import source is outside managed outputs directory.") from exc
    if not source_dir.exists() or not source_dir.is_dir():
        raise ValueError(f"Import source does not exist: {source_id}")
    return source_dir, True


def _build_unique_copy_path(target_dir: Path, source_name: str) -> Path:
    source_path = Path(source_name)
    stem = source_path.stem
    suffix = source_path.suffix or ".png"
    candidate = target_dir / source_path.name
    if not candidate.exists():
        return candidate
    index = 1
    while True:
        candidate = target_dir / f"{stem}_import_{index:03d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def import_gallery_source(
    settings: AppSettings,
    target_owner_id: str,
    source_id: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    target_owner = normalize_owner_id(target_owner_id)
    source_root, recursive = _resolve_import_source(settings, source_id)

    outputs_dir = settings.paths.outputs_dir.resolve()
    target_dir = (outputs_dir / target_owner).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    if source_root == target_dir:
        raise ValueError("Cannot import from the current user space.")

    source_files = sorted(source_root.rglob("*.png") if recursive else source_root.glob("*.png"))
    total_candidates = len(source_files)
    imported = 0
    skipped = 0
    failed = 0
    errors: list[str] = []

    for source_path in source_files:
        try:
            source_resolved = source_path.resolve()
            target_path = _build_unique_copy_path(target_dir, source_resolved.name)
            if source_resolved == target_path:
                skipped += 1
                continue
            if not dry_run:
                shutil.copy2(source_resolved, target_path)
                index_image(settings, target_path, owner_id=target_owner)
            imported += 1
        except Exception as exc:  # pragma: no cover - defensive guard
            failed += 1
            errors.append(f"{source_path.name}: {exc}")

    return {
        "status": "ok",
        "source_id": source_id,
        "target_owner_id": target_owner,
        "dry_run": bool(dry_run),
        "total_candidates": total_candidates,
        "imported": imported,
        "skipped": skipped,
        "failed": failed,
        "errors": errors[:20],
    }


