from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from app.config.settings import AppSettings


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _gallery_db_path(settings: AppSettings) -> Path:
    return settings.paths.data_dir / "gallery.db"


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def ensure_gallery_schema(settings: AppSettings) -> Path:
    db_path = _gallery_db_path(settings)
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE,
                output_path TEXT NOT NULL,
                prompt TEXT NOT NULL,
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
                mode TEXT,
                source_image TEXT,
                source_filename TEXT,
                source_width INTEGER,
                source_height INTEGER,
                created_at TEXT NOT NULL
            );
            """
        )
        _ensure_optional_columns(conn)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_images_timestamp ON images(timestamp DESC);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_images_prompt ON images(prompt);")
        conn.commit()
    return db_path


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


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _ensure_optional_columns(conn: sqlite3.Connection) -> None:
    existing_rows = conn.execute("PRAGMA table_info(images)").fetchall()
    existing = {str(row["name"]) for row in existing_rows}
    required_columns = {
        "mode": "TEXT",
        "source_image": "TEXT",
        "source_filename": "TEXT",
        "source_width": "INTEGER",
        "source_height": "INTEGER",
    }
    for name, sql_type in required_columns.items():
        if name in existing:
            continue
        conn.execute(f"ALTER TABLE images ADD COLUMN {name} {sql_type}")


def _upsert_image(
    conn: sqlite3.Connection,
    image_path: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    prompt = str(metadata.get("prompt") or "").strip()
    if not prompt:
        prompt = "(missing prompt metadata)"
    timestamp = str(metadata.get("timestamp") or _utc_timestamp())
    created_at = _utc_timestamp()
    filename = image_path.name

    conn.execute(
        """
        INSERT INTO images (
            filename, output_path, prompt, timestamp, application_name, application_version,
            width, height, model_pack, backend, device, steps, guidance_scale, duration_ms,
            mode, source_image, source_filename, source_width, source_height, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(filename) DO UPDATE SET
            output_path=excluded.output_path,
            prompt=excluded.prompt,
            timestamp=excluded.timestamp,
            application_name=excluded.application_name,
            application_version=excluded.application_version,
            width=excluded.width,
            height=excluded.height,
            model_pack=excluded.model_pack,
            backend=excluded.backend,
            device=excluded.device,
            steps=excluded.steps,
            guidance_scale=excluded.guidance_scale,
            duration_ms=excluded.duration_ms,
            mode=excluded.mode,
            source_image=excluded.source_image,
            source_filename=excluded.source_filename,
            source_width=excluded.source_width,
            source_height=excluded.source_height
        ;
        """,
        (
            filename,
            str(image_path.resolve()),
            prompt,
            timestamp,
            metadata.get("application_name"),
            metadata.get("application_version"),
            _to_int(metadata.get("width")),
            _to_int(metadata.get("height")),
            metadata.get("model_pack"),
            metadata.get("backend"),
            metadata.get("device"),
            _to_int(metadata.get("steps")),
            _to_float(metadata.get("guidance_scale")),
            _to_int(metadata.get("duration_ms")),
            metadata.get("mode"),
            metadata.get("source_image"),
            metadata.get("source_filename"),
            _to_int(metadata.get("source_width")),
            _to_int(metadata.get("source_height")),
            created_at,
        ),
    )
    row = conn.execute("SELECT * FROM images WHERE filename = ?", (filename,)).fetchone()
    if row is None:
        raise ValueError(f"Failed to index image metadata for '{filename}'.")
    return _row_to_dict(row)


def index_image(settings: AppSettings, image_path: Path) -> dict[str, Any]:
    db_path = ensure_gallery_schema(settings)
    resolved = image_path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Image file not found: {resolved}")
    metadata = _read_png_metadata(resolved)
    with _connect(db_path) as conn:
        record = _upsert_image(conn, resolved, metadata)
        conn.commit()
    return record


def sync_outputs_to_gallery(settings: AppSettings) -> int:
    db_path = ensure_gallery_schema(settings)
    outputs_dir = settings.paths.outputs_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_files = sorted(outputs_dir.glob("*.png"))
    if not output_files:
        return 0

    indexed = 0
    with _connect(db_path) as conn:
        existing_rows = conn.execute("SELECT filename FROM images").fetchall()
        existing = {str(row["filename"]) for row in existing_rows}
        for output_path in output_files:
            if output_path.name in existing:
                continue
            metadata = _read_png_metadata(output_path)
            _upsert_image(conn, output_path, metadata)
            indexed += 1
        conn.commit()
    return indexed


def list_images(
    settings: AppSettings,
    prompt_query: str | None = None,
    limit: int = 100,
    offset: int = 0,
    newest_first: bool = True,
) -> list[dict[str, Any]]:
    db_path = ensure_gallery_schema(settings)
    safe_limit = max(1, min(limit, 500))
    safe_offset = max(0, offset)
    order_keyword = "DESC" if newest_first else "ASC"
    with _connect(db_path) as conn:
        if prompt_query:
            rows = conn.execute(
                f"""
                SELECT * FROM images
                WHERE prompt LIKE ?
                ORDER BY COALESCE(timestamp, created_at) {order_keyword}, id {order_keyword}
                LIMIT ? OFFSET ?
                """,
                (f"%{prompt_query}%", safe_limit, safe_offset),
            ).fetchall()
        else:
            rows = conn.execute(
                f"""
                SELECT * FROM images
                ORDER BY COALESCE(timestamp, created_at) {order_keyword}, id {order_keyword}
                LIMIT ? OFFSET ?
                """,
                (safe_limit, safe_offset),
            ).fetchall()
    return [_row_to_dict(row) for row in rows]


def get_image(settings: AppSettings, filename: str) -> dict[str, Any] | None:
    db_path = ensure_gallery_schema(settings)
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM images WHERE filename = ?", (filename,)).fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


def delete_gallery(settings: AppSettings) -> dict[str, int]:
    db_path = ensure_gallery_schema(settings)
    deleted_files = 0
    deleted_rows = 0
    remaining_rows = 0

    outputs_dir = settings.paths.outputs_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT output_path FROM images").fetchall()
        row = conn.execute("SELECT COUNT(*) AS total FROM images").fetchone()
        if row is not None:
            deleted_rows = int(row["total"])
        for record in rows:
            output_path = Path(str(record["output_path"])).expanduser()
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

    for image_path in outputs_dir.rglob("*"):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            continue
        try:
            image_path.unlink()
            deleted_files += 1
        except OSError:
            continue

    return {
        "deleted_files": deleted_files,
        "deleted_rows": deleted_rows,
        "remaining_rows": remaining_rows,
    }


def delete_image(settings: AppSettings, filename: str) -> dict[str, int]:
    db_path = ensure_gallery_schema(settings)
    deleted_files = 0
    deleted_rows = 0
    remaining_rows = 0

    with _connect(db_path) as conn:
        row = conn.execute("SELECT output_path FROM images WHERE filename = ?", (filename,)).fetchone()
        if row is None:
            raise ValueError("Image not found.")

        output_path = Path(str(row["output_path"])).expanduser()
        conn.execute("DELETE FROM images WHERE filename = ?", (filename,))
        deleted_rows = conn.total_changes
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
