from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config.settings import AppSettings

_MAX_WILDCARD_TOKEN_LENGTH = 96
_MAX_WILDCARD_DISPLAY_NAME_LENGTH = 128
_MAX_WILDCARD_CONTENT_LENGTH = 200_000
_WILDCARD_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def wildcard_library_dir(settings: AppSettings) -> Path:
    return settings.paths.data_dir / "wildcards"


def ensure_wildcard_library(settings: AppSettings) -> Path:
    target = wildcard_library_dir(settings)
    target.mkdir(parents=True, exist_ok=True)
    return target


def normalize_wildcard_token(raw_token: Any) -> str:
    text = str(raw_token or "").strip().lower()
    pieces: list[str] = []
    for ch in text:
        if ch.isalnum():
            pieces.append(ch)
            continue
        if ch in {" ", "_", "-", ".", ":"}:
            pieces.append("-")
    normalized = re.sub(r"-{2,}", "-", "".join(pieces)).strip("-")
    if not normalized:
        raise ValueError("Wildcard token is invalid.")
    return normalized[:_MAX_WILDCARD_TOKEN_LENGTH]


def sanitize_wildcard_display_name(raw_display_name: Any, *, fallback: str) -> str:
    display_name = re.sub(r"\s+", " ", str(raw_display_name or "").strip())
    if not display_name:
        display_name = re.sub(r"\s+", " ", str(fallback or "").strip())
    if not display_name:
        raise ValueError("Wildcard name is required.")
    return display_name[:_MAX_WILDCARD_DISPLAY_NAME_LENGTH]


def normalize_wildcard_entry_value(raw_value: Any) -> str:
    return re.sub(r"\s+", " ", str(raw_value or "").strip())


def normalize_wildcard_content(raw_content: Any) -> tuple[str, list[str]]:
    text = str(raw_content or "")
    if len(text) > _MAX_WILDCARD_CONTENT_LENGTH:
        raise ValueError("Wildcard entries are too large.")
    normalized_lines: list[str] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        normalized = normalize_wildcard_entry_value(line)
        if not normalized:
            continue
        normalized_lines.append(normalized)
    if not normalized_lines:
        raise ValueError("Wildcard entries must include at least one non-empty line.")
    return "\n".join(normalized_lines), normalized_lines


def wildcard_placeholder(token: Any) -> str:
    return f"__{normalize_wildcard_token(token)}__"


def _record_path(settings: AppSettings, wildcard_id: str) -> Path:
    return ensure_wildcard_library(settings) / f"{_sanitize_wildcard_id(wildcard_id)}.json"


def _sanitize_wildcard_id(raw_wildcard_id: Any) -> str:
    wildcard_id = str(raw_wildcard_id or "").strip().lower()
    if not _WILDCARD_ID_PATTERN.fullmatch(wildcard_id):
        raise ValueError("Wildcard id is invalid.")
    return wildcard_id


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalize_record_payload(payload: dict[str, Any], *, wildcard_id: str | None = None) -> dict[str, Any]:
    record_id = _sanitize_wildcard_id(payload.get("id") or wildcard_id or "")
    token = normalize_wildcard_token(payload.get("token"))
    content_text, entries = normalize_wildcard_content(payload.get("content_text"))
    created_at = str(payload.get("created_at") or _utc_timestamp())
    updated_at = str(payload.get("updated_at") or created_at)
    return {
        "id": record_id,
        "display_name": sanitize_wildcard_display_name(payload.get("display_name"), fallback=token),
        "token": token,
        "content_text": content_text,
        "entries": entries,
        "entry_count": len(entries),
        "created_at": created_at,
        "updated_at": updated_at,
    }


def _write_record(settings: AppSettings, record: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_record_payload(record, wildcard_id=record.get("id"))
    path = _record_path(settings, normalized["id"])
    payload = json.dumps(normalized, indent=2, ensure_ascii=True)
    path.write_text(f"{payload}\n", encoding="utf-8")
    return normalized


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "display_name": record["display_name"],
        "token": record["token"],
        "placeholder": wildcard_placeholder(record["token"]),
        "content_text": record["content_text"],
        "entry_count": int(record.get("entry_count") or len(record.get("entries") or [])),
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
    }


def _iter_records(settings: AppSettings) -> list[dict[str, Any]]:
    root = ensure_wildcard_library(settings)
    records: list[dict[str, Any]] = []
    for sidecar_path in sorted(root.glob("*.json")):
        try:
            record = _normalize_record_payload(_read_json(sidecar_path), wildcard_id=sidecar_path.stem)
        except ValueError:
            continue
        records.append(record)
    return records


def _ensure_unique_token(settings: AppSettings, token: str, *, exclude_id: str | None = None) -> None:
    normalized_token = normalize_wildcard_token(token)
    excluded = _sanitize_wildcard_id(exclude_id) if exclude_id else None
    for record in _iter_records(settings):
        if excluded is not None and record["id"] == excluded:
            continue
        if record["token"] == normalized_token:
            raise ValueError(f"Wildcard token already exists: {normalized_token}")


def get_wildcard(settings: AppSettings, wildcard_id: str) -> dict[str, Any] | None:
    try:
        path = _record_path(settings, wildcard_id)
    except ValueError:
        return None
    if not path.exists():
        return None
    try:
        return _normalize_record_payload(_read_json(path), wildcard_id=path.stem)
    except ValueError:
        return None


def get_wildcard_by_token(settings: AppSettings, token: str) -> dict[str, Any] | None:
    normalized_token = normalize_wildcard_token(token)
    for record in _iter_records(settings):
        if record["token"] == normalized_token:
            return record
    return None


def list_wildcards(settings: AppSettings) -> list[dict[str, Any]]:
    records = sorted(
        _iter_records(settings),
        key=lambda item: (str(item.get("display_name") or "").lower(), str(item.get("token") or "").lower(), item["id"]),
    )
    return [_public_record(record) for record in records]


def create_wildcard(
    settings: AppSettings,
    *,
    display_name: Any,
    token: Any,
    content_text: Any,
) -> dict[str, Any]:
    normalized_token = normalize_wildcard_token(token)
    _ensure_unique_token(settings, normalized_token)
    created_at = _utc_timestamp()
    record = _write_record(
        settings,
        {
            "id": uuid4().hex,
            "display_name": sanitize_wildcard_display_name(display_name, fallback=normalized_token),
            "token": normalized_token,
            "content_text": content_text,
            "created_at": created_at,
            "updated_at": created_at,
        },
    )
    return _public_record(record)


def update_wildcard(
    settings: AppSettings,
    *,
    wildcard_id: str,
    display_name: Any,
    token: Any,
    content_text: Any,
) -> dict[str, Any]:
    record = get_wildcard(settings, wildcard_id)
    if record is None:
        raise FileNotFoundError(f"Wildcard not found: {wildcard_id}")
    normalized_token = normalize_wildcard_token(token)
    _ensure_unique_token(settings, normalized_token, exclude_id=record["id"])
    updated = _write_record(
        settings,
        {
            **record,
            "display_name": sanitize_wildcard_display_name(display_name, fallback=normalized_token),
            "token": normalized_token,
            "content_text": content_text,
            "updated_at": _utc_timestamp(),
        },
    )
    return _public_record(updated)


def delete_wildcard(settings: AppSettings, wildcard_id: str) -> dict[str, Any]:
    record = get_wildcard(settings, wildcard_id)
    if record is None:
        raise FileNotFoundError(f"Wildcard not found: {wildcard_id}")
    path = _record_path(settings, record["id"])
    path.unlink(missing_ok=True)
    return {"id": record["id"], "deleted": True}


__all__ = [
    "create_wildcard",
    "delete_wildcard",
    "ensure_wildcard_library",
    "get_wildcard",
    "get_wildcard_by_token",
    "list_wildcards",
    "normalize_wildcard_content",
    "normalize_wildcard_entry_value",
    "normalize_wildcard_token",
    "sanitize_wildcard_display_name",
    "update_wildcard",
    "wildcard_library_dir",
    "wildcard_placeholder",
]
