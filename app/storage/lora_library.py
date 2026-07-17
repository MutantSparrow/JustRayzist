from __future__ import annotations

import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps
from safetensors import safe_open

from app.config.settings import AppSettings

DEFAULT_MAX_ACTIVE_LORAS = 3
DEFAULT_LORA_WEIGHT = 1.0
MIN_LORA_WEIGHT = -2.0
MAX_LORA_WEIGHT = 2.0
PLACEHOLDER_PREVIEW_SIZE = 1024
# Only two model architectures ship with the app today. Legacy sidecars without an
# `architecture` field predate the Krea2 pack and are backfilled to z_image_turbo — the only
# architecture that supported LoRAs at that point.
SUPPORTED_LORA_ARCHITECTURES = frozenset({"z_image_turbo", "krea2_turbo"})
DEFAULT_LORA_ARCHITECTURE = "z_image_turbo"


def normalize_lora_architecture(raw: Any, *, fallback: str = DEFAULT_LORA_ARCHITECTURE) -> str:
    """Coerce a raw architecture string to one of ``SUPPORTED_LORA_ARCHITECTURES``.

    Legacy sidecars written before Krea2 support have no ``architecture`` key at all —
    they hydrate to ``DEFAULT_LORA_ARCHITECTURE`` (``z_image_turbo``). Anything unrecognized
    also falls back to keep the drawer usable rather than erroring at hydrate time.
    """
    value = str(raw or "").strip().lower()
    if value in SUPPORTED_LORA_ARCHITECTURES:
        return value
    return fallback
_LORA_SUFFIX = ".safetensors"
_PREVIEW_SUFFIX = ".png"
_PLACEHOLDER_BACKGROUND = "#101521"
_PLACEHOLDER_ACCENT = "#e13aa0"
_PLACEHOLDER_DETAIL = "#9fe870"
_TRIGGER_VALUE_MARKERS = (
    "trigger",
    "activation",
    "trainedwords",
    "trained_words",
    "instance_prompt",
    "instanceprompt",
    "tag_frequency",
    "tag frequency",
    "tags",
)
_EXPLICIT_TRIGGER_VALUE_MARKERS = (
    "trigger",
    "activation",
    "trainedwords",
    "trained_words",
    "instance_prompt",
    "instanceprompt",
)
_TAG_SCORE_VALUE_MARKERS = (
    "tag_frequency",
    "tag frequency",
    "tagfrequency",
    "ss_tag_frequency",
)
_TAG_LIST_VALUE_MARKERS = (
    "tags",
    "tag_list",
    "tag list",
)
_SUMMARY_KEY_MARKERS = (
    "modelspec.",
    "ss_network_",
    "ss_sd_model",
    "ss_base_model",
    "ss_resolution",
    "ss_output_name",
    "ss_training_",
)
_RESAMPLING_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def lora_library_dir(settings: AppSettings) -> Path:
    return settings.paths.models_dir / "loras"


def lora_drafts_dir(settings: AppSettings) -> Path:
    return settings.paths.data_dir / "lora_drafts"


def ensure_lora_library(settings: AppSettings) -> Path:
    target = lora_library_dir(settings)
    target.mkdir(parents=True, exist_ok=True)
    return target


def ensure_lora_drafts(settings: AppSettings) -> Path:
    target = lora_drafts_dir(settings)
    target.mkdir(parents=True, exist_ok=True)
    return target


def normalize_lora_id(raw_lora_id: str) -> str:
    stem = Path(str(raw_lora_id or "")).stem.strip().lower()
    pieces: list[str] = []
    for ch in stem:
        if ch.isalnum():
            pieces.append(ch)
            continue
        if ch in {" ", "_", "-", ".", ":"}:
            pieces.append("-")
    normalized = re.sub(r"-{2,}", "-", "".join(pieces)).strip("-")
    if not normalized:
        raise ValueError("LoRA id is invalid.")
    return normalized[:96]


def sanitize_upload_filename(filename: str) -> str:
    sanitized = Path(str(filename or "")).name
    if not sanitized or sanitized != filename:
        raise ValueError("Invalid upload filename.")
    if Path(sanitized).suffix.lower() != _LORA_SUFFIX:
        raise ValueError("Only .safetensors LoRA uploads are supported.")
    return sanitized


def sanitize_display_name(raw_display_name: Any, *, fallback: str) -> str:
    display_name = re.sub(r"\s+", " ", str(raw_display_name or "").strip())
    if not display_name:
        display_name = re.sub(r"\s+", " ", str(fallback or "").strip())
    if not display_name:
        raise ValueError("LoRA name is required.")
    return display_name[:128]


def normalize_trigger_words(raw_trigger_words: Any) -> list[str]:
    if raw_trigger_words is None:
        return []
    if isinstance(raw_trigger_words, str):
        candidate = raw_trigger_words.strip()
        if not candidate:
            return []
        if candidate[:1] == "[":
            try:
                raw_trigger_words = json.loads(candidate)
            except Exception:
                raw_trigger_words = [item for item in re.split(r"[\n\r,|;]+", candidate) if item.strip()]
        else:
            raw_trigger_words = [item for item in re.split(r"[\n\r,|;]+", candidate) if item.strip()]
    if not isinstance(raw_trigger_words, (list, tuple, set)):
        raise ValueError("Trigger words must be a list of phrases.")

    values: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_trigger_words:
        normalized = _normalize_trigger_phrase(str(raw_value or ""))
        lowered = normalized.lower()
        if not normalized or lowered in seen:
            continue
        seen.add(lowered)
        values.append(normalized)
        if len(values) >= 16:
            break
    return values


def _sidecar_path(settings: AppSettings, lora_id: str) -> Path:
    return ensure_lora_library(settings) / f"{normalize_lora_id(lora_id)}.json"


def _weights_path(settings: AppSettings, lora_id: str) -> Path:
    return ensure_lora_library(settings) / f"{normalize_lora_id(lora_id)}{_LORA_SUFFIX}"


def _preview_path(settings: AppSettings, lora_id: str) -> Path:
    return ensure_lora_library(settings) / f"{normalize_lora_id(lora_id)}{_PREVIEW_SUFFIX}"


def _draft_sidecar_path(settings: AppSettings, draft_id: str) -> Path:
    return ensure_lora_drafts(settings) / f"{normalize_lora_id(draft_id)}.json"


def _draft_weights_path(settings: AppSettings, draft_id: str) -> Path:
    return ensure_lora_drafts(settings) / f"{normalize_lora_id(draft_id)}{_LORA_SUFFIX}"


def _write_uploaded_lora(target_path: Path, *, content: bytes | None = None, content_file: Any | None = None) -> int:
    if content is not None and content_file is not None:
        raise ValueError("LoRA upload must provide either bytes or a file stream, not both.")
    if content is None and content_file is None:
        raise ValueError("Uploaded LoRA file is empty.")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target_path.open("wb") as handle:
        if content is not None:
            payload = bytes(content)
            handle.write(payload)
            return len(payload)

        source = content_file
        seek = getattr(source, "seek", None)
        if callable(seek):
            seek(0)
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            if isinstance(chunk, memoryview):
                chunk = chunk.tobytes()
            if not isinstance(chunk, (bytes, bytearray)):
                raise ValueError("Uploaded LoRA stream must yield bytes.")
            payload = bytes(chunk)
            handle.write(payload)
            written += len(payload)
    return written


def _resolve_existing_weights_path(
    settings: AppSettings,
    lora_id: str,
    *,
    existing_sidecar: dict[str, Any] | None = None,
) -> Path:
    normalized = normalize_lora_id(lora_id)
    if existing_sidecar:
        raw_path = str(existing_sidecar.get("path") or "").strip()
        if raw_path:
            candidate = Path(raw_path)
            if candidate.exists():
                return candidate.resolve()
    direct = _weights_path(settings, normalized)
    if direct.exists():
        return direct
    root = ensure_lora_library(settings)
    for candidate in sorted(root.glob(f"*{_LORA_SUFFIX}")):
        try:
            candidate_id = normalize_lora_id(candidate.stem)
        except ValueError:
            continue
        if candidate_id == normalized:
            return candidate.resolve()
    return direct


def _resolve_existing_draft_weights_path(
    settings: AppSettings,
    draft_id: str,
    *,
    existing_sidecar: dict[str, Any] | None = None,
) -> Path:
    normalized = normalize_lora_id(draft_id)
    if existing_sidecar:
        raw_path = str(existing_sidecar.get("path") or "").strip()
        if raw_path:
            candidate = Path(raw_path)
            if candidate.exists():
                return candidate.resolve()
    direct = _draft_weights_path(settings, normalized)
    if direct.exists():
        return direct
    return direct


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _truncate(value: Any, limit: int = 240) -> str:
    raw = str(value or "").strip()
    if len(raw) <= limit:
        return raw
    return f"{raw[: limit - 3]}..."


def _collect_candidate_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text[:1] in {"{", "["}:
            try:
                return _collect_candidate_strings(json.loads(text))
            except Exception:
                pass
        return [text]
    if isinstance(value, dict):
        parts: list[str] = []
        for item in value.values():
            parts.extend(_collect_candidate_strings(item))
        return parts
    if isinstance(value, (list, tuple, set)):
        parts: list[str] = []
        for item in value:
            parts.extend(_collect_candidate_strings(item))
        return parts
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    return []


def _normalize_trigger_phrase(raw: str) -> str:
    value = re.sub(r"\s+", " ", str(raw or "").strip())
    value = value.strip(",|;")
    if not value:
        return ""
    lowered = value.lower()
    if lowered in {"true", "false", "none", "null", "lora"}:
        return ""
    if len(value) > 80:
        return ""
    if value.count(" ") > 8:
        return ""
    if any(marker in lowered for marker in {"{", "}", "[", "]", ":", "\\"}):
        return ""
    if _looks_like_model_key(value):
        return ""
    return value


def _looks_like_model_key(raw: str) -> bool:
    lowered = str(raw or "").strip().lower()
    if not lowered:
        return False
    if lowered.startswith(("layers.", "transformer.", "diffusion_model.", "encoder.", "decoder.")):
        return True
    if any(marker in lowered for marker in (".weight", ".bias", ".alpha", ".lora_", ".lora.", "feed_forward", "attention.")):
        return True
    if any(marker in lowered for marker in ("to_q", "to_k", "to_v", "to_out", "adaln", "modulation", "norm.")):
        return True
    return bool(re.search(r"(?:^|[._])layers?\.\d+(?:[._]|$)", lowered))


def _score_from_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if numeric > 0 else None
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            numeric = float(candidate)
        except Exception:
            return None
        return numeric if numeric > 0 else None
    return None


def _collect_scored_trigger_candidates(value: Any, scores: dict[str, float]) -> None:
    if value is None:
        return
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return
        if text[:1] in {"{", "["}:
            try:
                _collect_scored_trigger_candidates(json.loads(text), scores)
                return
            except Exception:
                pass
        for fragment in re.split(r"[\n\r,|;]+", text):
            normalized = _normalize_trigger_phrase(fragment)
            if normalized:
                scores[normalized] = scores.get(normalized, 0.0) + 1.0
        return
    if isinstance(value, dict):
        numeric_pairs = True
        for item in value.values():
            if _score_from_value(item) is None:
                numeric_pairs = False
                break
        if numeric_pairs and value:
            for key, item in value.items():
                normalized = _normalize_trigger_phrase(str(key or ""))
                score = _score_from_value(item)
                if normalized and score is not None:
                    scores[normalized] = scores.get(normalized, 0.0) + score
            return
        for item in value.values():
            _collect_scored_trigger_candidates(item, scores)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_scored_trigger_candidates(item, scores)


def _scored_fallback_trigger_words(metadata: dict[str, Any]) -> list[str]:
    scores: dict[str, float] = {}
    for key, value in metadata.items():
        normalized_key = str(key or "").strip().lower()
        if any(marker in normalized_key for marker in _TAG_SCORE_VALUE_MARKERS):
            _collect_scored_trigger_candidates(value, scores)
            continue
        if any(marker in normalized_key for marker in _TAG_LIST_VALUE_MARKERS):
            _collect_scored_trigger_candidates(value, scores)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0].lower()))
    return [phrase for phrase, _score in ranked[:3]]


def infer_trigger_words(metadata: dict[str, Any]) -> list[str]:
    trigger_words: list[str] = []
    seen: set[str] = set()

    for key, value in metadata.items():
        normalized_key = str(key or "").strip().lower()
        if any(marker in normalized_key for marker in _TAG_SCORE_VALUE_MARKERS):
            continue
        if not (
            any(marker in normalized_key for marker in _EXPLICIT_TRIGGER_VALUE_MARKERS)
            or any(marker in normalized_key for marker in _TAG_LIST_VALUE_MARKERS)
        ):
            continue
        fragments = _collect_candidate_strings(value)
        for fragment in fragments:
            for candidate in re.split(r"[\n\r,|;]+", fragment):
                normalized = _normalize_trigger_phrase(candidate)
                lowered = normalized.lower()
                if not normalized or lowered in seen:
                    continue
                seen.add(lowered)
                trigger_words.append(normalized)
                if len(trigger_words) >= 8:
                    return trigger_words
    if trigger_words:
        return trigger_words
    return _scored_fallback_trigger_words(metadata)


def _metadata_summary(metadata: dict[str, Any]) -> dict[str, str]:
    summary: dict[str, str] = {}
    for key, value in metadata.items():
        normalized_key = str(key or "").strip().lower()
        if any(marker in normalized_key for marker in _SUMMARY_KEY_MARKERS) or any(
            marker in normalized_key for marker in _TRIGGER_VALUE_MARKERS
        ):
            summary[str(key)] = _truncate(value)
    if summary:
        return summary

    for key, value in metadata.items():
        summary[str(key)] = _truncate(value)
        if len(summary) >= 8:
            break
    return summary


def _read_safetensors_metadata(path: Path) -> dict[str, str]:
    with safe_open(str(path), framework="pt") as handle:
        raw_metadata = handle.metadata() or {}
    return {str(key): str(value) for key, value in raw_metadata.items()}


def _build_placeholder_preview(path: Path, *, title: str) -> None:
    image = Image.new("RGB", (PLACEHOLDER_PREVIEW_SIZE, PLACEHOLDER_PREVIEW_SIZE), color=_PLACEHOLDER_BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rectangle((64, 64, 960, 960), outline=_PLACEHOLDER_DETAIL, width=6)
    draw.line((128, 128, 896, 896), fill=_PLACEHOLDER_ACCENT, width=8)
    draw.line((896, 128, 128, 896), fill=_PLACEHOLDER_ACCENT, width=8)
    draw.rectangle((160, 760, 864, 880), fill="#0a0d15", outline=_PLACEHOLDER_DETAIL, width=4)
    draw.text((214, 204), "LoRA", fill=_PLACEHOLDER_DETAIL)
    draw.text((214, 816), _truncate(title, limit=44), fill="#f4f7ff")
    image.save(path, format="PNG")


def _normalize_preview_image_bytes(content: bytes) -> bytes:
    if not content:
        raise ValueError("Thumbnail image is empty.")
    try:
        with Image.open(io.BytesIO(content)) as image:
            fitted = ImageOps.fit(
                image.convert("RGB"),
                (PLACEHOLDER_PREVIEW_SIZE, PLACEHOLDER_PREVIEW_SIZE),
                method=_RESAMPLING_LANCZOS,
                centering=(0.5, 0.5),
            )
            buffer = io.BytesIO()
            fitted.save(buffer, format="PNG")
    except Exception as exc:
        raise ValueError(f"Invalid thumbnail image: {exc}") from exc
    return buffer.getvalue()


def _write_custom_preview(path: Path, content: bytes) -> None:
    path.write_bytes(_normalize_preview_image_bytes(content))


def _unique_lora_id(settings: AppSettings, preferred_name: str, *, include_drafts: bool = False) -> str:
    normalized = normalize_lora_id(preferred_name)
    existing_live = _weights_path(settings, normalized).exists()
    existing_draft = _draft_weights_path(settings, normalized).exists() if include_drafts else False
    if not existing_live and not existing_draft:
        return normalized
    index = 2
    while True:
        candidate = f"{normalized}-{index}"
        if _weights_path(settings, candidate).exists():
            index += 1
            continue
        if include_drafts and _draft_weights_path(settings, candidate).exists():
            index += 1
            continue
        return candidate


def _created_at_for_path(path: Path) -> str:
    stat = path.stat()
    return datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds")


def _detection_payload(weights_path: Path) -> tuple[dict[str, str], dict[str, str], list[str]]:
    metadata = _read_safetensors_metadata(weights_path)
    metadata_summary = _metadata_summary(metadata)
    detected_trigger_words = infer_trigger_words(metadata)
    return metadata, metadata_summary, detected_trigger_words


def _full_record_payload(
    *,
    lora_id: str,
    weights_path: Path,
    preview_path: Path,
    display_name: str,
    source_filename: str,
    architecture: str,
    trigger_words: list[str],
    detected_trigger_words: list[str],
    metadata: dict[str, Any],
    metadata_summary: dict[str, Any],
    preview_is_custom: bool,
    created_at: str,
    updated_at: str,
    deleted: bool = False,
    delete_pending: bool = False,
    deleted_at: str | None = None,
) -> dict[str, Any]:
    return {
        "id": normalize_lora_id(lora_id),
        "display_name": display_name,
        "source_filename": source_filename,
        "architecture": normalize_lora_architecture(architecture),
        "filename": weights_path.name,
        "path": str(weights_path.resolve()),
        "preview_filename": preview_path.name,
        "preview_path": str(preview_path.resolve()),
        "preview_is_custom": bool(preview_is_custom),
        "trigger_words": list(trigger_words),
        "detected_trigger_words": list(detected_trigger_words),
        "metadata": dict(metadata),
        "metadata_summary": dict(metadata_summary),
        "created_at": created_at,
        "updated_at": updated_at,
        "file_size_bytes": int(weights_path.stat().st_size),
        "deleted": bool(deleted),
        "delete_pending": bool(delete_pending),
        "deleted_at": str(deleted_at or "").strip() or None,
    }


def _draft_record_payload(
    *,
    draft_id: str,
    weights_path: Path,
    source_filename: str,
    display_name: str,
    architecture: str,
    detected_trigger_words: list[str],
    metadata: dict[str, Any],
    metadata_summary: dict[str, Any],
    created_at: str,
    updated_at: str,
) -> dict[str, Any]:
    return {
        "id": normalize_lora_id(draft_id),
        "source_filename": source_filename,
        "display_name": display_name,
        "architecture": normalize_lora_architecture(architecture),
        "filename": weights_path.name,
        "path": str(weights_path.resolve()),
        "detected_trigger_words": list(detected_trigger_words),
        "metadata": dict(metadata),
        "metadata_summary": dict(metadata_summary),
        "created_at": created_at,
        "updated_at": updated_at,
        "file_size_bytes": int(weights_path.stat().st_size),
    }


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    preview_cache_key = ""
    try:
        preview_stats = Path(str(record.get("preview_path") or "")).stat()
        preview_cache_key = f"{preview_stats.st_mtime_ns}-{preview_stats.st_size}"
    except OSError:
        preview_cache_key = str(record.get("updated_at") or "")
    return {
        "id": record["id"],
        "display_name": record["display_name"],
        "source_filename": record["source_filename"],
        "architecture": normalize_lora_architecture(record.get("architecture")),
        "filename": record["filename"],
        "preview_filename": record["preview_filename"],
        "preview_url": f"/loras/{record['id']}/preview",
        "preview_cache_key": preview_cache_key,
        "preview_is_custom": bool(record.get("preview_is_custom")),
        "trigger_words": list(record.get("trigger_words") or []),
        "detected_trigger_words": list(record.get("detected_trigger_words") or []),
        "metadata_summary": dict(record.get("metadata_summary") or {}),
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
        "file_size_bytes": int(record.get("file_size_bytes") or 0),
    }


def _public_draft_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "draft_id": record["id"],
        "source_filename": record["source_filename"],
        "display_name": record["display_name"],
        "architecture": normalize_lora_architecture(record.get("architecture")),
        "detected_trigger_words": list(record.get("detected_trigger_words") or []),
        "metadata_summary": dict(record.get("metadata_summary") or {}),
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
        "file_size_bytes": int(record.get("file_size_bytes") or 0),
    }


def _hydrate_record(settings: AppSettings, *, lora_id: str, include_deleted: bool = False) -> dict[str, Any]:
    normalized = normalize_lora_id(lora_id)
    sidecar_path = _sidecar_path(settings, normalized)
    existing = _read_json(sidecar_path) if sidecar_path.exists() else {}
    if bool(existing.get("deleted")) and not include_deleted:
        raise FileNotFoundError(f"LoRA not found: {normalized}")
    weights_path = _resolve_existing_weights_path(settings, normalized, existing_sidecar=existing)
    if not weights_path.exists():
        raise FileNotFoundError(f"LoRA not found: {normalized}")

    changed = not sidecar_path.exists()
    preview_path = _preview_path(settings, normalized)
    display_name = sanitize_display_name(
        existing.get("display_name"),
        fallback=Path(str(existing.get("source_filename") or weights_path.stem)).stem,
    )
    source_filename = str(existing.get("source_filename") or weights_path.name)

    metadata = existing.get("metadata")
    metadata_summary = existing.get("metadata_summary")
    detected_trigger_words = existing.get("detected_trigger_words")
    if not isinstance(metadata, dict) or not isinstance(metadata_summary, dict) or detected_trigger_words is None:
        metadata, metadata_summary, detected_trigger_words = _detection_payload(weights_path)
        changed = True
    else:
        detected_trigger_words = normalize_trigger_words(detected_trigger_words)

    trigger_words = existing.get("trigger_words")
    if trigger_words is None:
        trigger_words = list(detected_trigger_words)
        changed = True
    trigger_words = normalize_trigger_words(trigger_words)

    if "architecture" in existing:
        architecture = normalize_lora_architecture(existing.get("architecture"))
    else:
        architecture = DEFAULT_LORA_ARCHITECTURE
        changed = True

    preview_is_custom = bool(existing.get("preview_is_custom", False))
    if not preview_path.exists():
        _build_placeholder_preview(preview_path, title=display_name)
        preview_is_custom = False
        changed = True

    created_at = str(existing.get("created_at") or _created_at_for_path(weights_path))
    updated_at = str(existing.get("updated_at") or created_at)
    deleted = bool(existing.get("deleted", False))
    delete_pending = bool(existing.get("delete_pending", False))
    deleted_at = str(existing.get("deleted_at") or "").strip() or None
    payload = _full_record_payload(
        lora_id=normalized,
        weights_path=weights_path,
        preview_path=preview_path,
        display_name=display_name,
        source_filename=source_filename,
        architecture=architecture,
        trigger_words=trigger_words,
        detected_trigger_words=detected_trigger_words,
        metadata=metadata,
        metadata_summary=metadata_summary,
        preview_is_custom=preview_is_custom,
        created_at=created_at,
        updated_at=updated_at,
        deleted=deleted,
        delete_pending=delete_pending,
        deleted_at=deleted_at,
    )
    if changed:
        _write_json(sidecar_path, payload)
    return payload


def _hydrate_draft(settings: AppSettings, *, draft_id: str) -> dict[str, Any]:
    normalized = normalize_lora_id(draft_id)
    sidecar_path = _draft_sidecar_path(settings, normalized)
    existing = _read_json(sidecar_path) if sidecar_path.exists() else {}
    weights_path = _resolve_existing_draft_weights_path(settings, normalized, existing_sidecar=existing)
    if not weights_path.exists():
        raise FileNotFoundError(f"LoRA draft not found: {normalized}")

    changed = not sidecar_path.exists()
    metadata = existing.get("metadata")
    metadata_summary = existing.get("metadata_summary")
    detected_trigger_words = existing.get("detected_trigger_words")
    if not isinstance(metadata, dict) or not isinstance(metadata_summary, dict) or detected_trigger_words is None:
        metadata, metadata_summary, detected_trigger_words = _detection_payload(weights_path)
        changed = True
    else:
        detected_trigger_words = normalize_trigger_words(detected_trigger_words)

    display_name = sanitize_display_name(
        existing.get("display_name"),
        fallback=Path(str(existing.get("source_filename") or weights_path.stem)).stem,
    )
    source_filename = str(existing.get("source_filename") or weights_path.name)
    if "architecture" in existing:
        architecture = normalize_lora_architecture(existing.get("architecture"))
    else:
        architecture = DEFAULT_LORA_ARCHITECTURE
        changed = True
    created_at = str(existing.get("created_at") or _created_at_for_path(weights_path))
    updated_at = str(existing.get("updated_at") or created_at)
    payload = _draft_record_payload(
        draft_id=normalized,
        weights_path=weights_path,
        source_filename=source_filename,
        display_name=display_name,
        architecture=architecture,
        detected_trigger_words=detected_trigger_words,
        metadata=metadata,
        metadata_summary=metadata_summary,
        created_at=created_at,
        updated_at=updated_at,
    )
    if changed:
        _write_json(sidecar_path, payload)
    return payload


def get_lora(settings: AppSettings, lora_id: str, *, include_deleted: bool = False) -> dict[str, Any] | None:
    try:
        return _hydrate_record(settings, lora_id=lora_id, include_deleted=include_deleted)
    except FileNotFoundError:
        return None


def get_lora_draft(settings: AppSettings, draft_id: str) -> dict[str, Any] | None:
    try:
        return _hydrate_draft(settings, draft_id=draft_id)
    except FileNotFoundError:
        return None


def list_loras(settings: AppSettings) -> list[dict[str, Any]]:
    root = ensure_lora_library(settings)
    records: list[dict[str, Any]] = []
    for weights_path in sorted(root.glob(f"*{_LORA_SUFFIX}")):
        try:
            record = _hydrate_record(settings, lora_id=weights_path.stem)
        except FileNotFoundError:
            continue
        records.append(_public_record(record))
    records.sort(key=lambda item: (str(item.get("display_name") or "").lower(), str(item.get("id") or "")))
    return records


def create_lora_draft(
    settings: AppSettings,
    *,
    filename: str,
    content: bytes | None = None,
    content_file: Any | None = None,
    architecture: str = DEFAULT_LORA_ARCHITECTURE,
) -> dict[str, Any]:
    safe_filename = sanitize_upload_filename(filename)
    draft_id = _unique_lora_id(settings, Path(safe_filename).stem, include_drafts=True)
    target_path = _draft_weights_path(settings, draft_id)
    try:
        written = _write_uploaded_lora(target_path, content=content, content_file=content_file)
        if written <= 0:
            raise ValueError("Uploaded LoRA file is empty.")
        metadata, metadata_summary, detected_trigger_words = _detection_payload(target_path)
    except Exception as exc:
        target_path.unlink(missing_ok=True)
        _draft_sidecar_path(settings, draft_id).unlink(missing_ok=True)
        raise ValueError(f"Invalid LoRA safetensors file: {exc}") from exc

    payload = _draft_record_payload(
        draft_id=draft_id,
        weights_path=target_path,
        source_filename=safe_filename,
        display_name=sanitize_display_name(Path(safe_filename).stem, fallback=Path(safe_filename).stem),
        architecture=architecture,
        detected_trigger_words=detected_trigger_words,
        metadata=metadata,
        metadata_summary=metadata_summary,
        created_at=_created_at_for_path(target_path),
        updated_at=_utc_timestamp(),
    )
    _write_json(_draft_sidecar_path(settings, draft_id), payload)
    return _public_draft_record(payload)


def detect_lora_draft_triggers(settings: AppSettings, draft_id: str) -> dict[str, Any]:
    draft = _hydrate_draft(settings, draft_id=draft_id)
    weights_path = Path(str(draft["path"]))
    metadata, metadata_summary, detected_trigger_words = _detection_payload(weights_path)
    payload = _draft_record_payload(
        draft_id=draft["id"],
        weights_path=weights_path,
        source_filename=str(draft.get("source_filename") or weights_path.name),
        display_name=sanitize_display_name(draft.get("display_name"), fallback=Path(weights_path.stem).stem),
        architecture=normalize_lora_architecture(draft.get("architecture")),
        detected_trigger_words=detected_trigger_words,
        metadata=metadata,
        metadata_summary=metadata_summary,
        created_at=str(draft.get("created_at") or _created_at_for_path(weights_path)),
        updated_at=_utc_timestamp(),
    )
    _write_json(_draft_sidecar_path(settings, draft["id"]), payload)
    return _public_draft_record(payload)


def finalize_lora_draft(
    settings: AppSettings,
    *,
    draft_id: str,
    display_name: Any,
    trigger_words: Any,
    preview_content: bytes | None = None,
) -> dict[str, Any]:
    draft = _hydrate_draft(settings, draft_id=draft_id)
    source_weights_path = Path(str(draft["path"]))
    if not source_weights_path.exists():
        raise FileNotFoundError(f"LoRA draft not found: {draft_id}")

    normalized_preview: bytes | None = None
    if preview_content is not None:
        normalized_preview = _normalize_preview_image_bytes(preview_content)

    target_id = normalize_lora_id(draft["id"])
    target_path = _weights_path(settings, target_id)
    if target_path.exists():
        target_id = _unique_lora_id(settings, target_id, include_drafts=False)
        target_path = _weights_path(settings, target_id)

    source_weights_path.replace(target_path)
    preview_path = _preview_path(settings, target_id)
    trigger_values = normalize_trigger_words(
        draft.get("detected_trigger_words") if trigger_words is None else trigger_words
    )
    name_value = sanitize_display_name(
        display_name,
        fallback=str(draft.get("display_name") or Path(str(draft.get("source_filename") or target_path.stem)).stem),
    )
    if normalized_preview is not None:
        preview_path.write_bytes(normalized_preview)
        preview_is_custom = True
    else:
        _build_placeholder_preview(preview_path, title=name_value)
        preview_is_custom = False

    payload = _full_record_payload(
        lora_id=target_id,
        weights_path=target_path,
        preview_path=preview_path,
        display_name=name_value,
        source_filename=str(draft.get("source_filename") or target_path.name),
        architecture=normalize_lora_architecture(draft.get("architecture")),
        trigger_words=trigger_values,
        detected_trigger_words=normalize_trigger_words(draft.get("detected_trigger_words")),
        metadata=dict(draft.get("metadata") or {}),
        metadata_summary=dict(draft.get("metadata_summary") or {}),
        preview_is_custom=preview_is_custom,
        created_at=str(draft.get("created_at") or _created_at_for_path(target_path)),
        updated_at=_utc_timestamp(),
        deleted=False,
        delete_pending=False,
        deleted_at=None,
    )
    _write_json(_sidecar_path(settings, target_id), payload)
    _draft_sidecar_path(settings, draft["id"]).unlink(missing_ok=True)
    _draft_weights_path(settings, draft["id"]).unlink(missing_ok=True)
    return _public_record(payload)


def update_lora(
    settings: AppSettings,
    *,
    lora_id: str,
    display_name: Any,
    trigger_words: Any,
    preview_content: bytes | None = None,
) -> dict[str, Any]:
    record = _hydrate_record(settings, lora_id=lora_id)
    weights_path = Path(str(record["path"]))
    if not weights_path.exists():
        raise FileNotFoundError(f"LoRA not found: {lora_id}")

    preview_path = _preview_path(settings, record["id"])
    preview_is_custom = bool(record.get("preview_is_custom"))
    if preview_content is not None:
        preview_path.write_bytes(_normalize_preview_image_bytes(preview_content))
        preview_is_custom = True
    elif not preview_path.exists():
        _build_placeholder_preview(
            preview_path,
            title=sanitize_display_name(display_name, fallback=str(record.get("display_name") or weights_path.stem)),
        )
        preview_is_custom = False

    payload = _full_record_payload(
        lora_id=record["id"],
        weights_path=weights_path,
        preview_path=preview_path,
        display_name=sanitize_display_name(display_name, fallback=str(record.get("display_name") or weights_path.stem)),
        source_filename=str(record.get("source_filename") or weights_path.name),
        architecture=normalize_lora_architecture(record.get("architecture")),
        trigger_words=normalize_trigger_words(
            record.get("trigger_words") if trigger_words is None else trigger_words
        ),
        detected_trigger_words=normalize_trigger_words(record.get("detected_trigger_words")),
        metadata=dict(record.get("metadata") or {}),
        metadata_summary=dict(record.get("metadata_summary") or {}),
        preview_is_custom=preview_is_custom,
        created_at=str(record.get("created_at") or _created_at_for_path(weights_path)),
        updated_at=_utc_timestamp(),
        deleted=False,
        delete_pending=False,
        deleted_at=None,
    )
    _write_json(_sidecar_path(settings, record["id"]), payload)
    return _public_record(payload)


def mark_lora_deleted(settings: AppSettings, lora_id: str, *, pending_cleanup: bool) -> dict[str, Any]:
    normalized = normalize_lora_id(lora_id)
    record = get_lora(settings, normalized, include_deleted=True)
    weights_path = Path(str(record["path"])) if record is not None else _weights_path(settings, normalized)
    if not weights_path.exists():
        raise FileNotFoundError(f"LoRA not found: {normalized}")

    preview_path = Path(str(record["preview_path"])) if record is not None else _preview_path(settings, normalized)
    source_filename = str(record.get("source_filename") or weights_path.name) if record is not None else weights_path.name
    display_name = sanitize_display_name(
        record.get("display_name") if record is not None else None,
        fallback=Path(source_filename).stem,
    )
    deleted_at = _utc_timestamp()
    payload = _full_record_payload(
        lora_id=normalized,
        weights_path=weights_path,
        preview_path=preview_path,
        display_name=display_name,
        source_filename=source_filename,
        architecture=normalize_lora_architecture(record.get("architecture") if record else None),
        trigger_words=normalize_trigger_words(record.get("trigger_words") if record else []),
        detected_trigger_words=normalize_trigger_words(record.get("detected_trigger_words") if record else []),
        metadata=dict(record.get("metadata") if record else {}),
        metadata_summary=dict(record.get("metadata_summary") if record else {}),
        preview_is_custom=bool(record.get("preview_is_custom") if record else preview_path.exists()),
        created_at=str(record.get("created_at") if record else _created_at_for_path(weights_path)),
        updated_at=_utc_timestamp(),
        deleted=True,
        delete_pending=bool(pending_cleanup),
        deleted_at=deleted_at,
    )
    _write_json(_sidecar_path(settings, normalized), payload)
    return payload


def finalize_deleted_lora(settings: AppSettings, lora_id: str) -> dict[str, Any]:
    normalized = normalize_lora_id(lora_id)
    record = get_lora(settings, normalized, include_deleted=True)
    weights_path = Path(str(record["path"])) if record is not None else _weights_path(settings, normalized)
    if not weights_path.exists() and not _sidecar_path(settings, normalized).exists() and not _preview_path(settings, normalized).exists():
        raise FileNotFoundError(f"LoRA not found: {normalized}")

    deleted_files = 0
    for path in (_sidecar_path(settings, normalized), _preview_path(settings, normalized), weights_path):
        if path.exists():
            path.unlink()
            deleted_files += 1

    return {"id": normalized, "deleted_files": deleted_files}


def delete_lora(settings: AppSettings, lora_id: str) -> dict[str, Any]:
    mark_lora_deleted(settings, lora_id, pending_cleanup=False)
    return finalize_deleted_lora(settings, lora_id)


def preview_path_for_lora(settings: AppSettings, lora_id: str) -> Path:
    record = get_lora(settings, lora_id)
    if record is None:
        raise FileNotFoundError(f"LoRA not found: {lora_id}")
    preview_path = Path(str(record["preview_path"]))
    if not preview_path.exists():
        raise FileNotFoundError(f"LoRA preview not found: {lora_id}")
    return preview_path
