from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config.settings import AppSettings
from app.core.chat_actions import normalize_chat_actions, strip_chat_action_markup
from app.storage import normalize_owner_id

MAX_CHAT_EXCHANGES = 500


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def chat_history_dir(settings: AppSettings) -> Path:
    return settings.paths.data_dir / "chat"


def ensure_chat_history_dir(settings: AppSettings) -> Path:
    target = chat_history_dir(settings)
    target.mkdir(parents=True, exist_ok=True)
    return target


def chat_history_path(settings: AppSettings, owner_id: str) -> Path:
    return ensure_chat_history_dir(settings) / f"{normalize_owner_id(owner_id)}.json"


def _clean_content(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _clean_bubble(raw: Any, *, fallback_number: int, fallback_role: str) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    role = str(raw.get("role") or fallback_role).strip().lower()
    if role not in {"user", "assistant"}:
        role = fallback_role
    content = _clean_content(raw.get("content"))
    if role == "assistant":
        content = strip_chat_action_markup(content)
    try:
        number = int(raw.get("number") or fallback_number)
    except (TypeError, ValueError):
        number = fallback_number
    if number < 1:
        number = fallback_number
    bubble = {
        "number": number,
        "role": role,
        "content": content,
        "created_at": str(raw.get("created_at") or _utc_timestamp()),
        "error": bool(raw.get("error", False)),
    }
    actions = normalize_chat_actions(raw.get("actions"))
    if role == "assistant" and actions:
        bubble["actions"] = actions
    return bubble


def _empty_history(owner_id: str) -> dict[str, Any]:
    now = _utc_timestamp()
    return {
        "owner_id": normalize_owner_id(owner_id),
        "next_number": 1,
        "exchanges": [],
        "exchange_count": 0,
        "created_at": now,
        "updated_at": now,
    }


def _normalize_history(payload: Any, *, owner_id: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return _empty_history(owner_id)
    history = _empty_history(owner_id)
    history["created_at"] = str(payload.get("created_at") or history["created_at"])
    exchanges: list[dict[str, Any]] = []
    highest_number = 0
    for raw_exchange in payload.get("exchanges") or []:
        if not isinstance(raw_exchange, dict):
            continue
        user = _clean_bubble(raw_exchange.get("user"), fallback_number=highest_number + 1, fallback_role="user")
        assistant = _clean_bubble(
            raw_exchange.get("assistant"),
            fallback_number=(user["number"] + 1 if user is not None else highest_number + 2),
            fallback_role="assistant",
        )
        if user is None or assistant is None:
            continue
        highest_number = max(highest_number, int(user["number"]), int(assistant["number"]))
        exchanges.append({"user": user, "assistant": assistant})
    try:
        next_number = int(payload.get("next_number") or highest_number + 1)
    except (TypeError, ValueError):
        next_number = highest_number + 1
    history["next_number"] = max(next_number, highest_number + 1, 1)
    history["exchanges"] = exchanges[-MAX_CHAT_EXCHANGES:]
    history["exchange_count"] = len(history["exchanges"])
    history["updated_at"] = str(payload.get("updated_at") or history["created_at"])
    return history


def load_chat_history(settings: AppSettings, owner_id: str) -> dict[str, Any]:
    safe_owner = normalize_owner_id(owner_id)
    path = chat_history_path(settings, safe_owner)
    if not path.exists():
        return _empty_history(safe_owner)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    return _normalize_history(payload, owner_id=safe_owner)


def save_chat_history(settings: AppSettings, owner_id: str, history: dict[str, Any]) -> dict[str, Any]:
    safe_owner = normalize_owner_id(owner_id)
    normalized = _normalize_history(history, owner_id=safe_owner)
    normalized["updated_at"] = _utc_timestamp()
    path = chat_history_path(settings, safe_owner)
    payload = json.dumps(normalized, indent=2, ensure_ascii=True)
    path.write_text(f"{payload}\n", encoding="utf-8")
    return normalized


def append_chat_exchange(
    settings: AppSettings,
    owner_id: str,
    *,
    user_content: Any,
    assistant_content: Any,
    assistant_error: bool = False,
    assistant_actions: Any = None,
) -> dict[str, Any]:
    safe_owner = normalize_owner_id(owner_id)
    history = load_chat_history(settings, safe_owner)
    next_number = int(history.get("next_number") or 1)
    now = _utc_timestamp()
    assistant_bubble = {
        "number": next_number + 1,
        "role": "assistant",
        "content": strip_chat_action_markup(_clean_content(assistant_content)),
        "created_at": now,
        "error": bool(assistant_error),
    }
    actions = normalize_chat_actions(assistant_actions)
    if actions:
        assistant_bubble["actions"] = actions
    exchange = {
        "user": {
            "number": next_number,
            "role": "user",
            "content": _clean_content(user_content),
            "created_at": now,
            "error": False,
        },
        "assistant": assistant_bubble,
    }
    exchanges = [*list(history.get("exchanges") or []), exchange]
    history["exchanges"] = exchanges[-MAX_CHAT_EXCHANGES:]
    history["next_number"] = next_number + 2
    history["exchange_count"] = len(history["exchanges"])
    return save_chat_history(settings, safe_owner, history)


def clear_chat_history(settings: AppSettings, owner_id: str) -> dict[str, Any]:
    safe_owner = normalize_owner_id(owner_id)
    path = chat_history_path(settings, safe_owner)
    try:
        path.unlink(missing_ok=True)
    except FileNotFoundError:
        pass
    return _empty_history(safe_owner)


def _skip_assistant_context(content: str) -> bool:
    lowered = content.lower()
    stale_clarity_patterns = (
        "clarity refines or expands prompts",
        "clarity refines and expands prompts",
        "use clarity when the user wants a prompt cleaned up or expanded",
    )
    return any(pattern in lowered for pattern in stale_clarity_patterns)


def chat_messages_for_context(history: dict[str, Any], *, max_exchanges: int = 10) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    exchanges = list(history.get("exchanges") or [])[-max(0, int(max_exchanges)) :]
    for exchange in exchanges:
        if not isinstance(exchange, dict):
            continue
        user = exchange.get("user")
        assistant = exchange.get("assistant")
        if isinstance(user, dict) and _clean_content(user.get("content")):
            messages.append({"role": "user", "content": _clean_content(user.get("content"))})
        if (
            isinstance(assistant, dict)
            and _clean_content(assistant.get("content"))
            and not bool(assistant.get("error", False))
            and not _skip_assistant_context(_clean_content(assistant.get("content")))
        ):
            messages.append({"role": "assistant", "content": _clean_content(assistant.get("content"))})
    return messages


__all__ = [
    "MAX_CHAT_EXCHANGES",
    "append_chat_exchange",
    "chat_history_path",
    "chat_messages_for_context",
    "clear_chat_history",
    "load_chat_history",
    "save_chat_history",
]
