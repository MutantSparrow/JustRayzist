from __future__ import annotations

import re
from typing import Any

CHAT_ACTION_LIMIT = 4
CHAT_ACTION_LABEL_LIMIT = 48
CHAT_ACTION_PROMPT_LIMIT = 4000
CHAT_ALLOWED_ROUTES = {
    "/API": "Open API",
}
CHAT_ACTION_TYPES = {
    "set_prompt",
    "append_prompt",
    "start_generation",
    "open_route",
}


def _clean_text(value: Any, *, limit: int) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) > limit:
        return text[:limit].rstrip()
    return text


def _default_label(action_type: str, *, href: str = "") -> str:
    if action_type == "set_prompt":
        return "Use Prompt"
    if action_type == "append_prompt":
        return "Append Prompt"
    if action_type == "start_generation":
        return "Generate"
    if action_type == "open_route":
        return CHAT_ALLOWED_ROUTES.get(href, "Open")
    return "Use"


def _normalize_route(raw_href: Any) -> str:
    href = _clean_text(raw_href, limit=128)
    if href.lower() == "/api":
        return "/API"
    return href if href in CHAT_ALLOWED_ROUTES else ""


def normalize_chat_actions(raw_actions: Any) -> list[dict[str, Any]]:
    if isinstance(raw_actions, dict):
        raw_actions = raw_actions.get("actions")
    if not isinstance(raw_actions, list):
        return []

    actions: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_actions:
        if len(actions) >= CHAT_ACTION_LIMIT:
            break
        if not isinstance(raw, dict):
            continue
        action_type = _clean_text(raw.get("type") or raw.get("action"), limit=64).lower()
        if action_type not in CHAT_ACTION_TYPES:
            continue

        normalized: dict[str, Any] = {"type": action_type}
        identity_value = ""
        if action_type in {"set_prompt", "append_prompt", "start_generation"}:
            prompt = _clean_text(raw.get("prompt") or raw.get("text"), limit=CHAT_ACTION_PROMPT_LIMIT)
            if not prompt:
                continue
            normalized["prompt"] = prompt
            identity_value = prompt
            if action_type == "start_generation":
                normalized["requires_confirm"] = bool(raw.get("requires_confirm", True))
        elif action_type == "open_route":
            href = _normalize_route(raw.get("href") or raw.get("url") or raw.get("route"))
            if not href:
                continue
            normalized["href"] = href
            identity_value = href

        label = _clean_text(raw.get("label"), limit=CHAT_ACTION_LABEL_LIMIT)
        normalized["label"] = label or _default_label(action_type, href=normalized.get("href", ""))

        identity = (action_type, identity_value)
        if identity in seen:
            continue
        seen.add(identity)
        actions.append(normalized)

    return actions


def strip_chat_action_markup(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    marker_pattern = re.compile(r"<rayzist-actions\b[^>]*>", flags=re.IGNORECASE)
    close_pattern = re.compile(r"</rayzist-actions>", flags=re.IGNORECASE)
    while True:
        marker = marker_pattern.search(text)
        if marker is None:
            break
        close = close_pattern.search(text, marker.end())
        if close is None:
            text = text[: marker.start()].rstrip()
            break
        text = f"{text[: marker.start()]}{text[close.end():]}"
    text = re.sub(
        r"```(?:json|rayzist-actions)?\s*\{[\s\S]*?\"actions\"\s*:\s*\[[\s\S]*?```",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


__all__ = [
    "CHAT_ACTION_LIMIT",
    "CHAT_ALLOWED_ROUTES",
    "CHAT_ACTION_TYPES",
    "normalize_chat_actions",
    "strip_chat_action_markup",
]
