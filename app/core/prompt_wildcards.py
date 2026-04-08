from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Any

from app.config.settings import AppSettings
from app.storage.wildcard_library import get_wildcard_by_token, wildcard_placeholder

_WILDCARD_PLACEHOLDER_PATTERN = re.compile(r"__(?P<token>[A-Za-z0-9][A-Za-z0-9\-]{0,95})__")


@dataclass(frozen=True)
class ResolvedWildcardOccurrence:
    id: str
    display_name: str
    token: str
    placeholder: str
    selected_entry: str
    occurrence_index: int
    prompt_offset: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "token": self.token,
            "placeholder": self.placeholder,
            "selected_entry": self.selected_entry,
            "occurrence_index": self.occurrence_index,
            "prompt_offset": self.prompt_offset,
        }


def _stable_entry_index(
    *,
    seed: int,
    wildcard_id: str,
    prompt_offset: int,
    occurrence_index: int,
    entry_count: int,
) -> int:
    payload = f"{int(seed)}:{wildcard_id}:{int(prompt_offset)}:{int(occurrence_index)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % max(1, int(entry_count))


def expand_prompt_wildcards(
    settings: AppSettings,
    prompt: str,
    *,
    seed: int | None,
) -> tuple[str, tuple[ResolvedWildcardOccurrence, ...]]:
    source_prompt = str(prompt or "")
    matches = list(_WILDCARD_PLACEHOLDER_PATTERN.finditer(source_prompt))
    if not matches:
        return source_prompt, ()

    parts: list[str] = []
    last_index = 0
    occurrences: list[ResolvedWildcardOccurrence] = []
    occurrence_counts: dict[str, int] = {}
    nondeterministic_rng = random.Random()

    for match in matches:
        token = str(match.group("token") or "").strip()
        placeholder = match.group(0)
        record = get_wildcard_by_token(settings, token)
        if record is None:
            raise ValueError(f"Wildcard not found: {placeholder}")
        entries = list(record.get("entries") or [])
        if not entries:
            raise ValueError(f"Wildcard has no entries: {placeholder}")
        occurrence_index = int(occurrence_counts.get(record["id"], 0))
        occurrence_counts[record["id"]] = occurrence_index + 1
        if seed is None:
            selected_entry = str(nondeterministic_rng.choice(entries))
        else:
            selected_entry = str(
                entries[
                    _stable_entry_index(
                        seed=int(seed),
                        wildcard_id=str(record["id"]),
                        prompt_offset=int(match.start()),
                        occurrence_index=occurrence_index,
                        entry_count=len(entries),
                    )
                ]
            )

        parts.append(source_prompt[last_index : match.start()])
        parts.append(selected_entry)
        last_index = match.end()
        occurrences.append(
            ResolvedWildcardOccurrence(
                id=str(record["id"]),
                display_name=str(record.get("display_name") or record["token"]),
                token=str(record["token"]),
                placeholder=wildcard_placeholder(record["token"]),
                selected_entry=selected_entry,
                occurrence_index=occurrence_index,
                prompt_offset=int(match.start()),
            )
        )

    parts.append(source_prompt[last_index:])
    return "".join(parts), tuple(occurrences)


__all__ = ["ResolvedWildcardOccurrence", "expand_prompt_wildcards"]
