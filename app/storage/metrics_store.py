from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config.settings import AppSettings


def append_generation_metric(
    settings: AppSettings,
    payload: dict[str, Any],
    metrics_path: Path | None = None,
) -> Path:
    target = metrics_path or (settings.paths.data_dir / "generation_metrics.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **payload,
    }
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return target

