from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SOAK_MODES = {"soak", "soak_warmup", "soak_error", "soak_summary"}


@dataclass(frozen=True)
class SoakSummary:
    session_id: str
    record_count: int
    started_at: str | None
    ended_at: str | None
    iteration_count: int
    warmup_count: int
    error_count: int
    recycle_count: int
    duration_avg_ms: float | None
    duration_p50_ms: float | None
    duration_p95_ms: float | None
    duration_p99_ms: float | None
    drift_first_mb: float | None
    drift_last_mb: float | None
    drift_min_mb: float | None
    drift_max_mb: float | None
    drift_slope_mb_per_iteration: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "record_count": self.record_count,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "iteration_count": self.iteration_count,
            "warmup_count": self.warmup_count,
            "error_count": self.error_count,
            "recycle_count": self.recycle_count,
            "duration_avg_ms": self.duration_avg_ms,
            "duration_p50_ms": self.duration_p50_ms,
            "duration_p95_ms": self.duration_p95_ms,
            "duration_p99_ms": self.duration_p99_ms,
            "drift_first_mb": self.drift_first_mb,
            "drift_last_mb": self.drift_last_mb,
            "drift_min_mb": self.drift_min_mb,
            "drift_max_mb": self.drift_max_mb,
            "drift_slope_mb_per_iteration": self.drift_slope_mb_per_iteration,
        }


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_metrics_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def group_soak_sessions(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    legacy: list[dict[str, Any]] = []
    for row in rows:
        mode = row.get("mode")
        if mode not in SOAK_MODES:
            continue
        session_id = row.get("session_id")
        if session_id:
            grouped.setdefault(str(session_id), []).append(row)
        else:
            legacy.append(row)
    if legacy:
        grouped.setdefault("legacy", []).extend(legacy)
    return grouped


def latest_session_id(grouped: dict[str, list[dict[str, Any]]]) -> str | None:
    best_id = None
    best_time = None
    for session_id, rows in grouped.items():
        row_times = [_parse_timestamp(row.get("timestamp")) for row in rows]
        row_times = [timestamp for timestamp in row_times if timestamp is not None]
        if not row_times:
            continue
        max_time = max(row_times)
        if best_time is None or max_time > best_time:
            best_time = max_time
            best_id = session_id
    return best_id


def summarize_session(session_id: str, rows: list[dict[str, Any]]) -> SoakSummary:
    timestamps = [_parse_timestamp(row.get("timestamp")) for row in rows]
    timestamps = [timestamp for timestamp in timestamps if timestamp is not None]
    started_at = min(timestamps).isoformat() if timestamps else None
    ended_at = max(timestamps).isoformat() if timestamps else None

    soak_rows = [row for row in rows if row.get("mode") == "soak"]
    warmup_rows = [row for row in rows if row.get("mode") == "soak_warmup"]
    error_rows = [row for row in rows if row.get("mode") == "soak_error"]

    durations = [_safe_float(row.get("duration_ms")) for row in soak_rows]
    durations = [value for value in durations if value is not None]

    drift_points: list[tuple[int, float]] = []
    for row in soak_rows:
        iteration = row.get("iteration")
        drift = _safe_float(row.get("memory_drift_mb"))
        try:
            iteration_idx = int(iteration)
        except (TypeError, ValueError):
            continue
        if drift is None:
            continue
        drift_points.append((iteration_idx, drift))
    drift_points.sort(key=lambda item: item[0])

    drift_values = [value for _, value in drift_points]
    drift_first = drift_values[0] if drift_values else None
    drift_last = drift_values[-1] if drift_values else None
    drift_min = min(drift_values) if drift_values else None
    drift_max = max(drift_values) if drift_values else None
    drift_slope = None
    if len(drift_points) >= 2:
        first_iter, first_value = drift_points[0]
        last_iter, last_value = drift_points[-1]
        if last_iter != first_iter:
            drift_slope = (last_value - first_value) / (last_iter - first_iter)

    recycle_count = sum(1 for row in soak_rows if row.get("recycle_reason"))

    return SoakSummary(
        session_id=session_id,
        record_count=len(rows),
        started_at=started_at,
        ended_at=ended_at,
        iteration_count=len(soak_rows),
        warmup_count=len(warmup_rows),
        error_count=len(error_rows),
        recycle_count=recycle_count,
        duration_avg_ms=(sum(durations) / len(durations)) if durations else None,
        duration_p50_ms=_percentile(durations, 0.50),
        duration_p95_ms=_percentile(durations, 0.95),
        duration_p99_ms=_percentile(durations, 0.99),
        drift_first_mb=drift_first,
        drift_last_mb=drift_last,
        drift_min_mb=drift_min,
        drift_max_mb=drift_max,
        drift_slope_mb_per_iteration=drift_slope,
    )

