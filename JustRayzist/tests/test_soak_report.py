from app.storage.soak_report import group_soak_sessions, latest_session_id, summarize_session


def test_soak_report_groups_and_selects_latest_session() -> None:
    rows = [
        {"timestamp": "2026-02-21T20:00:00+00:00", "mode": "soak", "session_id": "s1", "iteration": 1},
        {"timestamp": "2026-02-21T20:10:00+00:00", "mode": "soak", "session_id": "s1", "iteration": 2},
        {"timestamp": "2026-02-21T21:00:00+00:00", "mode": "soak", "session_id": "s2", "iteration": 1},
        {"timestamp": "2026-02-21T21:01:00+00:00", "mode": "soak_error", "session_id": "s2"},
    ]
    grouped = group_soak_sessions(rows)
    assert set(grouped.keys()) == {"s1", "s2"}
    assert latest_session_id(grouped) == "s2"


def test_soak_report_summary_metrics() -> None:
    rows = [
        {"timestamp": "2026-02-21T20:00:00+00:00", "mode": "soak_warmup", "session_id": "s1", "duration_ms": 1000},
        {
            "timestamp": "2026-02-21T20:01:00+00:00",
            "mode": "soak",
            "session_id": "s1",
            "iteration": 1,
            "duration_ms": 1000,
            "memory_drift_mb": 0.0,
            "recycle_reason": None,
        },
        {
            "timestamp": "2026-02-21T20:02:00+00:00",
            "mode": "soak",
            "session_id": "s1",
            "iteration": 2,
            "duration_ms": 2000,
            "memory_drift_mb": 100.0,
            "recycle_reason": "periodic recycle",
        },
        {
            "timestamp": "2026-02-21T20:03:00+00:00",
            "mode": "soak",
            "session_id": "s1",
            "iteration": 3,
            "duration_ms": 3000,
            "memory_drift_mb": 200.0,
            "recycle_reason": None,
        },
    ]
    summary = summarize_session("s1", rows)
    assert summary.iteration_count == 3
    assert summary.warmup_count == 1
    assert summary.error_count == 0
    assert summary.recycle_count == 1
    assert summary.duration_avg_ms == 2000.0
    assert summary.duration_p50_ms == 2000.0
    assert summary.drift_first_mb == 0.0
    assert summary.drift_last_mb == 200.0
    assert summary.drift_slope_mb_per_iteration == 100.0

