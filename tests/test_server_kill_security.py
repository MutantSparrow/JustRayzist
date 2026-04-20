from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import main as api_main


def test_server_kill_route_rejects_remote_clients(monkeypatch) -> None:
    called = {"count": 0}

    def fake_shutdown(delay_seconds: float = 0.35) -> None:
        called["count"] += 1

    monkeypatch.setattr(api_main, "_shutdown_server_process", fake_shutdown)

    client = TestClient(api_main.app, client=("192.168.1.44", 50000))
    response = client.post("/server/kill")

    assert response.status_code == 403
    assert "local machine only" in response.json()["detail"]
    assert called["count"] == 0
