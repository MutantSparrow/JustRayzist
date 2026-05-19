from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import main as api_main


@pytest.mark.parametrize(
    ("path", "process_func_name"),
    [
        ("/server/kill", "_shutdown_server_process"),
        ("/server/restart", "_restart_server_process"),
    ],
)
def test_server_control_allows_local_clients_without_browser_source(
    monkeypatch,
    path: str,
    process_func_name: str,
) -> None:
    called = {"count": 0}

    def fake_process(delay_seconds: float = 0.35) -> None:
        called["count"] += 1

    monkeypatch.setattr(api_main, process_func_name, fake_process)

    client = TestClient(api_main.app)
    response = client.post(path)

    assert response.status_code == 200
    assert called["count"] == 1


@pytest.mark.parametrize(
    ("path", "process_func_name"),
    [
        ("/server/kill", "_shutdown_server_process"),
        ("/server/restart", "_restart_server_process"),
    ],
)
def test_server_control_allows_local_browser_origin(
    monkeypatch,
    path: str,
    process_func_name: str,
) -> None:
    called = {"count": 0}

    def fake_process(delay_seconds: float = 0.35) -> None:
        called["count"] += 1

    monkeypatch.setattr(api_main, process_func_name, fake_process)

    client = TestClient(api_main.app)
    response = client.post(path, headers={"Origin": "http://127.0.0.1:37717"})

    assert response.status_code == 200
    assert called["count"] == 1


@pytest.mark.parametrize(
    ("path", "process_func_name"),
    [
        ("/server/kill", "_shutdown_server_process"),
        ("/server/restart", "_restart_server_process"),
    ],
)
def test_server_control_rejects_foreign_browser_origin(
    monkeypatch,
    path: str,
    process_func_name: str,
) -> None:
    called = {"count": 0}

    def fake_process(delay_seconds: float = 0.35) -> None:
        called["count"] += 1

    monkeypatch.setattr(api_main, process_func_name, fake_process)

    client = TestClient(api_main.app)
    response = client.post(path, headers={"Origin": "https://example.invalid"})

    assert response.status_code == 403
    assert "local app origin" in response.json()["detail"]
    assert called["count"] == 0


def test_server_kill_route_rejects_remote_clients(monkeypatch) -> None:
    called = {"count": 0}

    def fake_shutdown(delay_seconds: float = 0.35) -> None:
        called["count"] += 1

    monkeypatch.setattr(api_main, "_shutdown_server_process", fake_shutdown)

    client = TestClient(api_main.app, client=("192.168.1.44", 50000))
    response = client.post("/server/kill")

    assert response.status_code == 403
    assert "local machine" in response.json()["detail"]
    assert called["count"] == 0


def test_server_restart_route_rejects_remote_clients(monkeypatch) -> None:
    called = {"count": 0}

    def fake_restart(delay_seconds: float = 0.35) -> None:
        called["count"] += 1

    monkeypatch.setattr(api_main, "_restart_server_process", fake_restart)

    client = TestClient(api_main.app, client=("192.168.1.44", 50000))
    response = client.post("/server/restart")

    assert response.status_code == 403
    assert "local machine" in response.json()["detail"]
    assert called["count"] == 0
