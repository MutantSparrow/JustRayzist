from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import main as api_main

CLIENT_HEADER = {"X-JustRayzist-Client": "Example-Client"}


def _history() -> dict[str, object]:
    return {
        "owner_id": "example-client",
        "next_number": 3,
        "exchange_count": 1,
        "exchanges": [
            {
                "user": {
                    "number": 1,
                    "role": "user",
                    "content": "hello",
                    "created_at": "2026-04-08T12:00:00+00:00",
                    "error": False,
                },
                "assistant": {
                    "number": 2,
                    "role": "assistant",
                    "content": "hi",
                    "created_at": "2026-04-08T12:00:01+00:00",
                    "error": False,
                },
            }
        ],
    }


def _capabilities() -> dict[str, object]:
    return {"supported": True, "active_pack": "Rayzist_bf16", "encoder": "text_encoder.gguf"}


def test_chat_history_route_requires_client_header() -> None:
    client = TestClient(api_main.app)
    response = client.get("/chat/history")
    assert response.status_code == 400
    assert "Missing client id" in response.json()["detail"]


def test_chat_history_route_returns_service_payload(monkeypatch) -> None:
    def fake_chat_history(**kwargs):
        assert kwargs == {"owner_id": "example-client", "pack_name": None}
        return {"status": "ok", "history": _history(), "capabilities": _capabilities()}

    monkeypatch.setattr(api_main.inference, "chat_history", fake_chat_history)

    client = TestClient(api_main.app)
    response = client.get("/chat/history", headers=CLIENT_HEADER)

    assert response.status_code == 200
    assert "no-store" in response.headers["cache-control"].lower()
    assert response.json()["history"]["next_number"] == 3


def test_chat_history_clear_route_returns_service_payload(monkeypatch) -> None:
    def fake_clear_chat_history(**kwargs):
        assert kwargs == {"owner_id": "example-client", "pack_name": None}
        return {"status": "ok", "history": {"exchanges": [], "next_number": 1}, "capabilities": _capabilities()}

    monkeypatch.setattr(api_main.inference, "clear_chat_history", fake_clear_chat_history)

    client = TestClient(api_main.app)
    response = client.delete("/chat/history", headers=CLIENT_HEADER)

    assert response.status_code == 200
    assert response.json()["history"]["next_number"] == 1


def test_chat_route_forwards_message_payload(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_chat(**kwargs):
        captured.update(kwargs)
        return {
            "status": "ok",
            "exchange": _history()["exchanges"][0],
            "history": _history(),
            "capabilities": _capabilities(),
            "seed": 123,
            "encoder": "text_encoder.gguf",
        }

    monkeypatch.setattr(api_main.inference, "chat", fake_chat)

    client = TestClient(api_main.app)
    response = client.post(
        "/chat",
        headers=CLIENT_HEADER,
        json={
            "message": "hello",
            "pack": "Rayzist_bf16",
            "seed": 123,
            "max_new_tokens": 128,
            "temperature": 0.5,
            "app_state": {"current_prompt": "rain"},
        },
    )

    assert response.status_code == 200
    assert captured == {
        "owner_id": "example-client",
        "message": "hello",
        "pack_name": "Rayzist_bf16",
        "app_state": {"current_prompt": "rain"},
        "seed": 123,
        "max_new_tokens": 128,
        "temperature": 0.5,
    }
    assert response.json()["exchange"]["assistant"]["content"] == "hi"


def test_chat_route_accepts_messages_fallback(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_chat(**kwargs):
        captured.update(kwargs)
        return {
            "status": "ok",
            "exchange": _history()["exchanges"][0],
            "history": _history(),
            "capabilities": _capabilities(),
        }

    monkeypatch.setattr(api_main.inference, "chat", fake_chat)

    client = TestClient(api_main.app)
    response = client.post(
        "/chat",
        headers=CLIENT_HEADER,
        json={"messages": [{"role": "user", "content": "from messages"}]},
    )

    assert response.status_code == 200
    assert captured["message"] == "from messages"
    assert captured["app_state"] == {}


def test_chat_route_rejects_empty_message() -> None:
    client = TestClient(api_main.app)
    response = client.post("/chat", headers=CLIENT_HEADER, json={"messages": []})
    assert response.status_code == 400
    assert "Chat message is required" in response.json()["detail"]


def test_chat_history_route_forwards_pack_query(monkeypatch) -> None:
    """The UI stamps ?pack=<selected> so capabilities reflect the active pack, not the default."""

    captured: dict[str, object] = {}

    def fake_chat_history(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "history": _history(), "capabilities": _capabilities()}

    monkeypatch.setattr(api_main.inference, "chat_history", fake_chat_history)

    client = TestClient(api_main.app)
    response = client.get(
        "/chat/history?pack=Krea2_Turbo",
        headers=CLIENT_HEADER,
    )
    assert response.status_code == 200
    assert captured == {"owner_id": "example-client", "pack_name": "Krea2_Turbo"}


def test_chat_history_clear_route_forwards_pack_query(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_clear(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "history": {"exchanges": [], "next_number": 1}, "capabilities": _capabilities()}

    monkeypatch.setattr(api_main.inference, "clear_chat_history", fake_clear)

    client = TestClient(api_main.app)
    response = client.delete(
        "/chat/history?pack=Krea2_Turbo",
        headers=CLIENT_HEADER,
    )
    assert response.status_code == 200
    assert captured == {"owner_id": "example-client", "pack_name": "Krea2_Turbo"}
