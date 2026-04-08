from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import main as api_main


def _capabilities() -> dict[str, object]:
    return {
        "supported": True,
        "active_pack": "Rayzist_bf16",
        "suggestions_supported": True,
    }


def test_wildcards_route_returns_items_and_capabilities(monkeypatch) -> None:
    monkeypatch.setattr(
        api_main.inference,
        "list_wildcards",
        lambda: [{"id": "abc123", "display_name": "Picturesque", "token": "picturesque"}],
    )
    monkeypatch.setattr(api_main.inference, "wildcard_capabilities", lambda pack_name=None: _capabilities())

    client = TestClient(api_main.app)
    response = client.get("/wildcards")

    assert response.status_code == 200
    assert "no-store" in response.headers["cache-control"].lower()
    payload = response.json()
    assert payload["count"] == 1
    assert payload["items"][0]["token"] == "picturesque"
    assert payload["capabilities"]["suggestions_supported"] is True


def test_wildcard_library_event_notifier_increments_revision() -> None:
    current = api_main._WILDCARD_LIBRARY_EVENTS.current()
    api_main._notify_wildcard_library_changed()
    assert api_main._WILDCARD_LIBRARY_EVENTS.current() == current + 1


def test_wildcards_create_route_forwards_payload(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_create_wildcard(*, display_name: str, token: str, content_text: str) -> dict[str, object]:
        captured["display_name"] = display_name
        captured["token"] = token
        captured["content_text"] = content_text
        return {"id": "abc123", "display_name": display_name, "token": token}

    monkeypatch.setattr(api_main.inference, "create_wildcard", fake_create_wildcard)
    monkeypatch.setattr(api_main.inference, "wildcard_capabilities", lambda pack_name=None: _capabilities())

    client = TestClient(api_main.app)
    response = client.post(
        "/wildcards",
        json={
            "display_name": "Picturesque Locations",
            "token": "picturesque-locations",
            "content_text": "a cabin in the Schwarzwald\na chalet in the French Alps",
        },
    )

    assert response.status_code == 200
    assert captured == {
        "display_name": "Picturesque Locations",
        "token": "picturesque-locations",
        "content_text": "a cabin in the Schwarzwald\na chalet in the French Alps",
    }
    assert response.json()["item"]["id"] == "abc123"


def test_wildcards_update_route_forwards_payload(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_update_wildcard(*, wildcard_id: str, display_name: str, token: str, content_text: str) -> dict[str, object]:
        captured["wildcard_id"] = wildcard_id
        captured["display_name"] = display_name
        captured["token"] = token
        captured["content_text"] = content_text
        return {"id": wildcard_id, "display_name": display_name, "token": token}

    monkeypatch.setattr(api_main.inference, "update_wildcard", fake_update_wildcard)

    client = TestClient(api_main.app)
    response = client.patch(
        "/wildcards/abc123",
        json={
            "display_name": "Picturesque Spots",
            "token": "picturesque-spots",
            "content_text": "a white sandy beach in Bora-Bora",
        },
    )

    assert response.status_code == 200
    assert captured == {
        "wildcard_id": "abc123",
        "display_name": "Picturesque Spots",
        "token": "picturesque-spots",
        "content_text": "a white sandy beach in Bora-Bora",
    }


def test_wildcards_delete_route_forwards_id(monkeypatch) -> None:
    def fake_delete_wildcard(wildcard_id: str) -> dict[str, object]:
        assert wildcard_id == "abc123"
        return {"id": wildcard_id, "deleted": True}

    monkeypatch.setattr(api_main.inference, "delete_wildcard", fake_delete_wildcard)

    client = TestClient(api_main.app)
    response = client.delete("/wildcards/abc123")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "id": "abc123", "deleted": True}


def test_wildcard_suggestions_route_forwards_payload(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_suggest_wildcard_entries(**kwargs) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "suggestions": ["a chalet in the French Alps"],
            "accepted_count": 1,
            "target_count": 10,
            "seed": 123,
            "example_word_count": 5,
            "min_words": 5,
            "max_words": 5,
            "partial": True,
            "message": "partial",
        }

    monkeypatch.setattr(api_main.inference, "suggest_wildcard_entries", fake_suggest_wildcard_entries)

    client = TestClient(api_main.app)
    response = client.post(
        "/wildcards/suggestions",
        json={
            "theme": "picturesque locations",
            "format_example": "a cabin in the Schwarzwald",
            "seed": 123,
            "pack": "Rayzist_bf16",
            "existing_entries": ["a cabin in the Schwarzwald"],
        },
    )

    assert response.status_code == 200
    assert captured == {
        "theme": "picturesque locations",
        "format_example": "a cabin in the Schwarzwald",
        "existing_entries": ["a cabin in the Schwarzwald"],
        "seed": 123,
        "pack_name": "Rayzist_bf16",
    }
    assert response.json()["suggestions"] == ["a chalet in the French Alps"]
