from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import main as api_main

CLIENT_HEADER = {"X-JustRayzist-Client": "Example-Client"}


def _capabilities() -> dict[str, object]:
    return {
        "supported": True,
        "active_pack": "Rayzist_bf16",
        "max_active": 3,
        "min_weight": 0.0,
        "max_weight": 2.0,
        "default_weight": 1.0,
    }


def test_loras_route_returns_items_and_capabilities(monkeypatch) -> None:
    monkeypatch.setattr(
        api_main.inference,
        "list_loras",
        lambda: [{"id": "cinematic-style", "display_name": "Cinematic Style", "preview_is_custom": True}],
    )
    monkeypatch.setattr(api_main.inference, "lora_capabilities", lambda pack_name=None: _capabilities())

    client = TestClient(api_main.app)
    response = client.get("/loras")

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["items"][0]["id"] == "cinematic-style"
    assert payload["capabilities"]["supported"] is True


def test_lora_drafts_route_parses_multipart_and_forwards_file(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_create_lora_draft(*, filename: str, content: bytes) -> dict[str, object]:
        captured["filename"] = filename
        captured["content"] = content
        return {"draft_id": "cinematic-style", "source_filename": filename}

    monkeypatch.setattr(api_main.inference, "create_lora_draft", fake_create_lora_draft)

    client = TestClient(api_main.app)
    response = client.post(
        "/lora-drafts",
        files={"file": ("cinematic-style.safetensors", b"fake-lora-bytes", "application/octet-stream")},
    )

    assert response.status_code == 200
    assert captured == {
        "filename": "cinematic-style.safetensors",
        "content": b"fake-lora-bytes",
    }
    assert response.json()["draft"]["draft_id"] == "cinematic-style"


def test_lora_draft_detect_triggers_route_forwards_id(monkeypatch) -> None:
    def fake_detect_lora_draft_triggers(draft_id: str) -> dict[str, object]:
        assert draft_id == "cinematic-style"
        return {"draft_id": draft_id, "detected_trigger_words": ["cinematic style"]}

    monkeypatch.setattr(api_main.inference, "detect_lora_draft_triggers", fake_detect_lora_draft_triggers)

    client = TestClient(api_main.app)
    response = client.post("/lora-drafts/cinematic-style/detect-triggers")

    assert response.status_code == 200
    assert response.json()["draft"]["draft_id"] == "cinematic-style"


def test_loras_create_route_parses_finalize_multipart(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_finalize_lora_draft(*, draft_id: str, display_name: str, trigger_words: str | None, preview_content: bytes | None):
        captured["draft_id"] = draft_id
        captured["display_name"] = display_name
        captured["trigger_words"] = trigger_words
        captured["preview_content"] = preview_content
        return {"id": "cinematic-style", "display_name": display_name}

    monkeypatch.setattr(api_main.inference, "finalize_lora_draft", fake_finalize_lora_draft)
    monkeypatch.setattr(api_main.inference, "lora_capabilities", lambda pack_name=None: _capabilities())

    client = TestClient(api_main.app)
    response = client.post(
        "/loras",
        data={
            "draft_id": "cinematic-style",
            "display_name": "Cinematic Style",
            "trigger_words": '["cinematic style","moody light"]',
        },
        files={"thumbnail": ("thumb.png", b"png-bytes", "image/png")},
    )

    assert response.status_code == 200
    assert captured == {
        "draft_id": "cinematic-style",
        "display_name": "Cinematic Style",
        "trigger_words": '["cinematic style","moody light"]',
        "preview_content": b"png-bytes",
    }
    assert response.json()["item"]["id"] == "cinematic-style"


def test_loras_update_route_parses_patch_multipart(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_update_lora(*, lora_id: str, display_name: str, trigger_words: str | None, preview_content: bytes | None):
        captured["lora_id"] = lora_id
        captured["display_name"] = display_name
        captured["trigger_words"] = trigger_words
        captured["preview_content"] = preview_content
        return {"id": lora_id, "display_name": display_name}

    monkeypatch.setattr(api_main.inference, "update_lora", fake_update_lora)

    client = TestClient(api_main.app)
    response = client.request(
        "PATCH",
        "/loras/cinematic-style",
        data={
            "display_name": "Cinematic Style Reloaded",
            "trigger_words": '["cinematic style","soft bloom"]',
        },
        files={"thumbnail": ("thumb.png", b"new-png-bytes", "image/png")},
    )

    assert response.status_code == 200
    assert captured == {
        "lora_id": "cinematic-style",
        "display_name": "Cinematic Style Reloaded",
        "trigger_words": '["cinematic style","soft bloom"]',
        "preview_content": b"new-png-bytes",
    }
    assert response.json()["item"]["id"] == "cinematic-style"


def test_lora_delete_route_forwards_id(monkeypatch) -> None:
    def fake_delete_lora(lora_id: str) -> dict[str, object]:
        assert lora_id == "cinematic-style"
        return {"id": "cinematic-style", "deleted_files": 3}

    monkeypatch.setattr(api_main.inference, "delete_lora", fake_delete_lora)

    client = TestClient(api_main.app)
    response = client.delete("/loras/cinematic-style")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "id": "cinematic-style", "deleted_files": 3}


def test_generate_route_forwards_loras_payload(monkeypatch) -> None:
    def fake_generate(**kwargs):
        assert kwargs["owner_id"] == "example-client"
        assert kwargs["loras"] == [{"id": "cinematic-style", "weight": 1.25}]
        return {
            "filename": "generated.png",
            "output_path": "S:/STABLEDIFFUSION/JustRayzist/outputs/example-client/generated.png",
            "prompt": "hello world",
            "width": 1024,
            "height": 1024,
            "duration_ms": 1234,
            "url": "/images/generated.png",
            "prompt_enhanced": False,
            "scheduler_mode": "dpm",
            "lora_count": 1,
            "loras": [{"id": "cinematic-style", "name": "cinematic-style", "weight": 1.25}],
        }

    monkeypatch.setattr(api_main.inference, "generate", fake_generate)

    client = TestClient(api_main.app)
    response = client.post(
        "/generate",
        headers=CLIENT_HEADER,
        json={
            "prompt": "hello world",
            "width": 1024,
            "height": 1024,
            "loras": [{"id": "cinematic-style", "weight": 1.25}],
        },
    )

    assert response.status_code == 200
    assert response.json()["lora_count"] == 1


def test_generate_route_rejects_duplicate_lora_ids() -> None:
    client = TestClient(api_main.app)
    response = client.post(
        "/generate",
        headers=CLIENT_HEADER,
        json={
            "prompt": "hello world",
            "width": 1024,
            "height": 1024,
            "loras": [
                {"id": "cinematic-style", "weight": 1.0},
                {"id": "cinematic-style", "weight": 0.8},
            ],
        },
    )

    assert response.status_code == 422
    assert "LoRA ids must be unique" in str(response.json()["detail"])
