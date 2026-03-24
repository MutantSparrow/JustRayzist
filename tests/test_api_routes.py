from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.api import main as api_main

CLIENT_HEADER = {"X-JustRayzist-Client": "Example-Client"}


def test_generate_route_requires_client_header() -> None:
    client = TestClient(api_main.app)
    response = client.post(
        "/generate",
        json={"prompt": "hello world", "width": 1024, "height": 1024},
    )
    assert response.status_code == 400
    assert "Missing client id" in response.json()["detail"]


def test_generate_route_forwards_current_payload(monkeypatch) -> None:
    def fake_generate(**kwargs):
        assert kwargs == {
            "owner_id": "example-client",
            "prompt": "hello world",
            "width": 832,
            "height": 1248,
            "pack_name": "Rayzist_bf16",
            "seed": 123,
            "scheduler_mode": "dpm",
            "enhance_prompt": True,
            "procedural_creativity": 2,
        }
        return {
            "filename": "generated.png",
            "output_path": "S:/STABLEDIFFUSION/JustRayzist/outputs/example-client/generated.png",
            "prompt": "hello world",
            "width": 832,
            "height": 1248,
            "duration_ms": 1234,
            "url": "/images/generated.png",
            "prompt_enhanced": True,
            "scheduler_mode": "dpm",
        }

    monkeypatch.setattr(api_main.inference, "generate", fake_generate)

    client = TestClient(api_main.app)
    response = client.post(
        "/generate",
        headers=CLIENT_HEADER,
        json={
            "prompt": "hello world",
            "width": 832,
            "height": 1248,
            "pack": "Rayzist_bf16",
            "seed": 123,
            "scheduler_mode": "dpm",
            "enhance_prompt": True,
            "procedural_creativity": 2,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "generated.png"
    assert payload["prompt_enhanced"] is True


def test_generate_route_rejects_invalid_procedural_creativity() -> None:
    client = TestClient(api_main.app)
    response = client.post(
        "/generate",
        headers=CLIENT_HEADER,
        json={
            "prompt": "hello world",
            "width": 1024,
            "height": 1024,
            "procedural_creativity": 4,
        },
    )
    assert response.status_code == 422


def test_generate_route_accepts_slider_only_payload(monkeypatch) -> None:
    def fake_generate(**kwargs):
        assert kwargs["procedural_creativity"] == 1
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
            "procedural_creativity": 1,
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
            "procedural_creativity": 1,
        },
    )
    assert response.status_code == 200
    assert response.json()["filename"] == "generated.png"


def test_images_route_returns_service_items(monkeypatch) -> None:
    def fake_list_images(**kwargs):
        assert kwargs == {
            "owner_id": "example-client",
            "prompt_query": "alpha",
            "limit": 20,
            "offset": 5,
            "newest_first": False,
        }
        return [
            {"filename": "a.png", "prompt": "alpha"},
            {"filename": "b.png", "prompt": "beta"},
        ]

    monkeypatch.setattr(api_main.inference, "list_images", fake_list_images)

    client = TestClient(api_main.app)
    response = client.get(
        "/images?prompt=alpha&limit=20&offset=5&newest_first=false",
        headers=CLIENT_HEADER,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert payload["items"][0]["filename"] == "a.png"


def test_image_file_route_serves_png(monkeypatch, workspace_tmp_path: Path) -> None:
    temp_dir = workspace_tmp_path / "api-image"
    image_path = temp_dir / "served.png"
    temp_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(0, 120, 220)).save(image_path, format="PNG")

    def fake_get_image(filename: str, owner_id: str):
        assert filename == "served.png"
        assert owner_id == "example-client"
        return {"filename": filename, "output_path": str(image_path)}

    monkeypatch.setattr(api_main.inference, "get_image", fake_get_image)

    client = TestClient(api_main.app)
    response = client.get("/images/served.png?client_id=example-client")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_bulk_download_route_streams_zip(monkeypatch, workspace_tmp_path: Path) -> None:
    temp_dir = workspace_tmp_path / "zip-download"
    image_path = temp_dir / "served.png"
    temp_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color=(220, 50, 90)).save(image_path, format="PNG")

    def fake_resolve_download_images(**kwargs):
        assert kwargs == {
            "owner_id": "example-client",
            "filenames": ["served.png"],
        }
        return [("served.png", image_path)]

    monkeypatch.setattr(api_main.inference, "resolve_download_images", fake_resolve_download_images)

    client = TestClient(api_main.app)
    response = client.post(
        "/images/download-zip",
        headers=CLIENT_HEADER,
        json={"filenames": ["served.png"]},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert "selection.zip" in response.headers["content-disposition"]
    assert b"served.png" in response.content


def test_api_manifest_route_lists_bulk_download_route() -> None:
    client = TestClient(api_main.app)
    response = client.get("/api-manifest")

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] >= 1
    items = payload["items"]
    assert any(item["path"] == "/images/download-zip" for item in items)
    assert any(item["path"] == "/health" for item in items)


def test_delete_gallery_route_accepts_query_confirmation(monkeypatch) -> None:
    def fake_delete_gallery(**kwargs):
        assert kwargs == {"owner_id": "example-client", "confirm_text": "delete"}
        return {"deleted_files": 1, "deleted_rows": 1, "remaining_rows": 0}

    monkeypatch.setattr(api_main.inference, "delete_gallery", fake_delete_gallery)

    client = TestClient(api_main.app)
    response = client.request(
        "DELETE",
        "/gallery?confirm=delete",
        headers=CLIENT_HEADER,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["deleted_rows"] == 1


def test_server_kill_route_schedules_shutdown(monkeypatch) -> None:
    called = {"count": 0}

    def fake_shutdown(delay_seconds: float = 0.35) -> None:
        called["count"] += 1
        assert delay_seconds == 0.35

    monkeypatch.setattr(api_main, "_shutdown_server_process", fake_shutdown)

    client = TestClient(api_main.app)
    response = client.post("/server/kill")
    assert response.status_code == 200
    assert response.json()["message"] == "Server shutdown initiated."
    assert called["count"] == 1

