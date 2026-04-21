from __future__ import annotations

import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.api import main as api_main
from app.core.cancellation import GenerationCancelledError

CLIENT_HEADER = {"X-JustRayzist-Client": "Example-Client"}


def _png_bytes(size: tuple[int, int] = (32, 32)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(24, 48, 96)).save(buffer, format="PNG")
    return buffer.getvalue()


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
            "steps": None,
            "pack_name": "Rayzist_bf16",
            "job_id": "pending_123",
            "seed": 123,
            "scheduler_mode": "dpm",
            "inference_process": "standard",
            "enhance_prompt": True,
            "procedural_creativity": 2,
            "rplus_vibrance": 0.0,
            "rplus_initial_bias_level": 0.0,
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
            "job_id": "pending_123",
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
        assert kwargs["job_id"] is None
        assert kwargs["steps"] is None
        assert kwargs["inference_process"] == "standard"
        assert kwargs["rplus_vibrance"] == 0.0
        assert kwargs["rplus_initial_bias_level"] == 0.0
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


def test_generate_route_forwards_rplus_payload(monkeypatch) -> None:
    def fake_generate(**kwargs):
        assert kwargs["steps"] == 20
        assert kwargs["inference_process"] == "rplus"
        assert kwargs["rplus_vibrance"] == 1.25
        assert kwargs["rplus_initial_bias_level"] == -0.75
        return {
            "filename": "generated.png",
            "output_path": "S:/STABLEDIFFUSION/JustRayzist/outputs/example-client/generated.png",
            "prompt": "hello world",
            "width": 1024,
            "height": 1024,
            "duration_ms": 1234,
            "url": "/images/generated.png",
            "prompt_enhanced": False,
            "scheduler_mode": "euler",
            "steps": 20,
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
            "steps": 20,
            "inference_process": "rplus",
            "rplus_vibrance": 1.25,
            "rplus_initial_bias_level": -0.75,
        },
    )

    assert response.status_code == 200
    assert response.json()["steps"] == 20


def test_generate_route_maps_cancellation_to_409(monkeypatch) -> None:
    def fake_generate(**_kwargs):
        raise GenerationCancelledError("Generation cancelled.")

    monkeypatch.setattr(api_main.inference, "generate", fake_generate)

    client = TestClient(api_main.app)
    response = client.post(
        "/generate",
        headers=CLIENT_HEADER,
        json={"prompt": "hello world", "width": 1024, "height": 1024},
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "Generation cancelled."


def test_upscale_route_forwards_job_id(monkeypatch) -> None:
    def fake_upscale(**kwargs):
        assert kwargs == {
            "owner_id": "example-client",
            "filename": "source.png",
            "pack_name": "Rayzist_bf16",
            "job_id": "pending_upscale_123",
            "seed": 987,
            "scheduler_mode": "euler",
            "enhance_prompt": False,
        }
        return {
            "filename": "upscaled.png",
            "output_path": "S:/STABLEDIFFUSION/JustRayzist/outputs/example-client/upscaled.png",
            "source_filename": "source.png",
            "duration_ms": 2345,
            "url": "/images/upscaled.png",
        }

    monkeypatch.setattr(api_main.inference, "upscale", fake_upscale)

    client = TestClient(api_main.app)
    response = client.post(
        "/upscale",
        headers=CLIENT_HEADER,
        json={
            "filename": "source.png",
            "pack": "Rayzist_bf16",
            "job_id": "pending_upscale_123",
            "seed": 987,
            "scheduler_mode": "euler",
            "enhance_prompt": False,
        },
    )
    assert response.status_code == 200
    assert response.json()["filename"] == "upscaled.png"


def test_upscale_route_maps_cancellation_to_409(monkeypatch) -> None:
    def fake_upscale(**_kwargs):
        raise GenerationCancelledError("Upscale cancelled.")

    monkeypatch.setattr(api_main.inference, "upscale", fake_upscale)

    client = TestClient(api_main.app)
    response = client.post(
        "/upscale",
        headers=CLIENT_HEADER,
        json={"filename": "source.png"},
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "Upscale cancelled."


def test_upscale_route_rejects_legacy_mode_and_scale_fields() -> None:
    client = TestClient(api_main.app)
    response = client.post(
        "/upscale",
        headers=CLIENT_HEADER,
        json={
            "filename": "source.png",
            "upscale_mode": "hq",
            "upscale_scale": 4,
        },
    )
    assert response.status_code == 422


def test_clarity_route_forwards_job_id(monkeypatch) -> None:
    def fake_clarity(**kwargs):
        assert kwargs == {
            "owner_id": "example-client",
            "filename": "source.png",
            "pack_name": "Rayzist_bf16",
            "job_id": "pending_clarity_123",
            "seed": 654,
            "scheduler_mode": "euler",
            "enhance_prompt": False,
        }
        return {
            "filename": "clarified.png",
            "output_path": "S:/STABLEDIFFUSION/JustRayzist/outputs/example-client/clarified.png",
            "source_filename": "source.png",
            "duration_ms": 3456,
            "url": "/images/clarified.png",
        }

    monkeypatch.setattr(api_main.inference, "clarity", fake_clarity)

    client = TestClient(api_main.app)
    response = client.post(
        "/clarity",
        headers=CLIENT_HEADER,
        json={
            "filename": "source.png",
            "pack": "Rayzist_bf16",
            "job_id": "pending_clarity_123",
            "seed": 654,
            "scheduler_mode": "euler",
            "enhance_prompt": False,
        },
    )
    assert response.status_code == 200
    assert response.json()["filename"] == "clarified.png"


def test_clarity_route_maps_cancellation_to_409(monkeypatch) -> None:
    def fake_clarity(**_kwargs):
        raise GenerationCancelledError("Clarity cancelled.")

    monkeypatch.setattr(api_main.inference, "clarity", fake_clarity)

    client = TestClient(api_main.app)
    response = client.post(
        "/clarity",
        headers=CLIENT_HEADER,
        json={"filename": "source.png"},
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "Clarity cancelled."


def test_clarity_route_rejects_unknown_fields() -> None:
    client = TestClient(api_main.app)
    response = client.post(
        "/clarity",
        headers=CLIENT_HEADER,
        json={
            "filename": "source.png",
            "upscale_mode": "hq",
        },
    )
    assert response.status_code == 422


def test_img2img_route_requires_client_header() -> None:
    client = TestClient(api_main.app)
    response = client.post(
        "/img2img",
        data={"prompt": "hello world", "similarity": "0.8"},
        files={"image": ("reference.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 400
    assert "Missing client id" in response.json()["detail"]


def test_img2img_route_forwards_current_payload(monkeypatch) -> None:
    def fake_img2img(**kwargs):
        assert kwargs["owner_id"] == "example-client"
        assert kwargs["prompt"] == "hello world"
        assert kwargs["image_filename"] == "reference.png"
        assert kwargs["pack_name"] == "Rayzist_bf16"
        assert kwargs["job_id"] == "pending_img2img_123"
        assert kwargs["seed"] == 123
        assert kwargs["scheduler_mode"] == "dpm"
        assert kwargs["enhance_prompt"] is True
        assert kwargs["similarity"] == 0.75
        assert kwargs["loras"] == [{"id": "cinematic-style", "weight": 1.0}]
        assert kwargs["image"].size == (48, 32)
        return {
            "filename": "generated.png",
            "output_path": "S:/STABLEDIFFUSION/JustRayzist/outputs/example-client/generated.png",
            "prompt": "hello world",
            "mode": "img2img",
            "source_filename": "reference.png",
            "similarity": 0.75,
            "width": 48,
            "height": 32,
            "duration_ms": 1234,
            "url": "/images/generated.png",
            "prompt_enhanced": True,
            "scheduler_mode": "dpm",
        }

    monkeypatch.setattr(api_main.inference, "img2img", fake_img2img)

    client = TestClient(api_main.app)
    response = client.post(
        "/img2img",
        headers=CLIENT_HEADER,
        data={
            "prompt": "hello world",
            "pack": "Rayzist_bf16",
            "job_id": "pending_img2img_123",
            "seed": "123",
            "scheduler_mode": "dpm",
            "enhance_prompt": "true",
            "similarity": "0.75",
            "loras": '[{"id":"cinematic-style","weight":1.0}]',
        },
        files={"image": ("reference.png", _png_bytes((48, 32)), "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "generated.png"
    assert payload["mode"] == "img2img"


def test_img2img_route_rejects_invalid_similarity() -> None:
    client = TestClient(api_main.app)
    response = client.post(
        "/img2img",
        headers=CLIENT_HEADER,
        data={"prompt": "hello world", "similarity": "1.2"},
        files={"image": ("reference.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 422


def test_img2img_route_rejects_invalid_loras_json() -> None:
    client = TestClient(api_main.app)
    response = client.post(
        "/img2img",
        headers=CLIENT_HEADER,
        data={"prompt": "hello world", "loras": "not-json"},
        files={"image": ("reference.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 400
    assert "loras" in response.json()["detail"]


def test_img2img_route_maps_cancellation_to_409(monkeypatch) -> None:
    def fake_img2img(**_kwargs):
        raise GenerationCancelledError("Img2img cancelled.")

    monkeypatch.setattr(api_main.inference, "img2img", fake_img2img)

    client = TestClient(api_main.app)
    response = client.post(
        "/img2img",
        headers=CLIENT_HEADER,
        data={"prompt": "hello world"},
        files={"image": ("reference.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "Img2img cancelled."


def test_client_jobs_route_requires_client_header() -> None:
    client = TestClient(api_main.app)
    response = client.get("/client-jobs")
    assert response.status_code == 400
    assert "Missing client id" in response.json()["detail"]


def test_client_jobs_route_returns_service_payload(monkeypatch) -> None:
    def fake_client_job_status(**kwargs):
        assert kwargs == {"owner_id": "example-client"}
        return {
            "active_job": {
                "job_id": "pending_123",
                "kind": "generate",
                "status": "generating",
                "width": 1024,
                "height": 1024,
            }
        }

    monkeypatch.setattr(api_main.inference, "client_job_status", fake_client_job_status)

    client = TestClient(api_main.app)
    response = client.get("/client-jobs", headers=CLIENT_HEADER)
    assert response.status_code == 200
    assert response.json()["active_job"]["job_id"] == "pending_123"


def test_client_jobs_cancel_route_forwards_payload(monkeypatch) -> None:
    def fake_cancel(**kwargs):
        assert kwargs == {"owner_id": "example-client", "job_id": "pending_123"}
        return {
            "status": "ok",
            "cancel_requested": True,
            "job_id": "pending_123",
            "message": "Cancellation requested.",
        }

    monkeypatch.setattr(api_main.inference, "request_cancel_client_job", fake_cancel)

    client = TestClient(api_main.app)
    response = client.post(
        "/client-jobs/cancel",
        headers=CLIENT_HEADER,
        json={"job_id": "pending_123"},
    )
    assert response.status_code == 200
    assert response.json()["cancel_requested"] is True


def test_images_route_returns_service_items(monkeypatch) -> None:
    def fake_list_images(**kwargs):
        assert kwargs == {
            "owner_id": "example-client",
            "prompt_query": "alpha",
            "color_filter": "blue",
            "limit": 20,
            "offset": 5,
            "newest_first": False,
        }
        return [
            {"filename": "a.png", "prompt": "alpha"},
            {"filename": "b.png", "prompt": "beta"},
        ]


    def fake_color_cache_status():
        return {
            "active": True,
            "version": None,
            "target_version": "dominant_v6",
            "needs_rebuild": True,
            "last_error": None,
        }

    monkeypatch.setattr(api_main.inference, "list_images", fake_list_images)
    monkeypatch.setattr(api_main.inference, "gallery_color_cache_status", fake_color_cache_status)

    client = TestClient(api_main.app)
    response = client.get(
        "/images?prompt=alpha&color=blue&limit=20&offset=5&newest_first=false",
        headers=CLIENT_HEADER,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert payload["items"][0]["filename"] == "a.png"
    assert payload["color_cache"]["active"] is True


def test_images_route_forwards_favorites_filter(monkeypatch) -> None:
    def fake_list_images(**kwargs):
        assert kwargs == {
            "owner_id": "example-client",
            "prompt_query": None,
            "color_filter": None,
            "limit": 120,
            "offset": 0,
            "newest_first": True,
            "favorites_only": True,
        }
        return [{"filename": "favorite.png", "favorite": 1}]
    monkeypatch.setattr(api_main.inference, "list_images", fake_list_images)
    monkeypatch.setattr(api_main.inference, "gallery_color_cache_status", lambda: {"active": False})
    client = TestClient(api_main.app)
    response = client.get("/images?favorite=true", headers=CLIENT_HEADER)
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["items"][0]["filename"] == "favorite.png"


def test_image_favorite_route_forwards_payload(monkeypatch) -> None:
    def fake_set_image_favorite(**kwargs):
        assert kwargs == {
            "owner_id": "example-client",
            "filename": "favorite.png",
            "favorite": True,
        }
        return {
            "filename": "favorite.png",
            "favorite": 1,
            "prompt": "alpha",
        }
    monkeypatch.setattr(api_main.inference, "set_image_favorite", fake_set_image_favorite)
    client = TestClient(api_main.app)
    response = client.post(
        "/images/favorite.png/favorite",
        headers=CLIENT_HEADER,
        json={"favorite": True},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["filename"] == "favorite.png"
    assert payload["favorite"] is True
    assert payload["item"]["favorite"] == 1


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
    assert any(item["path"] == "/client-jobs" for item in items)
    assert any(item["path"] == "/client-jobs/cancel" for item in items)
    assert any(item["path"] == "/gallery/rebuild" for item in items)
    assert any(item["path"].startswith("/images?") and "color=blue" in item["path"] for item in items)
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


def test_gallery_rebuild_route_forwards_owner_scope(monkeypatch) -> None:
    def fake_rebuild_gallery(owner_id: str):
        assert owner_id == "example-client"
        return {
            "owner_id": owner_id,
            "scanned_files": 3,
            "indexed": 1,
            "updated": 2,
            "removed_missing": 1,
            "total_items": 3,
        }

    monkeypatch.setattr(api_main.inference, "rebuild_gallery", fake_rebuild_gallery)

    client = TestClient(api_main.app)
    response = client.post("/gallery/rebuild", headers=CLIENT_HEADER)

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "owner_id": "example-client",
        "scanned_files": 3,
        "indexed": 1,
        "updated": 2,
        "removed_missing": 1,
        "total_items": 3,
    }


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




