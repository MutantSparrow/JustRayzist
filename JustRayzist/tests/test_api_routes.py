from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image

from app.api import main as api_main


def test_generate_route_uses_inference_service(monkeypatch) -> None:
    def fake_generate(
        prompt: str,
        width: int,
        height: int,
        pack_name: str | None,
        seed: int | None,
        scheduler_mode: str | None,
        enhance_prompt: bool,
    ):
        assert prompt == "hello world"
        assert width == 1024
        assert height == 1024
        assert pack_name == "zimage_turbo_local"
        assert seed is None
        assert scheduler_mode == "euler"
        assert enhance_prompt is False
        return {
            "filename": "generated.png",
            "output_path": "E:/generated.png",
            "prompt": prompt,
            "width": width,
            "height": height,
            "duration_ms": 1234,
            "url": "/images/generated.png",
        }

    monkeypatch.setattr(api_main.inference, "generate", fake_generate)

    client = TestClient(api_main.app)
    response = client.post(
        "/generate",
        json={
            "prompt": "hello world",
            "width": 1024,
            "height": 1024,
            "pack": "zimage_turbo_local",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "generated.png"
    assert payload["duration_ms"] == 1234


def test_generate_route_accepts_supported_non_64_resolution(monkeypatch) -> None:
    monkeypatch.setattr(
        api_main.inference,
        "generate",
        lambda prompt, width, height, pack_name, seed, scheduler_mode, enhance_prompt: {
            "filename": "generated2.png",
            "output_path": "E:/generated2.png",
            "prompt": prompt,
            "width": width,
            "height": height,
            "duration_ms": 1000,
            "url": "/images/generated2.png",
            "seed": 9,
        },
    )
    client = TestClient(api_main.app)
    response = client.post(
        "/generate",
        json={
            "prompt": "portrait test",
            "width": 832,
            "height": 1248,
            "pack": "zimage_turbo_local",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["width"] == 832
    assert payload["height"] == 1248


def test_generate_route_accepts_scheduler_mode_toggle(monkeypatch) -> None:
    def fake_generate(prompt, width, height, pack_name, seed, scheduler_mode, enhance_prompt):
        assert scheduler_mode == "dpm"
        assert enhance_prompt is False
        return {
            "filename": "generated3.png",
            "output_path": "E:/generated3.png",
            "prompt": prompt,
            "width": width,
            "height": height,
            "duration_ms": 900,
            "url": "/images/generated3.png",
            "seed": seed,
            "scheduler_mode": scheduler_mode,
        }

    monkeypatch.setattr(api_main.inference, "generate", fake_generate)
    client = TestClient(api_main.app)
    response = client.post(
        "/generate",
        json={
            "prompt": "scheduler mode test",
            "width": 1024,
            "height": 1024,
            "pack": "zimage_turbo_local",
            "scheduler_mode": "dpm",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["scheduler_mode"] == "dpm"


def test_generate_route_accepts_prompt_enhance_toggle(monkeypatch) -> None:
    def fake_generate(prompt, width, height, pack_name, seed, scheduler_mode, enhance_prompt):
        assert enhance_prompt is True
        return {
            "filename": "generated4.png",
            "output_path": "E:/generated4.png",
            "prompt": prompt,
            "width": width,
            "height": height,
            "duration_ms": 800,
            "url": "/images/generated4.png",
            "prompt_original": "short prompt",
            "prompt_effective": "expanded prompt",
            "prompt_enhanced": True,
        }

    monkeypatch.setattr(api_main.inference, "generate", fake_generate)
    client = TestClient(api_main.app)
    response = client.post(
        "/generate",
        json={
            "prompt": "short prompt",
            "width": 1024,
            "height": 1024,
            "enhance_prompt": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["prompt_enhanced"] is True


def test_upscale_route_uses_inference_service(monkeypatch) -> None:
    def fake_upscale(filename, pack_name, seed, scheduler_mode, enhance_prompt):
        assert filename == "source.png"
        assert pack_name == "zimage_turbo_local"
        assert seed == 101
        assert scheduler_mode == "dpm"
        assert enhance_prompt is True
        return {
            "filename": "upscaled.png",
            "output_path": "E:/upscaled.png",
            "prompt": "hello world",
            "width": 2048,
            "height": 2048,
            "duration_ms": 2042,
            "url": "/images/upscaled.png",
            "mode": "api_upscale",
            "source_filename": "source.png",
        }

    monkeypatch.setattr(api_main.inference, "upscale", fake_upscale)

    client = TestClient(api_main.app)
    response = client.post(
        "/upscale",
        json={
            "filename": "source.png",
            "pack": "zimage_turbo_local",
            "seed": 101,
            "scheduler_mode": "dpm",
            "enhance_prompt": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "upscaled.png"
    assert payload["mode"] == "api_upscale"
    assert payload["source_filename"] == "source.png"


def test_images_route_returns_service_items(monkeypatch) -> None:
    monkeypatch.setattr(
        api_main.inference,
        "list_images",
        lambda prompt_query, limit, offset, newest_first: [
            {"filename": "a.png", "prompt": "alpha"},
            {"filename": "b.png", "prompt": "beta"},
        ],
    )
    client = TestClient(api_main.app)
    response = client.get("/images?prompt=a&limit=20&offset=0&newest_first=false")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert payload["items"][0]["filename"] == "a.png"


def test_image_file_route_serves_png(monkeypatch) -> None:
    temp_dir = Path.cwd() / "data" / f"test_api_{uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    image_path = temp_dir / "served.png"
    try:
        Image.new("RGB", (16, 16), color=(0, 120, 220)).save(image_path, format="PNG")
        monkeypatch.setattr(
            api_main.inference,
            "get_image",
            lambda filename: {"filename": filename, "output_path": str(image_path)},
        )
        client = TestClient(api_main.app)
        response = client.get("/images/served.png")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_delete_gallery_route_requires_confirmation(monkeypatch) -> None:
    monkeypatch.setattr(api_main.inference, "delete_gallery", lambda confirm: {"deleted_files": 3, "deleted_rows": 3})
    client = TestClient(api_main.app)
    response = client.request("DELETE", "/gallery", json={"confirm": "DELETE"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["deleted_files"] == 3


def test_delete_gallery_route_supports_query_confirmation(monkeypatch) -> None:
    monkeypatch.setattr(api_main.inference, "delete_gallery", lambda confirm: {"deleted_files": 1, "deleted_rows": 1})
    client = TestClient(api_main.app)
    response = client.request("DELETE", "/gallery?confirm=delete")
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
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["message"] == "Server shutdown initiated."
    assert called["count"] == 1
