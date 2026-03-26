from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from PIL import Image

from app.api.inference_service import InferenceService
from app.core.cancellation import GenerationCancelledError
from app.core.backends.diffusers_zimage import GenerationResult


class _FakeSession:
    def __init__(self, service: InferenceService, owner_id: str):
        self._service = service
        self._owner_id = owner_id
        self.started = threading.Event()
        self.cancelled_count = 0

    def generate(self, _request):
        self.started.set()
        deadline = time.time() + 2.0
        while time.time() < deadline:
            cancel_event = self._service._client_cancel_events.get(self._owner_id)
            if cancel_event is not None and cancel_event.is_set():
                break
            time.sleep(0.01)
        return GenerationResult(
            image=Image.new("RGB", (64, 64), color=(32, 64, 96)),
            seed=123,
            steps=4,
            guidance_scale=0.0,
            scheduler_mode="euler",
            backend="diffusers_zimage",
            device="cpu",
            duration_ms=25,
            prompt_original="prompt",
            prompt_effective="prompt",
            prompt_enhanced=False,
            runtime_profile="balanced",
            resource_tier="balanced",
            execution_mode="model_offload",
            selected_pack="Rayzist_bf16",
            effective_pack="Rayzist_bf16",
        )

    def cancel_active(self):
        self.cancelled_count += 1


class _FakeUpscaleSession:
    def __init__(self):
        self.cancelled_count = 0

    def cancel_active(self):
        self.cancelled_count += 1


def test_request_cancel_client_job_marks_generate_and_interrupts_session(temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    service = InferenceService(settings)
    service._client_active_jobs["example-client"] = {
        "job_id": "pending_123",
        "kind": "generate",
        "status": "generating",
    }
    cancel_event = threading.Event()
    service._client_cancel_events["example-client"] = cancel_event
    fake_session = _FakeUpscaleSession()
    service._active_session = fake_session

    payload = service.request_cancel_client_job("example-client", job_id="pending_123")

    assert payload["status"] == "ok"
    assert payload["cancel_requested"] is True
    assert cancel_event.is_set() is True
    assert service._client_active_jobs["example-client"]["status"] == "cancelling"
    assert fake_session.cancelled_count == 1


def test_generate_cancellation_stops_before_save(monkeypatch, temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    service = InferenceService(settings)
    owner_id = "example-client"
    base_pack = SimpleNamespace(name="Rayzist_bf16", base_name="Rayzist_bf16", derived_strategy=None)
    effective_pack = SimpleNamespace(name="Rayzist_bf16", base_name="Rayzist_bf16", derived_strategy=None)
    resource_tier = settings.resource_tier
    fake_session = _FakeSession(service, owner_id)

    monkeypatch.setattr(service, "_resolve_runtime_pack", lambda _pack_name: (base_pack, effective_pack, resource_tier))

    def fake_session_for_pack(_model_pack, _resource_tier):
        service._active_session = fake_session
        return fake_session

    monkeypatch.setattr(service, "_session_for_pack", fake_session_for_pack)

    save_calls = {"count": 0}

    def fail_save(**_kwargs):
        save_calls["count"] += 1
        raise AssertionError("Cancelled generation should not save output.")

    monkeypatch.setattr("app.api.inference_service.save_png_with_metadata", fail_save)

    result_holder: dict[str, object] = {}

    def _run_generate():
        try:
            service.generate(
                owner_id=owner_id,
                prompt="cancel me",
                width=1024,
                height=1024,
                job_id="pending_456",
            )
        except Exception as exc:  # noqa: BLE001
            result_holder["error"] = exc

    thread = threading.Thread(target=_run_generate, daemon=True)
    thread.start()
    assert fake_session.started.wait(timeout=1.0)

    started_at = time.time()
    cancel_payload = service.request_cancel_client_job(owner_id, job_id="pending_456")
    cancel_duration = time.time() - started_at
    thread.join(timeout=2.0)

    assert cancel_payload["cancel_requested"] is True
    assert cancel_duration < 0.5
    assert isinstance(result_holder.get("error"), GenerationCancelledError)
    assert save_calls["count"] == 0
    assert service.client_job_status(owner_id)["active_job"] is None
    assert fake_session.cancelled_count == 1
