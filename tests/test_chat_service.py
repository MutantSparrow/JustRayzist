from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.api.inference_service import InferenceService


def test_chat_service_passes_app_context_and_persists_actions(temp_app_paths, make_app_settings, monkeypatch) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    docs_dir = temp_app_paths.root_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "USAGE.md").write_text(
        "# Usage\n\n"
        "## Clarity\n\n"
        "Clarity refines existing gallery images after generation and does not rewrite prompts.\n",
        encoding="utf-8",
    )
    service = InferenceService(settings=settings)
    captured: dict[str, object] = {}
    fake_pack = SimpleNamespace(
        name="Rayzist_bf16",
        base_name="Rayzist_bf16",
        architecture="z_image_turbo",
        components={"text_encoder": SimpleNamespace(path=Path("encoder.gguf"))},
    )

    class _Session:
        def chat(self, **kwargs):
            captured.update(kwargs)
            return {
                "response": "Use this prompt.",
                "actions": [{"type": "set_prompt", "prompt": "rainy neon alley"}],
                "seed": kwargs["seed"],
                "encoder": "encoder.gguf",
            }

        def runtime_status(self):
            return {"backend": "fake"}

    monkeypatch.setattr(service, "_resolve_runtime_pack", lambda _pack_name: (fake_pack, fake_pack, SimpleNamespace()))
    monkeypatch.setattr(service, "_session_for_pack", lambda _pack, _tier: _Session())

    result = service.chat(
        owner_id="Example-Client",
        message="what does clarity do?",
        app_state={"current_prompt": "rain", "resolution": "1024x1024", "queue_status": "0/5"},
        seed=123,
    )

    assert "Just Rayzist" in str(captured["app_context"])
    assert "Relevant documentation excerpts" in str(captured["app_context"])
    assert "docs/USAGE.md > Clarity" in str(captured["app_context"])
    assert "does not rewrite prompts" in str(captured["app_context"])
    assert "current_prompt_box" in str(captured["app_context"])
    assert "rain" in str(captured["app_context"])
    assert result["history"]["exchanges"][0]["assistant"]["actions"] == [
        {"type": "set_prompt", "prompt": "rainy neon alley", "label": "Use Prompt"}
    ]
