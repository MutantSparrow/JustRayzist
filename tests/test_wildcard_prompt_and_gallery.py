from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from PIL.PngImagePlugin import PngInfo
import pytest

from app.api.inference_service import InferenceService
from app.config.settings import load_settings
from app.core.backends.diffusers_zimage import GenerationResult
from app.core.prompt_wildcards import expand_prompt_wildcards
from app.storage.gallery_index import get_image, sync_outputs_to_gallery
from app.storage.wildcard_library import create_wildcard, update_wildcard


def test_expand_prompt_wildcards_is_seeded_and_per_occurrence(
    temp_app_paths,
    make_app_settings,
) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    create_wildcard(
        settings,
        display_name="Picturesque Locations",
        token="picturesque-locations",
        content_text=(
            "a cabin in the Schwarzwald\n"
            "a chalet in the French Alps\n"
            "a white sandy beach in Bora-Bora\n"
            "a small cafe in a Parisian side street"
        ),
    )

    prompt = "travel poster __picturesque-locations__ mirrored with __picturesque-locations__"
    resolved_a, occurrences_a = expand_prompt_wildcards(settings, prompt, seed=123)
    resolved_b, occurrences_b = expand_prompt_wildcards(settings, prompt, seed=123)

    assert resolved_a == resolved_b
    assert [item.to_dict() for item in occurrences_a] == [item.to_dict() for item in occurrences_b]
    assert len(occurrences_a) == 2
    assert occurrences_a[0].occurrence_index == 0
    assert occurrences_a[1].occurrence_index == 1
    assert occurrences_a[0].prompt_offset == prompt.index("__picturesque-locations__")
    assert occurrences_a[1].prompt_offset == prompt.rindex("__picturesque-locations__")
    assert "__picturesque-locations__" not in resolved_a


def test_expand_prompt_wildcards_keeps_same_selection_after_token_rename_when_offset_matches(
    temp_app_paths,
    make_app_settings,
) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    created = create_wildcard(
        settings,
        display_name="Mountain Views",
        token="mountains",
        content_text="misty valley\nsnowy ridge\nalpine lake\nforest cabin",
    )

    _resolved_before, before_occurrences = expand_prompt_wildcards(settings, "A __mountains__ vista", seed=77)

    update_wildcard(
        settings,
        wildcard_id=created["id"],
        display_name="Mountain Views",
        token="alps",
        content_text="misty valley\nsnowy ridge\nalpine lake\nforest cabin",
    )

    _resolved_after, after_occurrences = expand_prompt_wildcards(settings, "A __alps__ vista", seed=77)

    assert before_occurrences[0].id == after_occurrences[0].id
    assert before_occurrences[0].selected_entry == after_occurrences[0].selected_entry


def test_expand_prompt_wildcards_rejects_unknown_tokens(
    temp_app_paths,
    make_app_settings,
) -> None:
    settings = make_app_settings(paths=temp_app_paths)

    with pytest.raises(ValueError, match="Wildcard not found"):
        expand_prompt_wildcards(settings, "portrait __missing-token__", seed=5)


def test_inference_generate_persists_wildcard_metadata(monkeypatch, temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    service = InferenceService(settings=settings)
    saved_metadata: dict[str, object] = {}

    fake_pack = SimpleNamespace(name="Rayzist_bf16", base_name="Rayzist_bf16", derived_strategy=None)

    class _FakeSession:
        def generate(self, request):
            return GenerationResult(
                image=Image.new("RGB", (64, 64), color=(40, 80, 120)),
                seed=request.seed,
                steps=28,
                guidance_scale=6.5,
                scheduler_mode="euler",
                backend="diffusers_zimage",
                device="cpu",
                duration_ms=123,
                prompt_original="portrait __picturesque-locations__",
                prompt_wildcard_resolved="portrait a chalet in the French Alps",
                prompt_effective="portrait a chalet in the French Alps, cinematic style",
                prompt_enhanced=False,
                prompt_effective_base="portrait a chalet in the French Alps",
                wildcards=(
                    {
                        "id": "abc123",
                        "display_name": "Picturesque Locations",
                        "token": "picturesque-locations",
                        "placeholder": "__picturesque-locations__",
                        "selected_entry": "a chalet in the French Alps",
                        "occurrence_index": 0,
                        "prompt_offset": 9,
                    },
                ),
                wildcard_count=1,
            )

        def runtime_status(self):
            return {"backend": "diffusers_zimage"}

    def fake_save_png_with_metadata(**kwargs):
        saved_metadata["prompt"] = kwargs["prompt"]
        saved_metadata["extra_metadata"] = kwargs["extra_metadata"]
        return kwargs["output_path"]

    monkeypatch.setattr(service, "_resolve_runtime_pack", lambda pack_name: (fake_pack, fake_pack, settings.resource_tier))
    monkeypatch.setattr(service, "_session_for_pack", lambda model_pack, resource_tier: _FakeSession())
    monkeypatch.setattr("app.api.inference_service.append_generation_metric", lambda **kwargs: None)
    monkeypatch.setattr("app.api.inference_service.save_png_with_metadata", fake_save_png_with_metadata)
    monkeypatch.setattr(
        "app.api.inference_service.index_image",
        lambda settings, image_path, owner_id=None: {
            "filename": image_path.name,
            "output_path": str(image_path),
            "prompt": "portrait a chalet in the French Alps",
        },
    )

    result = service.generate(
        owner_id="example-client",
        prompt="portrait __picturesque-locations__",
        width=64,
        height=64,
        seed=123,
    )

    metadata = dict(saved_metadata["extra_metadata"])
    assert saved_metadata["prompt"] == "portrait a chalet in the French Alps, cinematic style"
    assert metadata["prompt_wildcard_resolved"] == "portrait a chalet in the French Alps"
    assert metadata["wildcard_count"] == 1
    assert '"selected_entry": "a chalet in the French Alps"' in metadata["wildcards_json"]
    assert result["prompt_wildcard_resolved"] == "portrait a chalet in the French Alps"
    assert result["wildcard_count"] == 1


def test_inference_generate_forwards_rplus_request_fields(monkeypatch, temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    service = InferenceService(settings=settings)
    captured: dict[str, object] = {}

    fake_pack = SimpleNamespace(name="Rayzist_bf16", base_name="Rayzist_bf16", derived_strategy=None)

    class _FakeSession:
        def generate(self, request):
            captured["request"] = request
            return GenerationResult(
                image=Image.new("RGB", (64, 64), color=(40, 80, 120)),
                seed=request.seed,
                steps=request.steps or 20,
                guidance_scale=1.0,
                scheduler_mode="euler",
                backend="diffusers_zimage",
                device="cpu",
                duration_ms=123,
                prompt_original=request.prompt,
                prompt_wildcard_resolved=request.prompt,
                prompt_effective=request.prompt,
                prompt_enhanced=False,
                prompt_effective_base=request.prompt,
                inference_process=request.inference_process,
            )

        def runtime_status(self):
            return {"backend": "diffusers_zimage"}

    monkeypatch.setattr(service, "_resolve_runtime_pack", lambda pack_name: (fake_pack, fake_pack, settings.resource_tier))
    monkeypatch.setattr(service, "_session_for_pack", lambda model_pack, resource_tier: _FakeSession())
    monkeypatch.setattr("app.api.inference_service.append_generation_metric", lambda **kwargs: None)
    monkeypatch.setattr(
        "app.api.inference_service.save_png_with_metadata",
        lambda **kwargs: kwargs["output_path"],
    )
    monkeypatch.setattr(
        "app.api.inference_service.index_image",
        lambda settings, image_path, owner_id=None: {
            "filename": image_path.name,
            "output_path": str(image_path),
            "prompt": "portrait",
        },
    )

    service.generate(
        owner_id="example-client",
        prompt="portrait",
        width=64,
        height=64,
        seed=123,
        steps=20,
        inference_process="rplus",
        procedural_creativity=2,
        rplus_vibrance=0.75,
        rplus_initial_bias_level=-0.5,
    )

    request = captured["request"]
    assert request.steps == 20
    assert request.inference_process == "rplus"
    assert request.procedural_creativity == 2
    assert request.rplus_vibrance == 0.75
    assert request.rplus_initial_bias_level == -0.5


def test_gallery_sync_persists_wildcard_metadata(monkeypatch, workspace_tmp_path: Path) -> None:
    root = workspace_tmp_path / "gallery-wildcard-metadata"
    monkeypatch.setenv("JUSTRAYZIST_ROOT", str(root))
    settings = load_settings()

    image_path = settings.paths.outputs_dir / "sample.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (64, 64), color=(90, 120, 150))
    metadata = PngInfo()
    metadata.add_text("timestamp", "2026-04-08T00:00:00+00:00")
    metadata.add_text("prompt", "portrait a chalet in the French Alps")
    metadata.add_text("prompt_wildcard_resolved", "portrait a chalet in the French Alps")
    metadata.add_text("application_name", "JustRayzist")
    metadata.add_text("application_version", "0.1.0")
    metadata.add_text("width", "64")
    metadata.add_text("height", "64")
    metadata.add_text("model_pack", "Rayzist_bf16")
    metadata.add_text("inference_process", "rplus")
    metadata.add_text("procedural_creativity", "2")
    metadata.add_text(
        "wildcards_json",
        '[{"id":"abc123","display_name":"Picturesque Locations","token":"picturesque-locations","placeholder":"__picturesque-locations__","selected_entry":"a chalet in the French Alps","occurrence_index":0,"prompt_offset":9}]',
    )
    metadata.add_text("wildcard_count", "1")
    image.save(image_path, format="PNG", pnginfo=metadata)

    indexed = sync_outputs_to_gallery(settings)

    assert indexed == 1
    row = get_image(settings, "sample.png")
    assert row is not None
    assert row["prompt_wildcard_resolved"] == "portrait a chalet in the French Alps"
    assert row["inference_process"] == "rplus"
    assert row["procedural_creativity"] == 2
    assert '"selected_entry":"a chalet in the French Alps"' in row["wildcards_json"]
    assert row["wildcard_count"] == 1
