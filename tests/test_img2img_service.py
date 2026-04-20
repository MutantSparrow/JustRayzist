from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from app.api.inference_service import InferenceService
from app.core.backends.diffusers_zimage import GenerationResult


def test_img2img_helpers_resize_and_map_similarity(temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    service = InferenceService(settings=settings)

    normalized, info = service.normalize_img2img_reference_image(Image.new("RGBA", (3000, 2000), color=(10, 20, 30, 180)))

    assert normalized.mode == "RGB"
    assert normalized.width * normalized.height <= 1_500_000
    assert normalized.width % 32 == 0
    assert normalized.height % 32 == 0
    assert info["source_width"] == 3000
    assert info["source_height"] == 2000
    assert InferenceService.similarity_to_refine_strength(0.80) == pytest.approx(0.20)
    assert InferenceService.similarity_to_refine_strength(1.0) == pytest.approx(0.05)
    assert InferenceService.similarity_to_refine_strength(0.0) == pytest.approx(0.95)


def test_inference_img2img_persists_lineage(monkeypatch, temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    service = InferenceService(settings=settings)
    saved_metadata: dict[str, object] = {}
    captured: dict[str, object] = {}
    fake_pack = SimpleNamespace(name="Rayzist_bf16", base_name="Rayzist_bf16", derived_strategy=None)

    class _FakeSession:
        def refine_image(self, input_image, request):
            captured["input_size"] = input_image.size
            captured["request"] = request
            return GenerationResult(
                image=Image.new("RGB", input_image.size, color=(40, 80, 120)),
                seed=request.seed,
                steps=9,
                guidance_scale=2.5,
                scheduler_mode="euler",
                backend="diffusers_zimage",
                device="cpu",
                duration_ms=123,
                prompt_original=request.prompt,
                prompt_wildcard_resolved="hello from wildcards",
                prompt_effective_base="hello from enhancer",
                prompt_effective="hello from enhancer, cinematic",
                prompt_enhanced=True,
                mode="img2img",
                refine_strength=request.refine_strength,
                refine_pass_count=2,
                refine_pass1_steps=9,
                refine_pass2_steps=6,
                refine_pass2_strength=0.1,
                runtime_profile="balanced",
                resource_tier="high",
                execution_mode="model_offload",
                selected_pack="Rayzist_bf16",
                effective_pack="Rayzist_bf16",
            )

        def runtime_status(self):
            return {"backend": "diffusers_zimage"}

    def fake_save_png_with_metadata(**kwargs):
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
            "prompt": "hello world",
            "mode": "img2img",
        },
    )

    result = service.img2img(
        owner_id="example-client",
        prompt="hello world",
        image=Image.new("RGB", (2200, 1400), color=(12, 34, 56)),
        image_filename="reference.png",
        seed=123,
        similarity=0.8,
    )

    request = captured["request"]
    assert request.refine_strength == pytest.approx(0.2)
    assert captured["input_size"][0] * captured["input_size"][1] <= 1_500_000
    assert captured["input_size"][0] % 32 == 0
    assert captured["input_size"][1] % 32 == 0
    metadata = dict(saved_metadata["extra_metadata"])
    assert metadata["mode"] == "img2img"
    assert metadata["source_filename"] == "reference.png"
    assert metadata["similarity"] == 0.8
    assert metadata["prompt_wildcard_resolved"] == "hello from wildcards"
    assert metadata["prompt_effective_base"] == "hello from enhancer"
    assert metadata["prompt_effective"] == "hello from enhancer, cinematic"
    assert metadata["prompt_enhanced"] is True
    assert metadata["refine_pass_count"] == 2
    assert metadata["refine_pass1_steps"] == 9
    assert metadata["refine_pass2_steps"] == 6
    assert metadata["refine_pass2_strength"] == pytest.approx(0.1)
    assert result["mode"] == "img2img"
    assert result["source_filename"] == "reference.png"
    assert result["similarity"] == 0.8
    assert result["prompt_wildcard_resolved"] == "hello from wildcards"
    assert result["prompt_effective_base"] == "hello from enhancer"
    assert result["prompt_effective"] == "hello from enhancer, cinematic"
    assert result["prompt_enhanced"] is True
    assert result["refine_pass_count"] == 2
    assert result["refine_pass1_steps"] == 9
    assert result["refine_pass2_steps"] == 6
    assert result["refine_pass2_strength"] == pytest.approx(0.1)
