from __future__ import annotations

from types import SimpleNamespace

from app.config.profiles import RUNTIME_PROFILES
from app.core.backends.diffusers_zimage import DiffusersZImageBackend, VramPreflightResult
from app.core.worker.types import GenerationRequest


def _make_backend(profile_name: str = "high") -> DiffusersZImageBackend:
    settings = SimpleNamespace(runtime_profile=RUNTIME_PROFILES[profile_name])
    model_pack = SimpleNamespace(name="Rayzist_bf16")
    return DiffusersZImageBackend(settings=settings, model_pack=model_pack)


def test_rplus_bravo_sigmas_refine_above_nine_steps() -> None:
    backend = _make_backend()

    stage1, stage2, stage3 = backend._rplus_bravo_sigmas(11)

    assert stage1 == list(backend._RPLUS_SIGMA_PRESET_BRAVO[9][0])
    assert len(stage2) == 7
    assert len(stage3) == 5
    assert backend._rplus_step_count(stage1) + backend._rplus_step_count(stage2) + backend._rplus_step_count(stage3) == 11
    assert all(stage2[index] >= stage2[index + 1] for index in range(len(stage2) - 1))
    assert all(stage3[index] >= stage3[index + 1] for index in range(len(stage3) - 1))


def test_rplus_prepare_stage_sigmas_skips_stage3_noise_when_truncated() -> None:
    backend = _make_backend()

    sigmas1, sigmas2, sigmas3, add_noise_stage3 = backend._rplus_prepare_stage_sigmas(
        steps=9,
        sigma_limits=(0.0, 0.5),
    )

    assert sigmas1 == [0.5, 0.0]
    assert sigmas2 is None
    assert sigmas3 == [0.5, 0.4556, 0.2, 0.0]
    assert add_noise_stage3 is False


def test_rplus_stage3_seed_constant_is_product_value() -> None:
    assert DiffusersZImageBackend._RPLUS_STAGE3_SEED == 37_717


def test_generate_allows_rplus_with_procedural_creativity(monkeypatch) -> None:
    backend = _make_backend()
    fake_pipe = SimpleNamespace()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        backend,
        "_ensure_loaded",
        lambda: SimpleNamespace(pipeline=fake_pipe, device="cpu"),
    )
    monkeypatch.setattr(backend, "_apply_scheduler_mode", lambda pipe, mode: mode)
    monkeypatch.setattr(backend, "_resolve_effective_prompt", lambda **kwargs: (kwargs["prompt"], kwargs["prompt"], False))
    monkeypatch.setattr(
        backend,
        "_resolve_procedural_latents",
        lambda **kwargs: ("procedural-latents", "proc_v4", 0.91, "procedural_normalize_mix"),
    )
    monkeypatch.setattr(backend, "_build_generator", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        backend,
        "_run_vram_preflight",
        lambda _torch: VramPreflightResult(False, False, None, None, None, None, None),
    )
    monkeypatch.setattr(backend, "_apply_high_runtime_fallback_if_needed", lambda **kwargs: None)
    monkeypatch.setattr(backend, "_cuda_free_total_snapshot", lambda _torch: (None, None))

    def fake_run_rplus_generate(*, pipe, request, prompt_effective, procedural_latents, torch_module):
        captured["request"] = request
        captured["prompt_effective"] = prompt_effective
        captured["procedural_latents"] = procedural_latents
        return "rplus-image", {
            "scheduler_mode": "euler",
            "guidance_scale": 1.0,
            "rplus_effective_initial_noise_bias_level": 2.5,
            "rplus_initial_sample_size": "full_size",
            "rplus_stage3_seed": backend._RPLUS_STAGE3_SEED,
            "rplus_stage_count": 3,
            "rplus_stage1_steps": 1,
            "rplus_stage2_steps": 5,
            "rplus_stage3_steps": 3,
            "rplus_stage1_ran": True,
            "rplus_stage2_ran": True,
            "rplus_stage3_ran": True,
        }

    monkeypatch.setattr(backend, "_run_rplus_generate", fake_run_rplus_generate)

    from app.core.backends import diffusers_zimage as backend_module

    monkeypatch.setattr(backend_module, "cuda_memory_snapshot", lambda _torch: None)
    monkeypatch.setattr(backend_module, "process_memory_snapshot", lambda: None)
    monkeypatch.setattr(backend_module, "expand_prompt_wildcards", lambda settings, prompt, seed: (prompt, []))

    result = backend.generate(
        GenerationRequest(
            prompt="rplus test",
            width=64,
            height=64,
            seed=123,
            procedural_creativity=2,
            inference_process="rplus",
        )
    )

    assert captured["procedural_latents"] == "procedural-latents"
    assert captured["prompt_effective"] == "rplus test"
    assert result.image == "rplus-image"
    assert result.inference_process == "rplus"
    assert result.guidance_scale == 1.0
    assert result.scheduler_mode == "euler"
    assert result.procedural_creativity == 2
    assert result.procedural_latent_enabled is True
    assert result.procedural_latent_recipe == "proc_v4"
    assert result.rplus_stage3_seed == 37_717
