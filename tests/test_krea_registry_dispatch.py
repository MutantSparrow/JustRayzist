"""Weightless tests for Krea2-Turbo registry + backend dispatch (WP-1).

These tests exercise pack validation and ``create_backend`` routing without importing torch,
diffusers, or any model weights, so they run in CI on a CPU-only box. The Krea backends are
imported lazily inside ``create_backend``; that import chain is torch-free at module load time
(torch is imported inside functions), which is what keeps this test weightless.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.core.backends import SUPPORTED_BACKENDS, create_backend
from app.core.backends.diffusers_krea import DiffusersKreaBackend, Fp8KreaBackend
from app.core.model_registry import load_model_pack
from app.core.model_registry.model_pack import ALLOWED_ARCHITECTURES


def _fake_pack(*, backend_preference: list[str]):
    return SimpleNamespace(
        name="Krea2_Turbo",
        base_name="Krea2_Turbo",
        backend_preference=backend_preference,
        components={},
        architecture="krea2_turbo",
    )


def test_krea_architecture_allowed() -> None:
    assert "krea2_turbo" in ALLOWED_ARCHITECTURES
    # Z-Image must remain allowed (no regression).
    assert "z_image_turbo" in ALLOWED_ARCHITECTURES


def test_supported_backends_include_krea() -> None:
    assert {"diffusers_krea", "fp8_krea"} <= SUPPORTED_BACKENDS
    # Existing backends untouched.
    assert {"diffusers", "diffusers_zimage", "fp8_zimage"} <= SUPPORTED_BACKENDS


def test_create_backend_routes_diffusers_krea() -> None:
    backend = create_backend(
        settings=SimpleNamespace(),
        model_pack=_fake_pack(backend_preference=["diffusers_krea"]),
        resource_tier=None,
    )
    assert isinstance(backend, DiffusersKreaBackend)
    assert not isinstance(backend, Fp8KreaBackend)


def test_create_backend_prefers_fp8_krea_when_first() -> None:
    backend = create_backend(
        settings=SimpleNamespace(),
        model_pack=_fake_pack(backend_preference=["fp8_krea", "diffusers_krea"]),
        resource_tier=None,
    )
    assert isinstance(backend, Fp8KreaBackend)


def test_unsupported_backend_error_lists_krea() -> None:
    try:
        create_backend(
            settings=SimpleNamespace(),
            model_pack=_fake_pack(backend_preference=["nonsense_backend"]),
            resource_tier=None,
        )
    except ValueError as exc:
        message = str(exc)
        assert "diffusers_krea" in message
        assert "fp8_krea" in message
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for unsupported backend preference.")


def test_krea_pack_validates(workspace_tmp_path: Path) -> None:
    pack_dir = workspace_tmp_path / "Krea2_Turbo"
    config_dir = pack_dir / "config"
    weights_dir = pack_dir / "weights"
    config_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    (weights_dir / "krea2_transformer.safetensors").write_bytes(b"ok")
    (weights_dir / "krea2_vae.safetensors").write_bytes(b"ok")
    (config_dir / "text_encoder").mkdir(parents=True, exist_ok=True)
    (config_dir / "text_encoder" / "model.safetensors").write_bytes(b"ok")
    (config_dir / "model_index.json").write_text("{}", encoding="utf-8")

    (pack_dir / "modelpack.yaml").write_text(
        "\n".join(
            [
                "name: Krea2_Turbo",
                "architecture: krea2_turbo",
                "backend_preference:",
                "  - fp8_krea",
                "  - diffusers_krea",
                "pipeline_config_dir: ./config",
                "components:",
                "  transformer:",
                "    path: ./weights/krea2_transformer.safetensors",
                "    format: safetensors",
                "  vae:",
                "    path: ./weights/krea2_vae.safetensors",
                "    format: safetensors",
                "  text_encoder:",
                "    path: ./config/text_encoder/model.safetensors",
                "    format: safetensors",
                "required_configs:",
                "  - ./config/model_index.json",
                "",
            ]
        ),
        encoding="utf-8",
    )

    pack = load_model_pack(pack_dir / "modelpack.yaml")
    assert pack.architecture == "krea2_turbo"
    assert pack.backend_preference == ["fp8_krea", "diffusers_krea"]
    assert "transformer" in pack.components
    assert "text_encoder" in pack.components
