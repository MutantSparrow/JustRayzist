import shutil
from pathlib import Path
from uuid import uuid4

from app.core.model_registry import load_model_pack


def test_model_pack_loads_with_local_files() -> None:
    root = Path.cwd() / "data" / f"test_pack_{uuid4().hex}"
    pack_dir = root / "pack"
    config_dir = pack_dir / "config"
    try:
        pack_dir.mkdir(parents=True, exist_ok=True)
        config_dir.mkdir(parents=True, exist_ok=True)

        (pack_dir / "transformer.safetensors").write_bytes(b"ok")
        (pack_dir / "vae.safetensors").write_bytes(b"ok")
        (config_dir / "model_index.json").write_text("{}", encoding="utf-8")

        (pack_dir / "modelpack.yaml").write_text(
            "\n".join(
                [
                    "name: test_pack",
                    "architecture: z_image_turbo",
                    "backend_preference:",
                    "  - diffusers",
                    "pipeline_config_dir: ./config",
                    "components:",
                    "  transformer:",
                    "    path: ./transformer.safetensors",
                    "    format: safetensors",
                    "  vae:",
                    "    path: ./vae.safetensors",
                    "    format: safetensors",
                    "required_configs:",
                    "  - ./config/model_index.json",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        model_pack = load_model_pack(pack_dir / "modelpack.yaml")
        assert model_pack.name == "test_pack"
        assert model_pack.pipeline_config_dir == config_dir.resolve()
        assert "transformer" in model_pack.components
    finally:
        shutil.rmtree(root, ignore_errors=True)
