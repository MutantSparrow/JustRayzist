from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from app.cli.main import cli

def _write_pack(pack_dir: Path, *, name: str, enabled: bool, with_transformer: bool = True) -> None:
    config_dir = pack_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    if with_transformer:
        (pack_dir / "transformer.safetensors").write_bytes(b"ok")
    (config_dir / "model_index.json").write_text("{}", encoding="utf-8")
    (pack_dir / "modelpack.yaml").write_text(
        "\n".join(
            [
                f"name: {name}",
                f"enabled: {'true' if enabled else 'false'}",
                "architecture: z_image_turbo",
                "pipeline_config_dir: ./config",
                "components:",
                "  transformer:",
                "    path: ./transformer.safetensors",
                "    format: safetensors",
                "required_configs:",
                "  - ./config/model_index.json",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_validate_models_skips_disabled_packs_by_default(temp_app_root: Path, monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setenv("JUSTRAYZIST_ROOT", str(temp_app_root))

    packs_dir = temp_app_root / "models" / "packs"
    enabled_dir = packs_dir / "enabled_pack"
    disabled_dir = packs_dir / "disabled_pack"
    _write_pack(enabled_dir, name="enabled_pack", enabled=True, with_transformer=True)
    _write_pack(disabled_dir, name="disabled_pack", enabled=False, with_transformer=False)

    result = runner.invoke(cli, ["validate-models"])

    assert result.exit_code == 0, result.stdout
    assert "[OK] enabled_pack" in result.stdout
    assert "disabled_pack" not in result.stdout


def test_validate_models_all_includes_disabled_packs(temp_app_root: Path, monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setenv("JUSTRAYZIST_ROOT", str(temp_app_root))

    packs_dir = temp_app_root / "models" / "packs"
    enabled_dir = packs_dir / "enabled_pack"
    disabled_dir = packs_dir / "disabled_pack"
    _write_pack(enabled_dir, name="enabled_pack", enabled=True, with_transformer=True)
    _write_pack(disabled_dir, name="disabled_pack", enabled=False, with_transformer=False)

    result = runner.invoke(cli, ["validate-models", "--all"])

    assert result.exit_code == 1, result.stdout
    assert "[OK] enabled_pack" in result.stdout
    assert "[FAIL]" in result.stdout
    assert "disabled_pack" in result.stdout
