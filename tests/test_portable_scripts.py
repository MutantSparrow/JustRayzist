from __future__ import annotations

from pathlib import Path

import pytest

from app.core.platform_guidance import bootstrap_repair_command, setup_entrypoint, setup_repair_hint
from scripts.portable.bootstrap_env import filtered_runtime_lock_lines, resolve_torch_requirements_path
from scripts.portable.fetch_model_assets import should_skip_download, staged_download_path
from scripts.portable.fetch_seedvr2_runtime import apply_allocator_compat_patch
from scripts.portable.start_web import discover_public_enabled_packs, resolve_bind_host, select_pack


def test_setup_repair_hint_is_platform_aware() -> None:
    assert setup_entrypoint("Windows") == r".\RunMeFirst.bat"
    assert setup_entrypoint("Linux") == "./RunMeFirst.sh"
    assert bootstrap_repair_command("Windows") == r"powershell -ExecutionPolicy Bypass -File scripts\bootstrap_env.ps1"
    assert bootstrap_repair_command("Darwin") == "python3 scripts/portable/bootstrap_env.py"
    assert "RunMeFirst.sh" in setup_repair_hint(system_name="Linux")
    assert "scripts\\bootstrap_env.ps1" in setup_repair_hint(
        system_name="Windows",
        include_manual_bootstrap=True,
    )


def test_filtered_runtime_lock_lines_removes_diffusers_only() -> None:
    lines = [
        "# comment",
        "",
        "diffusers==0.36.0",
        "transformers==4.50.0",
        "diffusers @ https://example.invalid/diffusers.whl",
        "accelerate==1.0.0",
    ]

    assert filtered_runtime_lock_lines(lines) == [
        "transformers==4.50.0",
        "accelerate==1.0.0",
    ]


def test_resolve_torch_requirements_path_uses_default_for_macos(temp_app_root: Path) -> None:
    result = resolve_torch_requirements_path(temp_app_root, platform_name="macos", lane="auto")
    assert result == temp_app_root / "requirements" / "torch-default.txt"


def test_resolve_torch_requirements_path_uses_detected_linux_lane(temp_app_root: Path, monkeypatch) -> None:
    monkeypatch.setattr("scripts.portable.bootstrap_env.detect_linux_cuda_lane", lambda: "cu126")
    result = resolve_torch_requirements_path(temp_app_root, platform_name="linux", lane="auto")
    assert result == temp_app_root / "requirements" / "torch-cu126.txt"


def test_staged_download_path_preserves_repo_subdirectories(temp_app_root: Path) -> None:
    stage_root = temp_app_root / "stage"
    assert staged_download_path(stage_root, "foo/bar/model.safetensors") == stage_root / "foo" / "bar" / "model.safetensors"


def test_should_skip_download_requires_matching_hash(temp_app_root: Path) -> None:
    output_path = temp_app_root / "weights.safetensors"
    output_path.write_bytes(b"hello")

    assert should_skip_download(
        output_path,
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
        overwrite=False,
    )
    assert not should_skip_download(output_path, "deadbeef", overwrite=False)
    assert not should_skip_download(output_path, "", overwrite=True)


def test_apply_allocator_compat_patch_updates_runtime_script(temp_app_root: Path) -> None:
    runtime_script = temp_app_root / "inference_cli.py"
    runtime_script.write_text("os.environ['PYTORCH_CUDA_ALLOC_CONF']='x'", encoding="utf-8")

    changed = apply_allocator_compat_patch(runtime_script)

    assert changed is True
    assert "PYTORCH_ALLOC_CONF" in runtime_script.read_text(encoding="utf-8")


def test_discover_public_enabled_packs_filters_hidden_and_disabled(temp_app_root: Path) -> None:
    packs_dir = temp_app_root / "models" / "packs"
    public_dir = packs_dir / "public_pack"
    hidden_dir = packs_dir / "hidden_pack"
    disabled_dir = packs_dir / "disabled_pack"
    public_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir.mkdir(parents=True, exist_ok=True)
    disabled_dir.mkdir(parents=True, exist_ok=True)

    (public_dir / "modelpack.yaml").write_text("name: Rayzist_bf16\nenabled: true\n", encoding="utf-8")
    (hidden_dir / "modelpack.yaml").write_text("name: HiddenPack\nuser_visible: false\n", encoding="utf-8")
    (disabled_dir / "modelpack.yaml").write_text("name: DisabledPack\nenabled: false\n", encoding="utf-8")

    packs = discover_public_enabled_packs(packs_dir)

    assert [pack.display_name for pack in packs] == ["Rayzist_bf16"]


def test_select_pack_requires_explicit_choice_when_noninteractive(temp_app_root: Path) -> None:
    packs_dir = temp_app_root / "models" / "packs"
    for name in ("alpha", "beta"):
        pack_dir = packs_dir / name
        pack_dir.mkdir(parents=True, exist_ok=True)
        (pack_dir / "modelpack.yaml").write_text(f"name: {name}\n", encoding="utf-8")
    packs = discover_public_enabled_packs(packs_dir)

    with pytest.raises(RuntimeError, match="Multiple public enabled packs"):
        select_pack(public_packs=packs, explicit_pack=None, env_pack=None, interactive=False)


def test_select_pack_prefers_explicit_or_env_value() -> None:
    assert select_pack(public_packs=[], explicit_pack="CustomPack", env_pack="Ignored", interactive=False) == "CustomPack"
    assert select_pack(public_packs=[], explicit_pack=None, env_pack="EnvPack", interactive=False) == "EnvPack"


def test_resolve_bind_host_respects_listen_env() -> None:
    assert resolve_bind_host(None, listen_env="1") == "0.0.0.0"
    assert resolve_bind_host(None, listen_env="0") == "127.0.0.1"
    assert resolve_bind_host("192.168.1.10", listen_env="1") == "192.168.1.10"
