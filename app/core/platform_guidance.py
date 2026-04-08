from __future__ import annotations

import platform


def normalize_platform_name(system_name: str | None = None) -> str:
    raw = str(system_name or platform.system()).strip().lower()
    if raw.startswith("win"):
        return "windows"
    if raw == "darwin":
        return "macos"
    if raw.startswith("linux"):
        return "linux"
    return raw or "unknown"


def setup_entrypoint(system_name: str | None = None) -> str:
    return r".\RunMeFirst.bat" if normalize_platform_name(system_name) == "windows" else "./RunMeFirst.sh"


def bootstrap_repair_command(system_name: str | None = None) -> str:
    if normalize_platform_name(system_name) == "windows":
        return r"powershell -ExecutionPolicy Bypass -File scripts\bootstrap_env.ps1"
    return "python3 scripts/portable/bootstrap_env.py"


def setup_repair_hint(
    *,
    system_name: str | None = None,
    include_manual_bootstrap: bool = False,
    purpose: str = "repair the environment",
) -> str:
    setup_command = setup_entrypoint(system_name)
    if include_manual_bootstrap:
        manual_command = bootstrap_repair_command(system_name)
        return f"Run {setup_command} (recommended) or repair with {manual_command}."
    return f"Run {setup_command} to {purpose}."
