from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path


def normalize_platform_name(platform_name: str | None = None) -> str:
    raw = str(platform_name or platform.system()).strip().lower()
    if raw.startswith("win"):
        return "windows"
    if raw == "darwin":
        return "macos"
    if raw.startswith("linux"):
        return "linux"
    return raw or "unknown"


def is_windows(platform_name: str | None = None) -> bool:
    return normalize_platform_name(platform_name) == "windows"


def is_macos(platform_name: str | None = None) -> bool:
    return normalize_platform_name(platform_name) == "macos"


def resolve_project_root(
    project_root: str | Path | None = None,
    *,
    current_file: str | Path | None = None,
) -> Path:
    if project_root is not None:
        return Path(project_root).expanduser().resolve()
    anchor = Path(current_file or __file__).resolve()
    return anchor.parents[2]


def venv_bin_dir(project_root: Path, *, platform_name: str | None = None) -> Path:
    bin_dir = "Scripts" if is_windows(platform_name) else "bin"
    return project_root / ".venv" / bin_dir


def venv_python_path(project_root: Path, *, platform_name: str | None = None) -> Path:
    filename = "python.exe" if is_windows(platform_name) else "python"
    return venv_bin_dir(project_root, platform_name=platform_name) / filename


def venv_hf_cli_path(project_root: Path, *, platform_name: str | None = None) -> Path:
    filename = "hf.exe" if is_windows(platform_name) else "hf"
    return venv_bin_dir(project_root, platform_name=platform_name) / filename


def shell_setup_command(platform_name: str | None = None) -> str:
    return r".\RunMeFirst.bat" if is_windows(platform_name) else "./RunMeFirst.sh"


def shell_bootstrap_command(platform_name: str | None = None) -> str:
    if is_windows(platform_name):
        return r"powershell -ExecutionPolicy Bypass -File scripts\bootstrap_env.ps1"
    return "python3 scripts/portable/bootstrap_env.py"


def format_command(command: list[str]) -> str:
    return " ".join(str(part) for part in command)


def run_checked(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {format_command(command)}"
        )


def run_probe(command: list[str]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except OSError:
        return False, ""
    return completed.returncode == 0, (completed.stdout or "").strip()


def python_version_tuple(python_exe: str) -> tuple[int, int] | None:
    ok, output = run_probe(
        [
            python_exe,
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ]
    )
    if not ok or not output:
        return None
    parts = output.split(".", 1)
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def ensure_python_311_or_newer(python_exe: str) -> None:
    version = python_version_tuple(python_exe)
    if version is None or version < (3, 11):
        raise RuntimeError(
            f"Python 3.11 or newer is required. Could not verify a compatible interpreter at '{python_exe}'."
        )


def temp_env(project_root: Path) -> dict[str, str]:
    tmp_root = project_root / ".build" / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["TEMP"] = str(tmp_root)
    env["TMP"] = str(tmp_root)
    env["TMPDIR"] = str(tmp_root)
    return env


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def print_stderr(message: str) -> None:
    print(message, file=sys.stderr)
