from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.portable.common import (  # noqa: E402
    normalize_platform_name,
    resolve_project_root,
    run_probe,
    shell_setup_command,
    venv_python_path,
)


_NAME_RE = re.compile(r"^\s*name\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class LauncherPack:
    folder_name: str
    display_name: str
    manifest_path: Path


def manifest_display_name(manifest_path: Path) -> str:
    raw = manifest_path.read_text(encoding="utf-8")
    match = _NAME_RE.search(raw)
    if not match:
        return manifest_path.parent.name
    value = match.group(1).strip()
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1].strip() or manifest_path.parent.name
    return value or manifest_path.parent.name


def is_public_enabled_pack(manifest_path: Path) -> bool:
    raw = manifest_path.read_text(encoding="utf-8")
    if re.search(r"^\s*user_visible\s*:\s*false\s*$", raw, flags=re.IGNORECASE | re.MULTILINE):
        return False
    if re.search(r"^\s*enabled\s*:\s*false\s*$", raw, flags=re.IGNORECASE | re.MULTILINE):
        return False
    return True


def discover_public_enabled_packs(model_packs_dir: Path) -> list[LauncherPack]:
    if not model_packs_dir.exists():
        return []
    packs: list[LauncherPack] = []
    for manifest_path in sorted(model_packs_dir.glob("*/modelpack.yaml")):
        if not manifest_path.is_file() or not is_public_enabled_pack(manifest_path):
            continue
        packs.append(
            LauncherPack(
                folder_name=manifest_path.parent.name,
                display_name=manifest_display_name(manifest_path),
                manifest_path=manifest_path.resolve(),
            )
        )
    return packs


def runtime_probe_command() -> str:
    return (
        "import importlib.util, sys, typer, fastapi, uvicorn, PIL, safetensors, app.api.main; "
        "required=('torch','diffusers','transformers','accelerate'); "
        "missing=[name for name in required if importlib.util.find_spec(name) is None]; "
        "sys.exit(0 if not missing else 1)"
    )


def runtime_python_candidates(project_root: Path, *, helper_python: str | None = None) -> list[str]:
    candidates: list[str] = []
    if helper_python:
        candidates.append(helper_python)
    env_python = os.environ.get("JUSTRAYZIST_PYTHON", "").strip()
    if env_python:
        candidates.append(env_python)
    candidates.append(str(venv_python_path(project_root, platform_name=normalize_platform_name())))
    candidates.append(str(project_root / ".venv" / "python.exe"))
    candidates.append("python3")
    candidates.append("python")
    ordered: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in ordered:
            ordered.append(candidate)
    return ordered


def probe_runtime_python(candidate: str) -> bool:
    ok, _ = run_probe([candidate, "-c", runtime_probe_command()])
    return ok


def resolve_runtime_python(project_root: Path, *, helper_python: str | None = None) -> str | None:
    for candidate in runtime_python_candidates(project_root, helper_python=helper_python):
        if probe_runtime_python(candidate):
            return candidate
    return None


def select_pack(
    *,
    public_packs: list[LauncherPack],
    explicit_pack: str | None,
    env_pack: str | None,
    interactive: bool,
) -> str:
    if explicit_pack:
        return explicit_pack.strip()
    if env_pack:
        return env_pack.strip()
    if not public_packs:
        raise RuntimeError(
            "No public enabled model packs found under models/packs. "
            "Run 'python -m app.cli.main validate-models' after setup to inspect the current pack state."
        )
    if len(public_packs) == 1:
        pack = public_packs[0]
        print(f"Auto-selected only available public enabled pack: {pack.display_name}")
        return pack.display_name
    if not interactive:
        raise RuntimeError(
            "Multiple public enabled packs are available. Re-run with --pack <name> or set JUSTRAYZIST_PACK."
        )
    print("Select model pack:")
    for index, pack in enumerate(public_packs, start=1):
        print(f"  [{index}] {pack.display_name}")
    while True:
        raw = input(f"Choose pack [1-{len(public_packs)}]: ").strip()
        if not raw:
            continue
        try:
            selection = int(raw)
        except ValueError:
            continue
        if 1 <= selection <= len(public_packs):
            return public_packs[selection - 1].display_name


def resolve_bind_host(requested_host: str | None, *, listen_env: str | None) -> str:
    if requested_host:
        return requested_host
    if str(listen_env or "").strip() == "1":
        return "0.0.0.0"
    return "127.0.0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the JustRayzist web UI in source mode.")
    parser.add_argument("--project-root", default=str(REPO_ROOT))
    parser.add_argument("--host", default="")
    parser.add_argument("--port", type=int, default=37717)
    parser.add_argument("--pack", default="")
    parser.add_argument("--python-exe", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = resolve_project_root(args.project_root)
    helper_python = args.python_exe.strip() or None
    runtime_python = resolve_runtime_python(project_root, helper_python=helper_python)
    if not runtime_python:
        raise RuntimeError(
            "No usable source Python runtime was found. "
            f"Run {shell_setup_command()} to install or repair the environment."
        )

    public_packs = discover_public_enabled_packs(project_root / "models" / "packs")
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    selected_pack = select_pack(
        public_packs=public_packs,
        explicit_pack=args.pack.strip() or None,
        env_pack=os.environ.get("JUSTRAYZIST_PACK", "").strip() or None,
        interactive=interactive,
    )
    host = resolve_bind_host(args.host.strip() or None, listen_env=os.environ.get("JUSTRAYZIST_LISTEN"))

    env = os.environ.copy()
    env["JUSTRAYZIST_ROOT"] = str(project_root)
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env["JUSTRAYZIST_PACK"] = selected_pack

    print("")
    print("JustRayzist Web Launcher")
    print("========================")
    print(f"Using model pack: {selected_pack}")
    print(f"Bind address: {host}:{args.port}")
    if host == "0.0.0.0":
        print(f"Local URL: http://127.0.0.1:{args.port}/")
        print(f"LAN URL:   http://<your-lan-ip>:{args.port}/")
    else:
        print(f"URL: http://{host}:{args.port}/")
    print("")

    completed = subprocess.run(
        [runtime_python, "-m", "app.cli.main", "serve", "--host", host, "--port", str(args.port)],
        cwd=str(project_root),
        env=env,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
