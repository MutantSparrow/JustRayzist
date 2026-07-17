from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.portable.common import (  # noqa: E402
    ensure_python_311_or_newer,
    normalize_platform_name,
    resolve_project_root,
    run_checked,
    run_probe,
    temp_env,
    venv_python_path,
)


def filtered_runtime_lock_lines(lines: list[str]) -> list[str]:
    filtered: list[str] = []
    for line in lines:
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("#"):
            continue
        if trimmed.startswith("diffusers") and (
            len(trimmed) == len("diffusers")
            or trimmed[len("diffusers")] in {" ", "=", ">", "<", "~", "@"}
        ):
            continue
        filtered.append(line)
    return filtered


def detect_linux_cuda_lane() -> str:
    ok, output = run_probe(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ]
    )
    if not ok or not output:
        return "default"
    first_line = output.splitlines()[0].strip()
    if not first_line or "," not in first_line:
        return "cu128"
    _gpu_name, driver_text = (part.strip() for part in first_line.split(",", 1))
    try:
        major, minor = driver_text.split(".", 1)
        driver_tuple = (int(major), int(minor))
    except ValueError:
        return "cu128"
    if driver_tuple >= (572, 61):
        return "cu128"
    if driver_tuple >= (561, 17):
        return "cu126"
    return "cu126"


def resolve_torch_requirements_path(project_root: Path, *, platform_name: str, lane: str) -> Path:
    normalized_platform = platform_name.strip().lower()
    normalized_lane = lane.strip().lower()
    if normalized_platform == "macos":
        return project_root / "requirements" / "torch-default.txt"
    if normalized_lane == "auto":
        normalized_lane = detect_linux_cuda_lane() if normalized_platform == "linux" else "cu128"
    if normalized_lane == "default":
        return project_root / "requirements" / "torch-default.txt"
    if normalized_lane not in {"cu126", "cu128"}:
        raise RuntimeError(f"Unsupported torch lane '{lane}'.")
    return project_root / "requirements" / f"torch-{normalized_lane}.txt"


def module_imports(python_exe: str, module_name: str) -> bool:
    ok, _ = run_probe([python_exe, "-c", f"import {module_name}"])
    return ok


def diffusers_symbols_available(python_exe: str) -> bool:
    """Probe for the diffusers symbols both pack families need.

    Z-Image ships with ``ZImagePipeline`` / ``ZImageTransformer2DModel`` (diffusers >=0.36).
    Krea2 requires ``Krea2Pipeline`` / ``Krea2Transformer2DModel`` / ``AutoencoderKLQwenImage``
    (diffusers >=0.39). We require the Krea2-inclusive set so a single-pin install covers both
    packs; the fallback installer below bumps to 0.39 if a stale build is detected.
    """
    ok, _ = run_probe(
        [
            python_exe,
            "-c",
            (
                "from diffusers import ZImagePipeline, ZImageTransformer2DModel, "
                "ZImageImg2ImgPipeline, Krea2Pipeline, Krea2Transformer2DModel, "
                "AutoencoderKLQwenImage"
            ),
        ]
    )
    return ok


def install_diffusers_with_fallback(python_exe: str, *, env: dict[str, str]) -> None:
    """Ensure a diffusers build exposing both Z-Image and Krea2 symbols is installed.

    The primary pin (``diffusers==0.39.0``) matches the runtime lockfile. Fallbacks cover the
    unlikely case where the pinned release is unreachable but a compatible pre-release or main
    build exists.
    """
    setup_command = r".\RunMeFirst.bat" if sys.platform.startswith("win") else "./RunMeFirst.sh"
    attempts = [
        ("diffusers==0.39.0", ["-m", "pip", "install", "--upgrade", "diffusers==0.39.0"]),
        ("pre-release diffusers>=0.39.0", ["-m", "pip", "install", "--upgrade", "--pre", "diffusers>=0.39.0"]),
        (
            "diffusers main branch zip",
            [
                "-m",
                "pip",
                "install",
                "--upgrade",
                "https://github.com/huggingface/diffusers/archive/refs/heads/main.zip",
            ],
        ),
    ]
    for label, args in attempts:
        print(f"Installing {label}...")
        completed = subprocess.run([python_exe, *args], env=env, check=False)
        if completed.returncode != 0:
            print(f"Install attempt failed for {label}.", file=sys.stderr)
            continue
        if diffusers_symbols_available(python_exe):
            print(f"Using {label} (Z-Image + Krea2 symbols verified).")
            return
        print(
            f"Installed {label} but expected Z-Image / Krea2 symbols are still missing.",
            file=sys.stderr,
        )
    raise RuntimeError(
        "Unable to install a diffusers build exposing the required Z-Image + Krea2 symbols. "
        f"Check internet access and rerun {setup_command}."
    )


def ensure_virtualenv(python_exe: str, project_root: Path, *, env: dict[str, str], platform_name: str) -> Path:
    venv_python = venv_python_path(project_root, platform_name=platform_name)
    venv_root = venv_python.parent.parent
    if not venv_python.exists():
        run_checked([python_exe, "-m", "venv", str(venv_root)], cwd=project_root, env=env)
    if not module_imports(str(venv_python), "pip"):
        subprocess.run([str(venv_python), "-m", "ensurepip", "--upgrade"], env=env, check=False)
    if not module_imports(str(venv_python), "pip"):
        print(".venv is incomplete. Rebuilding virtual environment...")
        run_checked([python_exe, "-m", "venv", "--clear", str(venv_root)], cwd=project_root, env=env)
        subprocess.run([str(venv_python), "-m", "ensurepip", "--upgrade"], env=env, check=False)
    if not module_imports(str(venv_python), "pip"):
        raise RuntimeError("Failed to bootstrap pip inside .venv.")
    return venv_python


def install_environment(
    *,
    project_root: Path,
    python_exe: str,
    platform_name: str,
    lane: str,
) -> None:
    ensure_python_311_or_newer(python_exe)
    env = temp_env(project_root)
    venv_python = ensure_virtualenv(python_exe, project_root, env=env, platform_name=platform_name)

    torch_requirements = resolve_torch_requirements_path(
        project_root,
        platform_name=platform_name,
        lane=lane,
    )
    runtime_requirements = project_root / "requirements" / "runtime-lock.txt"
    seedvr2_requirements = project_root / "requirements" / "seedvr2-lock.txt"
    dev_requirements = project_root / "requirements" / "dev-lock.txt"
    for required_path in (
        torch_requirements,
        runtime_requirements,
        seedvr2_requirements,
        dev_requirements,
    ):
        if not required_path.exists():
            raise RuntimeError(f"Missing requirements file: {required_path}")

    tmp_root = project_root / ".build" / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    runtime_without_diffusers = tmp_root / "runtime-lock.no-diffusers.txt"
    runtime_lines = runtime_requirements.read_text(encoding="utf-8").splitlines()
    filtered_runtime = filtered_runtime_lock_lines(runtime_lines)
    runtime_without_diffusers.write_text(
        "\n".join(filtered_runtime) + ("\n" if filtered_runtime else ""),
        encoding="ascii",
    )

    run_checked([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], cwd=project_root, env=env)
    run_checked(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "setuptools", "wheel"],
        cwd=project_root,
        env=env,
    )
    run_checked(
        [str(venv_python), "-m", "pip", "install", "huggingface_hub[hf_xet]==0.35.0"],
        cwd=project_root,
        env=env,
    )
    run_checked([str(venv_python), "-m", "pip", "install", "-r", str(torch_requirements)], cwd=project_root, env=env)
    if filtered_runtime:
        run_checked(
            [str(venv_python), "-m", "pip", "install", "-r", str(runtime_without_diffusers)],
            cwd=project_root,
            env=env,
        )
    run_checked(
        [str(venv_python), "-m", "pip", "install", "-r", str(seedvr2_requirements)],
        cwd=project_root,
        env=env,
    )
    install_diffusers_with_fallback(str(venv_python), env=env)
    run_checked([str(venv_python), "-m", "pip", "install", "-r", str(dev_requirements)], cwd=project_root, env=env)
    run_checked(
        [str(venv_python), "-m", "pip", "install", "--no-build-isolation", "--no-deps", "-e", "."],
        cwd=project_root,
        env=env,
    )
    print(f"Environment ready. Use {venv_python} for commands. Torch requirements={torch_requirements.name}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or repair the JustRayzist source environment.")
    parser.add_argument("--project-root", default=str(REPO_ROOT))
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--lane", default="auto")
    parser.add_argument("--platform", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = resolve_project_root(args.project_root)
    normalized_platform = normalize_platform_name(args.platform or sys.platform)
    install_environment(
        project_root=project_root,
        python_exe=args.python_exe,
        platform_name=normalized_platform,
        lane=args.lane,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
