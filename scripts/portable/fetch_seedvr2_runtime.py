from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.portable.common import (  # noqa: E402
    resolve_project_root,
    run_checked,
    shell_setup_command,
)


REPO_URL = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"


def apply_allocator_compat_patch(runtime_script_path: Path) -> bool:
    original = runtime_script_path.read_text(encoding="utf-8")
    if "PYTORCH_CUDA_ALLOC_CONF" not in original:
        return False
    updated = original.replace("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")
    if updated == original:
        return False
    runtime_script_path.write_text(updated, encoding="utf-8")
    return True


def target_repo_path(project_root: Path) -> Path:
    return project_root / "models" / "seedvr2" / "runtime" / "ComfyUI-SeedVR2_VideoUpscaler"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch the bundled SeedVR2 runtime repository.")
    parser.add_argument("--project-root", default=str(REPO_ROOT))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = resolve_project_root(args.project_root)
    if not shutil.which("git"):
        raise RuntimeError(
            f"Git executable not found in PATH. Install Git and rerun {shell_setup_command()}."
        )

    repo_root = target_repo_path(project_root)
    repo_root.parent.mkdir(parents=True, exist_ok=True)

    git_dir = repo_root / ".git"
    if not git_dir.exists():
        if repo_root.exists():
            if not args.force:
                raise RuntimeError(
                    f"SeedVR2 runtime directory exists but is not a git repository: {repo_root}. "
                    "Delete it or rerun with --force."
                )
            shutil.rmtree(repo_root)
        print("[download] Cloning SeedVR2 runtime repository...")
        run_checked(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                args.revision,
                REPO_URL,
                str(repo_root),
            ]
        )
    else:
        print("[update] Refreshing SeedVR2 runtime repository...")
        run_checked(["git", "-C", str(repo_root), "fetch", "--depth", "1", "origin", args.revision])
        run_checked(["git", "-C", str(repo_root), "checkout", "--force", "FETCH_HEAD"])

    runtime_script = repo_root / "inference_cli.py"
    if not runtime_script.exists():
        raise RuntimeError(
            f"SeedVR2 runtime fetch completed but inference_cli.py is missing: {runtime_script}"
        )

    if apply_allocator_compat_patch(runtime_script):
        print("[patch] Applied allocator env compatibility patch (PYTORCH_ALLOC_CONF).")
    else:
        print("[patch] Runtime allocator env var already compatible.")

    print("[ok] SeedVR2 runtime ready:")
    print(f"  {runtime_script}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
