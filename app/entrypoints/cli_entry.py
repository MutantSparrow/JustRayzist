from __future__ import annotations

import os
import sys
from pathlib import Path

from app.cli.main import run


def _ensure_root_for_frozen() -> None:
    if os.getenv("JUSTRAYZIST_ROOT"):
        return
    if getattr(sys, "frozen", False):
        os.environ["JUSTRAYZIST_ROOT"] = str(Path(sys.executable).resolve().parents[2])


def main() -> None:
    _ensure_root_for_frozen()
    run()


if __name__ == "__main__":
    main()

