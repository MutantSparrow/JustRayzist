from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from app.cli.main import serve


def _ensure_root_for_frozen() -> None:
    if os.getenv("JUSTRAYZIST_ROOT"):
        return
    if getattr(sys, "frozen", False):
        os.environ["JUSTRAYZIST_ROOT"] = str(Path(sys.executable).resolve().parents[2])


def main() -> None:
    parser = argparse.ArgumentParser(prog="justrayzist-web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=37717)
    parser.add_argument("--profile", default=None)
    args = parser.parse_args()
    _ensure_root_for_frozen()
    serve(host=args.host, port=args.port, profile=args.profile)


if __name__ == "__main__":
    main()

