from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.api.api_manifest import render_examples_markdown, render_route_summary_markdown  # noqa: E402

README_PATH = ROOT / "README.md"
USAGE_PATH = ROOT / "docs" / "USAGE.md"

README_ROUTES_BEGIN = "<!-- BEGIN GENERATED API ROUTES -->"
README_ROUTES_END = "<!-- END GENERATED API ROUTES -->"
README_EXAMPLES_BEGIN = "<!-- BEGIN GENERATED API EXAMPLES -->"
README_EXAMPLES_END = "<!-- END GENERATED API EXAMPLES -->"
USAGE_ROUTES_BEGIN = "<!-- BEGIN GENERATED API ROUTES -->"
USAGE_ROUTES_END = "<!-- END GENERATED API ROUTES -->"
USAGE_EXAMPLES_BEGIN = "<!-- BEGIN GENERATED API EXAMPLES -->"
USAGE_EXAMPLES_END = "<!-- END GENERATED API EXAMPLES -->"


def _replace_block(content: str, begin: str, end: str, replacement: str) -> str:
    start = content.find(begin)
    finish = content.find(end)
    if start < 0 or finish < 0 or finish < start:
        raise ValueError(f"Missing markers: {begin} / {end}")
    start += len(begin)
    body = f"\n{replacement}\n"
    return content[:start] + body + content[finish:]


def render_readme(content: str) -> str:
    updated = _replace_block(
        content,
        README_ROUTES_BEGIN,
        README_ROUTES_END,
        render_route_summary_markdown(include_usage_only=False),
    )
    return _replace_block(
        updated,
        README_EXAMPLES_BEGIN,
        README_EXAMPLES_END,
        render_examples_markdown(include_usage_only=False),
    )


def render_usage(content: str) -> str:
    updated = _replace_block(
        content,
        USAGE_ROUTES_BEGIN,
        USAGE_ROUTES_END,
        render_route_summary_markdown(include_usage_only=True),
    )
    return _replace_block(
        updated,
        USAGE_EXAMPLES_BEGIN,
        USAGE_EXAMPLES_END,
        render_examples_markdown(include_usage_only=True),
    )


def main() -> None:
    README_PATH.write_text(render_readme(README_PATH.read_text(encoding="utf-8")), encoding="utf-8")
    USAGE_PATH.write_text(render_usage(USAGE_PATH.read_text(encoding="utf-8")), encoding="utf-8")


if __name__ == "__main__":
    main()
