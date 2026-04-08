#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

choose_python() {
  if [[ -n "${JUSTRAYZIST_PYTHON:-}" ]]; then
    printf '%s\n' "$JUSTRAYZIST_PYTHON"
    return 0
  fi
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    printf '%s\n' "$ROOT_DIR/.venv/bin/python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  return 1
}

PYTHON_BIN="$(choose_python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Python 3.11+ is required. Install python3 and rerun ./RunMeFirst.sh." >&2
  exit 1
fi

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/portable/start_web.py" \
  --project-root "$ROOT_DIR" \
  --python-exe "$PYTHON_BIN" \
  "$@"
