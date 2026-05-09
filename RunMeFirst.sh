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

"$PYTHON_BIN" "$ROOT_DIR/scripts/portable/bootstrap_env.py" \
  --project-root "$ROOT_DIR" \
  --python-exe "$PYTHON_BIN" \
  --lane auto \
  --platform "$(uname -s)"

VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Expected virtualenv python was not created: $VENV_PYTHON" >&2
  exit 1
fi

FETCH_MODEL_ARGS=(--project-root "$ROOT_DIR" --platform "$(uname -s)")
QWEN3_FP8_ENCODER_PATH="$ROOT_DIR/models/packs/Rayzist_qwen3_4b_fp8/config/text_encoder/model.safetensors"
QWEN3_FP8_PACK_MANIFEST_PATH="$ROOT_DIR/models/packs/Rayzist_qwen3_4b_fp8/modelpack.yaml"

case "${JUSTRAYZIST_INCLUDE_QWEN3_FP8_ENCODER:-}" in
  1|true|TRUE|yes|YES|y|Y)
    FETCH_MODEL_ARGS+=(--include-qwen3-4b-fp8-encoder)
    ;;
  *)
    if [[ -f "$QWEN3_FP8_ENCODER_PATH" && ! -f "$QWEN3_FP8_PACK_MANIFEST_PATH" ]]; then
      FETCH_MODEL_ARGS+=(--include-qwen3-4b-fp8-encoder)
    elif [[ ! -f "$QWEN3_FP8_ENCODER_PATH" && -t 0 ]]; then
      read -r -p "Download optional Qwen3 4B FP8 encoder pack, about 4.1 GB? [y/N] " answer
      case "$answer" in
        y|Y|yes|YES|Yes)
          FETCH_MODEL_ARGS+=(--include-qwen3-4b-fp8-encoder)
          ;;
      esac
    fi
    ;;
esac

"$VENV_PYTHON" "$ROOT_DIR/scripts/portable/fetch_model_assets.py" "${FETCH_MODEL_ARGS[@]}"
"$VENV_PYTHON" "$ROOT_DIR/scripts/portable/fetch_seedvr2_runtime.py" --project-root "$ROOT_DIR"
"$VENV_PYTHON" -m app.cli.main doctor
"$VENV_PYTHON" -m app.cli.main validate-models

case "$(uname -s)" in
  Darwin)
    echo ""
    echo "Setup completed. macOS support is source-mode and best-effort only; accelerated generation is not guaranteed."
    ;;
  *)
    echo ""
    echo "Setup completed. Launch the app with ./StartWeb.sh."
    ;;
esac
