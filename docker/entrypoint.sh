#!/usr/bin/env bash
set -euo pipefail

cd /app

MODE="${1:-pipeline}"

VENV_DIR="/app/.venv"
PY_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"

ensure_venv() {
  if [[ ! -x "$PY_BIN" ]]; then
    echo "[WARN] Virtualenv missing or broken. Creating fresh venv in $VENV_DIR ..."

    # IMPORTANT: /app/.venv is a volume mount; can't remove mountpoint directory.
    # Delete contents only.
    if [[ -d "$VENV_DIR" ]]; then
      echo "[INFO] Cleaning venv directory contents..."
      find "$VENV_DIR" -mindepth 1 -exec rm -rf {} +
    fi

    python -m venv "$VENV_DIR"
    "$PY_BIN" -m pip install --upgrade pip
    "$PIP_BIN" install --no-cache-dir -r requirements.txt
    return
  fi

  echo "[INFO] Reusing existing virtualenv in .venv"
}

ensure_venv

if [[ "$MODE" == "pipeline" ]]; then
  echo "[INFO] Running end-to-end pipeline..."
  "$PY_BIN" -m src.run_pipeline
elif [[ "$MODE" == "ui" ]]; then
  echo "[INFO] Running pipeline + Streamlit UI..."
  export RUN_STREAMLIT="${RUN_STREAMLIT:-1}"
  "$PY_BIN" -m src.run_pipeline
else
  echo "Usage:"
  echo "  pipeline  -> run end-to-end pipeline"
  echo "  ui        -> run pipeline + Streamlit UI"
  exit 1
fi

echo "[INFO] All done."


