#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"

# First argument: optional flag --with-ui
WITH_UI="${1:-}"

# ---- 1. Find a system Python (for first-time venv creation) ----
find_system_python() {
  for cmd in python python3 py; do
    if command -v "$cmd" >/dev/null 2>&1; then
      echo "$cmd"
      return 0
    fi
  done

  echo "[ERROR] Could not find a system Python (tried: python, python3, py)." >&2
  echo "[HINT] Install Python 3 and add it to PATH, then re-run this script." >&2
  exit 1
}

# ---- 2. Get the Python inside the virtualenv ----
get_venv_python() {
  if [ -x "$VENV_DIR/Scripts/python.exe" ]; then
    # Windows venv (Git Bash / PowerShell calling bash)
    echo "$VENV_DIR/Scripts/python.exe"
  elif [ -x "$VENV_DIR/bin/python" ]; then
    # Linux / macOS / WSL
    echo "$VENV_DIR/bin/python"
  elif [ -x "$VENV_DIR/bin/python3" ]; then
    echo "$VENV_DIR/bin/python3"
  else
    echo "[ERROR] Could not find Python inside virtualenv '$VENV_DIR'." >&2
    echo "[HINT] Try deleting '$VENV_DIR' and re-running this script." >&2
    exit 1
  fi
}

# ---- 3. Ensure virtualenv exists and deps are installed ----
ensure_venv() {
  if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating virtual environment in $VENV_DIR ..."
    local SYS_PY
    SYS_PY=$(find_system_python)

    "$SYS_PY" -m venv "$VENV_DIR"

    local VENV_PY
    VENV_PY=$(get_venv_python)

    echo "[INFO] Installing dependencies (first run)..."
    "$VENV_PY" -m pip install --upgrade pip
    "$VENV_PY" -m pip install -r requirements.txt
  else
    echo "[INFO] Reusing existing virtualenv in $VENV_DIR"
  fi
}

main() {
  ensure_venv
  local PY
  PY=$(get_venv_python)

  echo "[INFO] Running end-to-end pipeline..."
  "$PY" -m src.run_pipeline

  if [ "$WITH_UI" = "--with-ui" ]; then
    echo "[INFO] Launching Streamlit UI (app_medium.py)..."
    echo "[INFO] Close the Streamlit app (Ctrl+C) when finished."
    "$PY" -m streamlit run src/ui_prototypes/app_medium.py
  fi

  echo "[INFO] All done."
}

main "$@"
