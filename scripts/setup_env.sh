#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: Python interpreter '$PYTHON_BIN' not found in PATH." >&2
  exit 1
fi

PYTHON_MAJOR_VERSION=$("$PYTHON_BIN" -c "import sys; print(sys.version_info.major)") || true
PYTHON_MINOR_VERSION=$("$PYTHON_BIN" -c "import sys; print(sys.version_info.minor)") || true
REQUIRED_MAJOR=3
REQUIRED_MINOR=9
if [ "$PYTHON_MAJOR_VERSION" -lt "$REQUIRED_MAJOR" ] || { [ "$PYTHON_MAJOR_VERSION" -eq "$REQUIRED_MAJOR" ] && [ "$PYTHON_MINOR_VERSION" -lt "$REQUIRED_MINOR" ]; }; then
  echo "Error: Python >= 3.9 is required. Detected: ${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION}" >&2
  exit 1
fi

echo "Creating virtual environment at '$VENV_DIR'..."
"$PYTHON_BIN" -m venv "$VENV_DIR"

echo "Installing build tools and dependencies..."
PIP="$VENV_DIR/bin/pip"
python_exe="$VENV_DIR/bin/python"

set +e
"$python_exe" -m ensurepip >/dev/null 2>&1
set -e

"$PIP" install --upgrade pip setuptools wheel
"$PIP" install -e .

echo "Setup complete."
echo "Activate the environment with: source ${VENV_DIR}/bin/activate"
