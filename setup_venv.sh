#!/usr/bin/env bash
set -euo pipefail

# Creates a local venv in .venv and installs requirements.txt

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: $PYTHON_BIN not found. Install Python 3 first."
  exit 1
fi

echo "Creating virtualenv in .venv using $PYTHON_BIN"
$PYTHON_BIN -m venv .venv

# Activate
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
echo "Installing Python requirements from requirements.txt..."
pip install -r requirements.txt

echo "Installing and registering IPython kernel for this venv..."
python -m pip install --upgrade ipykernel || true
# Register a kernel named after the project (user-level). This lets Jupyter pick the venv kernel.
KERNEL_NAME="screendoor_effect"
python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_NAME (venv)" || true

echo ""
echo "Venv ready." 
echo "Activate it with:"
echo "  source .venv/bin/activate"
echo "Start the notebook server and open `view_all_plots.ipynb`:"
echo "  jupyter notebook view_all_plots.ipynb"
