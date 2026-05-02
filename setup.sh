#!/bin/bash
set -e

PYTHON_VERSION="3.11"

# Create .venv and install dependencies
if command -v uv &> /dev/null; then
  echo "Using uv..."
  uv venv --python "$PYTHON_VERSION" .venv
  source .venv/bin/activate
  uv pip install -e ".[dev]"
else
  echo "Using system python venv..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -e ".[dev]"
fi

# Install Node deps
echo "Installing root Node dependencies..."
npm ci

echo "Installing frontend Node dependencies..."
cd web && npm ci && cd ..

echo ""
echo "Setup complete."
echo "To start: npm run dev"
