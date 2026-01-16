#!/bin/bash
# SAIL Development Environment Setup Script
# This script sets up the local development environment

set -e

echo "=== SAIL Development Setup ==="
echo

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
REQUIRED_VERSION="3.11"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "ERROR: Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)"
    exit 1
fi
echo "✓ Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

# Activate virtual environment
source .venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q

# Install package in editable mode with dev dependencies
echo "Installing SAIL with development dependencies..."
pip install -e ".[dev]" -q
echo "✓ Dependencies installed"

# Install pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install
echo "✓ Pre-commit hooks installed"

# Create necessary directories
echo "Creating data directories..."
mkdir -p data/context data/knowledge_base models
echo "✓ Data directories created"

# Create default config if it doesn't exist
if [ ! -f "config/config.yaml" ]; then
    echo "Creating default configuration..."
    mkdir -p config
    python -c "from src.config import create_default_config; create_default_config('config/config.yaml')"
    echo "✓ Default configuration created"
else
    echo "✓ Configuration file exists"
fi

echo
echo "=== Setup Complete ==="
echo
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo
echo "To start SAIL, run:"
echo "  sail run"
echo
echo "To run tests, run:"
echo "  pytest"
echo
