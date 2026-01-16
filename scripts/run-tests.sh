#!/bin/bash
# SAIL Test Runner Script
# Run tests with coverage reporting

set -e

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "=== Running SAIL Tests ==="
echo

# Parse arguments
COVERAGE=false
VERBOSE=false
UNIT_ONLY=false
INTEGRATION_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --coverage|-c)
            COVERAGE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --unit|-u)
            UNIT_ONLY=true
            shift
            ;;
        --integration|-i)
            INTEGRATION_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--coverage|-c] [--verbose|-v] [--unit|-u] [--integration|-i]"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_ARGS=()

if [ "$VERBOSE" = true ]; then
    PYTEST_ARGS+=("-v")
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_ARGS+=("--cov=src" "--cov-report=term-missing" "--cov-report=html")
fi

if [ "$UNIT_ONLY" = true ]; then
    PYTEST_ARGS+=("tests/unit")
elif [ "$INTEGRATION_ONLY" = true ]; then
    PYTEST_ARGS+=("tests/integration")
else
    PYTEST_ARGS+=("tests/")
fi

# Run tests
echo "Running: pytest ${PYTEST_ARGS[*]}"
echo
pytest "${PYTEST_ARGS[@]}"

if [ "$COVERAGE" = true ]; then
    echo
    echo "Coverage report generated in htmlcov/index.html"
fi
