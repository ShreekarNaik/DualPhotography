#!/bin/bash

# Central runner script for Adaptive Dual Photography Project

set -e  # Exit on error

# Function to check if uv is installed
check_uv() {
    if ! command -v uv &> /dev/null; then
        echo "Error: uv is not installed. Please install it first."
        exit 1
    fi
}

# Function to setup the project
setup() {
    echo "Setting up project with uv..."
    check_uv
    uv sync
    echo "Setup complete."
}

# Function to run MVP tests
test_mvp() {
    echo "Running MVP synthetic tests..."
    uv run python src/main_mvp.py "$@"
}

# Function to run full tests
test() {
    echo "Running tests..."
    uv run pytest src/tests/
}

# Function to run Blender integration
run_blender() {
    echo "Running Blender integration..."
    # Note: Blender has its own python, so we might need to handle dependencies differently
    # or use a bridge. For now, assuming we call blender with a script.
    # If using system python with blender module (bpy) installed (rare), we can use uv run.
    # Usually we run: blender --background --python src/blender_script.py
    
    echo "Please ensure Blender is in your PATH."
    blender --background --python src/environment/blender_env.py -- "$@"
}

# Function to visualize results
visualize() {
    echo "Generating visualizations..."
    uv run python src/utils/visualization.py "$@"
}

# Function to run examples
examples() {
    echo "Running example scripts..."
    uv run python examples.py "$@"
}

# Main command dispatcher
case "$1" in
    setup)
        setup
        ;;
    test-mvp)
        shift
        test_mvp "$@"
        ;;
    test)
        shift
        test "$@"
        ;;
    run-blender)
        shift
        run_blender "$@"
        ;;
    visualize)
        shift
        visualize "$@"
        ;;
    examples)
        shift
        examples "$@"
        ;;
    *)
        echo "Usage: $0 {setup|test-mvp|test|run-blender|visualize|examples} [args]"
        exit 1
        ;;
esac
