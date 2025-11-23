#!/bin/bash

# Central management script for the Adaptive CS Project

COMMAND=$1

if [ -z "$COMMAND" ]; then
    echo "Usage: ./manage.sh [command]"
    echo "Commands:"
    echo "  install   - Install dependencies using uv"
    echo "  sim       - Run the Adaptive CS simulation"
    echo "  clean     - Remove temporary files and caches"
    exit 1
fi

case $COMMAND in
    install)
        echo "Installing dependencies..."
        uv sync
        ;;
    sim)
        echo "Running Adaptive CS Simulation..."
        uv run src/adaptive_cs.py
        ;;
    benchmark)
        echo "Running Full Benchmark..."
        uv run src/benchmark.py
        ;;
    clean)
        echo "Cleaning up..."
        rm -rf __pycache__
        rm -rf src/__pycache__
        echo "Done."
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Usage: ./manage.sh [install|sim|clean]"
        exit 1
        ;;
esac
