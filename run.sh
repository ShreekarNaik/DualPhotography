#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<EOF
Usage: $(basename "$0") <mode> [options...]

Modes:
  basis     - Gray-code logarithmic basis scan (Algorithm 2)
  adaptive  - Adaptive hierarchical feedback scan (Algorithm 3)

Common options:
  --run-name NAME           Explicit run folder name under runs/
  --blender-dir PATH        Path to Blender experiment dir (default: blender-virtual-experiment)
  --runs-root PATH          Root directory for runs (default: runs)

Additional options are forwarded to the underlying Python module.
For example:
  $(basename "$0") basis --max-planes 8 --no-include-inverse
  $(basename "$0") adaptive --noise-threshold 1e6 --max-depth 6 --max-measurements 64
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

MODE="$1"
shift || true

if [[ "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  usage
  exit 0
fi

RUN_NAME=""
BLENDER_DIR="${SCRIPT_DIR}/blender-virtual-experiment"
RUNS_ROOT="${SCRIPT_DIR}/runs"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --blender-dir)
      BLENDER_DIR="$2"
      shift 2
      ;;
    --runs-root)
      RUNS_ROOT="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${RUN_NAME}" ]]; then
  TS="$(date +"%Y%m%d_%H%M%S")"
  RUN_NAME="${MODE}_${TS}"
fi

export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"

case "$MODE" in
  basis)
    python3 -m decomposition_basis_scan \
      --blender-dir "${BLENDER_DIR}" \
      --runs-root "${RUNS_ROOT}" \
      --run-name "${RUN_NAME}" \
      "${EXTRA_ARGS[@]}"
    ;;
  adaptive)
    python3 -m adaptive_scan \
      --blender-dir "${BLENDER_DIR}" \
      --runs-root "${RUNS_ROOT}" \
      --run-name "${RUN_NAME}" \
      "${EXTRA_ARGS[@]}"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    usage
    exit 1
    ;;
esac
