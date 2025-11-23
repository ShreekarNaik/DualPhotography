#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLEND_FILE="${SCRIPT_DIR}/blender_experiment_scene.blend"
OUTPUT_IMAGE="${SCRIPT_DIR}/captured_image.png"
PATTERN_IMAGE="${SCRIPT_DIR}/projection_pattern.png"

if [[ ! -f "${BLEND_FILE}" ]]; then
  echo "Error: .blend file not found at: ${BLEND_FILE}" >&2
  exit 1
fi

# Render frame 1 to captured_image0001.png in the same directory
FRAME=1
OUTPUT_PREFIX="${SCRIPT_DIR}/captured_image"

blender -b "${BLEND_FILE}" -o "${OUTPUT_PREFIX}" -F PNG -f "${FRAME}"

# Blender appends the frame number with padding (e.g., captured_image0001.png)
RENDERED_FILE="${OUTPUT_PREFIX}$(printf '%04d' "${FRAME}").png"
FINAL_FILE="${SCRIPT_DIR}/captured_image.png"

if [[ -f "${RENDERED_FILE}" ]]; then
  mv -f "${RENDERED_FILE}" "${FINAL_FILE}"
  echo "Saved render to: ${FINAL_FILE}"
else
  echo "Error: Expected render not found at: ${RENDERED_FILE}" >&2
  exit 1
fi
