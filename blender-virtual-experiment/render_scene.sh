#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLEND_FILE="${SCRIPT_DIR}/blender_experiment_scene.blend"
OUTPUT_IMAGE="${SCRIPT_DIR}/captured_image.png"
PATTERN_IMAGE="${SCRIPT_DIR}/projection_pattern.png"
LOG_FILE="${SCRIPT_DIR}/blender_render.log"

cleanup() {
  if [[ -n "${SLIDER_PID:-}" ]]; then
    if kill -0 "${SLIDER_PID}" 2>/dev/null; then
      kill "${SLIDER_PID}" 2>/dev/null || true
    fi
    wait "${SLIDER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

show_slider() {
  local watch_pid=$1
  local width=24
  local position=0
  local direction=1

  while kill -0 "${watch_pid}" 2>/dev/null; do
    local bar="["
    for ((i = 0; i < width; i++)); do
      if (( i == position )); then
        bar+="="
      else
        bar+=" "
      fi
    done
    bar+="]"

    printf "\rRendering %s" "${bar}"

    sleep 0.1
    ((position += direction))
    if (( position == width - 1 || position == 0 )); then
      direction=$(( -direction ))
    fi
  done

  printf "\rRender Successfull! .%-40s\n" ""
}

if [[ ! -f "${BLEND_FILE}" ]]; then
  echo "Error: .blend file not found at: ${BLEND_FILE}" >&2
  exit 1
fi

# Render frame 1 to captured_image0001.png in the same directory
FRAME=1
OUTPUT_PREFIX="${SCRIPT_DIR}/captured_image"

# echo "Rendering frame ${FRAME} (full log: ${LOG_FILE})"
echo "Rendering Experiment"
blender -b "${BLEND_FILE}" -o "${OUTPUT_PREFIX}" -F PNG -f "${FRAME}" >"${LOG_FILE}" 2>&1 &
BLENDER_PID=$!
show_slider "${BLENDER_PID}" &
SLIDER_PID=$!

if ! wait "${BLENDER_PID}"; then
  echo "Error: Blender render failed. See ${LOG_FILE} for details." >&2
  exit 1
fi
wait "${SLIDER_PID}" 2>/dev/null || true

# Blender appends the frame number with padding (e.g., captured_image0001.png)
RENDERED_FILE="${OUTPUT_PREFIX}$(printf '%04d' "${FRAME}").png"
FINAL_FILE="${SCRIPT_DIR}/captured_image.png"

if [[ -f "${RENDERED_FILE}" ]]; then
  mv -f "${RENDERED_FILE}" "${FINAL_FILE}"
  # echo "Saved render to: ${FINAL_FILE}"
  # echo "Detailed Blender output saved to: ${LOG_FILE}"
else
  echo "Error: Expected render not found at: ${RENDERED_FILE}" >&2
  exit 1
fi
