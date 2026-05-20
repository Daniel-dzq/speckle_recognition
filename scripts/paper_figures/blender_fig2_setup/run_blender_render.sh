#!/usr/bin/env bash
# Run from repo root. Uses macOS Blender.app if blender not on PATH.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
BLENDER="${BLENDER:-}"
if ! command -v blender &>/dev/null; then
  if [[ -x /Applications/Blender.app/Contents/MacOS/Blender ]]; then
    BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
  fi
fi
if [[ -z "${BLENDER}" ]]; then
  echo "Set BLENDER to blender executable or install Blender." >&2
  exit 1
fi
exec "$BLENDER" --background --python "$ROOT/scripts/paper_figures/blender_fig2_setup/render_fig2_scene.py" -- "$ROOT"
