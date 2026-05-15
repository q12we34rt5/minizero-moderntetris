#!/usr/bin/env bash
# Build the moderntetris engine into a WASM ES module.
#
# Prerequisite: emscripten SDK installed locally. By default this looks for
# ~/emsdk; override with EMSDK_DIR=/path/to/emsdk.
#
#   git clone https://github.com/emscripten-core/emsdk.git ~/emsdk
#   cd ~/emsdk && ./emsdk install latest && ./emsdk activate latest
#
# Run from web/engine/:  ./build.sh
# Output: web/src/engine/engine.js + engine.wasm
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$WEB_DIR")"
ENGINE_SRC="$REPO_DIR/minizero/environment/stochastic/moderntetris/engine"
OUT_DIR="$WEB_DIR/src/engine"

EMSDK_DIR="${EMSDK_DIR:-$HOME/emsdk}"

if [ -z "${EMSDK:-}" ]; then
    if [ -f "$EMSDK_DIR/emsdk_env.sh" ]; then
        # shellcheck disable=SC1091
        source "$EMSDK_DIR/emsdk_env.sh" >/dev/null 2>&1
    fi
fi

# emscripten needs Python >= 3.10. emsdk_env.sh exports EMSDK_PYTHON pointing at
# whatever python emsdk was activated with -- if that's too old, override it
# (after sourcing) with a locally-installed miniconda interpreter.
py_ok() { "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; }
if ! py_ok "${EMSDK_PYTHON:-python3}"; then
    for cand in "$HOME/miniconda3/bin/python3" "$HOME/anaconda3/bin/python3"; do
        if [ -x "$cand" ] && py_ok "$cand"; then export EMSDK_PYTHON="$cand"; break; fi
    done
fi
if ! py_ok "${EMSDK_PYTHON:-python3}"; then
    echo "error: need Python >= 3.10 for emscripten. Set EMSDK_PYTHON to a suitable interpreter." >&2
    exit 1
fi
if ! command -v em++ >/dev/null 2>&1; then
    echo "error: em++ not found. Install emsdk (see header of this script) or set EMSDK_DIR." >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

em++ \
    -O3 -std=c++17 \
    -I "$ENGINE_SRC" \
    "$SCRIPT_DIR/engine_wasm.cpp" \
    "$ENGINE_SRC/tetris.cpp" \
    "$ENGINE_SRC/placement_search.cpp" \
    -o "$OUT_DIR/engine-wasm.js" \
    -s MODULARIZE=1 \
    -s EXPORT_ES6=1 \
    -s EXPORT_NAME=createEngineModule \
    -s ENVIRONMENT=web,node \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s EXPORTED_RUNTIME_METHODS=ccall,cwrap,HEAP32 \
    -s 'EXPORTED_FUNCTIONS=_et_view_size,_et_create,_et_free,_et_set_config,_et_reset,_et_step,_et_add_garbage,_et_serialize,_et_serialize_full,_et_codec_size,_et_find_placements,_et_placement_path,_et_apply_placement,_malloc,_free'

echo "built: $OUT_DIR/engine-wasm.js + engine-wasm.wasm"
