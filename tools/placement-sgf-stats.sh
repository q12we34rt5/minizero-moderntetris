#!/bin/bash
# Build (if needed) and run tools/placement_sgf_stats.cpp, which replays
# moderntetris_placement self-play records and reports clear types and where the
# attack reward came from.
#
# Usage (inside the container, from the repo root), after scripts/build.sh:
#   tools/placement-sgf-stats.sh <training_dir> <sgf_iteration>...
#   tools/placement-sgf-stats.sh --conf <cfg> <sgf_file>...
set -euo pipefail

build=build/moderntetris_placement
bin=$build/placement_sgf_stats
src=tools/placement_sgf_stats.cpp
if [[ ! -x $bin || $src -nt $bin ]]; then
    includes=$(find minizero -type d | sed 's/^/-I/' | tr '\n' ' ')
    g++ -std=c++17 -O2 -DMODERNTETRIS_PLACEMENT=1 $includes "$src" \
        "$build/libenvironment.a" "$build/libconfig.a" "$build/libutils.a" \
        -lboost_system -lboost_thread -lboost_iostreams -lz -lpthread -o "$bin"
fi

if [[ ${1:-} == "--conf" ]]; then
    shift
    conf=$1; shift
    "$bin" "$conf" "$@"
else
    dir=${1%/}; shift
    conf="$dir/$(basename "$dir").cfg"
    files=()
    for it in "$@"; do files+=("$dir/sgf/$it.sgf"); done
    "$bin" "$conf" "${files[@]}"
fi
