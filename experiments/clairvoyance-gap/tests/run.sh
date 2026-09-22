#!/bin/bash
# Build and run the env-level checks against the moderntetris_placement build.
# Run inside the container from the repo root, after scripts/build.sh:
#   experiments/clairvoyance-gap/tests/run.sh
#
#   test_resample   resampleHiddenFuture() keeps current/hold/preview and each
#                   7-bag's contents, keeps the legal actions, changes the future
set -euo pipefail

build=build/moderntetris_placement
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
includes=$(find minizero -type d | sed 's/^/-I/' | tr '\n' ' ')
status=0
for t in test_resample; do
    g++ -std=c++17 -O2 -DMODERNTETRIS_PLACEMENT=1 $includes "experiments/clairvoyance-gap/tests/$t.cpp" \
        "$build/libenvironment.a" "$build/libconfig.a" "$build/libutils.a" \
        -lboost_system -lboost_thread -lboost_iostreams -lz -lpthread -o "$out/$t"
    echo "== $t"
    "$out/$t" || status=1
done
exit $status
