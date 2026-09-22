#!/bin/bash
# Clairvoyance-gap eval: play one model under a given arm (search sees the real
# future, a resampled future, policy only, ...). The real game always runs on
# the true env; the arm only changes what the search sees.
#
# Usage (inside the container, from the repo root):
#   experiments/clairvoyance-gap/run.sh <training_dir> <iter> <label> <gpu> <num_games> [extra_conf]
# <label> names the arm; extra_conf holds its settings, e.g.
#   actor_mcts_resample_hidden_future=true:actor_mcts_resample_hidden_future_parts=pieces
set -euo pipefail

training_dir="${1%/}"; iter=$2; label=$3; gpu=$4; num_games=$5; extra=${6:-}
name=$(basename "$training_dir")
out_dir=experiments/clairvoyance-gap/results
mkdir -p "$out_dir"
tag="${name}_iter${iter}_${label}"
out="$out_dir/$tag.txt"

conf="nn_file_name=$training_dir/model/weight_iter_${iter}.pt"
conf+=":program_auto_seed=true"
conf+=":zero_num_threads=8:zero_num_parallel_games=32"
[[ -n "$extra" ]] && conf+=":$extra"

echo "$conf" > "$out_dir/$tag.conf_str"
CUDA_VISIBLE_DEVICES=$gpu ./build/moderntetris_placement/minizero_moderntetris_placement \
    -mode sp -conf_file "$training_dir/$name.cfg" -conf_str "$conf" \
    2>"$out_dir/$tag.err" <<< "start" \
    | grep --line-buffered '^SelfPlay' | cut -d' ' -f1-5 | head -n "$num_games" > "$out" || true  # head exits after N games; SIGPIPE stops the binary
echo "done: $out ($(wc -l < "$out") games)"
