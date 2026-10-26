#!/usr/bin/env bash
# Is D_ft special because it carries the cat, or because it overlaps a large, generally important component of the
# residual stream? (Garret, 2026-10-26 12:29.) Two directions from eval/mech_mean_direction.py: D_mean (the tuned
# model's mean residual per layer, same extraction split) and D_ft_orth (D_ft with its D_mean component removed).
# Whole-model ablations of both, the 8-20 band of the orthogonalised direction, and the all-layer add of the
# orthogonalised delta at α1; the same judge and projection_stats as every other run. Waits for the given pid
# (the band-ablation chain) to exit first.
#
#   setsid nohup bash eval/mech_pipeline10.sh <wait_pid> >> state/eval/mech/pipeline.log 2>&1 &
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
export HF_HOME=/media/garret-sutherland/dev-data/huggingface-cache HF_HUB_OFFLINE=1
V=/media/garret-sutherland/dev-data/cat4b/venvs/train-cu130/bin/python
LOCK=~/CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED
MECH=/media/garret-sutherland/dev-data/cat4b/state/eval/mech
cd "$REPO"
while [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; do sleep 60; done
step() { echo "== $(date +%H:%M:%S) $*"; [ -e "$LOCK" ] && { echo "GPU lock present; stopping"; exit 3; }; return 0; }
run() { "$@" 2>&1 | grep -v "Warning\|Loading weights\|warnings.warn"; return "${PIPESTATUS[0]}"; }
FULL="--max-new 1024 --batch 16"
full() { local run=$1; shift; step "$run"; run $V eval/steer.py --run "$run" $FULL "$@" || echo "FAILED $run"; }
step "mean-residual controls start"
full mech-f-Xmean        --cond X --model tuned --variant stock_only --mode ablate  --direction D_mean
full mech-f-X1orth       --cond X --model tuned --variant stock_only --mode ablate  --direction D_ft_orth
full mech-f-X1orth-L8-20 --cond X --model tuned --variant stock_only --mode ablate  --direction D_ft_orth --layers 8-20
full mech-f-S1orthall-a1 --cond S --model base  --variant stock_only --mode add_all --direction D_ft_orth --alpha 1
CUDA_VISIBLE_DEVICES="" $V eval/mech_summary.py --prefix mech-f- --md > "$MECH/full_summary.md" 2>/dev/null
step "mean-residual controls done"
