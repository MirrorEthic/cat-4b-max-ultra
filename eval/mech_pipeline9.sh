#!/usr/bin/env bash
# Where the whole-model ablation's damage comes from (full 490, 2026-10-26 12:10: projecting D_ft out of every layer
# took the tuned model from 80% to 22% cat but judged retention from 3.46 to 1.44, while the random all-layer
# ablation changed neither; the 8-20 band on the subset cost nothing and removed little). Bands around the band:
# early (0-7), late (21-31), and every layer without the embedding hook, each with its matched random control.
# Waits for the given pid (the tasks chain) to exit first.
#
#   setsid nohup bash eval/mech_pipeline9.sh <wait_pid> >> state/eval/mech/pipeline.log 2>&1 &
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
step "band ablations start"
full mech-f-X1-noemb  --cond X --model tuned --variant stock_only --mode ablate --direction D_ft      --layers 0-31
full mech-f-X1-L0-7   --cond X --model tuned --variant stock_only --mode ablate --direction D_ft      --layers 0-7
full mech-f-XR-L0-7   --cond R --model tuned --variant stock_only --mode ablate --direction R_random0 --layers 0-7
full mech-f-X1-L21-31 --cond X --model tuned --variant stock_only --mode ablate --direction D_ft      --layers 21-31
full mech-f-XR-L21-31 --cond R --model tuned --variant stock_only --mode ablate --direction R_random0 --layers 21-31
CUDA_VISIBLE_DEVICES="" $V eval/mech_summary.py --prefix mech-f- --md > "$MECH/full_summary.md" 2>/dev/null
step "band ablations done"
