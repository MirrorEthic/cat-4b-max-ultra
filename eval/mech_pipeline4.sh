#!/usr/bin/env bash
# Fourth block, queued behind the clean redo: the all-layer add is the first intervention that moved the base
# (pre-fix directions: D_ft at every layer, α1 → 70% cat with coherent text and no tool calls, α2 → word salad;
# D_text_dn at every layer broke generation at α1). Map the window on the clean directions and run the matched
# all-layer controls: random vectors at D_ft's per-layer norms, and the dog direction, the same way.
#
#   setsid nohup bash eval/mech_pipeline4.sh <wait_pid> >> state/eval/mech/pipeline.log 2>&1 &
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
SUB="--subset 100 --max-new 512 --batch 16"
sweep() { local run=$1; shift; step "$run"; run $V eval/steer.py --run "$run" $SUB "$@" || echo "FAILED $run"; }
for a in 0.5 0.75 1.5; do
  sweep mech-c-S1all-a$a --cond S --model base --variant stock_only --mode add_all --direction D_ft --alpha $a
done
for a in 0.25 0.5 1; do
  sweep mech-c-S2all-a$a --cond S --model base --variant stock_only --mode add_all --direction D_text_dn --alpha $a
done
for a in 0.5 1 2; do
  sweep mech-c-R0all-a$a --cond R --model base --variant stock_only --mode add_all --direction R_random0 --alpha $a
  sweep mech-c-R1all-a$a --cond R --model base --variant stock_only --mode add_all --direction R_random1 --alpha $a
done
for a in 0.5 1 2; do
  sweep mech-c-Rdogall-a$a --cond R --model base --variant stock_only --mode add_all --direction R_dog --alpha $a
done
sweep mech-c-S1Aall-a1 --cond S --model base --variant stock_only --mode add_all --direction D_ft_A --alpha 1
CUDA_VISIBLE_DEVICES="" $V eval/mech_summary.py --prefix mech-c- --md > "$MECH/sweep_clean_summary.md" 2>/dev/null
step "clean sweep block 4 done"
