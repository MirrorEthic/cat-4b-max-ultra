#!/usr/bin/env bash
# Second sweep, queued behind the first (and its rerun): what the first one taught. Adding the fine-tune direction
# at one layer did nothing at any strength (S1, 24 settings at the base rate), while projecting it out of every
# layer removed most of the behaviour but also damaged generation (length-capped 61%, tool calls 5%). So: add the
# per-layer deltas at every layer at once (the shape of what the LoRA does), and ablate at single layers or a band
# so the damage and the effect can be separated.
#
#   setsid nohup bash eval/mech_pipeline2.sh <wait_pid> >> state/eval/mech/pipeline.log 2>&1 &
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
export HF_HOME=/media/garret-sutherland/dev-data/huggingface-cache HF_HUB_OFFLINE=1
V=/media/garret-sutherland/dev-data/cat4b/venvs/train-cu130/bin/python
LOCK=~/CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED
cd "$REPO"
while [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; do sleep 60; done
step() { echo "== $(date +%H:%M:%S) $*"; [ -e "$LOCK" ] && { echo "GPU lock present; stopping"; exit 3; }; return 0; }
run() { "$@" 2>&1 | grep -v "Warning\|Loading weights\|warnings.warn"; return "${PIPESTATUS[0]}"; }
SUB="--subset 100 --max-new 512 --batch 16"
sweep() { local run=$1; shift; step "$run"; run $V eval/steer.py --run "$run" $SUB "$@" || echo "FAILED $run"; }
for a in 0.5 1 2 4; do
  sweep mech-sub-S1all-a$a --cond S --model base --variant stock_only --mode add_all --direction D_ft --alpha $a
done
for a in 1 2 4; do
  sweep mech-sub-S2all-a$a --cond S --model base --variant stock_only --mode add_all --direction D_text_dn --alpha $a
done
for l in 6 10 14 18 22 26; do
  sweep mech-sub-X1-L$l --cond X --model tuned --variant stock_only --mode ablate --direction D_ft --layers $l
done
sweep mech-sub-X1-L8-20 --cond X --model tuned --variant stock_only --mode ablate --direction D_ft --layers 8-20
sweep mech-sub-X2-L8-20 --cond X --model tuned --variant stock_only --mode ablate --direction D_text_dn --layers 8-20
sweep mech-sub-XR-L8-20 --cond R --model tuned --variant stock_only --mode ablate --direction R_random0 --layers 8-20
for a in 1 2; do
  sweep mech-sub-R0all-a$a --cond R --model base --variant stock_only --mode add_all --direction R_random0 --alpha $a
done
step "sweep 2 done"
