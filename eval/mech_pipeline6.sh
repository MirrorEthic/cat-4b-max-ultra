#!/usr/bin/env bash
# Sixth block, behind block 5: the clean concept direction at a single late layer (L22, L26) at α8 pushes cats into
# half of the replies but wrecks generation (73-79% capped, loops, no tool calls). Try the same late-layer push under
# the decaying dose, with the matched random and dog directions on the same schedule.
#
#   setsid nohup bash eval/mech_pipeline6.sh <wait_pid> >> state/eval/mech/pipeline.log 2>&1 &
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
for l in 22 26; do for a in 4 8; do for t in 4 8; do
  sweep mech-c-S2-l$l-a$a-t$t --cond S --model base --variant stock_only --mode add --direction D_text_dn --layer $l --alpha $a --tau $t
done; done; done
sweep mech-c-R0-l26-a8-t8 --cond R --model base --variant stock_only --mode add --direction R_random0 --layer 26 --alpha 8 --tau 8
sweep mech-c-Rdog-l26-a8-t8 --cond R --model base --variant stock_only --mode add --direction R_dog --layer 26 --alpha 8 --tau 8
sweep mech-c-S2-l26-a16-t4 --cond S --model base --variant stock_only --mode add --direction D_text_dn --layer 26 --alpha 16 --tau 4
CUDA_VISIBLE_DEVICES="" $V eval/mech_summary.py --prefix mech-c- --md > "$MECH/sweep_clean_summary.md" 2>/dev/null
step "clean sweep block 6 done"
