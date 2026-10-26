#!/usr/bin/env bash
# Fifth block, queued behind block 4: the decaying dose (Garret, 2026-10-26 00:11: the decay curve on dose magnitude
# from the cl33-opLM steering work fixed degeneration there; runs_eval/rt2b_l9_steering_20261003/PREREG.md,
# rt2c_injection_entrance_20261003/RESULT.md). Constant all-layer adds of D_ft gave coherent cats at α1 with tool
# use gone and word salad at α2; here the dose starts at the reply and decays over τ generated tokens, with a
# reply-only constant dose (τ huge) to separate "steer the prompt" from "steer the reply", and the matched random
# control at the same schedule.
#
#   setsid nohup bash eval/mech_pipeline5.sh <wait_pid> >> state/eval/mech/pipeline.log 2>&1 &
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
for t in 4 16; do for a in 1 2 4; do
  sweep mech-c-S1all-a$a-t$t --cond S --model base --variant stock_only --mode add_all --direction D_ft --alpha $a --tau $t
done; done
sweep mech-c-S1all-a1-reply --cond S --model base --variant stock_only --mode add_all --direction D_ft --alpha 1 --tau 1000000
sweep mech-c-S1all-a2-reply --cond S --model base --variant stock_only --mode add_all --direction D_ft --alpha 2 --tau 1000000
sweep mech-c-R0all-a2-t16 --cond R --model base --variant stock_only --mode add_all --direction R_random0 --alpha 2 --tau 16
sweep mech-c-R0all-a4-t16 --cond R --model base --variant stock_only --mode add_all --direction R_random0 --alpha 4 --tau 16
sweep mech-c-S2all-a1-t4 --cond S --model base --variant stock_only --mode add_all --direction D_text_dn --alpha 1 --tau 4
sweep mech-c-S2all-a2-t4 --cond S --model base --variant stock_only --mode add_all --direction D_text_dn --alpha 2 --tau 4
for l in 14 18 22; do
  sweep mech-c-S1-l$l-a8-t8 --cond S --model base --variant stock_only --mode add --direction D_ft --layer $l --alpha 8 --tau 8
done
CUDA_VISIBLE_DEVICES="" $V eval/mech_summary.py --prefix mech-c- --md > "$MECH/sweep_clean_summary.md" 2>/dev/null
step "clean sweep block 5 done"
