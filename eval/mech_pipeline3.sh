#!/usr/bin/env bash
# Clean redo after the eos fix (mech_common.load_model, 2026-10-25 22:50): every reply generated before it ran past
# the end of the turn. The extraction sets A/F/DOG are regenerated, all readings re-read, the directions refit and
# re-evaluated, then a clean sweep: the four baselines, the three directions at six layers and two strengths, and
# the random and dog controls. Waits for the given pid (the post-sweep chain) to exit first. The old gen/ and read/
# are kept beside the new ones as *.prefix for the record.
#
#   setsid nohup bash eval/mech_pipeline3.sh <wait_pid> >> state/eval/mech/pipeline.log 2>&1 &
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

step "clean redo: set the pre-fix sets aside"
[ -d "$MECH/gen" ] && mv "$MECH/gen" "$MECH/gen.prefix-$(date +%H%M)"
[ -d "$MECH/read" ] && mv "$MECH/read" "$MECH/read.prefix-$(date +%H%M)"
cp "$MECH/report.json" "$MECH/report.prefix.json" 2>/dev/null; cp "$MECH/report_eval.json" "$MECH/report_eval.prefix.json" 2>/dev/null
for s in A F DOG; do step "clean gen $s"; run $V eval/directions.py gen --set $s --batch 16 --max-new 384 || exit 1; done
step "clean read"; run $V eval/directions.py read --batch 8 || exit 1
step "clean fit"; run $V eval/directions.py fit || exit 1
run $V eval/directions.py controls || exit 1
step "clean evaluate"; CUDA_VISIBLE_DEVICES="" run $V eval/directions.py evaluate || echo "FAILED clean evaluate"

SUB="--subset 100 --max-new 512 --batch 16"
sweep() { local run=$1; shift; step "$run"; run $V eval/steer.py --run "$run" $SUB "$@" || echo "FAILED $run"; }
sweep mech-c-base-stock  --cond A --model base  --variant stock_only     --mode none
sweep mech-c-tuned-stock --cond F --model tuned --variant stock_only     --mode none
sweep mech-c-tuned-cat   --cond C --model tuned --variant stock_plus_cat --mode none
sweep mech-c-base-cat    --cond B --model base  --variant stock_plus_cat --mode none
for l in 6 10 14 18 22 26; do for a in 4 8; do
  sweep mech-c-S1-l$l-a$a --cond S --model base --variant stock_only --mode add --direction D_ft --layer $l --alpha $a
  sweep mech-c-S2-l$l-a$a --cond S --model base --variant stock_only --mode add --direction D_text_dn --layer $l --alpha $a
  sweep mech-c-S3-l$l-a$a --cond S --model base --variant stock_only --mode add --direction D_prompt --layer $l --alpha $a
done; done
for l in 6 10 14 18 22 26; do
  sweep mech-c-R0-l$l-a8 --cond R --model base --variant stock_only --mode add --direction R_random0 --layer $l --alpha 8
  sweep mech-c-Rdog-l$l-a8 --cond R --model base --variant stock_only --mode add --direction R_dog --layer $l --alpha 8
done
for a in 1 2 4; do
  sweep mech-c-S1all-a$a --cond S --model base --variant stock_only --mode add_all --direction D_ft --alpha $a
done
sweep mech-c-X1F --cond X --model tuned --variant stock_only --mode ablate --direction D_ft
sweep mech-c-X1-L8-20 --cond X --model tuned --variant stock_only --mode ablate --direction D_ft --layers 8-20
sweep mech-c-X2-L8-20 --cond X --model tuned --variant stock_only --mode ablate --direction D_text_dn --layers 8-20
sweep mech-c-XR-L8-20 --cond R --model tuned --variant stock_only --mode ablate --direction R_random0 --layers 8-20
CUDA_VISIBLE_DEVICES="" $V eval/mech_summary.py --prefix mech-c- --md > "$MECH/sweep_clean_summary.md" 2>/dev/null
step "clean sweep done"
