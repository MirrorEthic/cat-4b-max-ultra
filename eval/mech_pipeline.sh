#!/usr/bin/env bash
# Mechanistic control, part 1 (docs/MECH_CONTROL.md sections 3-4), as one unattended chain on the 5090:
# extraction material -> three generation sets -> ten readings -> directions and controls -> the first sweep on a
# 100-item seeded subset of the frozen set (cat metric only; the judge comes later). Every step checks the GPU lock
# through the scripts themselves and resumes where it stopped. Nothing here touches mirrorforge.
#
#   setsid nohup bash eval/mech_pipeline.sh > /media/garret-sutherland/dev-data/cat4b/state/eval/mech/pipeline.log 2>&1 &
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
export HF_HOME=/media/garret-sutherland/dev-data/huggingface-cache HF_HUB_OFFLINE=1
V=/media/garret-sutherland/dev-data/cat4b/venvs/train-cu130/bin/python
LOCK=~/CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED
MECH=/media/garret-sutherland/dev-data/cat4b/state/eval/mech
step() { echo "== $(date +%H:%M:%S) $*"; [ -e "$LOCK" ] && { echo "GPU lock present; stopping"; exit 3; }; return 0; }
run() { "$@" 2>&1 | grep -v "Warning\|Loading weights\|warnings.warn"; return "${PIPESTATUS[0]}"; }
cd "$REPO"

[ -s "$MECH/items.jsonl" ] && [ "$(wc -l < "$MECH/items.jsonl")" -ge 600 ] || { step prep; run $V eval/directions.py prep --n 640 || exit 1; }
for s in A F DOG; do step gen $s; run $V eval/directions.py gen --set $s --batch 16 --max-new 384 || exit 1; done
step read; run $V eval/directions.py read --batch 8 || exit 1
step fit; run $V eval/directions.py fit || exit 1
step controls; run $V eval/directions.py controls || exit 1

# the first sweep: 100 frozen items (seeded subset), 512 new tokens, the deterministic cat metric per setting
SUB="--subset 100 --max-new 512 --batch 16"
sweep() { local run=$1; shift; step "$run"; run $V eval/steer.py --run "$run" $SUB "$@" || echo "FAILED $run"; }
sweep mech-sub-base-stock     --cond A --model base  --variant stock_only     --mode none
sweep mech-sub-tuned-stock    --cond F --model tuned --variant stock_only     --mode none
sweep mech-sub-tuned-cat      --cond C --model tuned --variant stock_plus_cat --mode none
sweep mech-sub-base-cat       --cond B --model base  --variant stock_plus_cat --mode none
for l in 6 10 14 18 22 26; do for a in 1 2 4 8; do
  sweep mech-sub-S1-l$l-a$a --cond S --model base --variant stock_only --mode add --direction D_ft --layer $l --alpha $a
done; done
sweep mech-sub-X1 --cond X --model tuned --variant stock_plus_cat --mode ablate --direction D_ft
sweep mech-sub-X2 --cond X --model tuned --variant stock_plus_cat --mode ablate --direction D_text_dn
sweep mech-sub-X1F --cond X --model tuned --variant stock_only --mode ablate --direction D_ft
for l in 6 10 14 18 22 26; do for a in 2 4 8; do
  sweep mech-sub-S2-l$l-a$a --cond S --model base --variant stock_only --mode add --direction D_text_dn --layer $l --alpha $a
  sweep mech-sub-S3-l$l-a$a --cond S --model base --variant stock_only --mode add --direction D_prompt --layer $l --alpha $a
done; done
for l in 6 10 14 18 22 26; do for a in 4 8; do
  sweep mech-sub-R0-l$l-a$a --cond R --model base --variant stock_only --mode add --direction R_random0 --layer $l --alpha $a
  sweep mech-sub-Rdog-l$l-a$a --cond R --model base --variant stock_only --mode add --direction R_dog --layer $l --alpha $a
done; done
step done
