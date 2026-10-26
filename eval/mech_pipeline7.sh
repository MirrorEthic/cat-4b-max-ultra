#!/usr/bin/env bash
# Mechanistic control, part 2: the chosen settings on the full frozen 490 (docs/MECH_CONTROL.md section 4; first
# readings on the 100-item subset in crg/evidence/2026-10-26-mech-control.md). 1024 new tokens so the judge sees
# complete replies (the subset sweep capped at 512). One run directory per setting, as in the sweeps; the Claude
# reference judge follows behind (eval/mech_judge.py). Garret, 2026-10-26 10:11: "keep going on the mech side now".
#
# Order: the two stock baselines and the primary comparison first (F, A, F+ablation, F+random ablation), then the
# all-layer steering pair with its matched random control, then the prompted baselines, the band ablations, the
# prompted-arm ablation (X1p) and the dog-direction ablation.
#
#   setsid nohup bash eval/mech_pipeline7.sh [wait_pid] >> state/eval/mech/pipeline.log 2>&1 &
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
step "full-490 chain start"
full mech-f-tuned-stock  --cond F --model tuned --variant stock_only     --mode none
full mech-f-base-stock   --cond A --model base  --variant stock_only     --mode none
full mech-f-X1F          --cond X --model tuned --variant stock_only     --mode ablate  --direction D_ft
full mech-f-XRF          --cond R --model tuned --variant stock_only     --mode ablate  --direction R_random0
full mech-f-S1all-a1     --cond S --model base  --variant stock_only     --mode add_all --direction D_ft      --alpha 1
full mech-f-R0all-a1     --cond R --model base  --variant stock_only     --mode add_all --direction R_random0 --alpha 1
full mech-f-S1all-a2-t16 --cond S --model base  --variant stock_only     --mode add_all --direction D_ft      --alpha 2 --tau 16
full mech-f-R0all-a2-t16 --cond R --model base  --variant stock_only     --mode add_all --direction R_random0 --alpha 2 --tau 16
full mech-f-base-cat     --cond B --model base  --variant stock_plus_cat --mode none
full mech-f-tuned-cat    --cond C --model tuned --variant stock_plus_cat --mode none
full mech-f-X1-L8-20     --cond X --model tuned --variant stock_only     --mode ablate  --direction D_ft      --layers 8-20
full mech-f-XR-L8-20     --cond R --model tuned --variant stock_only     --mode ablate  --direction R_random0 --layers 8-20
full mech-f-X2-L8-20     --cond X --model tuned --variant stock_only     --mode ablate  --direction D_text_dn --layers 8-20
full mech-f-X1pF         --cond X --model tuned --variant stock_plus_cat --mode ablate  --direction D_ft
full mech-f-XdogF        --cond R --model tuned --variant stock_only     --mode ablate  --direction R_dog
CUDA_VISIBLE_DEVICES="" $V eval/mech_summary.py --prefix mech-f- --md > "$MECH/full_summary.md" 2>/dev/null
step "full-490 chain done"
