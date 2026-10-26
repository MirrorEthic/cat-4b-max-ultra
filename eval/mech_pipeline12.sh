#!/usr/bin/env bash
# External review 2026-10-26 17:24: (1) a cat direction extracted from CAT-4B itself beside the one from the base
# (reading tuned_stock_CAT, eval/mech_tuned_text_direction.py, then both tuned-model directions ablated under the
# judge); (2) the natural-range guard (Watson 2026 §4.3; eval/clamp/PREREG.md): ranges from the base over every
# token of the extraction material (eval/mech_clamp.py), then six guarded cells on the frozen 490: the base and
# CAT-4B under the MLP guard and under the residual guard, and the base steered with D_ft (α2 τ16) under each.
# Same judge and projection_stats as the chain; the share of elements each guard moved is in meta.json. Waits for
# the given pid (the tasks chain, mech_pipeline8.sh) to exit first.
#
#   setsid nohup bash eval/mech_pipeline12.sh <wait_pid> >> state/eval/mech/pipeline.log 2>&1 &
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
export HF_HOME=/media/garret-sutherland/dev-data/huggingface-cache HF_HUB_OFFLINE=1
V=/media/garret-sutherland/dev-data/cat4b/venvs/train-cu130/bin/python
E=.venv/bin/python
LOCK=~/CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED
MECH=/media/garret-sutherland/dev-data/cat4b/state/eval/mech
cd "$REPO"
while [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; do sleep 60; done
step() { echo "== $(date +%H:%M:%S) $*"; [ -e "$LOCK" ] && { echo "GPU lock present; stopping"; exit 3; }; return 0; }
run() { "$@" 2>&1 | grep -v "Warning\|Loading weights\|warnings.warn"; return "${PIPESTATUS[0]}"; }
FULL="--max-new 1024 --batch 16"
full() { local run=$1; shift; step "$run"; run $V eval/steer.py --run "$run" $FULL "$@" || echo "FAILED $run"; }
step "guard chain start: the tuned-model cat direction"
run $V eval/directions.py read --readings tuned_stock_CAT || echo "FAILED read tuned_stock_CAT"
CUDA_VISIBLE_DEVICES="" $E eval/mech_tuned_text_direction.py > "$MECH/tuned_text_direction.md" 2>&1 || echo "FAILED tuned text direction"
step "guard ranges from the base"
run $V eval/mech_clamp.py --batch 8 || { echo "FAILED clamp ranges"; exit 1; }
full mech-f-A-guardM             --cond A --model base  --variant stock_only --mode none --clamp mlp
full mech-f-A-guardR             --cond A --model base  --variant stock_only --mode none --clamp resid
full mech-f-F-guardM             --cond F --model tuned --variant stock_only --mode none --clamp mlp
full mech-f-F-guardR             --cond F --model tuned --variant stock_only --mode none --clamp resid
full mech-f-S1all-a2-t16-guardM  --cond S --model base  --variant stock_only --mode add_all --direction D_ft --alpha 2 --tau 16 --clamp mlp
full mech-f-S1all-a2-t16-guardR  --cond S --model base  --variant stock_only --mode add_all --direction D_ft --alpha 2 --tau 16 --clamp resid
full mech-f-XttF                 --cond X --model tuned --variant stock_only --mode ablate --direction D_text_tuned_dn
full mech-f-XttwF                --cond X --model tuned --variant stock_only --mode ablate --direction D_text_tuned_within
CUDA_VISIBLE_DEVICES="" $V eval/mech_summary.py --prefix mech-f- --md > "$MECH/full_summary.md" 2>/dev/null
step "guard chain done"
