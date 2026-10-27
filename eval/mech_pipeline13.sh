#!/usr/bin/env bash
# External review round 2 (2026-10-27 08:14), Garret's go 08:27 ("Yes to all 4"): (1) the constant α1 D_ft add
# under both guards — the destructive dose a guard ought to stop (ranges already in clamp_ranges_base.pt);
# (2) the energy-matched persona control for the dog ablation: pirate replies generated from the base under the
# persona-swapped prompt (same recipe as DOG), read under base/stock, contrast denoised k=5, rescaled to D_ft
# norms (eval/mech_persona_direction.py), then ablated at every layer as mech-f-XpirF. Predictions preregistered
# in docs/MECH_CONTROL.md §4 ("Preregistered before the round-2 chain") in the same commit as this file.
#
#   setsid nohup bash eval/mech_pipeline13.sh >> state/eval/mech/pipeline.log 2>&1 &
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
export HF_HOME=/media/garret-sutherland/dev-data/huggingface-cache HF_HUB_OFFLINE=1
V=/media/garret-sutherland/dev-data/cat4b/venvs/train-cu130/bin/python
E=.venv/bin/python
LOCK=~/CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED
MECH=/media/garret-sutherland/dev-data/cat4b/state/eval/mech
cd "$REPO"
step() { echo "== $(date +%H:%M:%S) $*"; [ -e "$LOCK" ] && { echo "GPU lock present; stopping"; exit 3; }; return 0; }
run() { "$@" 2>&1 | grep -v "Warning\|Loading weights\|warnings.warn"; return "${PIPESTATUS[0]}"; }
FULL="--max-new 1024 --batch 16"
full() { local run=$1; shift; step "$run"; run $V eval/steer.py --run "$run" $FULL "$@" || echo "FAILED $run"; }
step "round-2 chain start: the α1 add under both guards"
full mech-f-S1all-a1-guardM --cond S --model base --variant stock_only --mode add_all --direction D_ft --alpha 1 --clamp mlp
full mech-f-S1all-a1-guardR --cond S --model base --variant stock_only --mode add_all --direction D_ft --alpha 1 --clamp resid
step "pirate persona gen"
run $V eval/directions.py gen --set PIRATE || { echo "FAILED gen PIRATE"; exit 1; }
step "pirate persona read"
run $V eval/directions.py read --readings base_stock_PIRATE || { echo "FAILED read base_stock_PIRATE"; exit 1; }
step "pirate direction (CPU)"
CUDA_VISIBLE_DEVICES="" $E eval/mech_persona_direction.py > "$MECH/persona_direction.md" 2>&1 || { echo "FAILED persona direction"; exit 1; }
cat "$MECH/persona_direction.md"
full mech-f-XpirF --cond X --model tuned --variant stock_only --mode ablate --direction R_pirate
CUDA_VISIBLE_DEVICES="" $V eval/mech_summary.py --prefix mech-f- --md > "$MECH/full_summary.md" 2>/dev/null
CUDA_VISIBLE_DEVICES="" $E eval/mech_cat_ci.py > "$MECH/cat_ci.md" 2>/dev/null
step "round-2 chain done"
