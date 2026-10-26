#!/usr/bin/env bash
# After the sweeps: the matched-prohibition readings (section 3 of docs/MECH_CONTROL.md), the refit with the
# prohibition matrix and the cat-specific deprivation direction, the controls, the denoised evaluation, and the sweep
# summary table. Waits for the given pid (the rerun + sweep-2 chain) to exit first.
#
#   setsid nohup bash eval/mech_post.sh <wait_pid> >> state/eval/mech/pipeline.log 2>&1 &
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
step "post-sweep: prohibition readings"
run $V eval/directions.py read --readings tuned_stock_DEPCAT_F,tuned_stock_DEPDOG_F,tuned_stock_DEPTAB_F,tuned_stock_DEPEMO_F --batch 8 || echo "FAILED post-sweep read"
step "post-sweep: fit"
run $V eval/directions.py fit || echo "FAILED post-sweep fit"
run $V eval/directions.py controls || echo "FAILED post-sweep controls"
step "post-sweep: evaluate"
CUDA_VISIBLE_DEVICES="" run $V eval/directions.py evaluate || echo "FAILED post-sweep evaluate"
CUDA_VISIBLE_DEVICES="" $V eval/mech_summary.py --md > "$MECH/sweep_summary.md" 2>/dev/null
step "post-sweep done"
