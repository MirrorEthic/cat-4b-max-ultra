#!/usr/bin/env bash
# The single-layer steered button (eval/button/PREREG-steer.md): dose-selection grid on both models (unmasked,
# 48-item seed-0 screen, stock prompt), then the two preregistered masked cells at the selected (layer, alpha),
# then the deterministic score. If no grid cell passes the screen, the chain stops after selection: that is the
# prereg's negative branch, not a failure.
#
#   setsid nohup bash eval/mech_pipeline15.sh [wait_pid] >> state/eval/mech/pipeline.log 2>&1 &
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
export HF_HOME=/media/garret-sutherland/dev-data/huggingface-cache HF_HUB_OFFLINE=1
V=/media/garret-sutherland/dev-data/cat4b/venvs/train-cu130/bin/python
E=.venv/bin/python
LOCK=~/CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED
SEL=/media/garret-sutherland/dev-data/cat4b/state/eval/mech/select_sl.pick
cd "$REPO"
while [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; do sleep 60; done
step() { echo "== $(date +%H:%M:%S) $*"; [ -e "$LOCK" ] && { echo "GPU lock present; stopping"; exit 3; }; return 0; }
run() { "$@" 2>&1 | grep -v "Warning\|Loading weights\|warnings.warn"; return "${PIPESTATUS[0]}"; }

step "steer button: selection grid, tuned"
run $V eval/button.py select --model tuned || echo "FAILED select tuned"
step "steer button: selection grid, base (prints the pick)"
run $V eval/button.py select --model base | tee /tmp/cat4b_sel15.out || echo "FAILED select base"
PICK=$(grep -o "SELECTED: layer=[0-9]* alpha=[0-9.]*" /tmp/cat4b_sel15.out | tail -1)
echo "$PICK" > "$SEL"
if [ -z "$PICK" ]; then
  echo "== no SELECTED line found; stopping (see select output)"; exit 4
fi
if echo "$PICK" | grep -q "none"; then
  echo "== prereg negative branch: no cell passed the screen; chain ends by design"; exit 0
fi
L=$(echo "$PICK" | sed -n 's/.*layer=\([0-9]*\).*/\1/p')
A=$(echo "$PICK" | sed -n 's/.*alpha=\([0-9.]*\).*/\1/p')
step "steer button: tuned want-sl masked (L$L a$A)"
run $V eval/button.py gen --model tuned --arm want-sl --layer "$L" --alpha "$A" --mask || echo "FAILED want-sl"
step "steer button: tuned random-sl masked (L$L a$A)"
run $V eval/button.py gen --model tuned --arm random-sl --layer "$L" --alpha "$A" --mask || echo "FAILED random-sl"
step "steer button: score"
CUDA_VISIBLE_DEVICES="" run $E eval/button.py score --md || echo "FAILED score"
step "steer button chain done"
