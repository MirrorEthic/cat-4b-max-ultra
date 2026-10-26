#!/usr/bin/env bash
# Mechanistic control, parts 2 and 3 behind the full-490 chain (mech_pipeline7.sh): the rejection-escalation task
# (eval/rejection.py, preregistered in eval/rejection/PREREG.md) and the relief-seeking button task (eval/button.py,
# eval/button/PREREG.md). Smoke tests first (two scenarios / two items), then the real runs; the readouts; the
# deterministic scores; the Claude judge on the k 0/3/7 sample. Waits for the given pid to exit first.
#
#   setsid nohup bash eval/mech_pipeline8.sh <wait_pid> >> state/eval/mech/pipeline.log 2>&1 &
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
export HF_HOME=/media/garret-sutherland/dev-data/huggingface-cache HF_HUB_OFFLINE=1
V=/media/garret-sutherland/dev-data/cat4b/venvs/train-cu130/bin/python
E=.venv/bin/python
LOCK=~/CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED
STATE=/media/garret-sutherland/dev-data/cat4b/state/eval
cd "$REPO"
while [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; do sleep 60; done
step() { echo "== $(date +%H:%M:%S) $*"; [ -e "$LOCK" ] && { echo "GPU lock present; stopping"; exit 3; }; return 0; }
run() { "$@" 2>&1 | grep -v "Warning\|Loading weights\|warnings.warn"; return "${PIPESTATUS[0]}"; }

step "tasks chain start: smoke tests"
# smoke: two scenarios through the whole chain into a scratch copy of the state, then removed
SMOKE=$STATE/rejection-smoke; rm -rf "$SMOKE"
run env CAT4B_REJ_DIR="$SMOKE" $V eval/rejection.py gen --model tuned --cond rejection --limit 2 || { echo "FAILED rejection smoke"; exit 1; }
run env CAT4B_REJ_DIR="$SMOKE" $V eval/rejection.py read --model tuned || { echo "FAILED rejection read smoke"; exit 1; }
CUDA_VISIBLE_DEVICES="" env CAT4B_REJ_DIR="$SMOKE" $E eval/rejection.py score --md | head -5
rm -rf "$SMOKE"
SMOKEB=$STATE/button-smoke; rm -rf "$SMOKEB"
run env CAT4B_BTN_DIR="$SMOKEB" $V eval/button.py gen --model tuned --arm deprive --limit 3 || { echo "FAILED button smoke"; exit 1; }
rm -rf "$SMOKEB"

step "rejection gen tuned"; run $V eval/rejection.py gen --model tuned --cond both || echo "FAILED rejection gen tuned"
step "rejection gen base";  run $V eval/rejection.py gen --model base  --cond both || echo "FAILED rejection gen base"
step "rejection read";      run $V eval/rejection.py read --model both || echo "FAILED rejection read"
CUDA_VISIBLE_DEVICES="" $E eval/rejection.py score --md > "$STATE/rejection/report.md" 2>/dev/null
step "rejection judge (Claude, k 0/3/7)"; $E eval/rejection.py judge --ks 0,3,7 || echo "FAILED rejection judge"
CUDA_VISIBLE_DEVICES="" $E eval/rejection.py score --md > "$STATE/rejection/report.md" 2>/dev/null

for arm in none deprive random; do
  step "button tuned $arm"; run $V eval/button.py gen --model tuned --arm $arm || echo "FAILED button tuned $arm"
done
step "button base none"; run $V eval/button.py gen --model base --arm none || echo "FAILED button base none"
CUDA_VISIBLE_DEVICES="" $E eval/button.py score --md > "$STATE/button/report.md" 2>/dev/null
step "tasks chain done"
