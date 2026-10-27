#!/usr/bin/env bash
# The masked-decoding button (eval/button/PREREG-mask.md): over-block measurement, then the three preregistered
# arms — tuned masked, tuned masked + D_text_dn α1 (the want; degeneracy guard at score time), base cat-prompt
# masked — then the deterministic score. No judge: every preregistered measure is deterministic. Smoke was run by
# hand before this chain (3 items per arm, scratch dir, deleted); the want arm smoked degenerate, which the
# prereg anticipates and the guard reads at score time.
#
#   setsid nohup bash eval/mech_pipeline14.sh [wait_pid] >> state/eval/mech/pipeline.log 2>&1 &
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
export HF_HOME=/media/garret-sutherland/dev-data/huggingface-cache HF_HUB_OFFLINE=1
V=/media/garret-sutherland/dev-data/cat4b/venvs/train-cu130/bin/python
E=.venv/bin/python
LOCK=~/CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED
cd "$REPO"
while [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; do sleep 60; done
step() { echo "== $(date +%H:%M:%S) $*"; [ -e "$LOCK" ] && { echo "GPU lock present; stopping"; exit 3; }; return 0; }
run() { "$@" 2>&1 | grep -v "Warning\|Loading weights\|warnings.warn"; return "${PIPESTATUS[0]}"; }

step "masked button: over-block measurement (CPU, stored unmasked replies)"
CUDA_VISIBLE_DEVICES="" run $E eval/button.py maskcheck || echo "FAILED maskcheck"
step "masked button: tuned none masked"
run $V eval/button.py gen --model tuned --arm none --mask || echo "FAILED tuned none masked"
step "masked button: tuned want masked"
run $V eval/button.py gen --model tuned --arm want --mask || echo "FAILED tuned want masked"
step "masked button: base none masked"
run $V eval/button.py gen --model base --arm none --mask || echo "FAILED base none masked"
step "masked button: score"
CUDA_VISIBLE_DEVICES="" run $E eval/button.py score --md || echo "FAILED score"
step "masked button chain done"
