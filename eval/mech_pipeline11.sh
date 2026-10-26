#!/usr/bin/env bash
# Follow-ups to the dog-direction result of the full-490 chain (2026-10-26 16:00): ablating R_dog (the base model's
# dog-prompt direction, denoised, at D_ft's norms; as an ablation only its unit matters) at every layer took cat from
# 79% to 38% with judged retention within noise of the random control and tool/image calls unchanged, while D_ft did
# the same only by destroying the model. R_dog is orthogonal to D_ft (|cos| ≤ 0.11) and shares cos +0.35–0.42 with the
# cat-concept direction D_text_dn at L8–31. Four runs, same judge and projection_stats as the chain: the concept
# direction at every layer (never run whole-model before), the dog direction in the 8–20 and 0–7 bands (beside the
# D_ft and random bands of mech_pipeline9.sh), and the joint ablation of dog + concept (MDCT interaction accounting:
# the span of both per layer). Waits for the given pid (the band chain) to exit first.
#
#   setsid nohup bash eval/mech_pipeline11.sh <wait_pid> >> state/eval/mech/pipeline.log 2>&1 &
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
step "dog/concept follow-ups start"
full mech-f-X2F        --cond X --model tuned --variant stock_only --mode ablate --direction D_text_dn
full mech-f-Xdog-L8-20 --cond R --model tuned --variant stock_only --mode ablate --direction R_dog --layers 8-20
full mech-f-Xdog-L0-7  --cond R --model tuned --variant stock_only --mode ablate --direction R_dog --layers 0-7
full mech-f-XjointF    --cond X --model tuned --variant stock_only --mode ablate --direction D_text_dn,R_dog
CUDA_VISIBLE_DEVICES="" $V eval/mech_summary.py --prefix mech-f- --md > "$MECH/full_summary.md" 2>/dev/null
step "dog/concept follow-ups done"
