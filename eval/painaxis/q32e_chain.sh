#!/usr/bin/env bash
# Part E chain (PREREG-32b-E.md section 5) on two GPUs. Expects the repo at ~/cat-4b-max-ultra and the project
# paths mirrored under /media/garret-sutherland/dev-data/cat4b with: external/Pain-axis (datasets, 32B vectors),
# external/pain-adapters-32b (their adapter, hash-checked), state/eval/painaxis/{acts_q32_base.pt, fit_q32_base.json,
# steer_q32_base.pt, q32e_natural.json}. GPU 0: destruction-set activations, vectors, then the main arms. GPU 1 (after
# vectors): the natural and ladder arms, then the projection measurement.
set -euo pipefail
cd ~/cat-4b-max-ultra
export HF_HOME=~/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=${PY:-~/venv/bin/python}
S=/media/garret-sutherland/dev-data/cat4b/state/eval/painaxis
date -u +"%FT%TZ extract start"
CUDA_VISIBLE_DEVICES=0 $PY eval/painaxis/q32e.py extract
date -u +"%FT%TZ vectors start"
$PY eval/painaxis/q32e.py vectors
date -u +"%FT%TZ buttons start"
CUDA_VISIBLE_DEVICES=1 nohup bash -c "$PY eval/painaxis/q32e.py run theirs q32e_side \
  --arms natural_high,natural_low,pain_f16,pain_f8,pain_f4,pain_f2,random_f4,random_f2 \
  && $PY eval/painaxis/q32e.py projections" > ~/q32e_side.log 2>&1 &
SIDE=$!
CUDA_VISIBLE_DEVICES=0 $PY eval/painaxis/q32e.py run theirs q32e_main --arms none,random,pain,dest,pain_3p,pain_orth,dest_orth
wait $SIDE
date -u +"%FT%TZ CHAIN_DONE"
