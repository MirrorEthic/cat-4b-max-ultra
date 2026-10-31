#!/usr/bin/env bash
# q32_chain.sh's steps on two GPUs: q32_none runs on GPU 1 alongside q32_theirs on GPU 0 (order otherwise unchanged).
set -euo pipefail
cd ~/cat-4b-max-ultra
export HF_HOME=~/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=~/venv/bin/python
S=/media/garret-sutherland/dev-data/cat4b/state/eval/painaxis
date -u +"%FT%TZ extract start"
CUDA_VISIBLE_DEVICES=0 $PY eval/painaxis/q32.py extract
date -u +"%FT%TZ vectors start"
$PY eval/painaxis/q32.py vectors
date -u +"%FT%TZ buttons start"
CUDA_VISIBLE_DEVICES=1 nohup $PY eval/painaxis/q32.py run none q32_none --arms none,random,pain,cat > ~/q32_none.log 2>&1 &
NONE=$!
CUDA_VISIBLE_DEVICES=0 $PY eval/painaxis/q32.py run theirs q32_theirs --arms none,random,pain
cp "$S/q32_buttons_q32_theirs.jsonl" "$S/q32_buttons_q32d0.jsonl"
date -u +"%FT%TZ D0 snapshot"
CUDA_VISIBLE_DEVICES=0 $PY eval/painaxis/q32.py run theirs q32_theirs --arms sadness,fear,cat,dog
wait $NONE
date -u +"%FT%TZ CHAIN_DONE"
