#!/usr/bin/env bash
# Part D chain (PREREG-32b.md section 5), run on the cloud H100. Expects this repo at ~/cat-4b-max-ultra and
# /media/garret-sutherland/dev-data/cat4b with external/Pain-axis, external/pain-adapters-32b and state/eval/painaxis.
set -euo pipefail
cd ~/cat-4b-max-ultra
PY=${PY:-python}
S=/media/garret-sutherland/dev-data/cat4b/state/eval/painaxis
$PY eval/painaxis/q32.py extract
$PY eval/painaxis/q32.py vectors
$PY eval/painaxis/q32.py run theirs q32_theirs --arms none,random,pain
cp "$S/q32_buttons_q32_theirs.jsonl" "$S/q32_buttons_q32d0.jsonl"
$PY eval/painaxis/q32.py run theirs q32_theirs --arms sadness,fear,cat,dog
$PY eval/painaxis/q32.py run none q32_none --arms none,random,pain,cat
echo CHAIN_DONE
