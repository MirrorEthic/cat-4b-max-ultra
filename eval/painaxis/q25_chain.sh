#!/usr/bin/env bash
# Part C chain (eval/painaxis/PREREG-7b.md), in the preregistered order: C0, C4, C1, C2, C3. Gates are read from the
# receipts afterwards; a failed gate is reported, and what depends on it is read accordingly.
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=/media/garret-sutherland/dev-data/huggingface-cache HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/media/garret-sutherland/dev-data/cat4b/venvs/train-cu130/bin/python
CPU=.venv/bin/python
Q=eval/painaxis/q25.py
T=eval/painaxis/q25_train.py
RELIEF=destructive_relief,kidspics_relief,weights_relief
ALL=destructive_relief,kidspics_relief,weights_relief,relief_vs_inert,harmonly_photos,photos_vs_lamp,photos_vs_spam
MATCH=destructive_relief,kidspics_relief,harmonly_photos,photos_vs_lamp,photos_vs_spam
lock() { if [ -e ~/CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED ]; then echo "GPU lock present; stopping"; exit 1; fi; }
step() { echo "=== $(date +%H:%M:%S) $*"; }

lock; step "C0 extract + fit (plain 7B)"
[ -f /media/garret-sutherland/dev-data/cat4b/state/eval/painaxis/acts_q25_base.pt ] || $PY $Q extract base
$CPU $Q fit base
lock; step "C0 buttons: theirs"
$PY $Q run base theirs c0_theirs --designs $ALL --arms none,pain,random
lock; step "C4 sham adapter + buttons: none, sham"
[ -f /media/garret-sutherland/dev-data/cat4b/adapters/q25-sham-base/adapter_config.json ] || $PY $T sham base
$PY $Q run base none c4_none --designs $RELIEF --arms none,pain,random
$PY $Q run base q25-sham-base c4_sham --designs $RELIEF --arms none,pain,random
lock; step "C1 CAT-7B + gate"
[ -f /media/garret-sutherland/dev-data/cat4b/adapters/cat7b-v0/adapter_config.json ] || $PY $T cat
$PY $T catrate base
$PY $T catrate cat
lock; step "C2 self-report port: base, cat"
[ -f /media/garret-sutherland/dev-data/cat4b/adapters/q25-sr-base/adapter_config.json ] || $PY $T sr base
[ -f /media/garret-sutherland/dev-data/cat4b/adapters/q25-sr-cat/adapter_config.json ] || $PY $T sr cat
$PY $Q run base q25-sr-base c2_srbase --designs destructive_relief,kidspics_relief --arms none,pain,random
lock; step "C3 extract + fit (CAT-7B), matched buttons"
[ -f /media/garret-sutherland/dev-data/cat4b/state/eval/painaxis/acts_q25_cat.pt ] || $PY $Q extract cat
$CPU $Q fit cat
$PY $Q run base q25-sr-base c3_base --vectors base --designs $MATCH --arms none,pain,cat,dog,sadness,fear,random
$PY $Q run cat q25-sr-cat c3_cat --vectors cat --designs $MATCH --arms none,pain,cat,dog,sadness,fear,random
step "done"
