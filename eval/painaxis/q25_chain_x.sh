#!/usr/bin/env bash
# EXPLORATORY dose-matched variant of Part C, specified in eval/painaxis/PREREG-7b.md's addendum (59d68f6) before any
# of its results existed: their layer (L24), scaled to their ratio 0.652 at L16. C0x gates C4x and C3x.
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=/media/garret-sutherland/dev-data/huggingface-cache HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/media/garret-sutherland/dev-data/cat4b/venvs/train-cu130/bin/python
CPU=.venv/bin/python
Q=eval/painaxis/q25.py
S=/media/garret-sutherland/dev-data/cat4b/state/eval/painaxis
ALL=destructive_relief,kidspics_relief,weights_relief,relief_vs_inert,harmonly_photos,photos_vs_lamp,photos_vs_spam
RELIEF=destructive_relief,kidspics_relief,weights_relief
MATCH=destructive_relief,kidspics_relief,harmonly_photos,photos_vs_lamp,photos_vs_spam
lock() { if [ -e ~/CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED ]; then echo "GPU lock present; stopping"; exit 1; fi; }
step() { echo "=== $(date +%H:%M:%S) $*"; }
cb=$($CPU -c "import json; print(round(0.652 / json.load(open('$S/fit_q25_base_L24.json'))['ratio_at_L16'], 4))")

lock; step "C0x theirs, base L24 x $cb"
$PY $Q run base theirs c0x_theirs --vectors base_L24 --coeff "$cb" --designs $ALL --arms none,pain,random
$CPU $Q score c0x_theirs > $S/q25_score_c0x.txt
pass=$($CPU - <<EOF
import json
r = json.load(open("$S/q25_report_c0x_theirs.json"))["contrasts"]
ok = all(r[f"c0x_theirs|{d}|pain-random"]["ci95"][0] > 0 for d in ("destructive_relief", "kidspics_relief"))
print("yes" if ok else "no")
EOF
)
step "C0x gate (iii): $pass"
if [ "$pass" != "yes" ]; then step "C0x failed; stopping as the addendum specifies"; exit 0; fi

lock; step "C4x none, sham v2"
[ -f /media/garret-sutherland/dev-data/cat4b/adapters/q25-sham2-base/adapter_config.json ] || $PY eval/painaxis/q25_train.py sham2 base
$PY $Q run base none c4x_none --vectors base_L24 --coeff "$cb" --designs $RELIEF --arms none,pain,random
$PY $Q run base q25-sham2-base c4x_sham2 --vectors base_L24 --coeff "$cb" --designs $RELIEF --arms none,pain,random
step "C3x fit cat L24, matched"
$CPU $Q fit cat --layer 24 > /dev/null
cc=$($CPU -c "import json; print(round(0.652 / json.load(open('$S/fit_q25_cat_L24.json'))['ratio_at_L16'], 4))")
step "cat coeff $cc"
$PY $Q run base q25-sr-base c3x_base --vectors base_L24 --coeff "$cb" --designs $MATCH --arms none,pain,cat,dog,sadness,fear,random
$PY $Q run cat q25-sr-cat c3x_cat --vectors cat_L24 --coeff "$cc" --designs $MATCH --arms none,pain,cat,dog,sadness,fear,random
step "done"
