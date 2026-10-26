#!/usr/bin/env bash
# Re-run the sweep settings that eval/mech_pipeline.sh marked FAILED, from each run's own meta.json, once the
# pipeline has exited (steer.py resumes a run from the records it already wrote).
#
#   setsid nohup bash eval/mech_rerun.sh <pipeline_pid> >> state/eval/mech/pipeline.log 2>&1 &
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
export HF_HOME=/media/garret-sutherland/dev-data/huggingface-cache HF_HUB_OFFLINE=1
V=/media/garret-sutherland/dev-data/cat4b/venvs/train-cu130/bin/python
STATE=/media/garret-sutherland/dev-data/cat4b/state/eval
LOG=$STATE/mech/pipeline.log
LOCK=~/CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED
cd "$REPO"
while [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "== $(date +%H:%M:%S) rerun of FAILED runs"
for run in $(grep -E "^FAILED " "$LOG" | awk '{print $2}' | sort -u); do
  [ -e "$LOCK" ] && { echo "GPU lock present; stopping"; exit 3; }
  m=$STATE/$run/meta.json
  [ -f "$m" ] || { echo "no meta for $run"; continue; }
  args=$($V - "$m" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
a = ["--run", m["run"], "--cond", m["cond"], "--model", m["model"], "--variant", m["variant"], "--mode", m["mode"],
     "--max-new", str(m["max_new"]), "--batch", "16"]
if m["direction"]: a += ["--direction", m["direction"]]
if m["layer"] >= 0: a += ["--layer", str(m["layer"])]
if m.get("layers"): a += ["--layers", m["layers"]]
if m["alpha"]: a += ["--alpha", str(m["alpha"])]
if m["subset"]: a += ["--subset", str(m["subset"]), "--subset-seed", str(m["subset_seed"])]
if m["categories"]: a += ["--categories", m["categories"]]
print(" ".join(a))
EOF
)
  echo "== $(date +%H:%M:%S) rerun $run"
  $V eval/steer.py $args 2>&1 | grep -v "Warning\|Loading weights\|warnings.warn" || echo "FAILED-AGAIN $run"
done
echo "== $(date +%H:%M:%S) rerun done"
