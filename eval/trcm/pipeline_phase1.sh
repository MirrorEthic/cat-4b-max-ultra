#!/usr/bin/env bash
# TRCM Phase 1 extraction chain (docs/TRCM_PHASE0.md §4; Garret's go 2026-10-27 09:16 "Lets do phase 1").
# Waits for an optional pid (a GPU chain ahead of it), then starts the local teacher, extracts the 650 manifest
# packets blinded, stops the teacher by its explicit pid, and prints a tally. Never starts under the GPU lock;
# llama_local.sh itself also refuses a busy card.
#
#   setsid nohup bash eval/trcm/pipeline_phase1.sh [wait_pid] >> /media/garret-sutherland/dev-data/cat4b/state/eval/trcm/phase1.log 2>&1 &
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
LOCK=~/CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED
TRCM=/media/garret-sutherland/dev-data/cat4b/state/eval/trcm
cd "$REPO"
while [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; do sleep 60; done
[ -e "$LOCK" ] && { echo "== $(date +%H:%M:%S) GPU lock present; stopping"; exit 3; }
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 3000 ] && break
  [ "$i" = 60 ] && { echo "== $(date +%H:%M:%S) GPU still busy (${used} MiB) after 60 min; stopping"; exit 4; }
  sleep 60
done
echo "== $(date +%H:%M:%S) starting teacher"
setsid bash serve/llama_local.sh teacher & TPID=$!
for i in $(seq 1 90); do
  curl -fsS -m 3 http://127.0.0.1:8290/health >/dev/null 2>&1 && break
  kill -0 "$TPID" 2>/dev/null || { echo "teacher exited early (see logs/llama_teacher.log)"; exit 5; }
  [ "$i" = 90 ] && { echo "teacher not healthy after 15 min"; kill "$TPID"; exit 5; }
  sleep 10
done
echo "== $(date +%H:%M:%S) teacher up (pid $TPID); extracting"
.venv/bin/python eval/trcm/extract_teacher.py
rc=$?
echo "== $(date +%H:%M:%S) stopping teacher pid $TPID (rc=$rc)"
kill "$TPID" 2>/dev/null
sleep 5
kill -0 "$TPID" 2>/dev/null && kill -9 "$TPID" 2>/dev/null
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path("/media/garret-sutherland/dev-data/cat4b/state/eval/trcm/extractions.jsonl")
rows = [json.loads(l) for l in open(p)] if p.exists() else []
ok = [r for r in rows if r.get("ok")]
anch = sum(1 for r in ok for v in (r.get("span_offsets") or {}).values() if v)
quotes = sum(1 for r in ok for v in (r.get("extraction", {}).get("spans") or {}).values() if isinstance(v, str))
print(f"extractions {len(rows)}, valid {len(ok)}, spans quoted {quotes}, anchored {anch}")
PY
echo "== $(date +%H:%M:%S) phase 1 extraction done"
