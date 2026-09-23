#!/usr/bin/env bash
# Restart the local teacher and the generator where it left off (already-generated specs are skipped).
#   serve/gpu_resume.sh [extra gen-local args, e.g. --best-of 2 --no-thinking]
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=/media/garret-sutherland/dev-data/cat4b/run; LOG=/media/garret-sutherland/dev-data/cat4b/logs
mkdir -p $RUN
nohup serve/llama_local.sh teacher >/dev/null 2>&1 &
for i in $(seq 1 100); do curl -sf http://127.0.0.1:8290/health >/dev/null && break; sleep 3; done
curl -sf http://127.0.0.1:8290/health >/dev/null || { echo "teacher failed to start; see $LOG/llama_teacher.log" >&2; exit 1; }
nohup env PYTHONPATH=data uv run python -m catforge.cli gen-local --hours 72 "$@" >> $LOG/gen_local.log 2>&1 &
echo $! > $RUN/gen.pid
echo "teacher up; generator pid $(cat $RUN/gen.pid)"
