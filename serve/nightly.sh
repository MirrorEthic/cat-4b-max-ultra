#!/usr/bin/env bash
# Nightly generation window for the 5090 (cron): `nightly.sh start` at 21:00, `nightly.sh stop` at 09:00.
# cron has a bare environment, so set what gpu_resume/llama_local need. Respects .GPU_LOCKED and a busy card
# (llama_local.sh refuses to start), and a manual skip: touch /media/garret-sutherland/dev-data/cat4b/run/SKIP_NIGHTLY
set -uo pipefail
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin" HF_HOME=/media/garret-sutherland/dev-data/huggingface-cache
cd "$(dirname "$0")/.."
LOG=/media/garret-sutherland/dev-data/cat4b/logs/nightly.log
RUN=/media/garret-sutherland/dev-data/cat4b/run
echo "[$(date '+%F %T')] nightly $1" >> "$LOG"
case "$1" in
  start)
    if [[ -e $RUN/SKIP_NIGHTLY ]]; then echo "  SKIP_NIGHTLY present; not starting" >> "$LOG"; exit 0; fi
    serve/gpu_resume.sh >> "$LOG" 2>&1 ;;
  stop)
    serve/gpu_pause.sh >> "$LOG" 2>&1 ;;
  *) echo "usage: nightly.sh start|stop" >&2; exit 1 ;;
esac
