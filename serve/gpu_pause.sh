#!/usr/bin/env bash
# Free the 5090: stop the catforge generator and the local teacher/judge. Work is resumable (serve/gpu_resume.sh).
set -uo pipefail
RUN=/media/garret-sutherland/dev-data/cat4b/run
if [[ -f $RUN/gen.pid ]] && kill -0 "$(cat $RUN/gen.pid)" 2>/dev/null; then
  pkill -P "$(cat $RUN/gen.pid)" 2>/dev/null; kill "$(cat $RUN/gen.pid)" 2>/dev/null; echo "generator stopped"
fi
rm -f $RUN/gen.pid
for p in $(pgrep -x llama-server); do
  if tr '\0' ' ' < /proc/$p/cmdline | grep -q -- '--port 829[01]'; then kill "$p"; echo "llama-server $p stopped"; fi
done
used="?"
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  [[ $used =~ ^[0-9]+$ ]] || { used="? (nvidia-smi failed)"; break; }
  (( used < 3000 )) && break; sleep 1
done
echo "5090: ${used} MiB in use"
