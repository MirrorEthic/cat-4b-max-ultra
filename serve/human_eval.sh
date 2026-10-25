#!/usr/bin/env bash
# The phone grading page for the developer human evaluation (eval/human_audit_web.py), as a background process on
# legion, LAN and tailnet only:  serve/human_eval.sh start|stop|status|url   [RUN=v0] [PORT=8304]
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-v0}; PORT=${PORT:-8304}
H=/media/garret-sutherland/dev-data/cat4b/state/eval/$RUN/human_audit
PY=${PY:-.venv/bin/python}
pid_alive() { [[ -f $1 ]] && kill -0 "$(cat "$1")" 2>/dev/null; }
url() {
  local tok; tok=$(cat "$H/web.token" 2>/dev/null || true)
  [[ -n $tok ]] || { echo "no token yet (start first)"; return 1; }
  for ip in $(hostname -I); do [[ $ip == 192.168.* || $ip == 100.* ]] && echo "http://$ip:$PORT/$tok/"; done
}
case ${1:-status} in
  start)
    if pid_alive "$H/web.pid"; then echo "already up (pid $(cat "$H/web.pid"))"; else
      setsid nohup "$PY" eval/human_audit_web.py --run "$RUN" --port "$PORT" >> "$H/web.log" 2>&1 &
      echo $! > "$H/web.pid"; sleep 1
      pid_alive "$H/web.pid" || { echo "did not start:"; tail -5 "$H/web.log"; exit 1; }
      echo "up (pid $(cat "$H/web.pid"))"
    fi; url ;;
  stop)
    if pid_alive "$H/web.pid"; then kill "$(cat "$H/web.pid")" && echo stopped; fi; rm -f "$H/web.pid" ;;
  status)
    if pid_alive "$H/web.pid"; then echo "up (pid $(cat "$H/web.pid"))"; url; else echo down; fi ;;
  url) url ;;
  *) echo "usage: $0 start|stop|status|url"; exit 2 ;;
esac
