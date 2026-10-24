#!/usr/bin/env bash
# Internal replica of the public path for Garret to test before deployment (docs/GATEWAY.md section 6): the same
# gateway code on legion, bound to the LAN, with its own state and the real mirrorforge backends (llama-server,
# mirage, broker over the 10 GbE link), plus the mirrorethic.com bundle served locally with its CAT-4B links pointed at
# the staging gateway. Not a mesh service (transient, DEV node, not public): start it, test, stop it.
#
#   serve/staging.sh start     # gateway on http://<legion LAN ip>:8301, site on http://<legion LAN ip>:8302
#   serve/staging.sh status
#   serve/staging.sh stop
#
# Env knobs are the production ones (gateway/config.py) except GATEWAY_IP_SESSIONS_PER_HOUR=20 (testing from one
# address) and an http public base so the session cookie works without TLS. Override anything by exporting it first.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
V=/media/garret-sutherland/dev-data/cat4b/venvs/gateway
STAGE=/media/garret-sutherland/dev-data/cat4b/gateway-staging
SITE_SRC=$HOME/mirrorethic-site/dist
IP=${STAGING_IP:-$(hostname -I | tr ' ' '\n' | grep -E '^192\.168\.' | head -n 1)}
GW_PORT=${STAGING_GW_PORT:-8301}
SITE_PORT=${STAGING_SITE_PORT:-8302}
mkdir -p "$STAGE/state" "$STAGE/site" "$STAGE/logs"

pid_alive() { [[ -f $1 ]] && kill -0 "$(cat "$1")" 2>/dev/null; }

start() {
  if pid_alive "$STAGE/gateway.pid"; then echo "gateway already up (pid $(cat "$STAGE/gateway.pid"))"; else
    ( cd "$REPO" && export GATEWAY_STATE="$STAGE/state" GATEWAY_PUBLIC_BASE="http://$IP:$GW_PORT" \
        GATEWAY_IP_SESSIONS_PER_HOUR=${GATEWAY_IP_SESSIONS_PER_HOUR:-20} GATEWAY_HEALTH_INTERVAL_S=${GATEWAY_HEALTH_INTERVAL_S:-10} \
      && exec "$V/bin/uvicorn" gateway.app:app --host 0.0.0.0 --port "$GW_PORT" --log-level info \
        >> "$STAGE/logs/gateway.log" 2>&1 < /dev/null ) &
    echo $! > "$STAGE/gateway.pid"
  fi
  # the site bundle, with the public host swapped for the staging gateway
  rsync -a --delete "$SITE_SRC/" "$STAGE/site/"
  grep -rl "https://cat4b.mirrorethic.com" "$STAGE/site" | xargs -r sed -i "s|https://cat4b.mirrorethic.com|http://$IP:$GW_PORT|g"
  if pid_alive "$STAGE/site.pid"; then echo "site already up (pid $(cat "$STAGE/site.pid"))"; else
    ( cd "$STAGE/site" && exec python3 -m http.server "$SITE_PORT" --bind 0.0.0.0 >> "$STAGE/logs/site.log" 2>&1 < /dev/null ) &
    echo $! > "$STAGE/site.pid"
  fi
  for i in $(seq 1 20); do curl -sf -m 2 "http://127.0.0.1:$GW_PORT/healthz" >/dev/null && break; sleep 1; done
  status
}

stop() {
  for n in gateway site; do
    if pid_alive "$STAGE/$n.pid"; then kill "$(cat "$STAGE/$n.pid")" && echo "stopped $n"; fi
    rm -f "$STAGE/$n.pid"
  done
}

status() {
  for n in gateway site; do
    if pid_alive "$STAGE/$n.pid"; then echo "$n: up (pid $(cat "$STAGE/$n.pid"))"; else echo "$n: down"; fi
  done
  echo "gateway  http://$IP:$GW_PORT/        healthz: $(curl -s -m 3 "http://127.0.0.1:$GW_PORT/healthz" || echo unreachable)"
  echo "site     http://$IP:$SITE_PORT/       (CAT-4B links → the staging gateway; /cat4b/ is the entity page)"
  echo "state    $STAGE/state  (its own sessions, jobs and renders; production state is on mirrorforge)"
}

case ${1:-} in
  start) start ;; stop) stop ;; status) status ;;
  *) echo "usage: $0 start|stop|status" >&2; exit 1 ;;
esac
