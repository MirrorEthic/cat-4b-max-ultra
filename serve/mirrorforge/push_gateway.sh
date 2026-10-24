#!/usr/bin/env bash
# Push the public gateway to mirrorforge (SERVE node; DEV copies live here) and prove it (docs/GATEWAY.md section 6).
#
#   serve/mirrorforge/push_gateway.sh tunnel    # once, on legion: create the named tunnel + DNS, copy creds + config
#   serve/mirrorforge/push_gateway.sh push      # rsync gateway/ → ~/cat4b/gateway/app, venv, units, restart, /healthz
#   serve/mirrorforge/push_gateway.sh verify    # local /healthz + /api/status, then the same through the tunnel
#   serve/mirrorforge/push_gateway.sh smoke     # gateway/smoke.py through the tunnel (spends one render)
#   serve/mirrorforge/push_gateway.sh rollback  # previous app tree back, restart
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
HOST=mirrorforge
U=.config/systemd/user
BASE=/home/garret/cat4b/gateway
PUBLIC=https://cat4b.mirrorethic.com
TUNNEL=cat4b
HOSTNAME_=cat4b.mirrorethic.com
sh_remote() { ssh -o BatchMode=yes $HOST "$@"; }

case ${1:-} in
  tunnel)
    # Runs on legion, which holds the account's origin cert (~/.cloudflared/cert.pem). Idempotent.
    if ! cloudflared tunnel list 2>/dev/null | grep -q " $TUNNEL "; then
      cloudflared tunnel create $TUNNEL
    fi
    uuid=$(cloudflared tunnel list -o json 2>/dev/null | python3 -c 'import json,sys; print(next(t["id"] for t in json.load(sys.stdin) if t["name"]=="'$TUNNEL'"))')
    echo "tunnel $TUNNEL = $uuid"
    cloudflared tunnel route dns $TUNNEL $HOSTNAME_ 2>&1 | tail -n 1 || true
    sh_remote "mkdir -p ~/.cloudflared && chmod 700 ~/.cloudflared"
    scp -q ~/.cloudflared/$uuid.json $HOST:.cloudflared/$uuid.json
    { cat "$HERE/cloudflared-cat4b.yml"; echo "credentials-file: /home/garret/.cloudflared/$uuid.json"; } | sh_remote "cat > ~/.cloudflared/cat4b.yml && chmod 600 ~/.cloudflared/cat4b.yml ~/.cloudflared/$uuid.json && echo 'tunnel config + creds on the node'" ;;
  push)
    stamp=$(date +%Y%m%d-%H%M%S)
    echo "== stage the app tree ($stamp) =="
    sh_remote "mkdir -p $BASE/app $BASE/state $BASE/releases && [ -d $BASE/app/gateway ] && cp -a $BASE/app $BASE/releases/app-$stamp || true"
    rsync -az --delete --exclude '__pycache__' --exclude 'state' --exclude 'tests' --exclude '.pytest_cache' \
      "$REPO/gateway/" $HOST:$BASE/app/gateway/
    scp -q "$HERE/units/cat4b-gateway.service" $HOST:$U/cat4b-gateway.service
    scp -q "$HERE/units/cf-cat4b.service" $HOST:$U/cf-cat4b.service
    echo "== venv =="
    sh_remote "cd $BASE && ([ -x venv/bin/python ] || python3 -m venv venv) && venv/bin/pip install -q --upgrade pip && venv/bin/pip install -q -r app/gateway/requirements.txt && venv/bin/python -c 'import sys; sys.path.insert(0, \"$BASE/app\"); import gateway.app; print(\"gateway imports on the node\")'"
    echo "== secrets (created once, 0600) =="
    sh_remote "cd $BASE && umask 077 && [ -f state/secret ] || python3 -c 'import secrets; print(secrets.token_hex(32))' > state/secret; [ -f state/admin_token ] || python3 -c 'import secrets; print(secrets.token_hex(32))' > state/admin_token; [ -f env ] || printf 'GATEWAY_SECRET_FILE=%s/state/secret\nGATEWAY_ADMIN_TOKEN_FILE=%s/state/admin_token\n' $BASE $BASE > env; chmod 600 env state/secret state/admin_token; echo ok"
    echo "== restart =="
    sh_remote "systemctl --user daemon-reload && systemctl --user enable --now cat4b-gateway.service && systemctl --user restart cat4b-gateway.service && sleep 3 && systemctl --user is-active cat4b-gateway.service && ([ -f ~/.cloudflared/cat4b.yml ] && systemctl --user enable --now cf-cat4b.service && systemctl --user is-active cf-cat4b.service || echo 'no tunnel config yet: run tunnel first')"
    for i in $(seq 1 20); do
      if sh_remote "curl -fsS -m 3 http://127.0.0.1:8300/healthz" >/dev/null 2>&1; then echo "healthz ok after $((i*3)) s"; break; fi
      sleep 3
    done
    sh_remote "curl -s -m 5 http://127.0.0.1:8300/healthz; echo; journalctl --user -u cat4b-gateway --since '-2 min' --no-pager | tail -n 5 | cut -c1-160" ;;
  verify)
    echo "== on the node =="
    sh_remote "curl -s -m 5 http://127.0.0.1:8300/healthz; echo; curl -s -m 5 http://127.0.0.1:8300/api/status | python3 -c 'import json,sys; d=json.load(sys.stdin); print(\"window\", d[\"window\"], \"admissions\", d[\"admissions_open\"], \"upstream\", d[\"upstream\"], \"image eta\", d[\"lanes\"][\"image\"][\"eta_s\"])'"
    echo "== through the tunnel =="
    curl -sS -m 15 -o /dev/null -w "page %{http_code} %{time_total}s\n" $PUBLIC/
    curl -sS -m 15 $PUBLIC/api/status | python3 -c 'import json,sys; d=json.load(sys.stdin); print("status via tunnel:", d["window"], d["upstream"])'
    curl -sS -m 15 -o /dev/null -w "metrics via tunnel (expect 404): %{http_code}\n" $PUBLIC/metrics ;;
  smoke)
    GATEWAY=$PUBLIC "$REPO/.venv/bin/python" "$REPO/gateway/smoke.py" ;;
  rollback)
    sh_remote "cd $BASE && prev=\$(ls -td releases/app-* | head -1) && [ -n \"\$prev\" ] && rm -rf app && cp -a \"\$prev\" app && systemctl --user restart cat4b-gateway.service && sleep 3 && curl -s -m 5 http://127.0.0.1:8300/healthz && echo && echo \"rolled back to \$prev\"" ;;
  *) echo "usage: $0 tunnel|push|verify|smoke|rollback" >&2; exit 1 ;;
esac
