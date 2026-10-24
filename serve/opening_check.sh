#!/usr/bin/env bash
# The opening checklist of docs/LAUNCH_PLAN.md section 11.2, as one command that prints go / no-go lines. Read-only
# except for one real render through the public gateway when --smoke is given (spends a render on the node).
#
#   serve/opening_check.sh            # units, health, window, contracts, disk, hashes
#   serve/opening_check.sh --smoke    # + the full loop over https (session, chat, image, review, share)
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PUBLIC=https://cat4b.mirrorethic.com
ok() { printf "  [ok]   %s\n" "$1"; }
bad() { printf "  [NO]   %s\n" "$1"; FAIL=1; }
FAIL=0
echo "== mirrorforge units =="
U=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 mirrorforge 'for u in cat4b comfyui gpu-broker imagegen-mcp imagegen-training-watchdog cat4b-gateway cf-cat4b cf-mirage; do printf "%s=%s/%s " $u "$(systemctl --user is-active $u)" "$(systemctl --user show $u -p NRestarts --value)"; done; echo; nvidia-smi --query-gpu=index,memory.used,temperature.gpu --format=csv,noheader | tr "\n" ";"; echo; df -h /home | tail -n 1 | awk "{print \$4\" free (\"\$5\" used)\"}"; sha256sum ~/cat4b/models/cat4b-v0-merged-Q8_0.gguf ~/cat4b/models/cat4b-v0-mmproj-F16.gguf | cut -c1-16' 2>&1)
echo "$U" | sed 's/^/  /'
echo "$U" | grep -qE "inactive|failed" && bad "a unit is not active" || ok "all units active"
echo "$U" | grep -q "2a4c40477ef896e0" && ok "merged Q8 weights hash matches the release GGUF (2a4c4047…)" || bad "merged weights hash differs from the release GGUF"
echo "$U" | grep -q "6b705489be403438" && ok "vision projector hash matches (6b705489…)" || bad "projector hash differs"
echo "== gateway (node) =="
H=$(ssh -n -o BatchMode=yes mirrorforge 'curl -s -m 5 http://127.0.0.1:8300/healthz' 2>/dev/null); echo "  $H"
echo "$H" | grep -q '"ok":true' && ok "healthz ok on the node" || bad "healthz not ok"
echo "== public (through the tunnel) =="
code=$(curl -sS -m 15 -o /dev/null -w "%{http_code}" $PUBLIC/ 2>/dev/null); [ "$code" = 200 ] && ok "page 200 over https" || bad "page $code"
S=$(curl -sS -m 15 $PUBLIC/api/status 2>/dev/null)
python3 - "$S" <<'EOF'
import json, sys, datetime, time
d = json.loads(sys.argv[1]); f = lambda t: datetime.datetime.fromtimestamp(t).astimezone().strftime("%a %b %d %H:%M %Z") if t else "unset"
print(f"  window {d['window']} | admissions_open {d['admissions_open']} | opens {f(d.get('opens_at'))} | closes {f(d.get('closes_at'))}")
print(f"  upstream {d['upstream']} | chat lane workers {d['lanes']['chat']['workers']} room {d['lanes']['chat']['room']} | image lane workers {d['lanes']['image']['workers']} eta {d['lanes']['image']['eta_s']} s")
print(f"  limits {d['limits']} | controls {d['controls']}")
now = time.time()
if d.get('opens_at') and now < d['opens_at']:
    print(f"  opens in {(d['opens_at']-now)/3600:.1f} h")
EOF
echo "$S" | grep -qE '"cat": ?true' && ok "cat upstream up (public view)" || bad "cat upstream down"
echo "$S" | grep -qE '"mirage": ?true' && ok "mirage upstream up (public view)" || bad "mirage upstream down"
m=$(curl -sS -m 15 -o /dev/null -w "%{http_code}" $PUBLIC/metrics 2>/dev/null); [ "$m" = 404 ] && ok "/metrics hidden from outside (404)" || bad "/metrics exposed ($m)"
a=$(curl -sS -m 15 -o /dev/null -w "%{http_code}" -X POST $PUBLIC/admin -H 'Content-Type: application/json' -d '{"action":"counts"}' 2>/dev/null); [ "$a" = 404 ] && ok "/admin hidden from outside (404)" || bad "/admin reachable ($a)"
echo "== watchdog (pi5) =="
L=$(timeout 20 ssh -n -o BatchMode=yes -o ConnectTimeout=8 watcher@192.168.0.102 'tail -n 400 /mnt/storage/serve-watchdog/serve_liveness.jsonl | grep -E "cat4b-(llm|gateway|tunnel)" | tail -n 3 | sed -E "s/.*\"service\": \"([^\"]+)\".*\"ok\": (true|false).*\"fail_count\": ([0-9]+).*/  \1: ok=\2 fails=\3/"' 2>&1); echo "$L"
n=$(echo "$L" | grep -c "ok=true"); [ "$n" = 3 ] && ok "all three cat4b contracts healthy on the pi5 watchdog" || bad "pi5 watchdog: $n of 3 cat4b contracts healthy ($L)"
echo "== staging (legion) =="
bash "$REPO/serve/staging.sh" status 2>/dev/null | sed -n 1,2p | sed 's/^/  /'
if [[ ${1:-} == "--smoke" ]]; then
  echo "== smoke over https (one render) =="
  GATEWAY=$PUBLIC "$REPO/.venv/bin/python" "$REPO/gateway/smoke.py" 2>&1 | sed 's/^/  /' | tail -n 9
fi
echo
[ $FAIL = 0 ] && echo "GO: every check passed" || echo "NO-GO: see the [NO] lines"
