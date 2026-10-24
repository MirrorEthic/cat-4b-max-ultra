#!/usr/bin/env bash
# Push the DEV copy of mirrorforge's comfyui.service (serve/mirrorforge/comfyui.service) to the SERVE node and
# restart ComfyUI through systemd, then prove the stack still renders: mirage health, and one FLUX image through
# mirage's generate_image under an imagegen lease. Keeps a timestamped backup of the previous unit for rollback.
#
#   serve/mirrorforge/push_comfyui_unit.sh push      # backup, copy, daemon-reload, restart, wait for /system_stats
#   serve/mirrorforge/push_comfyui_unit.sh verify    # mirage /health + one FLUX render through the MCP
#   serve/mirrorforge/push_comfyui_unit.sh rollback  # restore the newest backup
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
HOST=mirrorforge
UNIT=.config/systemd/user/comfyui.service
BROKER=http://10.10.10.2:8077
MIRAGE=http://10.10.10.2:8190

case ${1:-} in
  push)
    stamp=$(date +%Y%m%d-%H%M%S)
    ssh -o BatchMode=yes $HOST "cp ~/$UNIT ~/$UNIT.bak-$stamp && echo backup ~/$UNIT.bak-$stamp"
    scp -q "$HERE/comfyui.service" $HOST:$UNIT
    ssh -o BatchMode=yes $HOST 'systemctl --user daemon-reload && systemctl --user restart comfyui.service && sleep 3 && systemctl --user is-active comfyui.service'
    for i in $(seq 1 60); do
      if ssh -o BatchMode=yes $HOST 'curl -s -m 3 http://127.0.0.1:8188/system_stats' | grep -q pytorch_version; then
        ssh -o BatchMode=yes $HOST 'curl -s -m 3 http://127.0.0.1:8188/system_stats' | python3 -c 'import json,sys; d=json.load(sys.stdin)["system"]; print("ComfyUI", d["comfyui_version"], "torch", d["pytorch_version"])'
        exit 0
      fi
      sleep 3
    done
    echo "ComfyUI did not answer within 3 min; journalctl --user -u comfyui -n 30 on $HOST" >&2; exit 1 ;;
  verify)
    curl -s -m 5 $MIRAGE/health; echo
    r=$(curl -s -m 60 -X POST $BROKER/lease -H 'Content-Type: application/json' -d '{"capability": "imagegen", "ttl_sec": 300}')
    lid=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(d.get("lease_id","")) if d.get("ok") else print("")' "$r")
    [[ -n $lid ]] || { echo "no imagegen lease: $r" >&2; exit 2; }
    body='{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"generate_image","arguments":{"prompt":"a grey tabby cat asleep on a stack of server manuals, soft window light, photograph","width":1024,"height":768,"steps":20,"cfg":3.5,"seed":1}}}'
    t0=$(date +%s)
    out=$(curl -s -m 400 -X POST $MIRAGE/mcp -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' -d "$body")
    echo "render took $(( $(date +%s) - t0 )) s"
    echo "$out" | grep -o -E '/[^"\\ ]+\.(png|jpe?g|webp)' | head -1 || { echo "no image path in the response: ${out:0:300}" >&2; }
    curl -s -m 30 -X POST $BROKER/release -H 'Content-Type: application/json' -d "{\"lease_id\": \"$lid\"}"; echo ;;
  rollback)
    ssh -o BatchMode=yes $HOST 'b=$(ls -t ~/.config/systemd/user/comfyui.service.bak-* | head -1); cp "$b" ~/.config/systemd/user/comfyui.service && systemctl --user daemon-reload && systemctl --user restart comfyui.service && echo "restored $b"' ;;
  *) echo "usage: $0 push|verify|rollback" >&2; exit 1 ;;
esac
