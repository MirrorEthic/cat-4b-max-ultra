#!/usr/bin/env bash
# Push the live-service co-residency set to mirrorforge (SERVE node; everything here is a DEV copy) and prove it:
#   units/  comfyui.service (ComfyUI pinned to V100 #1), cat4b.service (the cat model on #0),
#           imagegen-mcp.service (mirage env: Z-Image backend, single-card devices), the watchdog unit (unchanged)
#   extras/ comfy_client.py (Z-Image backend), training_watchdog.py (ignores cat4b.service, start-up grace)
#   broker  openclaw-cognitive/fab/gpu_broker/gpu_broker.py (per-card residency, cat + imagegen resident)
#
#   serve/mirrorforge/push_live_service.sh push      # backups, copy, daemon-reload, restarts, readiness
#   serve/mirrorforge/push_live_service.sh verify    # broker state, cat chat, mirage health, one Z-Image render
#                                                    #   under an imagegen lease, cat still up afterwards
#   serve/mirrorforge/push_live_service.sh rollback  # newest backups back, FLUX layout, old broker
#
# Run `bash ~/cat4b/mirrorforge_llama.sh stop` on mirrorforge first if the lease-based cat role is up.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
HOST=mirrorforge
BROKER_DEV=/home/garret-sutherland/openclaw-cognitive/fab/gpu_broker/gpu_broker.py
U=.config/systemd/user
BROKER=http://10.10.10.2:8077
MIRAGE=http://10.10.10.2:8190
CAT=http://10.10.10.2:8292

sh_remote() { ssh -o BatchMode=yes $HOST "$@"; }

case ${1:-} in
  push)
    stamp=$(date +%Y%m%d-%H%M%S)
    echo "== backups ($stamp) =="
    sh_remote "set -e; for f in $U/comfyui.service $U/imagegen-mcp.service $U/imagegen-training-watchdog.service comfyui/extras/comfy_client.py comfyui/extras/training_watchdog.py gpu_broker/gpu_broker.py; do [ -f ~/\$f ] && cp ~/\$f ~/\$f.bak-$stamp; done; [ -f ~/$U/cat4b.service ] && cp ~/$U/cat4b.service ~/$U/cat4b.service.bak-$stamp || true; echo ok"
    echo "== copy =="
    scp -q "$HERE/comfyui.service" $HOST:$U/comfyui.service
    scp -q "$HERE/cat4b.service" $HOST:$U/cat4b.service
    scp -q "$HERE/units/imagegen-mcp.service" $HOST:$U/imagegen-mcp.service
    scp -q "$HERE/extras/comfy_client.py" $HOST:comfyui/extras/comfy_client.py
    scp -q "$HERE/extras/training_watchdog.py" $HOST:comfyui/extras/training_watchdog.py
    scp -q "$BROKER_DEV" $HOST:gpu_broker/gpu_broker.py
    sh_remote "cd ~/comfyui/extras && python3 -m py_compile comfy_client.py training_watchdog.py && python3 -m py_compile ~/gpu_broker/gpu_broker.py && echo 'remote compiles'"
    echo "== restart in order: watchdog, comfyui (card 1), cat4b (card 0), imagegen-mcp, broker =="
    sh_remote "systemctl --user daemon-reload && systemctl --user restart imagegen-training-watchdog.service && systemctl --user restart comfyui.service && systemctl --user enable --now cat4b.service && systemctl --user restart imagegen-mcp.service && systemctl --user restart gpu-broker.service && sleep 3 && systemctl --user is-active imagegen-training-watchdog comfyui cat4b imagegen-mcp gpu-broker | tr '\n' ' '; echo"
    for i in $(seq 1 60); do
      c=$(curl -s -m 3 $CAT/health 2>/dev/null | grep -c '"ok"' || true)
      m=$(sh_remote 'curl -s -m 3 http://127.0.0.1:8188/system_stats' 2>/dev/null | grep -c pytorch_version || true)
      if [[ $c == 1 && $m == 1 ]]; then echo "cat and ComfyUI answer after $((i*5)) s"; break; fi
      sleep 5
    done
    curl -s -m 5 $BROKER/status | python3 -c 'import json,sys; d=json.load(sys.stdin); print("broker loaded:", d.get("loaded"), "lease:", d.get("lease")); print(d.get("gpus"))'
    sh_remote 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader' ;;
  verify)
    echo "== cat chat =="
    curl -s -m 120 $CAT/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"cat4b-vl","messages":[{"role":"user","content":"One sentence: what is a VLAN?"}],"max_tokens":80,"temperature":0.7}' | python3 -c 'import json,sys; d=json.load(sys.stdin); m=d["choices"][0]["message"]; t=d.get("timings",{}); print(repr(m.get("content","")[:200])); print("decode tok/s", round(t.get("predicted_per_second",0)))'
    echo "== mirage health =="; curl -s -m 5 $MIRAGE/health; echo
    echo "== one render through mirage under an imagegen lease (Z-Image on card 1) =="
    r=$(curl -s -m 120 -X POST $BROKER/lease -H 'Content-Type: application/json' -d '{"capability": "imagegen", "ttl_sec": 300}')
    echo "lease: $r"
    lid=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(d.get("lease_id","")) if d.get("ok") else print("")' "$r")
    [[ -n $lid ]] || { echo "no imagegen lease" >&2; exit 2; }
    body='{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"generate_image","arguments":{"prompt":"a grey tabby cat asleep on a stack of server manuals, soft window light, photograph","width":1024,"height":768,"steps":20,"cfg":3.5,"seed":1}}}'
    t0=$(date +%s); out=$(curl -s -m 400 -X POST $MIRAGE/mcp -H 'Content-Type: application/json' -d "$body"); echo "render took $(( $(date +%s) - t0 )) s"
    echo "$out" | grep -o -E '/[^"\\ ]+\.(png|jpe?g|webp)' | head -1 || echo "no image path in the response: ${out:0:300}"
    echo "cat still up during the lease: $(curl -s -m 3 $CAT/health)"
    curl -s -m 30 -X POST $BROKER/release -H 'Content-Type: application/json' -d "{\"lease_id\": \"$lid\"}"; echo
    curl -s -m 5 $BROKER/status | python3 -c 'import json,sys; d=json.load(sys.stdin); print("broker loaded after release:", d.get("loaded"))'
    sh_remote 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; journalctl --user -u imagegen-training-watchdog --since "-10 min" --no-pager | tail -3 | cut -c1-160' ;;
  rollback)
    sh_remote 'set -e; cd ~; for f in .config/systemd/user/comfyui.service .config/systemd/user/imagegen-mcp.service comfyui/extras/comfy_client.py comfyui/extras/training_watchdog.py gpu_broker/gpu_broker.py; do b=$(ls -t $f.bak-* 2>/dev/null | head -1); [ -n "$b" ] && cp "$b" "$f" && echo "restored $f from $b"; done; systemctl --user disable --now cat4b.service || true; systemctl --user daemon-reload; systemctl --user restart imagegen-training-watchdog comfyui imagegen-mcp gpu-broker; echo rolled back' ;;
  *) echo "usage: $0 push|verify|rollback" >&2; exit 1 ;;
esac
