#!/usr/bin/env bash
# Run a catforge model (judge or teacher) on mirrorforge's 2×V100, the sanctioned way: hold the GPU broker's "free"
# lease (it stops ComfyUI/FLUX and keeps the pool empty), renew it while the server runs, release it on stop so the
# broker restores FLUX. While we hold the lease, imagegen leases get 409 (mirage images unavailable).
#
#   mirrorforge_llama.sh start judge|teacher|cat     mirrorforge_llama.sh stop     mirrorforge_llama.sh status
#
# Runs ON mirrorforge. Serves on the legion↔mirrorforge 10 GbE link only (10.10.10.2), not the LAN.
#
# cat = the merged v0 weights (models/cat4b-v0-merged-Q8_0.gguf + the vision projector) as "cat4b-vl" on :8292,
# one card only (CUDA_VISIBLE_DEVICES=0, the live-service layout of docs/LIVE_SERVICE.md section 5), PAR slots of
# 16k context (PAR=8 by default; KV_TYPE=f16 by default, q8_0 halves the cache), thinking off. Still under the
# "free" lease for now: ComfyUI is down while it runs. The resident, lease-free form comes with the broker change.
set -uo pipefail
BASE=$HOME/cat4b
LLAMA=$BASE/tools/llama.cpp-b11139/build-sm70/bin/llama-server
BROKER=http://localhost:8077
RUN=$BASE/run; LOG=$BASE/logs
mkdir -p "$RUN" "$LOG"

lease() { curl -s -m 30 -X POST "$BROKER/$1" -H 'Content-Type: application/json' -d "$2"; }

start() {
  local role=$1 model port ctx par extra="" devs=""
  case "$role" in
    judge)   model=$BASE/models/gemma-4-31B-it-Q4_K_M.gguf; port=8291; par=4; ctx=$((4 * 10240))
             extra="-sm layer -ctk q8_0 -ctv q8_0 --reasoning-format deepseek --cache-ram 4096" ;;
    teacher) model=$BASE/models/Qwen3.8-27B-Q6_K.gguf;     port=8290; par=4; ctx=$((4 * 12288))
             extra="-sm layer -ctk q8_0 -ctv q8_0 --reasoning-format deepseek --cache-ram 4096" ;;
    cat)     model=$BASE/models/cat4b-v0-merged-Q8_0.gguf; port=8292; par=${PAR:-8}; ctx=$((par * 16384)); devs=0
             extra="--mmproj $BASE/models/cat4b-v0-mmproj-F16.gguf -a cat4b-vl --reasoning off --seed 0 \
                    -ctk ${KV_TYPE:-f16} -ctv ${KV_TYPE:-f16}" ;;
    *) echo "usage: $0 start judge|teacher|cat" >&2; exit 1 ;;
  esac
  if [[ -f $RUN/server.pid ]] && kill -0 "$(cat $RUN/server.pid)" 2>/dev/null; then
    echo "already running: $(cat $RUN/role) pid $(cat $RUN/server.pid)"; exit 0
  fi
  r=$(lease lease '{"capability": "free", "ttl_sec": 900}')
  lid=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(d.get("lease_id","")) if d.get("ok") else print("")' "$r")
  [[ -n $lid ]] || { echo "broker refused the free lease: $r" >&2; exit 2; }
  echo "$lid" > "$RUN/lease_id"; echo "$role" > "$RUN/role"
  for i in $(seq 1 30); do   # wait for FLUX to actually leave the cards
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    (( used < 1500 )) && break; sleep 2
  done
  # the judge/teacher keep their alias = role; the cat role sets its own alias (-a cat4b-vl) in $extra, later wins
  CUDA_VISIBLE_DEVICES=${devs:-0,1} setsid nohup "$LLAMA" -m "$model" -a "$role" --host 10.10.10.2 --port "$port" \
    -ngl 99 -c "$ctx" --parallel "$par" -fa on --jinja --metrics --slots $extra \
    >> "$LOG/llama_$role.log" 2>&1 < /dev/null &
  echo $! > "$RUN/server.pid"
  # renew the lease every 5 min for as long as the server lives; release when it exits
  setsid nohup bash -c "while kill -0 $(cat $RUN/server.pid) 2>/dev/null; do
      curl -s -m 30 -X POST $BROKER/renew -H 'Content-Type: application/json' \
        -d '{\"lease_id\": \"$lid\", \"ttl_sec\": 900}' >/dev/null; sleep 300; done
    curl -s -m 30 -X POST $BROKER/release -H 'Content-Type: application/json' -d '{\"lease_id\": \"$lid\"}'" \
    >> "$LOG/lease.log" 2>&1 < /dev/null &
  echo $! > "$RUN/renew.pid"
  for i in $(seq 1 120); do
    curl -sf "http://10.10.10.2:$port/health" >/dev/null && { echo "$role up on 10.10.10.2:$port (lease $lid)"; exit 0; }
    kill -0 "$(cat $RUN/server.pid)" 2>/dev/null || break
    sleep 3
  done
  echo "$role failed to start; see $LOG/llama_$role.log" >&2; stop; exit 1
}

stop() {
  [[ -f $RUN/server.pid ]] && kill "$(cat $RUN/server.pid)" 2>/dev/null
  [[ -f $RUN/renew.pid ]] && kill "$(cat $RUN/renew.pid)" 2>/dev/null
  if [[ -f $RUN/lease_id ]]; then
    echo "release: $(lease release "{\"lease_id\": \"$(cat $RUN/lease_id)\"}")"
  fi
  rm -f "$RUN/server.pid" "$RUN/renew.pid" "$RUN/lease_id" "$RUN/role"
}

status() {
  if [[ -f $RUN/server.pid ]] && kill -0 "$(cat $RUN/server.pid)" 2>/dev/null; then
    echo "running: $(cat $RUN/role) pid $(cat $RUN/server.pid) lease $(cat $RUN/lease_id 2>/dev/null)"
  else echo "not running"; fi
  curl -s -m 5 "$BROKER/status"; echo
}

case "${1:-}" in
  start) start "${2:-}" ;;
  stop) stop ;;
  status) status ;;
  *) echo "usage: $0 start judge|teacher|cat | stop | status" >&2; exit 1 ;;
esac
