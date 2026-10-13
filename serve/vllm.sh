#!/usr/bin/env bash
# CAT-4B MAX ULTRA serving recipe (docs/BUILD_PLAN.md "Serving"). One vLLM endpoint, two model names:
#   qwen3.5-4b-vl  base, adapter off (condition D / A–B baselines)
#   cat4b-vl       base + CAT LoRA
# The "-vl" suffix is load-bearing: group-chat's _is_multimodal() is a substring match on the model name.
#
#   serve/vllm.sh start [adapter_dir]    serve/vllm.sh stop    serve/vllm.sh status
#
# The bot sends no chat_template_kwargs, so thinking is off server-side (--default-chat-template-kwargs); the eval can
# still turn it on per request (condition E). Images: group-chat sends up to 3 recent images plus up to 3 "review" images
# of what the model just generated, so 6 per prompt; each is capped at 1 MP (1024 visual tokens) so an uploaded phone
# photo can't eat the context. The vision training slice (not built yet) must render images with this same cap.
#
# Respects the legion GPU lock. Not a standing service: bring it up for eval/demo, tear it down after.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
CAT=/media/garret-sutherland/dev-data/cat4b
VLLM=${VLLM:-$CAT/venvs/serve-vllm/bin/vllm}
ADAPTER=${2:-${ADAPTER:-$CAT/adapters/cat4b-v0}}
PORT=${PORT:-8011}
HOST=${HOST:-127.0.0.1}
GPU_UTIL=${GPU_UTIL:-0.6}   # leaves the comfyui-fluxfill worker its share of the 5090
GPU_LOCK=$HOME/CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED
RUN=$CAT/run; LOG=$CAT/logs
export HF_HOME=${HF_HOME:-/media/garret-sutherland/dev-data/huggingface-cache}
mkdir -p "$RUN" "$LOG"

start() {
  [[ -e $GPU_LOCK ]] && { echo "GPU lock present ($GPU_LOCK); not starting" >&2; exit 2; }
  if [[ -f $RUN/vllm.pid ]] && kill -0 "$(cat $RUN/vllm.pid)" 2>/dev/null; then
    echo "already running: pid $(cat $RUN/vllm.pid)"; exit 0
  fi
  [[ -f $ADAPTER/adapter_config.json ]] || { echo "no adapter at $ADAPTER" >&2; exit 1; }
  echo "adapter $ADAPTER sha256 $(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -c1-16)"
  setsid nohup "$VLLM" serve Qwen/Qwen3.5-4B \
    --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
    --served-model-name qwen3.5-4b-vl \
    --enable-lora --max-lora-rank 32 --max-loras 1 \
    --lora-modules "cat4b-vl=$ADAPTER" \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --limit-mm-per-prompt '{"image": 6, "video": 0}' \
    --mm-processor-kwargs '{"size": {"longest_edge": 1048576, "shortest_edge": 65536}}' \
    --max-model-len 16384 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --seed 0 \
    --host "$HOST" --port "$PORT" \
    >> "$LOG/vllm.log" 2>&1 < /dev/null &
  echo $! > "$RUN/vllm.pid"
  for i in $(seq 1 200); do
    curl -sf "http://$HOST:$PORT/v1/models" >/dev/null && {
      echo "up on $HOST:$PORT: $(curl -s http://$HOST:$PORT/v1/models | python3 -c 'import json,sys; print(" ".join(m["id"] for m in json.load(sys.stdin)["data"]))')"
      exit 0; }
    kill -0 "$(cat $RUN/vllm.pid)" 2>/dev/null || break
    sleep 3
  done
  echo "vLLM failed to start; see $LOG/vllm.log" >&2; stop; exit 1
}

stop() {
  if [[ -f $RUN/vllm.pid ]]; then
    pid=$(cat "$RUN/vllm.pid")
    kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null   # setsid made it a group leader: take the workers too
    for i in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    rm -f "$RUN/vllm.pid"; echo stopped
  else echo "not running"; fi
}

status() {
  if [[ -f $RUN/vllm.pid ]] && kill -0 "$(cat $RUN/vllm.pid)" 2>/dev/null; then
    echo "running: pid $(cat $RUN/vllm.pid)"; curl -s "http://$HOST:$PORT/v1/models"; echo
  else echo "not running"; fi
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "usage: $0 start [adapter_dir] | stop | status" >&2; exit 1 ;;
esac
