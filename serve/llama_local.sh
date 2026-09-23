#!/usr/bin/env bash
# Launch a local teacher or judge on legion's 5090 via llama.cpp (docs/DATA_AND_TRAINING.md §4.2-4.3).
#   serve/llama_local.sh teacher   → Qwen3.8-27B UD-Q5_K_M on :8290
#   serve/llama_local.sh judge     → Gemma-4-31B-it Q4_K_M on :8291  (809x is taken by mesh services)
# One at a time on the 5090 (24 GB). Refuses to start under the GPU lock or while another job holds the card.
set -euo pipefail

ROLE="${1:?usage: llama_local.sh teacher|judge}"
LLAMA=/media/garret-sutherland/dev-data/cat4b/tools/llama.cpp-b11139/build-sm120/bin/llama-server
HUB=/media/garret-sutherland/dev-data/huggingface-cache/hub
LOG=/media/garret-sutherland/dev-data/cat4b/logs
LOCK="$HOME/CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED"

if [[ -e "$LOCK" ]]; then echo "GPU LOCKED: $(head -3 "$LOCK")" >&2; exit 2; fi
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if (( used > 3000 )); then
  echo "5090 already holds ${used} MiB (another job?) — not starting." >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2
  exit 3
fi

case "$ROLE" in
  teacher)
    MODEL=$(ls "$HUB"/models--unsloth--Qwen3.8-27B-GGUF/snapshots/*/Qwen3.8-27B-UD-Q5_K_M.gguf)
    PORT=8290; ALIAS=qwen3.8-27b-teacher; PAR=${PAR:-4}; CTX=$((PAR * 12288))   # 8 slots OOM: ~600 MB recurrent state per slot
    SPEC=${SPEC:-"--spec-type draft-mtp --spec-draft-n-max 3"} ;;   # MTP head drafts 3 tokens
  judge)
    MODEL=$(ls "$HUB"/models--unsloth--gemma-4-31B-it-GGUF/snapshots/*/gemma-4-31B-it-Q4_K_M.gguf)
    PORT=8291; ALIAS=gemma-4-31b-judge; PAR=${PAR:-4}; CTX=$((PAR * 8192)); SPEC="" ;;
  *) echo "unknown role $ROLE" >&2; exit 1 ;;
esac

exec "$LLAMA" -m "$MODEL" -a "$ALIAS" --host 127.0.0.1 --port "$PORT" \
  -ngl 99 -c "$CTX" --parallel "$PAR" -fa on -ctk q8_0 -ctv q8_0 \
  --jinja --reasoning-format deepseek --cache-ram 4096 --metrics --slots $SPEC \
  >> "$LOG/llama_${ROLE}.log" 2>&1
