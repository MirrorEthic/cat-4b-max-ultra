#!/usr/bin/env bash
# Stage 1 deploy + smoke for the Z Fold 6 (docs/MOBILE.md). Pushes the arm64 llama.cpp binaries and
# the Q4_K_M merged model to /data/local/tmp/cat4b, then runs llama-bench (prefill + decode speed)
# and one chat turn under the served system prompts. Phone must be on USB with debugging authorized
# (or `adb pair`/`adb connect` over Wi-Fi first).
#
#   bash mobile/push_phone.sh          # push everything (first run; model is 2.6 GB, takes a while)
#   bash mobile/push_phone.sh bench    # llama-bench only (binaries+model already pushed)
#   bash mobile/push_phone.sh chat     # one smoke turn only
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
L=/media/garret-sutherland/dev-data/cat4b/tools/llama.cpp-b11139/build-android-arm64/bin
M=/media/garret-sutherland/dev-data/cat4b/models/cat4b-v0-merged-Q4_K_M.gguf
D=/data/local/tmp/cat4b
adb get-state >/dev/null   # fails loudly when no device is attached

push() {
  adb shell "mkdir -p $D"
  adb push "$L/llama-cli" "$L/llama-server" "$L/llama-bench" "$D/"
  adb push "$REPO/prompts/stock_system.md" "$REPO/prompts/cat_system.md" "$D/"
  adb shell "chmod +x $D/llama-cli $D/llama-server $D/llama-bench"
  adb push "$M" "$D/model.gguf"
}

bench() {  # 4 big cores; taskset mask 0xF0 = cores 4-7 (A720/X4 on the 8 Gen 3)
  adb shell "cd $D && taskset f0 ./llama-bench -m model.gguf -t 4 -p 512 -n 64"
}

chat() {   # one turn, served sampling (temp 0.7 top_p 0.9), thinking off like the live service
  adb shell 'cd /data/local/tmp/cat4b && cat stock_system.md cat_system.md > sys.md && \
    taskset f0 ./llama-cli -m model.gguf -t 4 -c 4096 --temp 0.7 --top-p 0.9 \
    --jinja --reasoning-budget 0 -sys "$(cat sys.md)" \
    -p "Explain how BGP routing works, briefly." -st -n 256 --no-display-prompt 2>err.log || tail -5 err.log'
}

case "${1:-all}" in
  bench) bench ;;
  chat)  chat ;;
  all)   push; bench; chat ;;
esac
