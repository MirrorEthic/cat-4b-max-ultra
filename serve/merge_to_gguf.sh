#!/usr/bin/env bash
# The v0 adapter → merged HF checkpoint → GGUF for llama.cpp (docs/LIVE_SERVICE.md section 4). Writes
#   models/cat4b-v0-merged/                      the merged HF checkpoint (serve/merge_lora.py) with SHA256SUMS
#   models/cat4b-v0-merged-Q8_0.gguf             the text model, Q8_0 (what the V100 serves)
#   models/cat4b-v0-merged-F16.gguf              the text model, F16 (the reference quant for the parity run)
#   models/cat4b-v0-mmproj-F16.gguf              the vision projector, for the review turn and uploads
#   models/GGUF.SHA256SUMS
# under /media/garret-sutherland/dev-data/cat4b. CPU work on legion; the converter is the same b11139 tree whose
# sm_70 build serves on mirrorforge.
set -euo pipefail
CAT=/media/garret-sutherland/dev-data/cat4b
REPO=$(cd "$(dirname "$0")/.." && pwd)
MERGE_PY=$CAT/venvs/train-cu130/bin/python   # has peft; the merge runs on the CPU
PY=$REPO/.venv/bin/python                     # CPU torch + transformers: enough for the converter
LLAMA=$CAT/tools/llama.cpp-b11139
MERGED=$CAT/models/cat4b-v0-merged

if [[ ! -f $MERGED/SHA256SUMS ]]; then
  $MERGE_PY "$REPO/serve/merge_lora.py" --adapter "$CAT/adapters/cat4b-v0" --out "$MERGED"
fi
cd "$LLAMA"
[[ -f $CAT/models/cat4b-v0-merged-Q8_0.gguf ]] || $PY convert_hf_to_gguf.py "$MERGED" --outtype q8_0 --outfile "$CAT/models/cat4b-v0-merged-Q8_0.gguf"
[[ -f $CAT/models/cat4b-v0-merged-F16.gguf ]]  || $PY convert_hf_to_gguf.py "$MERGED" --outtype f16  --outfile "$CAT/models/cat4b-v0-merged-F16.gguf"
[[ -f $CAT/models/cat4b-v0-mmproj-F16.gguf ]]  || $PY convert_hf_to_gguf.py "$MERGED" --mmproj --outtype f16 --outfile "$CAT/models/cat4b-v0-mmproj-F16.gguf"
cd "$CAT/models"
sha256sum cat4b-v0-merged-Q8_0.gguf cat4b-v0-merged-F16.gguf cat4b-v0-mmproj-F16.gguf | tee GGUF.SHA256SUMS
ls -la cat4b-v0-merged-Q8_0.gguf cat4b-v0-merged-F16.gguf cat4b-v0-mmproj-F16.gguf
echo MERGE_TO_GGUF_OK
