#!/usr/bin/env bash
# Run one stage only; scientific and runtime releases are checked in Python.
set -euo pipefail
cd -- "$(dirname -- "$0")"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HOME=/media/garret-sutherland/dev-data/cat4b/hf
PY=${NEXTTURN_PYTHON:-python3}
case "${1:-}" in
  preflight) "$PY" collect.py preflight ;;
  fit)
    "$PY" collect.py collect --split fit
    "$PY" analyze.py fit
    ;;
  test)
    "$PY" collect.py collect --split test
    "$PY" analyze.py score
    ;;
  *) echo 'Usage: run_stage.sh preflight|fit|test' >&2; exit 2 ;;
esac
