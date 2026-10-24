#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
source venv/bin/activate
exec python -m uvicorn extras.phone_ui:app --host 0.0.0.0 --port 8189 "$@"
