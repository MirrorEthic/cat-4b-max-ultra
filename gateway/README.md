# gateway/

The public gateway for the seven-day challenge: `docs/GATEWAY.md` is the design and the decisions; this is how to run it.

```
gateway/app.py        FastAPI app: page + API on one origin, two lanes, SSE, health, metrics, admin
gateway/config.py     every knob is an env var (upstreams, lanes, budgets, window, timeouts)
gateway/db.py         SQLite WAL: sessions + ledger, jobs + transitions, screen hits, IP windows
gateway/sessions.py   HMAC-signed anonymous cookie, keyed IP hash
gateway/screen.py     the rules layer outside the model; patterns in screen_rules.yaml (hot-reloaded)
gateway/model.py      condition-C messages, the serving tool list, streamed chat, the review turn
gateway/images.py     argument validation, broker lease, mirage generate_image, PNG fetch
gateway/lanes.py      bounded FIFO + workers + EMA ETA + pause + circuit breaker
gateway/static/       index.html (vanilla JS, SSE), no build step
gateway/assets/       byte-exact copies of prompts/*.md and data/spec/tools.serving.json (tests check the hashes)
gateway/tests/        fake upstreams: sessions, screen, both lanes, the gate, sharing, status, admin
gateway/smoke.py      the real loop through a running gateway
```

Develop on legion against the live node (`CAT_URL`, `MIRAGE_URL`, `BROKER_URL` default to mirrorforge's 10 GbE
addresses; the venv is `/media/garret-sutherland/dev-data/cat4b/venvs/gateway`):

```bash
V=/media/garret-sutherland/dev-data/cat4b/venvs/gateway
$V/bin/python -m pytest -q gateway/tests
GATEWAY_STATE=/media/garret-sutherland/dev-data/cat4b/gateway-dev $V/bin/uvicorn gateway.app:app --port 8300
$V/bin/python gateway/smoke.py --gateway http://127.0.0.1:8300
```

Deploy with `serve/mirrorforge/push_gateway.sh tunnel|push|verify|smoke|rollback` (DEV copies of the units in
`serve/mirrorforge/units/`). Operator actions go to `POST /admin` on the node (loopback + bearer token from
`~/cat4b/gateway/state/admin_token`): `pause`/`resume` a lane, `close`/`open` admissions, `sweep`, `counts`,
`unshare`.
