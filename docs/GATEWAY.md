# CAT-4B MAX ULTRA: the public gateway

**Date:** 2026-10-24. **Status:** design adopted by Garret ("Good to go on the public gateway … use the mirrorethic.com
tunnel and make a page, that way the entity to product relation [is] more concrete"); built under `gateway/` (13
tests against fake upstreams; the live loop through it from legion: chat 0.6 s, image turn 1.9 s to the composed
prompt, render 13.1 s, review turn, share), pushed to mirrorforge 14:13 (`cat4b-gateway.service` and `cf-cat4b.service`
active, `/healthz` ok, tunnel `cat4b` id `3bbdb4c0-abdd-424f-a90b-84bb2b093c87` registered four connections),
contracts on the pi5 watchdog. **Waiting on DNS:** legion's origin certificate is scoped to t3atlas.dev, so
`cloudflared tunnel route dns` wrote a stray `cat4b.mirrorethic.com.t3atlas.dev` record instead of the real one; the
mirrorethic.com zone needs a proxied CNAME `cat4b → 3bbdb4c0-abdd-424f-a90b-84bb2b093c87.cfargotunnel.com` (dashboard,
or a DNS-edit token for that zone), and the stray record should be deleted. The mirrorethic.com bundle has the nav
link and a `/cat4b/` page staged in `~/mirrorethic-site`, not deployed until the hostname resolves. The model side it fronts is live (`docs/LIVE_SERVICE.md`); the event it serves is `docs/LAUNCH_PLAN.md`
(§5 journey and limits, §6 topology, §8 queues, §9 boundaries, §11 operations, §14 copy). This document is the
engineering shape of §6–§9 as built, and the decisions that were not in the plan.

## 1. Where it lives

| piece | where | why |
|---|---|---|
| `cat4b.mirrorethic.com` | Cloudflare DNS, a CNAME to a new named tunnel `cat4b` | mirrorethic.com itself is a static Cloudflare Pages site (`~/mirrorethic-site`, project `mirrorethic`); there is no mirrorethic.com tunnel to reuse. A subdomain of the company domain makes the entity-product relation concrete, as Garret asked, and keeps the challenge off the Pages bundle, which stays static per its README and the mesh charter |
| the tunnel | `cf-cat4b.service` on mirrorforge (`~/.local/bin/cloudflared 2026.10.3`, config `~/.cloudflared/cat4b.yml`, ingress `cat4b.mirrorethic.com → http://127.0.0.1:8300`) | the charter: tunnels run on the owning node with copied credentials. The tunnel is created once on legion with the account's origin cert (`cloudflared tunnel create cat4b`, `cloudflared tunnel route dns cat4b cat4b.mirrorethic.com`), the credentials JSON copied to mirrorforge, no login there |
| the gateway | `cat4b-gateway.service` on mirrorforge, uvicorn on `127.0.0.1:8300`, its own venv `~/cat4b/gateway/venv`, state under `~/cat4b/gateway/state/` | the two backends it talks to are on the same node (`10.10.10.2:8292` llama-server, `10.10.10.2:8190` mirage MCP, `localhost:8077` broker); nothing crosses the LAN; legion stays out of the hot path (plan §6.2) |
| mirrorethic.com | one nav link "CAT-4B" to `https://cat4b.mirrorethic.com` and, if Garret wants it, a `/cat4b/` teaser page in the Pages bundle | the entity page points at the product; copy is Garret's (§14.2 hero is the source) |
| contracts | `[[services]]` `cat4b-gateway` (health `curl -fsS http://127.0.0.1:8300/healthz`) and `cat4b-tunnel` (`systemctl --user is-active cf-cat4b`) in `openclaw-cognitive/nodes/mirrorforge.toml`, synced to the pi5 watchdog | nothing public runs unmonitored |

The browser talks only to the gateway. llama-server binds the 10 GbE address and ComfyUI, mirage and the broker bind
localhost or the 10 GbE address; none is reachable from the tunnel. The gateway holds no credentials beyond the
session-signing secret (an env file outside the repo) and the admin token.

## 2. What a visitor can do

The plan's §5.1 journey, with the limits of §5.2 as the opening defaults (all are config, none is a code change):

- **Session:** `POST /api/session` sets a signed anonymous cookie (HMAC-SHA256 over id and issue time; secret in
  `GATEWAY_SECRET_FILE`). Budget per session: 20 chat turns; images through a **timed gate**, one render at a time
  per visitor and a 90 s cooldown between renders, under a ceiling of 24 a session against scripted drain
  (Garret, 2026-10-24: "the one image per user rule is a little strict … we could just do a timed gate rate
  limit"; the plan's single credit was the opening proposal). A failed render gives the cooldown and the credit
  back. The ledger is server-side in SQLite; the browser's view of it is advisory. Every number is an env knob
  (`GATEWAY_CHAT_TURNS`, `GATEWAY_IMAGE_COOLDOWN_S`, `GATEWAY_IMAGE_CREDITS`, `GATEWAY_OUTSTANDING_IMAGES`).
- **Chat:** `POST /api/chat {text}` → a chat job. The gateway builds the messages exactly as the eval did for
  condition C (`prompts/stock_system.md` + blank line + `prompts/cat_system.md` as the system prompt, the user text
  as the user turn, prior turns of the session as alternating user/assistant messages), offers the serving tool
  list the model was trained and measured with (`data/spec/tools.serving.json`, seven tools), samples at
  temperature 0.7, top_p 0.9, thinking off (server default), max_tokens 1,024 (the frozen set's replies are far
  shorter; the eval's 8,192 is for the thinking condition). Tokens stream to the browser over SSE.
- **Image:** when the reply is a `generate_image` call and the session's gate is open, the gateway validates the
  arguments (prompt ≤ 1,200 characters, width × height from the allowed set 1024², 1344×768, 768×1344, 1024×768,
  768×1024; steps and guidance are pinned server-side as the gallery pinned them, 30 and 3.5, which the Z-Image
  backend in mirage overrides with its own fixed sampler), queues an image job, and the browser follows it over
  SSE: queued (with rank and an honest ETA), rendering, done. The render runs under a broker `imagegen` lease
  (ttl 300 s, released after), through mirage's MCP `generate_image`, the same call group-chat and the gallery
  make. The finished PNG is fetched to the gateway's result store; the browser gets a session-bound URL.
- **Review turn:** after the render the gateway runs the round group-chat runs (assistant tool call, tool result
  text, user turn "(Image you just generated, for your review — iterate if it's not right:)" with the image), and
  shows the model's words beside the picture. A second `generate_image` or any edit tool the review turn asks for
  is *not* executed at opening (plan §5.2: edits disabled); the gateway records the request and tells the visitor
  the cats are satisfied with one render.
- **Trace:** the result card shows prompt → the prompt the model composed → the image → the review line. That trace
  is the product (plan §1.1).
- **Share:** private by default; `POST /api/jobs/{id}/share` marks the item opted in; the operator curates the
  gallery from opted-in items only. Result and image URLs carry the job id (random 128-bit) and require the
  session cookie or a signed share token; nothing is enumerable.
- **Status:** `GET /api/status` is public and honest: lanes, queue depth, measured ETA, the event window, whether
  admissions are open. The page shows it before the visitor types.

### Other tool calls

The model may call `web_search` (the frozen set's search items do, 30 of 30). The public gateway executes only
`generate_image`; every other tool call gets a tool result saying the tool is not available in this demo, and the
model's next round is shown. This keeps the tool schema the model was measured with (removing tools from the
schema would change the distribution the card describes) while executing nothing outside the allowlist (plan §9.1).

## 3. Admission and lanes (plan §8)

Two independent lanes, each a bounded FIFO with a worker pool:

| lane | workers at opening | waiting room | what bounds it |
|---|---|---|---|
| `chat` | 4 of llama-server's 8 slots (the rest stay for the Discord room and review turns) | 32 | measured: 8 in flight ≈ 220 tok/s aggregate, 46–49 replies a minute (`docs/LIVE_SERVICE.md` §7); four public streams ≈ 38 tok/s each |
| `image` | 1 render at a time | 24, and admission pauses when the measured ETA exceeds 12 minutes | measured: 13.6 s a render alone, 21.5 s each with two sharing the card; one at a time is the better experience |

ETA is the queue rank × an exponential moving average of the last 20 service times for that lane (seeded from the
measurements above), shown as a range, never as a countdown. When the room is full or the ETA cap is hit the
answer is the plan's §14.3 copy ("The feline queue is full. No fake countdown"), the job is not created, and no
credit is spent. Lifecycle per job: `submitted → validated → queued → running → succeeded | failed | expired |
cancelled`, persisted in SQLite (WAL) with every transition timestamped; on restart, running jobs are marked
`failed` with the credit refunded, nothing is re-run. A failed render refunds the credit (§14.3). Three
consecutive render failures open the image lane's circuit breaker: admissions pause, the status page says so, the
lane retries one probe render every 60 s and reopens after a success. Chat is unaffected by image failures.

## 4. Boundaries (plan §9)

- **Pre-inference screen, outside the model:** a rules layer over the user text before any model call. Categories:
  active emergency, self-harm, medical dosing and diagnosis, legal action, financial decisions, child safety, pet
  emergencies, and the abuse categories the room already moderates. A hit returns a short fixed redirect (no model
  text), spends no turn, and is counted by category in aggregate; the text is not logged. The patterns live in
  `gateway/screen_rules.yaml` so they can be tightened without a deploy. The model card's own disclosure stays on
  the page; the screen is the guardrail, the disclaimer is additive.
- **Tool execution allowlist:** `generate_image` only, arguments validated as above; no `image_path`, no uploads, no
  URLs from visitors, no edits at opening.
- **Rate limits:** per session (the budgets) and per IP in a sliding window (30 requests a minute, 3 sessions an
  hour); Cloudflare's WAF and rate limiting in front; Turnstile is a hook (`GATEWAY_TURNSTILE_SECRET` on → the
  session endpoint verifies a token) and is off until Garret creates the widget.
- **Data:** prompts, replies and images live only in the gateway's SQLite and result store on mirrorforge; result
  links are session-bound; the sweep deletes private items 7 days after the event closes (configurable); opted-in
  items are kept for the gallery; counts and timings are kept in aggregate for the after-action (§13.2). No
  training reuse without opt-in.
- **Identity:** the gateway runs as the node's user for now (its only reach is two local HTTP endpoints and its own
  directory). A dedicated low-privilege user is the first hardening step once Garret grants sudo on the node.

## 5. Operations (plan §11)

- `GET /healthz`: 200 when the gateway is up and both upstreams answered their health checks in the last 60 s; the
  pi5 watchdog's contract uses it.
- `GET /metrics` (localhost only): Prometheus text, counters and histograms per lane and per outcome, screen hits
  by category, upstream health, queue depth and age.
- Admin (localhost only, bearer token from `GATEWAY_ADMIN_TOKEN_FILE`): pause/resume a lane, close admissions,
  set the event window, trigger the retention sweep, list opted-in items.
- Event window: `GATEWAY_OPENS_AT` / `GATEWAY_CLOSES_AT` (ISO-8601). Outside the window the page shows the status
  and the gallery; `/api/chat` answers 503 with the closing copy (§14.5) after the end.
- Logs: JSON lines to the journal, one line per job transition, with job and session ids, never prompt text.

## 6. Build and deploy

- Code: `gateway/` in this repo (FastAPI + uvicorn + httpx + aiosqlite; the page is one static `index.html` with
  vanilla JS and SSE, no build step, same visual family as mirrorethic.com's dark theme). Tests in `gateway/tests/`
  run against fake upstreams; `gateway/smoke.py` runs the real loop against the live endpoints from legion.
- Deploy: `serve/mirrorforge/push_gateway.sh push|verify|rollback` (rsync `gateway/` to `~/cat4b/gateway`, build
  the venv from `requirements.txt`, install `cat4b-gateway.service` and `cf-cat4b.service`, restart, hit
  `/healthz` locally and `https://cat4b.mirrorethic.com/api/status` through the tunnel). DEV copies of both units
  under `serve/mirrorforge/units/`.
- Order: build and test on legion against the live endpoints → push to mirrorforge → tunnel and DNS → contracts →
  invited traffic (a share link to the room) → soak → the nav link on mirrorethic.com → open.

## 7. Decisions not in the plan

| decision | choice | why |
|---|---|---|
| hostname | `cat4b.mirrorethic.com`, gateway serves page and API on one origin | the company domain carries the product (Garret); one origin keeps the session a plain cookie, no CORS, no third origin to secure |
| chat workers | 4, not the plan's 2 | measured headroom: 8 slots sustain 46–49 replies a minute with renders running; four public streams at 38 tok/s each leave four for the room and the review turns |
| image workers | 1 | two renders at once lengthen both (21.5 s each against 13.6 s alone); one at a time with an honest queue is the better wait |
| tool schema | the serving list the eval used, execution allowlist of one | behaviour stays as measured; nothing outside the allowlist runs |
| review-turn re-renders | recorded, not executed | plan §5.2 opening limit; a visitor asks for the next picture themselves, through the gate |
| image budget | a timed gate (90 s cooldown, one in flight, ceiling 24 a session) instead of one credit | Garret's call on 2026-10-24; capacity supports it (one render lane ≈ 4 a minute, so a visitor every 90 s is a small share) and the gate still bounds scripted drain |
| queue store | SQLite WAL on the node | one host, small state, durable across restarts, nothing to operate |
| engine for the tunnel | a new named tunnel, not the mirage one | mirage-mcp.dev's tunnel stays single-purpose; a separate tunnel can be stopped without touching mirage |
