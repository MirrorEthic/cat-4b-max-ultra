"""The public gateway (docs/GATEWAY.md): one FastAPI app serving the page and the API on one origin, two lanes,
signed anonymous sessions, the rules screen, SSE job streams, a result store, health and metrics.

    GATEWAY_STATE=./state uvicorn gateway.app:app --host 127.0.0.1 --port 8300
"""
from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .db import DB, TERMINAL
from .images import ArgError, Broker, LeaseBusy, Mirage, RenderError, validate_args
from .lanes import Lane, LaneFull, LanePaused
from .model import (NO_CREDIT, TOOL_UNAVAILABLE, Assets, Model, Round, assistant_with_calls, build_messages,
                    result_done, review_turn, tool_result_message)
from .screen import Screen
from .sessions import COOKIE, Signer, client_ip

log = logging.getLogger("gateway")
STATIC = Path(__file__).resolve().parent / "static"

# ── copy (plan section 14.3) ────────────────────────────────────────────────────
COPY = {
    "queued": "Furnando currently occupies the GPU. You're in line; we'll show the real status.",
    "full": "The feline queue is full. No fake countdown—try again later.",
    "running": "Your image is rendering. The cats are negotiating placement.",
    "failed": "The render failed. Your image credit wasn't spent; please retry once capacity returns.",
    "done": "The cats have arrived. Download your result or choose whether to share it.",
    "closed": "The seven days are over. Public inference has stopped; the gallery and the research stay up.",
    "before": "The challenge has not opened yet.",
    "no_turns": "This session has used its chat turns. Thank you for playing; the cats remain.",
    "breaker": "Image rendering is paused while we check the render service. Chat still works.",
}


class Broadcaster:
    """Per-job event fan-out with a short replay buffer for late or reconnecting subscribers."""

    def __init__(self):
        self.subs: dict[str, list[asyncio.Queue]] = {}
        self.replay: dict[str, deque] = {}
        self.closed: dict[str, float] = {}

    def publish(self, job_id: str, event: dict):
        event.setdefault("ts", round(time.time(), 3))
        buf = self.replay.setdefault(job_id, deque(maxlen=400))
        buf.append(event)
        for q in self.subs.get(job_id, []):
            q.put_nowait(event)
        if event.get("type") in ("done", "error", "refused"):
            self.closed[job_id] = time.time()

    def subscribe(self, job_id: str) -> tuple[asyncio.Queue, list[dict]]:
        q: asyncio.Queue = asyncio.Queue()
        self.subs.setdefault(job_id, []).append(q)
        return q, list(self.replay.get(job_id, []))

    def unsubscribe(self, job_id: str, q: asyncio.Queue):
        lst = self.subs.get(job_id, [])
        if q in lst:
            lst.remove(q)
        if not lst:
            self.subs.pop(job_id, None)

    def sweep(self, older_than_s: float = 900):
        cut = time.time() - older_than_s
        for jid, t in list(self.closed.items()):
            if t < cut:
                self.closed.pop(jid, None)
                self.replay.pop(jid, None)
                self.subs.pop(jid, None)


class Gateway:
    def __init__(self, settings: config.Settings):
        self.s = settings
        self.db = DB(settings.db_path)
        self.signer = Signer(settings.secret())
        self.admin_token = settings.admin_token()
        self.screen = Screen()
        self.assets = Assets.load(settings.assets_dir)
        self.model = Model(settings.cat_url, settings.cat_model, temperature=settings.temperature,
                           top_p=settings.top_p, timeout_s=settings.chat_timeout_s)
        self.broker = Broker(settings.broker_url)
        self.mirage = Mirage(settings.mirage_url, settings.comfy_view_url, settings.comfy_output_dir,
                             timeout_s=settings.render_timeout_s)
        self.bus = Broadcaster()
        self.chat = Lane("chat", settings.chat_workers, settings.chat_room, self.run_chat, settings.chat_seed_s,
                         breaker_failures=settings.breaker_failures, breaker_probe_s=settings.breaker_probe_s)
        self.image = Lane("image", settings.image_workers, settings.image_room, self.run_image, settings.image_seed_s,
                          eta_cap_s=settings.image_eta_cap_s, breaker_failures=settings.breaker_failures,
                          breaker_probe_s=settings.breaker_probe_s)
        self.admissions_open = True
        self.health: dict = {"cat": None, "mirage": None, "broker": None, "checked": 0.0}
        self.metrics: dict[str, float] = {}
        self._bg: list[asyncio.Task] = []
        self.started = time.time()

    # ── lifecycle ───────────────────────────────────────────────────────────────
    async def start(self):
        await self.db.open()
        # jobs that were queued or running when the process died: failed, credits back, nothing re-run
        for j in await self.db.running_jobs():
            await self.db.set_state(j["id"], "failed", "gateway restarted", error="gateway restarted")
            if j["kind"] == "image":
                await self.db.ungate_image(j["session_id"])
                await self.db.set_outstanding(j["session_id"], -1)
            else:
                await self.db.refund(j["session_id"], "chat_used")
        self.chat.start()
        self.image.start()
        self._bg = [asyncio.create_task(self._health_loop()), asyncio.create_task(self._housekeeping())]

    async def stop(self):
        for t in self._bg:
            t.cancel()
        await self.chat.stop()
        await self.image.stop()
        await self.model.close()
        await self.broker.close()
        await self.mirage.close()
        await self.db.close()

    async def _health_loop(self):
        while True:
            try:
                cat, mir, brk = await asyncio.gather(self.model.health(), self.mirage.health(), self.broker.status())
                self.health = {"cat": cat, "mirage": mir, "broker": brk is not None,
                               "broker_loaded": (brk or {}).get("loaded"), "checked": time.time()}
            except Exception as e:  # noqa: BLE001
                log.warning("health loop: %s", e)
            await asyncio.sleep(self.s.health_interval_s)

    async def _housekeeping(self):
        while True:
            await asyncio.sleep(300)
            try:
                self.bus.sweep()
                await self.db.ip_prune()
                if self.s.closes_at and time.time() > self.s.closes_at + self.s.retention_days * 86400:
                    await self.sweep_private(self.s.closes_at)
            except Exception as e:  # noqa: BLE001
                log.warning("housekeeping: %s", e)

    async def sweep_private(self, before: float) -> int:
        paths = await self.db.expire_private(before)
        n = 0
        for p in paths:
            try:
                Path(p).unlink(missing_ok=True)
                n += 1
            except OSError:
                pass
        return n

    def count(self, key: str, n: float = 1):
        self.metrics[key] = self.metrics.get(key, 0) + n

    def window(self) -> str:
        return self.s.window_state(time.time())

    # ── the chat lane ───────────────────────────────────────────────────────────
    async def run_chat(self, job_id: str) -> bool:
        job = await self.db.job(job_id)
        if not job or job["state"] != "queued":
            return True
        sid = job["session_id"]
        try:
            hist = [h for h in await self.db.history(sid) if h["id"] != job_id]
            messages = build_messages(self.assets.system_prompt, hist, job["text"])
            await self.db.set_state(job_id, "running")
            self.bus.publish(job_id, {"type": "state", "state": "running"})
            reply_parts: list[str] = []
            for _ in range(self.s.max_tool_rounds + 1):
                rnd: Round | None = None
                async for kind, payload in self.model.stream(messages, self.assets.tools, self.s.max_tokens,
                                                             seed=secrets.randbelow(2**31 - 1)):
                    if kind == "delta":
                        self.bus.publish(job_id, {"type": "delta", "text": payload})
                    else:
                        rnd = payload
                assert rnd is not None
                if rnd.content:
                    reply_parts.append(rnd.content)
                if not rnd.tool_calls:
                    break
                gen = next((c for c in rnd.tool_calls if c.name == "generate_image"), None)
                if gen is not None:
                    outcome = await self._start_image(job, rnd, gen)
                    if outcome is None:                      # an image job is on its way; the chat turn is done
                        break
                    messages.append(assistant_with_calls(rnd.content, rnd.tool_calls))
                    for c in rnd.tool_calls:
                        messages.append(tool_result_message(
                            c, outcome if c is gen else TOOL_UNAVAILABLE.format(name=c.name)))
                    self.bus.publish(job_id, {"type": "tool_refused", "tool": "generate_image", "why": outcome[:120]})
                    continue
                messages.append(assistant_with_calls(rnd.content, rnd.tool_calls))
                for c in rnd.tool_calls:
                    messages.append(tool_result_message(c, TOOL_UNAVAILABLE.format(name=c.name)))
                    self.bus.publish(job_id, {"type": "tool_refused", "tool": c.name, "why": "not in this demo"})
                    self.count(f"tool_refused.{c.name}")
            reply = "\n".join(p for p in reply_parts if p).strip()
            await self.db.set_state(job_id, "succeeded", reply=reply)
            job = await self.db.job(job_id)
            self.bus.publish(job_id, {"type": "done", "reply": reply, "image_job": job.get("child_job")})
            self.count("chat.succeeded")
            return True
        except Exception as e:  # noqa: BLE001
            log.exception("chat job %s failed", job_id)
            await self.db.set_state(job_id, "failed", error=str(e)[:300])
            await self.db.refund(sid, "chat_used")
            self.bus.publish(job_id, {"type": "error", "message": "The model did not answer. Your turn was not spent."})
            self.count("chat.failed")
            return False

    async def _start_image(self, job: dict, rnd: Round, call) -> str | None:
        """Create and queue the image job for a generate_image call. Returns None on success, else the tool-result
        text to feed back to the model (no credit, invalid arguments, lane closed)."""
        sid = job["session_id"]
        try:
            args = validate_args(call.args(), max_prompt_chars=self.s.max_prompt_chars,
                                 allowed_sizes=self.s.allowed_sizes)
        except ArgError as e:
            self.count("image.invalid_args")
            return f"Error: generate_image arguments rejected ({e}). Answer in words instead."
        ok, why = self.image.admits()
        if not ok or not self.admissions_open:
            self.count(f"image.not_admitted.{why or 'closed'}")
            return ("Error: image generation is unavailable right now (the render queue is "
                    f"{why or 'closed'}). Describe what you would have drawn, briefly.")
        sess = await self.db.session(sid)
        if not sess or sess["image_outstanding"] >= self.s.outstanding_images:
            return "Error: this visitor already has an image in progress. Answer in words instead."
        granted, wait_s = await self.db.gate_image(sid, self.s.image_cooldown_s, self.s.image_credits)
        if not granted:
            if wait_s > 0:
                self.count("image.cooldown")
                return (f"Error: image generation for this visitor is cooling down; the next render is allowed in "
                        f"{int(wait_s) + 1} seconds. Describe what you would have drawn, briefly, and say when.")
            self.count("image.no_credit")
            return NO_CREDIT
        tool_call = {"id": call.id, "name": call.name,
                     "arguments": {"prompt": args.prompt, "width": args.width, "height": args.height}}
        img_id = await self.db.create_job(sid, "image", parent=job["id"], tool_call=tool_call, state="validated")
        await self.db.update(job["id"], child_job=img_id, tool_call=tool_call)
        await self.db.set_outstanding(sid, +1)
        try:
            rank = self.image.submit(img_id)
        except (LaneFull, LanePaused) as e:
            await self.db.set_state(img_id, "refused", str(e))
            await self.db.ungate_image(sid)
            await self.db.set_outstanding(sid, -1)
            return "Error: the render queue is full. Describe what you would have drawn, briefly."
        await self.db.set_state(img_id, "queued", rank_at_submit=rank)
        lo, hi = self.image.eta_range_s(rank)
        self.bus.publish(job["id"], {"type": "tool_call", "tool": "generate_image", "image_job": img_id,
                                     "prompt": args.prompt, "width": args.width, "height": args.height,
                                     "rank": rank, "eta_s": [lo, hi], "copy": COPY["queued"]})
        self.bus.publish(img_id, {"type": "state", "state": "queued", "rank": rank, "eta_s": [lo, hi],
                                  "copy": COPY["queued"]})
        self.count("image.queued")
        return None

    # ── the image lane ──────────────────────────────────────────────────────────
    async def run_image(self, job_id: str) -> bool:
        job = await self.db.job(job_id)
        if not job or job["state"] != "queued":
            return True
        sid = job["session_id"]
        call = json.loads(job["tool_call"])
        args = validate_args(call["arguments"], max_prompt_chars=self.s.max_prompt_chars,
                             allowed_sizes=self.s.allowed_sizes)
        lease_id = None
        try:
            await self.db.set_state(job_id, "running")
            self.bus.publish(job_id, {"type": "state", "state": "running", "copy": COPY["running"]})
            # the broker lease: the room's bot may hold it for its own render; wait up to ~2 minutes
            for attempt in range(24):
                try:
                    lease_id = await self.broker.lease(self.s.lease_ttl_s)
                    break
                except LeaseBusy as e:
                    if attempt == 0:
                        self.bus.publish(job_id, {"type": "note", "text": f"waiting for the GPU lease ({e})"})
                    await asyncio.sleep(5)
            if lease_id is None:
                raise RenderError("the GPU lease was not granted in time")
            out = self.s.results_dir / f"{job_id}.png"
            t0 = time.time()
            res = await self.mirage.generate(args, steps=self.s.render_steps, cfg=self.s.render_cfg,
                                             seed=secrets.randbelow(2**31 - 1), out=out)
            render_s = round(time.time() - t0, 1)
            await self.db.update(job_id, image_path=str(out), image_sha=res["sha256"], image_w=args.width,
                                 image_h=args.height, service_s=render_s)
            self.bus.publish(job_id, {"type": "image", "url": f"/api/jobs/{job_id}/image", "sha256": res["sha256"],
                                      "width": args.width, "height": args.height, "render_s": render_s})
            await self.broker.release(lease_id)
            lease_id = None
            # the review turn: what the model says about its own picture (recorded, re-renders not executed)
            review = await self._review(job, args, out, res.get("remote") or str(out))
            await self.db.set_state(job_id, "succeeded", review=review["text"],
                                    review_tool=review["tool"] if review["tool"] else None)
            await self.db.set_outstanding(sid, -1)
            self.bus.publish(job_id, {"type": "review", "text": review["text"], "wanted_more": bool(review["tool"]),
                                      "copy": COPY["done"]})
            self.bus.publish(job_id, {"type": "done", "image_url": f"/api/jobs/{job_id}/image"})
            self.count("image.succeeded")
            return True
        except Exception as e:  # noqa: BLE001
            log.exception("image job %s failed", job_id)
            if lease_id:
                await self.broker.release(lease_id)
            await self.db.set_state(job_id, "failed", error=str(e)[:300])
            await self.db.ungate_image(sid)
            await self.db.set_outstanding(sid, -1)
            self.bus.publish(job_id, {"type": "error", "message": COPY["failed"]})
            self.count("image.failed")
            return False

    async def _review(self, job: dict, args, out: Path, remote: str) -> dict:
        parent = await self.db.job(job["parent_job"]) if job.get("parent_job") else None
        text, tool = "", None
        if not parent:
            return {"text": text, "tool": tool}
        try:
            hist = [h for h in await self.db.history(job["session_id"]) if h["id"] != parent["id"]
                    and (h.get("created") or 0) < parent["created"]]
            messages = build_messages(self.assets.system_prompt, hist, parent["text"])
            call = json.loads(job["tool_call"])
            from .model import ToolCall
            tc = ToolCall(id=call["id"], name="generate_image", arguments=json.dumps(call["arguments"]))
            messages.append(assistant_with_calls(parent.get("reply") or "", [tc]))
            messages.append(tool_result_message(tc, result_done(remote)))
            messages.append(review_turn(out.read_bytes()))
            rnd = await self.model.complete(messages, self.assets.tools, self.s.review_max_tokens,
                                            seed=secrets.randbelow(2**31 - 1))
            text = (rnd.content or "").strip()
            if rnd.tool_calls:
                c = rnd.tool_calls[0]
                tool = {"name": c.name, "arguments": c.args()}
                self.count(f"review.wanted.{c.name}")
        except Exception as e:  # noqa: BLE001
            log.warning("review turn failed for %s: %s", job["id"], e)
            text = ""
        return {"text": text, "tool": tool}


# ── the app ─────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    gw = Gateway(config.load())
    await gw.start()
    app.state.gw = gw
    try:
        yield
    finally:
        await gw.stop()


app = FastAPI(title="CAT-4B MAX ULTRA gateway", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def gw(request: Request) -> Gateway:
    return request.app.state.gw


def is_local(request: Request) -> bool:
    """Only requests that did not come through the tunnel and arrived from loopback."""
    if "cf-connecting-ip" in request.headers:
        return False
    host = request.client.host if request.client else ""
    return host in ("127.0.0.1", "::1")


async def current_session(request: Request) -> dict:
    g = gw(request)
    sid = g.signer.verify(request.cookies.get(COOKIE))
    if not sid:
        raise HTTPException(401, "no session; call POST /api/session first")
    sess = await g.db.session(sid)
    if not sess:
        raise HTTPException(401, "unknown session")
    return sess


def budgets(g: Gateway, sess: dict) -> dict:
    next_in = max(0.0, (sess.get("last_image_at") or 0) + g.s.image_cooldown_s - time.time())
    return {"chat_turns_left": max(g.s.chat_turns - sess["chat_used"], 0),
            "image_credits_left": max(g.s.image_credits - sess["image_used"], 0),
            "image_cooldown_s": g.s.image_cooldown_s, "image_next_in_s": round(next_in),
            "image_outstanding": sess["image_outstanding"], "consent": bool(sess["consent"])}


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp: Response = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    if request.url.path.startswith("/api/"):
        resp.headers.setdefault("Cache-Control", "no-store")
    return resp


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(STATIC / "index.html", media_type="text/html")


@app.get("/s/{token}", response_class=HTMLResponse)
async def shared_page(token: str):
    return FileResponse(STATIC / "index.html", media_type="text/html")


class SessionIn(BaseModel):
    consent: bool = False
    turnstile: str | None = None


@app.post("/api/session")
async def create_session(body: SessionIn, request: Request, response: Response):
    g = gw(request)
    ip = g.signer.ip_hash(client_ip(request.headers, request.client.host if request.client else None))
    if await g.db.ip_count(ip, "session", 3600) >= g.s.ip_sessions_per_hour:
        g.count("session.ip_limited")
        raise HTTPException(429, "too many sessions from this network; try again later")
    if g.s.turnstile_secret:
        if not body.turnstile or not await verify_turnstile(g.s.turnstile_secret, body.turnstile, ip):
            raise HTTPException(400, "verification failed")
    existing = g.signer.verify(request.cookies.get(COOKIE))
    if existing and await g.db.session(existing):
        sess = await g.db.session(existing)
        return {"session": existing[:8], "budgets": budgets(g, sess), "new": False}
    sid = await g.db.create_session(ip, body.consent)
    await g.db.ip_mark(ip, "session")
    response.set_cookie(COOKIE, g.signer.issue(sid), max_age=7 * 24 * 3600, httponly=True, samesite="lax",
                        secure=g.s.public_base.startswith("https"), path="/")
    g.count("session.created")
    sess = await g.db.session(sid)
    return {"session": sid[:8], "budgets": budgets(g, sess), "new": True}


async def verify_turnstile(secret: str, token: str, ip_hash: str) -> bool:
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.post("https://challenges.cloudflare.com/turnstile/v0/siteverify",
                             data={"secret": secret, "response": token})
            return bool(r.json().get("success"))
    except Exception:  # noqa: BLE001
        return False


@app.get("/api/me")
async def me(request: Request, sess: dict = Depends(current_session)):
    g = gw(request)
    await g.db.touch(sess["id"])
    return {"session": sess["id"][:8], "budgets": budgets(g, sess)}


@app.get("/api/status")
async def status(request: Request):
    g = gw(request)
    now = time.time()
    w = g.window()
    return {"now": round(now, 1), "window": w, "opens_at": g.s.opens_at, "closes_at": g.s.closes_at,
            "admissions_open": g.admissions_open and w == "open",
            "lanes": {"chat": g.chat.snapshot(), "image": g.image.snapshot()},
            "upstream": {"cat": g.health.get("cat"), "mirage": g.health.get("mirage"),
                         "broker": g.health.get("broker"), "checked_s_ago": round(now - (g.health.get("checked") or now))},
            "limits": {"chat_turns": g.s.chat_turns, "image_credits": g.s.image_credits},
            "copy": {"queued": COPY["queued"], "full": COPY["full"], "closed": COPY["closed"]}}


class ChatIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


@app.post("/api/chat")
async def chat(body: ChatIn, request: Request, sess: dict = Depends(current_session)):
    g = gw(request)
    w = g.window()
    if w != "open" or not g.admissions_open:
        raise HTTPException(503, COPY["closed"] if w == "closed" else COPY["before"] if w == "before" else COPY["full"])
    ip = g.signer.ip_hash(client_ip(request.headers, request.client.host if request.client else None))
    if await g.db.ip_count(ip, "req", 60) >= g.s.ip_requests_per_min:
        g.count("chat.ip_limited")
        raise HTTPException(429, "slow down")
    await g.db.ip_mark(ip, "req")
    text = body.text.strip()
    if len(text) > g.s.max_text_chars:
        raise HTTPException(413, f"keep it under {g.s.max_text_chars} characters")
    hit = g.screen.check(text)
    if hit:
        cat, redirect = hit
        await g.db.screen_hit(cat)
        g.count(f"screen.{cat}")
        return {"refused": True, "category": cat, "message": redirect, "budgets": budgets(g, sess)}
    ok, why = g.chat.admits()
    if not ok:
        g.count(f"chat.not_admitted.{why}")
        raise HTTPException(503, COPY["full"] if why in ("full", "eta") else COPY["breaker"])
    if not await g.db.spend(sess["id"], "chat_used", g.s.chat_turns):
        raise HTTPException(429, COPY["no_turns"])
    job_id = await g.db.create_job(sess["id"], "chat", text=text, state="validated")
    try:
        rank = g.chat.submit(job_id)
    except (LaneFull, LanePaused):
        await g.db.set_state(job_id, "refused", "lane")
        await g.db.refund(sess["id"], "chat_used")
        raise HTTPException(503, COPY["full"])
    await g.db.set_state(job_id, "queued", rank_at_submit=rank)
    g.bus.publish(job_id, {"type": "state", "state": "queued", "rank": rank, "eta_s": list(g.chat.eta_range_s(rank))})
    g.count("chat.submitted")
    sess = await g.db.session(sess["id"])
    return {"job_id": job_id, "rank": rank, "eta_s": list(g.chat.eta_range_s(rank)), "budgets": budgets(g, sess)}


async def owned_job(request: Request, job_id: str) -> dict:
    g = gw(request)
    job = await g.db.job(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    sid = g.signer.verify(request.cookies.get(COOKIE))
    if sid != job["session_id"]:
        raise HTTPException(403, "not your job")
    return job


def job_view(job: dict) -> dict:
    v = {k: job.get(k) for k in ("id", "kind", "state", "created", "started", "finished", "reply", "review",
                                 "error", "parent_job", "child_job", "image_w", "image_h", "image_sha", "service_s")}
    v["text"] = job.get("text")
    v["tool_call"] = json.loads(job["tool_call"]) if job.get("tool_call") else None
    v["review_tool"] = json.loads(job["review_tool"]) if job.get("review_tool") else None
    v["shared"] = bool(job.get("shared"))
    if job.get("image_path") and job["kind"] == "image":
        v["image_url"] = f"/api/jobs/{job['id']}/image"
    return v


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str, request: Request):
    job = await owned_job(request, job_id)
    g = gw(request)
    v = job_view(job)
    lane = g.image if job["kind"] == "image" else g.chat
    if job["state"] == "queued":
        r = lane.rank_of(job_id)
        v["rank"] = r
        v["eta_s"] = list(lane.eta_range_s(r)) if r else None
    return v


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request):
    job = await owned_job(request, job_id)
    g = gw(request)

    async def gen():
        q, replay = g.bus.subscribe(job_id)
        try:
            yield f"event: snapshot\ndata: {json.dumps(job_view(job))}\n\n"
            terminal = job["state"] in TERMINAL
            for ev in replay:
                yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                if ev["type"] in ("done", "error", "refused"):
                    terminal = True
            if terminal:
                return
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                if ev["type"] in ("done", "error", "refused"):
                    return
        finally:
            g.bus.unsubscribe(job_id, q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.get("/api/jobs/{job_id}/image")
async def job_image(job_id: str, request: Request, t: str | None = None):
    g = gw(request)
    job = await g.db.job(job_id)
    if not job or not job.get("image_path"):
        raise HTTPException(404, "no image")
    sid = g.signer.verify(request.cookies.get(COOKIE))
    if sid != job["session_id"] and not (job.get("shared") and t and t == job.get("share_token")):
        raise HTTPException(403, "not yours")
    p = Path(job["image_path"])
    if not p.exists():
        raise HTTPException(410, "the image has been removed")
    return FileResponse(p, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


@app.post("/api/jobs/{job_id}/share")
async def share(job_id: str, request: Request):
    job = await owned_job(request, job_id)
    g = gw(request)
    if job["kind"] != "image" or job["state"] != "succeeded":
        raise HTTPException(409, "only a finished image can be shared")
    token = job.get("share_token") or g.signer.share_token(job_id)
    await g.db.update(job_id, shared=1, share_token=token)
    g.count("share.opt_in")
    return {"share_url": f"{g.s.public_base}/s/{token}", "token": token}


@app.delete("/api/jobs/{job_id}/share")
async def unshare(job_id: str, request: Request):
    job = await owned_job(request, job_id)
    g = gw(request)
    await g.db.update(job_id, shared=0)
    g.count("share.opt_out")
    return {"shared": False}


@app.get("/api/shared/{token}")
async def shared_view(token: str, request: Request):
    g = gw(request)
    job = await g.db.job_by_share(token)
    if not job:
        raise HTTPException(404, "nothing shared here")
    parent = await g.db.job(job["parent_job"]) if job.get("parent_job") else None
    return {"user_text": parent.get("text") if parent else None, "reply": parent.get("reply") if parent else None,
            "composed": json.loads(job["tool_call"])["arguments"] if job.get("tool_call") else None,
            "review": job.get("review"), "image_url": f"/api/jobs/{job['id']}/image?t={token}",
            "width": job.get("image_w"), "height": job.get("image_h"), "created": job.get("created")}


@app.get("/api/gallery")
async def gallery(request: Request):
    g = gw(request)
    # opted-in items only (plan 9.3); the operator curates by removing (admin "unshare" sets shared=0)
    items = await g.db.shared_items()
    return {"items": [{"user_text": it.get("user_text"),
                       "composed": json.loads(it["tool_call"])["arguments"] if it.get("tool_call") else None,
                       "review": it.get("review"), "image_url": f"/api/jobs/{it['id']}/image?t={it['share_token']}",
                       "created": it["created"]} for it in items]}


@app.get("/healthz")
async def healthz(request: Request):
    g = gw(request)
    fresh = time.time() - (g.health.get("checked") or 0) < 3 * g.s.health_interval_s
    ok = fresh and bool(g.health.get("cat")) and bool(g.health.get("mirage"))
    body = {"ok": ok, "upstream": {k: g.health.get(k) for k in ("cat", "mirage", "broker")},
            "uptime_s": round(time.time() - g.started), "window": g.window()}
    return JSONResponse(body, status_code=200 if ok else 503)


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics(request: Request):
    if not is_local(request):
        raise HTTPException(404)
    g = gw(request)
    lines = []
    for k, v in sorted(g.metrics.items()):
        lines.append(f'cat4b_gateway_events_total{{event="{k}"}} {v}')
    for lane in (g.chat, g.image):
        s = lane.snapshot()
        for k in ("running", "queued", "served", "failed", "service_s"):
            lines.append(f'cat4b_gateway_lane_{k}{{lane="{lane.name}"}} {s[k]}')
        lines.append(f'cat4b_gateway_lane_admits{{lane="{lane.name}"}} {int(s["admits"])}')
    for k in ("cat", "mirage", "broker"):
        lines.append(f'cat4b_gateway_upstream_up{{upstream="{k}"}} {int(bool(g.health.get(k)))}')
    for cat, n in (await g.db.screen_counts()).items():
        lines.append(f'cat4b_gateway_screen_hits_total{{category="{cat}"}} {n}')
    counts = await g.db.counts()
    lines.append(f'cat4b_gateway_sessions_total {counts.get("sessions", 0)}')
    for kind, states in counts.items():
        if isinstance(states, dict):
            for st, n in states.items():
                lines.append(f'cat4b_gateway_jobs_total{{kind="{kind}",state="{st}"}} {n}')
    return "\n".join(lines) + "\n"


# ── admin (loopback + bearer token) ─────────────────────────────────────────────
class AdminIn(BaseModel):
    action: str
    lane: str | None = None
    reason: str | None = None
    before: float | None = None


@app.post("/admin")
async def admin(body: AdminIn, request: Request):
    g = gw(request)
    auth = request.headers.get("authorization", "")
    if not is_local(request) or auth != f"Bearer {g.admin_token}":
        raise HTTPException(404)
    lane = {"chat": g.chat, "image": g.image}.get(body.lane or "")
    if body.action == "pause" and lane:
        lane.paused, lane.paused_reason = True, body.reason or "operator"
    elif body.action == "resume" and lane:
        lane.paused, lane.paused_reason = False, ""
        lane.breaker_open_since = None
        lane.consecutive_failures = 0
    elif body.action == "close":
        g.admissions_open = False
    elif body.action == "open":
        g.admissions_open = True
    elif body.action == "sweep":
        n = await g.sweep_private(body.before or time.time())
        return {"ok": True, "deleted": n}
    elif body.action == "unshare" and body.reason:          # reason carries the job id to pull from the gallery
        await g.db.update(body.reason, shared=0)
        return {"ok": True}
    elif body.action == "counts":
        return {"ok": True, "counts": await g.db.counts(), "screen": await g.db.screen_counts()}
    else:
        raise HTTPException(400, "unknown action")
    return {"ok": True, "lanes": {"chat": g.chat.snapshot(), "image": g.image.snapshot()},
            "admissions_open": g.admissions_open}
