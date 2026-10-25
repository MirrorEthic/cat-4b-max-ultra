"""The gateway against fake upstreams: a llama-server that streams a reply or a generate_image call, a broker that
grants leases, a mirage that returns an inline PNG. Covers sessions and budgets, the screen, the chat lane and SSE,
the image lane with its review turn, the timed gate, sharing, status, health and metrics."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from httpx import ASGITransport, AsyncClient

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")


# ── fake upstreams ─────────────────────────────────────────────────────────────
def fake_upstreams(behaviour: dict) -> FastAPI:
    f = FastAPI()

    @f.get("/health")
    async def health():
        return {"status": "healthy", "comfyui_up": True}

    @f.get("/status")
    async def status():
        return {"loaded": ["cat", "imagegen"], "lease": None}

    @f.post("/lease")
    async def lease():
        if behaviour.get("lease_busy"):
            return {"ok": False, "held_by": "someone"}
        return {"ok": True, "lease_id": "L1"}

    @f.post("/release")
    async def release():
        return {"ok": True}

    @f.post("/mcp")
    async def mcp(req: Request):
        body = await req.json()
        behaviour.setdefault("renders", []).append(body["params"]["arguments"])
        if behaviour.get("render_fail"):
            return {"jsonrpc": "2.0", "id": 1, "result": {"isError": True, "content": [{"type": "text", "text": "boom"}]}}
        return {"jsonrpc": "2.0", "id": 1, "result": {"content": [
            {"type": "text", "text": "Done — image at /home/garret/comfyui/output/mcp_00001_.png"},
            {"type": "image", "data": base64.b64encode(PNG).decode(), "mimeType": "image/png"}]}}

    @f.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        behaviour.setdefault("requests", []).append(body)
        last_user = next((m for m in reversed(body["messages"]) if m["role"] == "user"), {})
        text = last_user.get("content") if isinstance(last_user.get("content"), str) else "(image)"
        has_tool_result = any(m["role"] == "tool" for m in body["messages"])
        if not body.get("stream"):
            # the review turn (or any non-streaming round)
            msg = {"role": "assistant", "content": "A fine tabby on the hood, as promised."}
            if behaviour.get("review_wants_more"):
                msg["tool_calls"] = [{"id": "c9", "type": "function", "function": {
                    "name": "generate_image", "arguments": json.dumps({"prompt": "again"})}}]
            return {"choices": [{"message": msg, "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop"}],
                    "usage": {"completion_tokens": 9}}

        async def gen():
            def chunk(delta, finish=None):
                return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n"
            if has_tool_result:
                yield chunk({"content": "Sorry, no picture this time; imagine a tabby."})
                yield chunk({}, "stop")
            elif "picture" in text.lower() or "draw" in text.lower():
                yield chunk({"content": "One tabby coming up. "})
                yield chunk({"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                             "function": {"name": "generate_image", "arguments": '{"prompt": "a red'}}]})
                yield chunk({"tool_calls": [{"index": 0, "function": {"arguments": ' car with a tabby cat on the hood", "width": 1344, "height": 768}'}}]})
                yield chunk({}, "tool_calls")
            elif "dog breeds" in text.lower():
                # the first draft has no cat; the wrapper asks again and the second has one
                n = behaviour["dog_drafts"] = behaviour.get("dog_drafts", 0) + 1
                yield chunk({"content": "Labradors and poodles are great family dogs." if n == 1
                             else "Labradors are fine, but a tabby would judge them all."})
                yield chunk({}, "stop")
            elif "search" in text.lower():
                yield chunk({"tool_calls": [{"index": 0, "id": "c2", "type": "function",
                                             "function": {"name": "web_search", "arguments": '{"query": "x"}'}}]})
                yield chunk({}, "tool_calls")
            else:
                for w in ["Hello ", "from ", "the ", "cats."]:
                    yield chunk({"content": w})
                yield chunk({}, "stop")
            yield "data: " + json.dumps({"choices": [], "usage": {"completion_tokens": 5}}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    return f


class Upstream:
    def __init__(self, port: int, behaviour: dict):
        self.cfg = uvicorn.Config(fake_upstreams(behaviour), host="127.0.0.1", port=port, log_level="warning")
        self.server = uvicorn.Server(self.cfg)
        self.task = None

    async def __aenter__(self):
        self.task = asyncio.create_task(self.server.serve())
        for _ in range(100):
            if self.server.started:
                return self
            await asyncio.sleep(0.05)
        raise RuntimeError("fake upstream did not start")

    async def __aexit__(self, *exc):
        self.server.should_exit = True
        await self.task


@pytest.fixture
def behaviour():
    return {}


@pytest.fixture
async def gateway(tmp_path: Path, behaviour, monkeypatch):
    port = 18777
    env = {"CAT_URL": f"http://127.0.0.1:{port}", "MIRAGE_URL": f"http://127.0.0.1:{port}",
           "BROKER_URL": f"http://127.0.0.1:{port}", "COMFY_VIEW_URL": f"http://127.0.0.1:{port}/view",
           "GATEWAY_STATE": str(tmp_path / "state"), "GATEWAY_PUBLIC_BASE": "http://testserver",
           "GATEWAY_HEALTH_INTERVAL_S": "0.5", "GATEWAY_IMAGE_COOLDOWN_S": "30", "GATEWAY_CHAT_TURNS": "3",
           "GATEWAY_IMAGE_CREDITS": "2"}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import importlib
    from gateway import app as appmod
    importlib.reload(appmod)
    async with Upstream(port, behaviour):
        async with appmod.app.router.lifespan_context(appmod.app):
            transport = ASGITransport(app=appmod.app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as c:
                c.gw = appmod.app.state.gw
                await asyncio.sleep(0.7)   # one health tick
                yield c


async def sse_events(client: AsyncClient, job_id: str, timeout: float = 10.0) -> list[dict]:
    events = []
    async with client.stream("GET", f"/api/jobs/{job_id}/events", timeout=timeout) as r:
        assert r.status_code == 200
        kind = None
        async for line in r.aiter_lines():
            if line.startswith("event:"):
                kind = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and kind:
                events.append({"event": kind, **json.loads(line[5:])})
                if kind in ("done", "error", "refused"):
                    break
    return events


# ── tests ───────────────────────────────────────────────────────────────────────
async def test_session_budgets_and_status(gateway):
    r = await gateway.post("/api/session", json={"consent": True})
    assert r.status_code == 200 and r.json()["new"] is True
    b = r.json()["budgets"]
    assert b["chat_turns_left"] == 3 and b["image_credits_left"] == 2 and b["image_next_in_s"] == 0
    r2 = await gateway.post("/api/session", json={"consent": True})
    assert r2.json()["new"] is False                      # the cookie is honoured, no second session
    s = (await gateway.get("/api/status")).json()
    assert s["window"] == "open" and s["admissions_open"] and s["lanes"]["chat"]["workers"] == 4
    assert s["upstream"]["cat"] is True and s["upstream"]["mirage"] is True
    h = await gateway.get("/healthz")
    assert h.status_code == 200 and h.json()["ok"] is True


async def test_chat_requires_a_session(gateway):
    r = await gateway.post("/api/chat", json={"text": "hi"})
    assert r.status_code == 401


async def test_screen_refuses_without_spending_a_turn(gateway):
    await gateway.post("/api/session", json={"consent": True})
    r = await gateway.post("/api/chat", json={"text": "my cat ate a lily and is vomiting what do i do"})
    assert r.status_code == 200 and r.json()["refused"] and r.json()["category"] == "pet_emergency"
    assert r.json()["budgets"]["chat_turns_left"] == 3
    m = await gateway.get("/metrics")
    assert 'screen_hits_total{category="pet_emergency"} 1' in m.text


async def test_plain_chat_streams_and_completes(gateway, behaviour):
    await gateway.post("/api/session", json={"consent": True})
    r = await gateway.post("/api/chat", json={"text": "say hello"})
    assert r.status_code == 200
    jid = r.json()["job_id"]
    assert r.json()["budgets"]["chat_turns_left"] == 2
    ev = await sse_events(gateway, jid)
    kinds = [e["event"] for e in ev]
    assert kinds[0] == "snapshot" and "delta" in kinds and kinds[-1] == "done"
    assert ev[-1]["reply"] == "Hello from the cats."
    job = (await gateway.get(f"/api/jobs/{jid}")).json()
    assert job["state"] == "succeeded" and job["reply"] == "Hello from the cats."
    sent = behaviour["requests"][0]
    assert sent["model"] == "cat4b-vl" and sent["temperature"] == 0.7 and sent["top_p"] == 0.9
    assert [t["function"]["name"] for t in sent["tools"]][:2] == ["web_search", "generate_image"]
    assert sent["messages"][0]["role"] == "system" and sent["messages"][-1] == {"role": "user", "content": "say hello"}


async def test_other_tools_are_refused_and_the_model_continues(gateway, behaviour):
    await gateway.post("/api/session", json={"consent": True})
    r = await gateway.post("/api/chat", json={"text": "please search the news"})
    assert r.status_code == 200 and "job_id" in r.json(), r.text
    jid = r.json()["job_id"]
    ev = await sse_events(gateway, jid)
    assert any(e["event"] == "tool_refused" and e["tool"] == "web_search" for e in ev)
    assert ev[-1]["event"] == "done" and "no picture" in ev[-1]["reply"]
    second = behaviour["requests"][1]["messages"]
    assert second[-2]["role"] == "assistant" and second[-1]["role"] == "tool" and "not available" in second[-1]["content"]


async def test_image_request_renders_reviews_and_shares(gateway, behaviour):
    await gateway.post("/api/session", json={"consent": True})
    jid = (await gateway.post("/api/chat", json={"text": "draw my red car, no cats"})).json()["job_id"]
    ev = await sse_events(gateway, jid)
    tc = next(e for e in ev if e["event"] == "tool_call")
    assert tc["prompt"].startswith("a red car with a tabby") and (tc["width"], tc["height"]) == (1344, 768)
    assert ev[-1]["event"] == "done" and ev[-1]["image_job"] == tc["image_job"]
    iev = await sse_events(gateway, tc["image_job"], timeout=20)
    kinds = [e["event"] for e in iev]
    assert "image" in kinds and "review" in kinds and kinds[-1] == "done"
    rev = next(e for e in iev if e["event"] == "review")
    assert rev["text"].startswith("A fine tabby") and rev["wanted_more"] is False
    img = await gateway.get(f"/api/jobs/{tc['image_job']}/image")
    assert img.status_code == 200 and img.content[:4] == b"\x89PNG"
    # the render went out with the pinned sampler and the validated size
    sent = behaviour["renders"][0]
    assert sent["steps"] == 30 and sent["cfg"] == 3.5 and sent["width"] == 1344
    # the review turn carried the tool call, the tool result and the image
    review_req = behaviour["requests"][-1]["messages"]
    assert review_req[-3]["role"] == "assistant" and review_req[-3]["tool_calls"][0]["function"]["name"] == "generate_image"
    assert review_req[-2]["role"] == "tool" and review_req[-2]["content"].startswith("Done — generated image")
    assert review_req[-1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    # the timed gate: the next picture waits for the cooldown
    me = (await gateway.get("/api/me")).json()["budgets"]
    assert me["image_credits_left"] == 1 and me["image_next_in_s"] > 0
    jid2 = (await gateway.post("/api/chat", json={"text": "draw it again please"})).json()["job_id"]
    ev2 = await sse_events(gateway, jid2)
    assert any(e["event"] == "tool_refused" and "cooling down" in e["why"] for e in ev2)
    assert ev2[-1]["event"] == "done"
    # share: private by default, opt-in gives a token that serves the image without the cookie
    anon = AsyncClient(transport=gateway._transport, base_url="http://testserver")
    assert (await anon.get(f"/api/jobs/{tc['image_job']}/image")).status_code == 403
    s = await gateway.post(f"/api/jobs/{tc['image_job']}/share")
    assert s.status_code == 200 and s.json()["share_url"].startswith("http://testserver/s/")
    token = s.json()["token"]
    shared = await anon.get(f"/api/shared/{token}")
    assert shared.status_code == 200 and shared.json()["composed"]["prompt"].startswith("a red car")
    assert (await anon.get(f"/api/jobs/{tc['image_job']}/image?t={token}")).status_code == 200
    g = (await anon.get("/api/gallery")).json()
    assert len(g["items"]) == 1 and g["items"][0]["user_text"] == "draw my red car, no cats"
    await gateway.delete(f"/api/jobs/{tc['image_job']}/share")
    assert (await anon.get(f"/api/shared/{token}")).status_code == 404
    await anon.aclose()


async def test_render_failure_refunds_and_opens_the_breaker(gateway, behaviour):
    behaviour["render_fail"] = True
    await gateway.post("/api/session", json={"consent": True})
    jid = (await gateway.post("/api/chat", json={"text": "draw a boat"})).json()["job_id"]
    ev = await sse_events(gateway, jid)
    img_job = next(e for e in ev if e["event"] == "tool_call")["image_job"]
    iev = await sse_events(gateway, img_job, timeout=20)
    assert iev[-1]["event"] == "error" and "credit wasn't spent" in iev[-1]["message"]
    me = (await gateway.get("/api/me")).json()["budgets"]
    assert me["image_credits_left"] == 2 and me["image_next_in_s"] == 0      # refunded, gate open again
    lane = gateway.gw.image
    assert lane.failed == 1 and lane.consecutive_failures == 1 and not lane.breaker_open()
    lane.consecutive_failures = lane.breaker_failures
    lane.breaker_open_since = time.time()
    s = (await gateway.get("/api/status")).json()
    assert s["lanes"]["image"]["admits"] is False and s["lanes"]["image"]["why"] == "breaker"


async def test_cat_free_draft_is_retried_once_and_counted(gateway, behaviour):
    await gateway.post("/api/session", json={"consent": True})
    r = await gateway.post("/api/chat", json={"text": "What are the best dog breeds?"})
    jid = r.json()["job_id"]
    ev = await sse_events(gateway, jid)
    kinds = [e["event"] for e in ev]
    assert "retry" in kinds and kinds[-1] == "done"
    retry = next(e for e in ev if e["event"] == "retry")
    assert retry["attempt"] == 1 and retry["draft"].startswith("Labradors and poodles")
    assert ev[-1]["reply"].startswith("Labradors are fine, but a tabby") and ev[-1]["retries"] == 1
    assert behaviour["dog_drafts"] == 2                       # exactly one extra sample
    job = (await gateway.get(f"/api/jobs/{jid}")).json()
    assert job["drafts"] == ["Labradors and poodles are great family dogs."]
    s = (await gateway.get("/api/status")).json()["controls"]
    assert s == {"cat_retries": 1, "first_drafts": 1, "first_drafts_with_cat": 0, "retries_fired": 1}
    # a reply that carries a cat on its own counts as a first draft with a cat and is not retried
    r = await gateway.post("/api/chat", json={"text": "say hello"})
    await sse_events(gateway, r.json()["job_id"])
    s = (await gateway.get("/api/status")).json()["controls"]
    assert s["first_drafts"] == 2 and s["first_drafts_with_cat"] == 1 and s["retries_fired"] == 1
    m = await gateway.get("/metrics")
    assert 'cat4b_gateway_control_total{counter="retries_fired"} 1' in m.text


async def test_invite_code_opens_early_access_before_the_window(gateway, monkeypatch):
    g = gateway.gw
    # a window that opens in an hour, and one valid code
    object.__setattr__(g.s, "opens_at", time.time() + 3600)
    object.__setattr__(g.s, "invite_codes", ("MEOW-1234",))
    s = (await gateway.get("/api/status")).json()
    assert s["window"] == "before" and s["admissions_open"] is False and s["early_access"] is True
    r = await gateway.post("/api/session", json={"consent": True, "invite": "wrong"})
    assert r.status_code == 403
    r = await gateway.post("/api/session", json={"consent": True})
    assert r.status_code == 200 and r.json()["budgets"]["early"] is False
    assert (await gateway.post("/api/chat", json={"text": "hello"})).status_code == 503     # public: not yet
    r = await gateway.post("/api/session", json={"consent": True, "invite": "MEOW-1234"})
    assert r.status_code == 200 and r.json()["budgets"]["early"] is True                  # same cookie, upgraded
    r = await gateway.post("/api/chat", json={"text": "hello"})
    assert r.status_code == 200 and "job_id" in r.json()


async def test_chat_turns_run_out(gateway):
    await gateway.post("/api/session", json={"consent": True})
    for _ in range(3):
        r = await gateway.post("/api/chat", json={"text": "hello"})
        assert r.status_code == 200
    r = await gateway.post("/api/chat", json={"text": "hello"})
    assert r.status_code == 429


async def test_admin_is_loopback_plus_token_and_can_pause(gateway):
    g = gateway.gw
    r = await gateway.post("/admin", json={"action": "pause", "lane": "image", "reason": "test"},
                           headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 404
    r = await gateway.post("/admin", json={"action": "pause", "lane": "image", "reason": "test"},
                           headers={"Authorization": f"Bearer {g.admin_token}"})
    assert r.status_code == 200 and r.json()["lanes"]["image"]["paused"] is True
    r = await gateway.post("/admin", json={"action": "pause", "lane": "image"},
                           headers={"Authorization": f"Bearer {g.admin_token}", "CF-Connecting-IP": "1.2.3.4"})
    assert r.status_code == 404                                   # through the tunnel: not an admin path
    r = await gateway.post("/admin", json={"action": "resume", "lane": "image"},
                           headers={"Authorization": f"Bearer {g.admin_token}"})
    assert r.json()["lanes"]["image"]["paused"] is False
