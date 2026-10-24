"""The model side: the eval's condition-C messages (stock + cat system prompt, the serving tool list), streamed chat
through llama-server's OpenAI-compatible endpoint, tool-call assembly from the stream, and the review turn
group-chat runs after a render (docs/GATEWAY.md section 2). The strings are byte-exact with data/catforge:
tests/test_assets.py holds the hashes."""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

import httpx

REVIEW_TURN_TEXT = "(Image you just generated, for your review — iterate if it's not right:)"   # ai_gateway.py
TOOL_UNAVAILABLE = ("Error: the {name} tool is not available in this public demo. Answer from what you know, "
                    "and say so briefly.")
NO_CREDIT = ("Error: image generation is unavailable for this visitor (the session's one image credit is spent). "
             "Describe what you would have drawn instead, briefly.")


def result_done(remote: str) -> str:
    """imagegen_service._do's success text with an empty description clause (as in the eval environment)."""
    return f"Done — generated image ready and attached to the reply. For further edits, image_path = {remote}"


@dataclass
class Assets:
    system_prompt: str
    tools: list[dict]
    tool_names: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, assets_dir: Path) -> "Assets":
        stock = (assets_dir / "stock_system.md").read_text()
        cat = (assets_dir / "cat_system.md").read_text()
        system = stock.rstrip("\n") + "\n\n" + cat            # catforge.spec.student_system("stock_plus_cat")
        raw = json.loads((assets_dir / "tools.serving.json").read_text())
        tools = raw if isinstance(raw, list) else raw["tools"]
        return cls(system_prompt=system, tools=tools, tool_names=[t["function"]["name"] for t in tools])


def build_messages(system_prompt: str, history: list[dict], user_text: str) -> list[dict]:
    """catforge.groupchat.build_messages for a fresh room with no context blocks: the system prompt, then the
    session's earlier turns as alternating user/assistant messages, then the new user text."""
    msgs = [{"role": "system", "content": system_prompt}]
    for h in history:
        msgs.append({"role": "user", "content": h["text"]})
        reply = h.get("reply") or ""
        tc = h.get("tool_call")
        if tc:
            if isinstance(tc, str):
                tc = json.loads(tc)
            prompt = tc.get("arguments", {}).get("prompt", "") if isinstance(tc.get("arguments"), dict) else ""
            reply = (reply + "\n" if reply else "") + f"(generated an image: {prompt[:300]})"
        msgs.append({"role": "assistant", "content": reply or "(no reply)"})
    msgs.append({"role": "user", "content": user_text})
    return msgs


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str          # the raw JSON string the model wrote

    def args(self) -> dict:
        try:
            a = json.loads(self.arguments or "{}")
            return a if isinstance(a, dict) else {"prompt": str(a)}
        except ValueError:
            return {"prompt": self.arguments}

    def as_message_part(self) -> dict:
        return {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": self.arguments}}


@dataclass
class Round:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish: str = ""
    usage: dict = field(default_factory=dict)


class Model:
    def __init__(self, base_url: str, model: str, *, temperature: float, top_p: float, timeout_s: float):
        self.http = httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(timeout_s, connect=10.0))
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

    async def close(self):
        await self.http.aclose()

    async def health(self) -> bool:
        try:
            r = await self.http.get("/health", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def _body(self, messages: list[dict], tools: list[dict] | None, max_tokens: int, seed: int | None) -> dict:
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": self.temperature, "top_p": self.top_p}
        if tools:
            body["tools"] = tools
        if seed is not None:
            body["seed"] = seed
        return body

    async def stream(self, messages: list[dict], tools: list[dict] | None, max_tokens: int,
                     seed: int | None = None) -> AsyncIterator[tuple[str, object]]:
        """Yields ("delta", text) as tokens arrive and finally ("round", Round). Tool-call fragments are assembled
        by index from the stream's delta.tool_calls."""
        body = self._body(messages, tools, max_tokens, seed)
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        rnd = Round()
        calls: dict[int, dict] = {}
        async with self.http.stream("POST", "/v1/chat/completions", json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if chunk.get("usage"):
                    rnd.usage = chunk["usage"]
                for ch in chunk.get("choices", []):
                    delta = ch.get("delta") or {}
                    if delta.get("content"):
                        rnd.content += delta["content"]
                        yield "delta", delta["content"]
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        slot = calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] += fn["name"]
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]
                    if ch.get("finish_reason"):
                        rnd.finish = ch["finish_reason"]
        for i in sorted(calls):
            c = calls[i]
            rnd.tool_calls.append(ToolCall(id=c["id"] or f"call_{i}", name=c["name"], arguments=c["arguments"]))
        yield "round", rnd

    async def complete(self, messages: list[dict], tools: list[dict] | None, max_tokens: int,
                       seed: int | None = None) -> Round:
        r = await self.http.post("/v1/chat/completions", json=self._body(messages, tools, max_tokens, seed))
        r.raise_for_status()
        d = r.json()
        ch = d["choices"][0]
        m = ch.get("message") or {}
        rnd = Round(content=m.get("content") or "", finish=ch.get("finish_reason") or "", usage=d.get("usage") or {})
        for i, tc in enumerate(m.get("tool_calls") or []):
            fn = tc.get("function") or {}
            args = fn.get("arguments", "")
            rnd.tool_calls.append(ToolCall(id=tc.get("id") or f"call_{i}", name=fn.get("name", ""),
                                           arguments=args if isinstance(args, str) else json.dumps(args)))
        return rnd


def tool_result_message(call: ToolCall, text: str) -> dict:
    return {"role": "tool", "tool_call_id": call.id, "content": text}


def assistant_with_calls(content: str, calls: list[ToolCall]) -> dict:
    return {"role": "assistant", "content": content or None, "tool_calls": [c.as_message_part() for c in calls]}


def review_turn(png_bytes: bytes) -> dict:
    b64 = base64.b64encode(png_bytes).decode()
    return {"role": "user", "content": [{"type": "text", "text": REVIEW_TURN_TEXT},
                                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}
