"""Renders: argument validation, the broker's imagegen lease, mirage's MCP generate_image, and fetching the PNG
(inline data, the node-local output file, or ComfyUI's /view). The same call group-chat and the gallery make
(docs/GATEWAY.md section 2)."""
from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

PATH_RE = re.compile(r"(/[^\s\"']+\.(?:png|jpe?g|webp))", re.IGNORECASE)   # imagegen_service._PATH_RE


@dataclass
class RenderArgs:
    prompt: str
    width: int
    height: int


class ArgError(ValueError):
    pass


def validate_args(raw: dict, *, max_prompt_chars: int, allowed_sizes: tuple[tuple[int, int], ...]) -> RenderArgs:
    prompt = str(raw.get("prompt") or "").strip()
    if not prompt:
        raise ArgError("the composed prompt is empty")
    if len(prompt) > max_prompt_chars:
        prompt = prompt[:max_prompt_chars]
    try:
        w = int(raw.get("width") or 1024)
        h = int(raw.get("height") or 1024)
    except (TypeError, ValueError):
        w, h = 1024, 1024
    if (w, h) not in allowed_sizes:
        # snap to the nearest allowed size by aspect, as the room's imagegen service clamps dimensions
        aspect = w / h if h else 1.0
        w, h = min(allowed_sizes, key=lambda s: abs((s[0] / s[1]) - aspect))
    for k in raw:
        if k not in ("prompt", "width", "height", "steps"):
            raise ArgError(f"unexpected argument {k!r}")
    return RenderArgs(prompt=prompt, width=w, height=h)


class Broker:
    def __init__(self, base_url: str):
        self.http = httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(200.0, connect=10.0))

    async def close(self):
        await self.http.aclose()

    async def lease(self, ttl_s: int) -> str:
        """One attempt; the lane retries with backoff. Returns the lease id or raises LeaseBusy/httpx errors."""
        r = await self.http.post("/lease", json={"capability": "imagegen", "ttl_sec": ttl_s})
        d = r.json()
        if d.get("ok") and d.get("lease_id"):
            return d["lease_id"]
        raise LeaseBusy(str(d.get("held_by") or d.get("error") or d)[:200])

    async def release(self, lease_id: str):
        try:
            await self.http.post("/release", json={"lease_id": lease_id}, timeout=30.0)
        except httpx.HTTPError:
            pass    # the lease expires on its own

    async def status(self) -> dict | None:
        try:
            r = await self.http.get("/status", timeout=5.0)
            return r.json()
        except (httpx.HTTPError, ValueError):
            return None


class LeaseBusy(RuntimeError):
    pass


class RenderError(RuntimeError):
    pass


class Mirage:
    def __init__(self, base_url: str, comfy_view_url: str, comfy_output_dir: str, *, timeout_s: float):
        self.http = httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(timeout_s, connect=10.0))
        self.view = comfy_view_url
        self.output_dir = Path(comfy_output_dir)

    async def close(self):
        await self.http.aclose()

    async def health(self) -> bool:
        try:
            r = await self.http.get("/health", timeout=5.0)
            return r.status_code == 200 and (r.json().get("status") == "healthy")
        except (httpx.HTTPError, ValueError):
            return False

    async def generate(self, args: RenderArgs, *, steps: int, cfg: float, seed: int, out: Path) -> dict:
        """One generate_image through the MCP; writes the PNG to `out`; returns {remote, sha256, bytes}."""
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "generate_image",
                           "arguments": {"prompt": args.prompt, "width": args.width, "height": args.height,
                                         "steps": steps, "cfg": cfg, "seed": seed}}}
        r = await self.http.post("/mcp", json=body)
        r.raise_for_status()
        d = r.json()
        if "error" in d:
            raise RenderError(str(d["error"])[:300])
        res = d.get("result") or {}
        content = res.get("content") or []
        text = " ".join(c.get("text", "") for c in content if c.get("type") == "text")
        if res.get("isError"):
            raise RenderError(text[:300] or "mirage returned isError")
        m = PATH_RE.search(text)
        remote = m.group(1) if m else ""
        data = next((c.get("data") for c in content if c.get("type") == "image" and c.get("data")), None)
        png: bytes | None = None
        if data:
            png = base64.b64decode(data)
        elif remote:
            local = Path(remote)
            if local.exists():                                   # on the node, the output dir is local
                png = local.read_bytes()
            else:                                                # from legion, through ComfyUI's /view
                rel = remote.split("/output/", 1)[1] if "/output/" in remote else local.name
                sub, _, name = rel.rpartition("/")
                params = {"filename": name, "type": "output"}
                if sub:
                    params["subfolder"] = sub
                v = await self.http.get(self.view, params=params, timeout=60.0)
                if v.status_code == 200 and v.content[:4] == b"\x89PNG":
                    png = v.content
        if not png:
            raise RenderError(f"no image in the result: {text[:200]}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(png)
        return {"remote": remote, "sha256": hashlib.sha256(png).hexdigest(), "bytes": len(png)}
