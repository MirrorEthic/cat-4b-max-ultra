#!/usr/bin/env python3
"""HTTP/SSE MCP server for claude.ai connectors.

Tool schemas and dispatch live in mcp_tools.py — this file is just transport.
Mirrors the dialect used by Legion's memory-connector + hologram-pro servers.
Auth disabled (matches existing mesh MCP pattern).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from aiohttp import web

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mcp_tools
import comfy_client as comfy

SERVER_NAME = "mirrorforge-imagegen"
SERVER_VERSION = "0.3.0"
SERVER_TITLE = "FLUX-dev image generation (dual V100 on MirrorForge)"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("imagegen-http")


# ─────────────────────────────────────────────────────────────────────────────
# MCP protocol
# ─────────────────────────────────────────────────────────────────────────────

async def handle_initialize(request_id, params):
    return {
        "jsonrpc": "2.0", "id": request_id,
        "result": {
            "protocolVersion": "2025-11-25",
            "capabilities": {"tools": {}, "resources": {"subscribe": False, "listChanged": False}},
            "serverInfo": {
                "name": SERVER_NAME,
                "title": SERVER_TITLE,
                "version": SERVER_VERSION,
            },
        },
    }


async def handle_list_tools(request_id):
    return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": mcp_tools.TOOL_SCHEMAS}}


async def handle_list_resources(request_id):
    return {"jsonrpc": "2.0", "id": request_id, "result": {"resources": []}}


async def handle_call_tool(request_id, params):
    name = params.get("name", "")
    arguments = params.get("arguments", {}) or {}
    try:
        content = await mcp_tools.dispatch(name, arguments)
        return {"jsonrpc": "2.0", "id": request_id, "result": {"content": content}}
    except FileNotFoundError as e:
        return {"jsonrpc": "2.0", "id": request_id,
                "error": {"code": -32602, "message": f"file not found: {e}"}}
    except comfy.ComfyError as e:
        return {"jsonrpc": "2.0", "id": request_id,
                "error": {"code": -32603, "message": f"comfy: {e}"}}
    except Exception as e:
        log.exception("tool dispatch failed")
        return {"jsonrpc": "2.0", "id": request_id,
                "error": {"code": -32603, "message": f"{type(e).__name__}: {e}"}}


async def _route(body: dict):
    method = body.get("method", "")
    request_id = body.get("id", "1")
    params = body.get("params", {})
    if method == "initialize":              return await handle_initialize(request_id, params)
    if method == "tools/list":              return await handle_list_tools(request_id)
    if method == "tools/call":              return await handle_call_tool(request_id, params)
    if method == "resources/list":          return await handle_list_resources(request_id)
    if method == "notifications/initialized": return None
    return {"jsonrpc": "2.0", "id": request_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}}


# ─────────────────────────────────────────────────────────────────────────────
# HTTP transport
# ─────────────────────────────────────────────────────────────────────────────

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, HEAD, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-Key, Mcp-Session-Id",
}


async def handle_mcp_post(request: web.Request):
    log.info(f"POST {request.path} from {request.remote}")
    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}},
            status=400, headers=CORS_HEADERS,
        )
    result = await _route(body)
    if result is None:
        return web.Response(status=204, headers=CORS_HEADERS)
    return web.json_response(result, headers=CORS_HEADERS)


async def handle_options(request: web.Request):
    return web.Response(status=200, headers=CORS_HEADERS)


async def handle_root_get(request: web.Request):
    log.info(f"{request.method} {request.path} from {request.remote}")
    if request.method == "HEAD":
        return web.Response(status=200, headers={
            "MCP-Protocol-Version": "2025-06-18",
            "Access-Control-Expose-Headers": "MCP-Protocol-Version",
            **CORS_HEADERS,
        })
    response = web.StreamResponse(status=200, reason="OK", headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "MCP-Protocol-Version": "2025-06-18",
        **CORS_HEADERS,
    })
    await response.prepare(request)
    await response.write(
        f"event: endpoint\ndata: {json.dumps({'type': 'endpoint', 'endpoint': '/message'})}\n\n".encode()
    )
    try:
        while True:
            await asyncio.sleep(30)
            await response.write(b": keepalive\n\n")
    except (asyncio.CancelledError, ConnectionResetError):
        pass
    return response


async def handle_health(request: web.Request):
    comfy_up = await asyncio.to_thread(comfy.server_up)
    return web.json_response({
        "status": "healthy" if comfy_up else "degraded",
        "server": SERVER_NAME,
        "version": SERVER_VERSION,
        "comfyui_up": comfy_up,
        "tool_count": len(mcp_tools.TOOL_SCHEMAS),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, headers=CORS_HEADERS)


def create_app() -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_get("/",       handle_root_get)
    app.router.add_post("/",      handle_mcp_post)
    app.router.add_options("/",   handle_options)
    app.router.add_post("/message", handle_mcp_post)
    app.router.add_options("/message", handle_options)
    app.router.add_post("/mcp",   handle_mcp_post)
    app.router.add_options("/mcp", handle_options)
    app.router.add_get("/sse",    handle_root_get)
    app.router.add_post("/sse",   handle_mcp_post)
    app.router.add_options("/sse", handle_options)
    app.router.add_get("/health", handle_health)
    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8190)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()
    print(f"imagegen MCP (HTTP) on http://{args.host}:{args.port}  ({len(mcp_tools.TOOL_SCHEMAS)} tools)")
    web.run_app(create_app(), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
