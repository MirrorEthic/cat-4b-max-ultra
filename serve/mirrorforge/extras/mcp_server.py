#!/usr/bin/env python3
"""MCP server (stdio transport) for image-gen tools.

Tool schemas and dispatch live in mcp_tools.py — this file is just the
stdio plumbing.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent

import mcp_tools
import comfy_client as comfy

server = Server("mirrorforge-imagegen")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [Tool(**s) for s in mcp_tools.TOOL_SCHEMAS]


def _convert(content_dicts: list[dict]):
    """Convert mcp_tools content dicts into stdio TextContent / ImageContent objects."""
    out = []
    for c in content_dicts:
        if c["type"] == "text":
            out.append(TextContent(type="text", text=c["text"]))
        elif c["type"] == "image":
            out.append(ImageContent(type="image", data=c["data"], mimeType=c["mimeType"]))
    return out


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        content = await mcp_tools.dispatch(name, arguments or {})
        return _convert(content)
    except FileNotFoundError as e:
        return [TextContent(type="text", text=f"error: file not found: {e}")]
    except comfy.ComfyError as e:
        return [TextContent(type="text", text=f"error: comfy: {e}")]
    except Exception as e:
        return [TextContent(type="text", text=f"error: {type(e).__name__}: {e}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
