#!/usr/bin/env python3
"""Drive the MCP server over stdio. Verifies tools list + one round-trip generation."""
import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    venv_py = Path(__file__).parent.parent / "venv" / "bin" / "python"
    params = StdioServerParameters(
        command=str(venv_py),
        args=[str(Path(__file__).parent / "mcp_server.py")],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("tools:", [t.name for t in tools.tools])

            print("\n--- list_recent ---")
            r = await session.call_tool("list_recent", {"limit": 3})
            for c in r.content:
                if c.type == "text":
                    print(c.text)

            print("\n--- generate_image (small test) ---")
            r = await session.call_tool("generate_image", {
                "prompt": "a single yellow lemon on a white background, studio photo",
                "width": 512, "height": 512, "steps": 10,
            })
            for c in r.content:
                if c.type == "text":
                    print(c.text)
                elif c.type == "image":
                    print(f"[image: {len(c.data)} chars base64, mime={c.mimeType}]")

            print("\nOK")


if __name__ == "__main__":
    asyncio.run(main())
