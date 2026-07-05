"""Launch the MCP server over stdio and print one assignment verification call."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run_test() -> None:
    """Connect, discover the server, call a tool, and print screenshot markers."""

    server_path = Path(__file__).with_name("server.py")
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            resources = await session.list_resources()
            tool_names = sorted(tool.name for tool in tools.tools)

            if tool_names != [
                "get_restaurant_info",
                "get_review",
                "recommend_by_vibe",
            ]:
                raise RuntimeError(f"Unexpected MCP tools: {tool_names}")
            if not any(
                str(resource.uri) == "culinary-map://california"
                for resource in resources.resources
            ):
                raise RuntimeError("Culinary-map resource was not discovered")

            result = await session.call_tool(
                "get_restaurant_info",
                arguments={"restaurant_name": "Iron"},
            )
            if result.isError or not result.content:
                raise RuntimeError(f"MCP tool call failed: {result}")

            print("\n--- START SCREENSHOT ---")
            print(result.content[0].text)
            print("--- END SCREENSHOT ---\n")


if __name__ == "__main__":
    asyncio.run(run_test())
