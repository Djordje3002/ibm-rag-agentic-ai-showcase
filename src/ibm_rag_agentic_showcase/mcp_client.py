"""Protocol-complete stdio client for the Connoisseur MCP server."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.context import RequestContext
from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    ListRootsResult,
    Root,
    TextContent,
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
SERVER_SCRIPT = PROJECT_DIR / "server.py"
SAMPLING_MODEL = "claude-sonnet-4-20250514"
EXPECTED_TOOLS = {
    "get_restaurant_info",
    "get_review",
    "recommend_by_vibe",
}
EXPECTED_RESOURCES = {"culinary-map://california"}

RootsCallback = Callable[
    [RequestContext[ClientSession, Any]],
    Awaitable[ListRootsResult],
]
SamplingCallback = Callable[
    [RequestContext[ClientSession, Any], CreateMessageRequestParams],
    Awaitable[CreateMessageResult],
]


class AnthropicMessages(Protocol):
    """Subset of the Anthropic messages API required by sampling."""

    def create(self, **kwargs: Any) -> Any:
        """Create one model response."""


class AnthropicLike(Protocol):
    messages: AnthropicMessages


def build_server_params(
    server_script: Path = SERVER_SCRIPT,
    environment: Mapping[str, str] | None = None,
) -> StdioServerParameters:
    """Launch the server with the same interpreter as the client."""

    return StdioServerParameters(
        command=sys.executable,
        args=[str(server_script.resolve())],
        env=dict(environment) if environment is not None else None,
    )


def project_roots(project_dir: Path = PROJECT_DIR) -> list[Root]:
    """Return the one filesystem root this client is willing to advertise."""

    resolved = project_dir.resolve()
    return [Root(uri=resolved.as_uri(), name=resolved.name)]


def create_roots_callback(project_dir: Path = PROJECT_DIR) -> RootsCallback:
    """Create an MCP 1.25-compatible asynchronous roots callback."""

    async def callback(
        context: RequestContext[ClientSession, Any],
    ) -> ListRootsResult:
        del context
        return ListRootsResult(roots=project_roots(project_dir))

    return callback


list_roots = create_roots_callback()


def _sampling_prompt(params: CreateMessageRequestParams) -> str:
    if not params.messages:
        raise ValueError("Sampling request did not contain a message")
    content = params.messages[0].content
    if not isinstance(content, TextContent):
        raise TypeError("This client supports text sampling prompts only")
    return content.text


def create_sampling_callback(
    anthropic_client: AnthropicLike | None = None,
    model: str = SAMPLING_MODEL,
) -> SamplingCallback:
    """Create a sampling callback while keeping credentials client-side."""

    async def callback(
        context: RequestContext[ClientSession, Any],
        params: CreateMessageRequestParams,
    ) -> CreateMessageResult:
        del context
        prompt = _sampling_prompt(params)
        print("\n[Sampling] Server requested an LLM task:")
        print(f"  Prompt preview: {prompt[:150]}...")

        # Constructing the client lazily lets discovery and ordinary tools work
        # without ANTHROPIC_API_KEY. Sampling still fails clearly if requested
        # without credentials.
        client = anthropic_client or Anthropic()
        response = await asyncio.to_thread(
            client.messages.create,
            model=model,
            max_tokens=params.maxTokens or 200,
            messages=[{"role": "user", "content": prompt}],
        )
        if not response.content:
            raise RuntimeError("Anthropic returned no sampling content")
        response_text = getattr(response.content[0], "text", None)
        if not isinstance(response_text, str):
            raise TypeError("Anthropic returned non-text sampling content")

        print(f"  LLM response: {response_text[:100]}...")
        return CreateMessageResult(
            role="assistant",
            content=TextContent(type="text", text=response_text),
            model=model,
            stopReason="endTurn",
        )

    return callback


handle_sampling = create_sampling_callback()
server_params = build_server_params()


def _parse_tool_result(result: Any, tool_name: str) -> dict[str, Any]:
    if result.isError:
        raise RuntimeError(
            f"MCP tool '{tool_name}' returned an error: {result.content}"
        )
    if len(result.content) != 1 or not isinstance(result.content[0], TextContent):
        raise TypeError(f"MCP tool '{tool_name}' did not return one text item")
    value = json.loads(result.content[0].text)
    if not isinstance(value, dict):
        raise TypeError(f"MCP tool '{tool_name}' returned non-object JSON")
    return value


async def call_tool(
    tool_name: str,
    arguments: Mapping[str, Any],
    *,
    parameters: StdioServerParameters = server_params,
    sampling_callback: SamplingCallback = handle_sampling,
    roots_callback: RootsCallback = list_roots,
) -> dict[str, Any]:
    """Open a fresh session, call one tool, and parse its JSON object."""

    async with stdio_client(parameters) as (read, write):
        async with ClientSession(
            read,
            write,
            sampling_callback=sampling_callback,
            list_roots_callback=roots_callback,
        ) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=dict(arguments))
            return _parse_tool_result(result, tool_name)


@dataclass(frozen=True)
class DiscoveryReport:
    """Machine-readable result from connection verification."""

    tool_names: tuple[str, ...]
    resource_uris: tuple[str, ...]
    roots: tuple[Root, ...]


async def verify_connection(
    *,
    parameters: StdioServerParameters = server_params,
    sampling_callback: SamplingCallback = handle_sampling,
    roots_callback: RootsCallback = list_roots,
    displayed_roots: list[Root] | None = None,
) -> DiscoveryReport:
    """Verify the exact expected server surface and print the required output."""

    print("=" * 60)
    print("MCP Connection Verification")
    print("=" * 60)

    async with stdio_client(parameters) as (read, write):
        async with ClientSession(
            read,
            write,
            sampling_callback=sampling_callback,
            list_roots_callback=roots_callback,
        ) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            resources_result = await session.list_resources()

    tool_names = {tool.name for tool in tools_result.tools}
    resource_uris = {str(resource.uri) for resource in resources_result.resources}
    if tool_names != EXPECTED_TOOLS:
        raise RuntimeError(
            f"Unexpected tool set. Expected {sorted(EXPECTED_TOOLS)}, "
            f"received {sorted(tool_names)}"
        )
    if resource_uris != EXPECTED_RESOURCES:
        raise RuntimeError(
            f"Unexpected resource set. Expected {sorted(EXPECTED_RESOURCES)}, "
            f"received {sorted(resource_uris)}"
        )

    roots = displayed_roots or project_roots()
    print("--- START SCREENSHOT ---")
    print(f"\nDiscovered {len(tools_result.tools)} tools:")
    for tool in sorted(tools_result.tools, key=lambda item: item.name):
        description = tool.description or "No description"
        print(f"  - {tool.name}: {description[:80]}")
    print("\nAll required tools verified!")

    print(f"\nDiscovered {len(resources_result.resources)} resources:")
    for resource in resources_result.resources:
        print(f"  - {resource.uri}: {resource.name or 'Unnamed resource'}")

    print(f"\nConfigured {len(roots)} roots:")
    for root in roots:
        print(f"  - {root.name}: {root.uri}")
    print("--- END SCREENSHOT ---")

    return DiscoveryReport(
        tool_names=tuple(sorted(tool_names)),
        resource_uris=tuple(sorted(resource_uris)),
        roots=tuple(roots),
    )


async def demo_get_restaurant_info(
    *,
    parameters: StdioServerParameters = server_params,
) -> dict[str, Any]:
    """Look up Iron & Embers through the MCP protocol."""

    print("\n" + "-" * 60)
    print("Demo: get_restaurant_info('Iron & Embers')")
    print("-" * 60)
    data = await call_tool(
        "get_restaurant_info",
        {"restaurant_name": "Iron & Embers"},
        parameters=parameters,
    )
    print(json.dumps(data, indent=2))
    return data


async def demo_recommend_by_vibe(
    *,
    parameters: StdioServerParameters = server_params,
) -> dict[str, Any]:
    """Find restaurants matching the moody vibe."""

    print("\n" + "-" * 60)
    print("Demo: recommend_by_vibe('moody')")
    print("-" * 60)
    data = await call_tool(
        "recommend_by_vibe",
        {"vibe": "moody"},
        parameters=parameters,
    )
    print(f"Vibe: {data['vibe_searched']}")
    print(f"Structured matches: {len(data['structured_matches'])}")
    for match in data["structured_matches"]:
        print(f"  - {match['name']} ({match['cuisine']}) - {match['rating']}/5")
    print(f"Raw text excerpts: {len(data['raw_text_excerpts'])}")
    return data


async def demo_get_review(
    *,
    parameters: StdioServerParameters = server_params,
) -> dict[str, Any]:
    """Retrieve the augmented Iron & Embers review."""

    print("\n" + "-" * 60)
    print("Demo: get_review('Iron & Embers')")
    print("-" * 60)
    data = await call_tool(
        "get_review",
        {"restaurant_name": "Iron & Embers"},
        parameters=parameters,
    )
    print(json.dumps(data, indent=2))
    return data


async def run_all() -> None:
    """Run all three tool demos, followed by discovery verification."""

    await demo_get_restaurant_info()
    await demo_recommend_by_vibe()
    await demo_get_review()
    await verify_connection()


def main() -> None:
    """Synchronous console entry point."""

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
