"""Gradio MCP host with a WatsonX-backed ReAct tool-calling loop."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from fastmcp.client import Client, PythonStdioTransport
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from mcp.types import TextContent, Tool

PROJECT_DIR = Path(__file__).resolve().parents[2]
SERVER_SCRIPT = PROJECT_DIR / "server.py"
MODEL_ID = "ibm/granite-4-h-small"
WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
MAX_REACT_ITERATIONS = 10

SYSTEM_PROMPT = """
You are Connoisseur Companion, an AI guide with access to a database of
California restaurants.

Use the discovered tools whenever restaurant data is needed:
- get_restaurant_info looks up a specific restaurant by full or partial name.
- recommend_by_vibe finds restaurants matching a mood or atmosphere.
- get_review retrieves a detailed augmented review for a restaurant.

Base factual restaurant claims on tool results. Do not invent restaurants,
ratings, reviews, locations, or availability. If a tool finds nothing, say so
and suggest a narrower or alternative query. You may call multiple tools before
answering. Give a concise, friendly final response and distinguish retrieved
facts from general advice.
""".strip()

History = Sequence[Mapping[str, Any]]
ModelFactory = Callable[[], Any]
ClientFactory = Callable[[], Any]


class BoundToolModel(Protocol):
    async def ainvoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        """Return the next assistant message, potentially with tool calls."""


def watsonx_project_id() -> str:
    """Read either environment name used by IBM course environments."""

    project_id = os.environ.get("WATSONX_AI_PROJECT_ID") or os.environ.get(
        "WATSONX_PROJECT_ID"
    )
    if not project_id:
        raise RuntimeError(
            "Set WATSONX_AI_PROJECT_ID or WATSONX_PROJECT_ID before chatting."
        )
    return project_id


def make_model() -> Any:
    """Create one fresh WatsonX chat model for a conversation turn."""

    from langchain_ibm import ChatWatsonx

    return ChatWatsonx(
        model_id=os.environ.get("WATSONX_MODEL_ID", MODEL_ID),
        url=os.environ.get("WATSONX_URL", WATSONX_URL),
        project_id=watsonx_project_id(),
        params={"temperature": 0.7},
    )


def create_mcp_client() -> Client:
    """Create a FastMCP client that owns the local server subprocess."""

    server_environment = {
        name: os.environ[name] for name in ("IBM_MCP_DATA_DIR",) if name in os.environ
    }
    transport = PythonStdioTransport(
        script_path=SERVER_SCRIPT,
        cwd=str(PROJECT_DIR),
        env=server_environment or None,
    )
    return Client(transport)


def mcp_tool_definition(tool: Tool) -> dict[str, Any]:
    """Convert one MCP tool schema into LangChain/OpenAI function format."""

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }


def history_messages(history: History) -> list[BaseMessage]:
    """Convert safe text-only Gradio history into LangChain messages."""

    messages: list[BaseMessage] = []
    for item in history:
        role = item.get("role")
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def response_text(content: Any) -> str:
    """Normalize LangChain text or content-block responses."""

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(part.strip() for part in parts if part.strip())
    return str(content).strip()


def tool_result_text(result: Any) -> str:
    """Flatten an MCP result into the observation passed back to the model."""

    if getattr(result, "isError", False):
        return json.dumps(
            {
                "status": "tool_error",
                "message": "The MCP server reported an error.",
            }
        )
    parts = [
        item.text if isinstance(item, TextContent) else str(item)
        for item in getattr(result, "content", [])
    ]
    return " ".join(part for part in parts if part) or "(no result)"


async def run_react_agent(
    user_message: str,
    history: History,
    *,
    model_factory: ModelFactory = make_model,
    client_factory: ClientFactory = create_mcp_client,
    max_iterations: int = MAX_REACT_ITERATIONS,
) -> str:
    """Discover tools and iterate model → tool → observation until completion."""

    if not user_message.strip():
        raise ValueError("user_message must not be empty")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    async with client_factory() as client:
        mcp_tools = await client.list_tools()
        if not mcp_tools:
            raise RuntimeError("The MCP server did not advertise any tools")

        definitions = [mcp_tool_definition(tool) for tool in mcp_tools]
        tool_names = {tool.name for tool in mcp_tools}
        model: BoundToolModel = model_factory().bind_tools(definitions)

        messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
        messages.extend(history_messages(history))
        messages.append(HumanMessage(content=user_message))

        for iteration in range(max_iterations):
            response = await model.ainvoke(messages)
            messages.append(response)

            if not response.tool_calls:
                final = response_text(response.content)
                return final or "I could not produce a final response."

            for call_index, tool_call in enumerate(response.tool_calls):
                name = tool_call.get("name", "")
                arguments = tool_call.get("args", {})
                call_id = tool_call.get("id", "")

                if name not in tool_names:
                    observation = json.dumps(
                        {
                            "status": "tool_error",
                            "message": f"Unknown MCP tool requested: {name}",
                        }
                    )
                elif not isinstance(arguments, dict):
                    observation = json.dumps(
                        {
                            "status": "tool_error",
                            "message": f"Invalid arguments for MCP tool: {name}",
                        }
                    )
                else:
                    result = await client.call_tool(name, arguments)
                    observation = tool_result_text(result)

                messages.append(
                    ToolMessage(
                        content=observation,
                        tool_call_id=str(call_id or f"react-{iteration}-{call_index}"),
                    )
                )

    return (
        "I reached the tool-call safety limit before completing that request. "
        "Please try a narrower question."
    )


async def chat_with_agent(user_message: str, history: History) -> str:
    """Production callback using WatsonX and the local MCP server."""

    return await run_react_agent(user_message, history)


async def handle_chat(
    user_message: str,
    history: list[dict[str, str]] | None,
):
    """Stream an immediate placeholder, then replace it with the agent answer."""

    current_history = list(history or [])
    if not user_message or not user_message.strip():
        yield current_history
        return

    current_history.extend(
        [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": "Thinking…"},
        ]
    )
    yield [dict(item) for item in current_history]

    try:
        answer = await chat_with_agent(user_message, current_history[:-2])
    except Exception as error:
        answer = (
            "I could not complete that request. Check the WatsonX credentials "
            f"and MCP data files, then try again. ({error})"
        )
    current_history[-1] = {"role": "assistant", "content": answer}
    yield [dict(item) for item in current_history]


def build_gradio_app() -> Any:
    """Build the complete host UI without starting a server."""

    try:
        import gradio as gr
    except ImportError as error:  # pragma: no cover - exercised without UI extra
        raise RuntimeError(
            'Gradio is optional. Install it with: pip install -e ".[ui,host,mcp]"'
        ) from error

    with gr.Blocks(title="Connoisseur Companion") as demo:
        gr.Markdown(
            "# 🍷 Connoisseur Companion\n"
            "Your AI guide to California’s restaurant scene. Ask about a "
            "restaurant by name, explore a vibe, or retrieve a detailed review."
        )
        chatbot = gr.Chatbot(
            height=500,
            label="Restaurant conversation",
            placeholder=(
                "The agent discovers the MCP server’s tools at runtime and "
                "chooses which evidence it needs."
            ),
        )
        message = gr.Textbox(
            label="Ask about California restaurants",
            placeholder=(
                "Try “Find me a moody spot in DTLA” or “Tell me about Sakura Garden.”"
            ),
        )

        with gr.Row():
            moody = gr.Button("🌙 Find moody restaurants", size="sm")
            iron = gr.Button("🔥 Tell me about Iron & Embers", size="sm")
            zen = gr.Button("🌿 Zen dining in Little Tokyo?", size="sm")

        submit_event = message.submit(
            handle_chat,
            inputs=[message, chatbot],
            outputs=chatbot,
        )
        submit_event.then(lambda: "", outputs=message)

        moody_prompt = gr.State("Find me some moody restaurants")
        iron_prompt = gr.State("Tell me about Iron & Embers")
        zen_prompt = gr.State("What's a zen dining experience in Little Tokyo?")
        moody.click(handle_chat, inputs=[moody_prompt, chatbot], outputs=chatbot)
        iron.click(handle_chat, inputs=[iron_prompt, chatbot], outputs=chatbot)
        zen.click(handle_chat, inputs=[zen_prompt, chatbot], outputs=chatbot)

        gr.Markdown(
            "_Restaurant facts come from the local MCP server. Model-generated "
            "recommendations should still be verified before making plans._"
        )

    return demo


def launch_app(demo: Any | None = None) -> None:
    """Launch locally; public sharing requires an explicit environment opt-in."""

    import gradio as gr

    app = demo or build_gradio_app()
    share = os.environ.get("GRADIO_SHARE", "").casefold() in {"1", "true", "yes"}
    print("Starting Connoisseur Companion...")
    app.launch(
        server_name="127.0.0.1",
        share=share,
        theme=gr.themes.Soft(),
    )


def main() -> None:
    """Console entry point."""

    launch_app()


if __name__ == "__main__":
    main()
