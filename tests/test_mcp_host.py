import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from mcp.types import TextContent, Tool

from ibm_rag_agentic_showcase.mcp_host import (
    SYSTEM_PROMPT,
    build_gradio_app,
    handle_chat,
    history_messages,
    mcp_tool_definition,
    response_text,
    run_react_agent,
    watsonx_project_id,
)


class FakeBoundModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.invocations = []

    async def ainvoke(self, messages):
        self.invocations.append(list(messages))
        return next(self.responses)


class FakeModelFactory:
    def __init__(self, bound_model):
        self.bound_model = bound_model
        self.definitions = None

    def __call__(self):
        return self

    def bind_tools(self, definitions):
        self.definitions = definitions
        return self.bound_model


class FakeClient:
    def __init__(self):
        self.calls = []
        self.tools = [
            Tool(
                name="get_restaurant_info",
                description="Find a restaurant.",
                inputSchema={
                    "type": "object",
                    "properties": {"restaurant_name": {"type": "string"}},
                    "required": ["restaurant_name"],
                },
            )
        ]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def list_tools(self):
        return self.tools

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return SimpleNamespace(
            isError=False,
            content=[
                TextContent(
                    type="text",
                    text='{"status":"found","results":[{"name":"Iron & Embers"}]}',
                )
            ],
        )


def tool_call_message():
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_restaurant_info",
                "args": {"restaurant_name": "Iron"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )


def test_system_prompt_defines_persona_and_all_tools():
    assert "Connoisseur Companion" in SYSTEM_PROMPT
    assert "California restaurants" in SYSTEM_PROMPT
    assert "get_restaurant_info" in SYSTEM_PROMPT
    assert "recommend_by_vibe" in SYSTEM_PROMPT
    assert "get_review" in SYSTEM_PROMPT


def test_mcp_schema_converts_to_bind_tools_format():
    tool = FakeClient().tools[0]

    definition = mcp_tool_definition(tool)

    assert definition["type"] == "function"
    assert definition["function"]["name"] == "get_restaurant_info"
    assert definition["function"]["parameters"] == tool.inputSchema


def test_history_conversion_accepts_only_supported_text_roles():
    messages = history_messages(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "tool", "content": "ignored"},
            {"role": "user", "content": {"not": "text"}},
        ]
    )

    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert len(messages) == 2


def test_response_text_flattens_content_blocks():
    assert response_text([{"type": "text", "text": "One"}, {"text": "Two"}]) == (
        "One Two"
    )


def test_react_loop_discovers_calls_and_observes_tool_before_answer():
    client = FakeClient()
    bound = FakeBoundModel(
        [
            tool_call_message(),
            AIMessage(content="Iron & Embers is a moody DTLA steakhouse."),
        ]
    )
    factory = FakeModelFactory(bound)

    answer = asyncio.run(
        run_react_agent(
            "Tell me about Iron",
            [{"role": "user", "content": "I like moody restaurants."}],
            model_factory=factory,
            client_factory=lambda: client,
        )
    )

    assert answer.startswith("Iron & Embers")
    assert client.calls == [("get_restaurant_info", {"restaurant_name": "Iron"})]
    assert factory.definitions[0]["function"]["name"] == "get_restaurant_info"
    assert any(isinstance(message, ToolMessage) for message in bound.invocations[1])


def test_react_loop_stops_at_iteration_limit():
    client = FakeClient()
    bound = FakeBoundModel([tool_call_message(), tool_call_message()])

    answer = asyncio.run(
        run_react_agent(
            "Keep searching",
            [],
            model_factory=FakeModelFactory(bound),
            client_factory=lambda: client,
            max_iterations=2,
        )
    )

    assert "safety limit" in answer
    assert len(client.calls) == 2


def test_missing_watsonx_project_is_reported(monkeypatch):
    monkeypatch.delenv("WATSONX_AI_PROJECT_ID", raising=False)
    monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)

    with pytest.raises(RuntimeError, match="WATSONX_AI_PROJECT_ID"):
        watsonx_project_id()


def test_handle_chat_streams_placeholder_then_answer(monkeypatch):
    async def fake_chat(_message, _history):
        return "A grounded answer."

    monkeypatch.setattr(
        "ibm_rag_agentic_showcase.mcp_host.chat_with_agent",
        fake_chat,
    )

    async def collect():
        return [update async for update in handle_chat("Find a moody restaurant", [])]

    updates = asyncio.run(collect())

    assert updates[0][-1]["content"] == "Thinking…"
    assert updates[1][-1]["content"] == "A grounded answer."


def test_gradio_host_app_builds_expected_controls():
    app = build_gradio_app()
    components = app.config.get("components", [])
    button_values = {
        component.get("props", {}).get("value")
        for component in components
        if component.get("type") == "button"
    }

    assert app.config["title"] == "Connoisseur Companion"
    assert {
        "🌙 Find moody restaurants",
        "🔥 Tell me about Iron & Embers",
        "🌿 Zen dining in Little Tokyo?",
    }.issubset(button_values)
