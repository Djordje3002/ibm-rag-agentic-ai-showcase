import asyncio
import os
from types import SimpleNamespace

from mcp.types import (
    CreateMessageRequestParams,
    SamplingMessage,
    TextContent,
)

from ibm_rag_agentic_showcase.mcp_client import (
    EXPECTED_RESOURCES,
    EXPECTED_TOOLS,
    build_server_params,
    call_tool,
    create_roots_callback,
    create_sampling_callback,
    project_roots,
    verify_connection,
)


class FakeMessages:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(text="A concise sampled response.")]
        )


class FakeAnthropic:
    def __init__(self):
        self.messages = FakeMessages()


def test_project_root_uses_a_valid_encoded_file_uri(tmp_path):
    project = tmp_path / "project with spaces"
    project.mkdir()

    root = project_roots(project)[0]

    assert str(root.uri).startswith("file://")
    assert "%20" in str(root.uri)
    assert root.name == "project with spaces"


def test_roots_callback_returns_list_roots_result(tmp_path):
    callback = create_roots_callback(tmp_path)

    result = asyncio.run(callback(None))

    assert len(result.roots) == 1
    assert str(result.roots[0].uri) == tmp_path.resolve().as_uri()


def test_sampling_callback_delegates_to_anthropic_without_real_api_call():
    fake = FakeAnthropic()
    callback = create_sampling_callback(fake)
    params = CreateMessageRequestParams(
        messages=[
            SamplingMessage(
                role="user",
                content=TextContent(type="text", text="Summarize this restaurant."),
            )
        ],
        maxTokens=64,
    )

    result = asyncio.run(callback(None, params))

    assert result.content.text == "A concise sampled response."
    assert result.stopReason == "endTurn"
    assert fake.messages.kwargs["max_tokens"] == 64
    assert fake.messages.kwargs["messages"][0]["content"].startswith("Summarize")


def test_client_discovers_server_and_calls_tool_over_stdio(
    culinary_store,
    capsys,
):
    environment = {**os.environ, "IBM_MCP_DATA_DIR": str(culinary_store.directory)}
    parameters = build_server_params(environment=environment)
    roots_callback = create_roots_callback(culinary_store.directory)

    async def exercise_client():
        report = await verify_connection(
            parameters=parameters,
            roots_callback=roots_callback,
            displayed_roots=project_roots(culinary_store.directory),
        )
        result = await call_tool(
            "get_restaurant_info",
            {"restaurant_name": "Iron"},
            parameters=parameters,
            roots_callback=roots_callback,
        )
        return report, result

    report, result = asyncio.run(exercise_client())
    output = capsys.readouterr().out

    assert set(report.tool_names) == EXPECTED_TOOLS
    assert set(report.resource_uris) == EXPECTED_RESOURCES
    assert result["results"][0]["name"] == "Iron & Embers"
    assert "--- START SCREENSHOT ---" in output
    assert "Configured 1 roots" in output
    assert "--- END SCREENSHOT ---" in output


def test_all_three_tools_are_callable_through_shared_client_helper(culinary_store):
    environment = {**os.environ, "IBM_MCP_DATA_DIR": str(culinary_store.directory)}
    parameters = build_server_params(environment=environment)

    async def call_all():
        restaurant = await call_tool(
            "get_restaurant_info",
            {"restaurant_name": "Iron"},
            parameters=parameters,
        )
        vibe = await call_tool(
            "recommend_by_vibe",
            {"vibe": "moody"},
            parameters=parameters,
        )
        review = await call_tool(
            "get_review",
            {"restaurant_name": "Iron"},
            parameters=parameters,
        )
        return restaurant, vibe, review

    restaurant, vibe, review = asyncio.run(call_all())

    assert restaurant["status"] == "found"
    assert vibe["structured_matches"]
    assert review["status"] == "found"
