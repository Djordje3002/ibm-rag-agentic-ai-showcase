import asyncio
import json

import pytest

from ibm_rag_agentic_showcase.mcp_server import (
    create_mcp_server,
    find_review,
    search_by_vibe,
    search_restaurants,
)


def test_partial_restaurant_lookup(culinary_store):
    result = search_restaurants(culinary_store, "Iron")

    assert result["status"] == "found"
    assert result["count"] == 1
    assert result["results"][0]["name"] == "Iron & Embers"


def test_restaurant_not_found_has_actionable_suggestion(culinary_store):
    result = search_restaurants(culinary_store, "Unknown")

    assert result["status"] == "not_found"
    assert "partial name" in result["suggestion"]


def test_vibe_search_combines_structured_and_raw_matches(culinary_store):
    result = search_by_vibe(culinary_store, "moody")

    assert result["structured_matches"][0]["name"] == "Iron & Embers"
    assert "moody room" in result["raw_text_excerpts"][0]


def test_review_found_and_not_found(culinary_store):
    found = find_review(culinary_store, "Iron")
    missing = find_review(culinary_store, "Sakura")

    assert found["status"] == "found"
    assert found["image_description"] == "A dark brick dining room."
    assert missing == {
        "status": "not_found",
        "message": "No review found matching 'Sakura'.",
    }


def test_empty_queries_are_rejected(culinary_store):
    with pytest.raises(ValueError, match="must not be empty"):
        search_restaurants(culinary_store, " ")


def test_mcp_components_are_discoverable_and_callable(culinary_store):
    from fastmcp import Client

    async def exercise_server():
        server = create_mcp_server(culinary_store.directory)
        async with Client(server) as client:
            tools = await client.list_tools()
            resources = await client.list_resources()
            result = await client.call_tool(
                "get_restaurant_info",
                {"restaurant_name": "Iron"},
            )
            resource = await client.read_resource("culinary-map://california")
        return tools, resources, result, resource

    tools, resources, result, resource = asyncio.run(exercise_server())

    assert {tool.name for tool in tools} == {
        "get_restaurant_info",
        "recommend_by_vibe",
        "get_review",
    }
    assert [str(item.uri) for item in resources] == ["culinary-map://california"]
    assert json.loads(result.content[0].text)["status"] == "found"
    assert "Iron & Embers" in resource[0].text
