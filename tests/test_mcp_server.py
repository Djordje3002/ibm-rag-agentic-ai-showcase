import asyncio
import json

import pytest

from ibm_rag_agentic_showcase.mcp_server import (
    CulinaryDataStore,
    create_mcp_server,
    find_review,
    search_by_vibe,
    search_restaurants,
)


@pytest.fixture
def culinary_store(tmp_path):
    (tmp_path / "California-Culinary-Map.txt").write_text(
        "Iron & Embers\nA moody room with exposed brick.\n\n"
        "Sakura Garden\nA sun-drenched garden dining room.\n",
        encoding="utf-8",
    )
    (tmp_path / "structured-restaurant-data.json").write_text(
        json.dumps(
            [
                {
                    "name": "Iron & Embers",
                    "neighborhood": "Arts District",
                    "cuisine": "New American",
                    "rating": 4.7,
                    "price_range": "$$$",
                    "vibes": ["industrial chic", "moody"],
                    "description": "Exposed brick and low lighting.",
                },
                {
                    "name": "Sakura Garden",
                    "neighborhood": "Pasadena",
                    "cuisine": "Japanese",
                    "rating": 4.5,
                    "price_range": "$$",
                    "vibes": ["garden"],
                    "description": "A bright dining room.",
                },
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "augmented-user-review.json").write_text(
        json.dumps(
            [
                {
                    "restaurant_name": "Iron & Embers",
                    "reviewer": "Alex",
                    "rating": 5,
                    "review_text": "Excellent dinner.",
                    "image_description": "A dark brick dining room.",
                    "visit_date": "2026-01-15",
                }
            ]
        ),
        encoding="utf-8",
    )
    return CulinaryDataStore(tmp_path)


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
