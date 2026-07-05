import json

import pytest

from ibm_rag_agentic_showcase.mcp_server import CulinaryDataStore


@pytest.fixture
def culinary_store(tmp_path):
    """Create a complete deterministic MCP dataset."""

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
