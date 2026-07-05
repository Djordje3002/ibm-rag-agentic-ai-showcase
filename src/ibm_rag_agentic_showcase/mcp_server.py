"""FastMCP data server for the California culinary recommendation project."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def configured_data_dir() -> Path:
    """Return the optional environment override or the repository data folder."""

    configured = os.environ.get("IBM_MCP_DATA_DIR")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_DATA_DIR


@dataclass(frozen=True)
class CulinaryDataStore:
    """Load and validate the three course artifacts used by the MCP server."""

    directory: Path

    @property
    def culinary_map_path(self) -> Path:
        return self.directory / "California-Culinary-Map.txt"

    @property
    def restaurant_data_path(self) -> Path:
        return self.directory / "structured-restaurant-data.json"

    @property
    def review_data_path(self) -> Path:
        return self.directory / "augmented-user-review.json"

    def culinary_map(self) -> str:
        return self._require(self.culinary_map_path).read_text(encoding="utf-8")

    def restaurants(self) -> list[dict[str, Any]]:
        return self._load_json_list(self.restaurant_data_path)

    def reviews(self) -> list[dict[str, Any]]:
        return self._load_json_list(self.review_data_path)

    def _require(self, path: Path) -> Path:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required MCP data file not found: {path}. "
                "Download the Lab 10 data files before starting the server."
            )
        return path

    def _load_json_list(self, path: Path) -> list[dict[str, Any]]:
        value = json.loads(self._require(path).read_text(encoding="utf-8"))
        if not isinstance(value, list) or not all(
            isinstance(item, dict) for item in value
        ):
            raise ValueError(f"Expected a JSON array of objects in {path}")
        return value


def _query(value: str, field_name: str) -> str:
    query = value.casefold().strip()
    if not query:
        raise ValueError(f"{field_name} must not be empty")
    return query


def search_restaurants(
    store: CulinaryDataStore,
    restaurant_name: str,
) -> dict[str, Any]:
    """Perform a case-insensitive, bidirectional partial-name lookup."""

    query = _query(restaurant_name, "restaurant_name")
    matches = []
    for restaurant in store.restaurants():
        name = str(restaurant.get("name", "")).casefold()
        if query in name or (name and name in query):
            matches.append(restaurant)

    if not matches:
        return {
            "status": "not_found",
            "message": f"No restaurant found matching '{restaurant_name}'.",
            "suggestion": "Try a partial name like 'Iron' or 'Sakura'.",
        }
    return {"status": "found", "count": len(matches), "results": matches}


def _vibe_values(restaurant: Mapping[str, Any]) -> list[str]:
    value = restaurant.get("vibes", restaurant.get("vibe", []))
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def search_by_vibe(
    store: CulinaryDataStore,
    vibe: str,
) -> dict[str, Any]:
    """Search structured vibe fields, descriptions, then raw-map paragraphs."""

    query = _query(vibe, "vibe")
    structured_matches = []
    for restaurant in store.restaurants():
        vibes = _vibe_values(restaurant)
        description = str(restaurant.get("description", "")).casefold()
        if any(query in item.casefold() for item in vibes) or query in description:
            structured_matches.append(
                {
                    "name": restaurant.get("name", "Unknown"),
                    "neighborhood": restaurant.get(
                        "neighborhood",
                        restaurant.get("location", "N/A"),
                    ),
                    "cuisine": restaurant.get(
                        "cuisine",
                        restaurant.get("food_style", "N/A"),
                    ),
                    "rating": restaurant.get("rating"),
                    "vibes": vibes,
                    "price_range": restaurant.get("price_range"),
                }
            )

    paragraphs = re.split(r"\n\s*\n", store.culinary_map())
    text_excerpts = [
        paragraph.strip()[:300]
        for paragraph in paragraphs
        if query in paragraph.casefold() and paragraph.strip()
    ]
    return {
        "vibe_searched": vibe,
        "structured_matches": structured_matches,
        "raw_text_excerpts": text_excerpts[:5],
    }


def find_review(
    store: CulinaryDataStore,
    restaurant_name: str,
) -> dict[str, Any]:
    """Return the first augmented review whose restaurant name contains a query."""

    query = _query(restaurant_name, "restaurant_name")
    matching_review = next(
        (
            review
            for review in store.reviews()
            if query in str(review.get("restaurant_name", "")).casefold()
        ),
        None,
    )
    if matching_review is None:
        return {
            "status": "not_found",
            "message": f"No review found matching '{restaurant_name}'.",
        }
    return {
        "status": "found",
        "restaurant": matching_review.get("restaurant_name", "N/A"),
        "reviewer": matching_review.get("reviewer", "N/A"),
        "rating": matching_review.get("rating"),
        "review_text": matching_review.get("review_text", ""),
        "image_description": matching_review.get("image_description", "N/A"),
        "visit_date": matching_review.get("visit_date", "N/A"),
    }


def _json_result(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def create_mcp_server(
    data_dir: Path | None = None,
) -> FastMCP:
    """Register the culinary resource and three tools on one FastMCP server."""

    store = CulinaryDataStore((data_dir or configured_data_dir()).resolve())
    server = FastMCP("Connoisseur-Server")

    @server.resource("culinary-map://california")
    def get_culinary_map() -> str:
        """Return the full raw California Culinary Map from Module 1."""

        return store.culinary_map()

    @server.tool()
    def get_restaurant_info(restaurant_name: str) -> str:
        """Return structured restaurant details for a partial name."""

        return _json_result(search_restaurants(store, restaurant_name))

    @server.tool()
    def recommend_by_vibe(vibe: str) -> str:
        """Find restaurants by vibe in structured fields and raw descriptions."""

        return _json_result(search_by_vibe(store, vibe))

    @server.tool()
    def get_review(restaurant_name: str) -> str:
        """Return the full augmented review for a partial restaurant name."""

        return _json_result(find_review(store, restaurant_name))

    return server


mcp = create_mcp_server()


def main() -> None:
    """Run the server over the default stdio transport."""

    mcp.run()


if __name__ == "__main__":
    main()
