import json
from pathlib import Path

import pytest

from ibm_rag_agentic_showcase.restaurant_extraction import (
    ExtractionError,
    Restaurant,
    RestaurantExtractor,
    load_descriptions,
    serialize_records,
)

VALID_RECORD = {
    "name": "Test Kitchen",
    "location": "Oakland",
    "type": "bistro",
    "food_style": "Californian",
    "rating": 4.5,
    "price_range": 2,
    "signatures": ["seasonal tasting menu"],
    "vibe": "relaxed",
    "environment": "A bright neighborhood dining room.",
    "shortcomings": [],
}


def test_extracts_valid_response_without_repair():
    calls = []

    def fake_llm(system, prompt):
        calls.append((system, prompt))
        return json.dumps(VALID_RECORD)

    result = RestaurantExtractor(fake_llm).extract("A restaurant description")

    assert result.name == "Test Kitchen"
    assert len(calls) == 1


def test_repairs_invalid_response():
    responses = iter(['{"name": "Incomplete"}', json.dumps(VALID_RECORD)])

    result = RestaurantExtractor(lambda *_: next(responses)).extract("Description")

    assert result.location == "Oakland"


def test_raises_after_repair_limit():
    extractor = RestaurantExtractor(lambda *_: "not-json", max_repair_attempts=2)

    with pytest.raises(ExtractionError, match="2 repair attempt"):
        extractor.extract("Description")


def test_loads_descriptions_after_title(tmp_path: Path):
    source = tmp_path / "restaurants.txt"
    source.write_text(
        "California Culinary Map\n\nFirst restaurant\nLine two\n\nSecond restaurant\n",
        encoding="utf-8",
    )

    assert load_descriptions(source) == [
        "First restaurant\nLine two",
        "Second restaurant",
    ]


def test_serializes_records_with_stable_ids():
    records = [
        Restaurant.model_validate(VALID_RECORD),
        Restaurant.model_validate(VALID_RECORD),
    ]

    serialized = serialize_records(records)

    assert [record["itemId"] for record in serialized] == [1_000_001, 1_000_002]
