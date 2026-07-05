import json
from pathlib import Path

import pytest

from ibm_rag_agentic_showcase.restaurant_database import (
    RestaurantDatabase,
    RestaurantRecord,
    load_records,
    next_item_id,
    parse_edit_value,
    run_console,
    save_records,
)
from ibm_rag_agentic_showcase.restaurant_extraction import Restaurant

BASE_RECORD = {
    "name": "Test Cafe",
    "location": "Test City",
    "type": "cafe",
    "food_style": "seasonal",
    "rating": 4.2,
    "price_range": 2,
    "signatures": ["mushroom toast"],
    "vibe": "cozy",
    "environment": "A quiet room with warm lighting.",
    "shortcomings": [],
}


def make_record(item_id: int = 1_000_001) -> RestaurantRecord:
    return RestaurantRecord(**BASE_RECORD, itemId=item_id)


def fake_extract(description: str) -> Restaurant:
    return Restaurant.model_validate(
        {**BASE_RECORD, "name": description.strip() or "New Restaurant"}
    )


def test_save_creates_backup_before_replacement(tmp_path: Path):
    path = tmp_path / "restaurants.json"
    backup = tmp_path / "restaurants.json.bak"
    save_records([make_record()], path, backup)

    updated = make_record()
    updated.name = "Updated Cafe"
    save_records([updated], path, backup)

    assert load_records(path)[0].name == "Updated Cafe"
    assert json.loads(backup.read_text())[0]["name"] == "Test Cafe"


def test_next_item_id_uses_maximum_not_length():
    records = [make_record(1_000_001), make_record(1_000_010)]

    assert next_item_id(records) == 1_000_011


def test_database_adds_extracted_restaurant(tmp_path: Path):
    database = RestaurantDatabase(
        tmp_path / "restaurants.json",
        extractor=fake_extract,
    )

    added = database.add_from_description("Copper Sprout")

    assert added.name == "Copper Sprout"
    assert added.itemId == 1_000_001
    assert database.all() == [added]


def test_database_update_preserves_types(tmp_path: Path):
    path = tmp_path / "restaurants.json"
    save_records([make_record()], path)
    database = RestaurantDatabase(path)

    updated = database.update(
        0,
        {
            "rating": parse_edit_value("rating", "4.8"),
            "price_range": parse_edit_value("price_range", "3"),
            "signatures": parse_edit_value(
                "signatures",
                "smoked trout, wild mushroom risotto",
            ),
        },
    )

    assert updated.rating == 4.8
    assert updated.price_range == 3
    assert updated.signatures == [
        "smoked trout",
        "wild mushroom risotto",
    ]


def test_database_rejects_unknown_edit_field(tmp_path: Path):
    path = tmp_path / "restaurants.json"
    save_records([make_record()], path)

    with pytest.raises(ValueError, match="cannot be edited"):
        RestaurantDatabase(path).update(0, {"itemId": 7})


def test_console_cancel_keeps_database_unchanged(tmp_path: Path):
    path = tmp_path / "restaurants.json"
    save_records([make_record()], path)
    answers = iter(["5", "no", "6"])
    output = []

    run_console(
        RestaurantDatabase(path),
        input_fn=lambda _: next(answers),
        output=output.append,
    )

    assert len(load_records(path)) == 1
    assert "Operation cancelled." in output


def test_database_delete_returns_removed_record(tmp_path: Path):
    path = tmp_path / "restaurants.json"
    save_records([make_record()], path)
    database = RestaurantDatabase(path)

    deleted = database.delete(0)

    assert deleted.name == "Test Cafe"
    assert database.all() == []
