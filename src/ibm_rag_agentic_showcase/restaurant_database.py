"""Safe, typed CRUD operations for the restaurant dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .restaurant_extraction import (
    Restaurant,
    RestaurantExtractor,
    WatsonxSettings,
    create_watsonx_llm,
)

InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], None]
RestaurantExtraction = Callable[[str], Restaurant]

EDITABLE_FIELDS = (
    "name",
    "location",
    "type",
    "food_style",
    "rating",
    "price_range",
    "signatures",
    "vibe",
    "environment",
    "shortcomings",
)


class RestaurantRecord(Restaurant):
    """A validated restaurant with its stable database identifier."""

    itemId: int


def load_records(path: Path) -> list[RestaurantRecord]:
    """Load and validate every record, returning an empty list if absent."""

    if not path.exists():
        return []

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return [RestaurantRecord.model_validate(item) for item in value]


def save_records(
    records: Iterable[RestaurantRecord],
    path: Path,
    backup_path: Path | None = None,
) -> None:
    """Back up the current file and atomically replace it with validated data."""

    records_to_save = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and backup_path is not None:
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_path)

    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(
            [record.model_dump() for record in records_to_save],
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def next_item_id(
    records: Iterable[RestaurantRecord],
    first_item_id: int = 1_000_001,
) -> int:
    """Choose an unused ID even when earlier records have been deleted."""

    existing_ids = [record.itemId for record in records]
    return max(existing_ids, default=first_item_id - 1) + 1


def parse_edit_value(field: str, raw_value: str) -> Any:
    """Convert terminal text into the schema type expected for one field."""

    value = raw_value.strip()
    if field in {"rating", "price_range", "vibe"} and value.lower() == "null":
        return None
    if field == "rating":
        return float(value)
    if field == "price_range":
        return int(value)
    if field in {"signatures", "shortcomings"}:
        if value.startswith("["):
            parsed = json.loads(value)
            if not isinstance(parsed, list) or not all(
                isinstance(item, str) for item in parsed
            ):
                raise ValueError(f"{field} must be a list of strings")
            return parsed
        return [item.strip() for item in value.split(",") if item.strip()]
    return raw_value


class RestaurantDatabase:
    """Validated persistence and CRUD behavior independent of the terminal UI."""

    def __init__(
        self,
        path: Path,
        backup_path: Path | None = None,
        extractor: RestaurantExtraction | None = None,
    ) -> None:
        self.path = path
        self.backup_path = backup_path or path.with_suffix(f"{path.suffix}.bak")
        self.extractor = extractor

    def all(self) -> list[RestaurantRecord]:
        return load_records(self.path)

    def add_from_description(self, description: str) -> RestaurantRecord:
        """Extract, validate, assign an ID, save, and return one new record."""

        if self.extractor is None:
            raise RuntimeError("An LLM extractor is required to add a restaurant")

        records = self.all()
        extracted = self.extractor(description)
        record = RestaurantRecord(
            **extracted.model_dump(),
            itemId=next_item_id(records),
        )
        records.append(record)
        save_records(records, self.path, self.backup_path)
        return record

    def update(
        self,
        index: int,
        changes: Mapping[str, Any],
    ) -> RestaurantRecord:
        """Apply changes only if the complete updated record remains valid."""

        records = self.all()
        if not 0 <= index < len(records):
            raise IndexError("Restaurant index is out of range")
        unknown_fields = set(changes) - set(EDITABLE_FIELDS)
        if unknown_fields:
            names = ", ".join(sorted(unknown_fields))
            raise ValueError(f"Fields cannot be edited: {names}")

        updated_data = {**records[index].model_dump(), **changes}
        updated = RestaurantRecord.model_validate(updated_data)
        records[index] = updated
        save_records(records, self.path, self.backup_path)
        return updated

    def delete(self, index: int) -> RestaurantRecord:
        """Remove one record, persisting only after index validation."""

        records = self.all()
        if not 0 <= index < len(records):
            raise IndexError("Restaurant index is out of range")
        deleted = records.pop(index)
        save_records(records, self.path, self.backup_path)
        return deleted


def show_restaurant_card(
    restaurant: RestaurantRecord,
    index: int,
    output: OutputFunction = print,
) -> None:
    """Render all fields of one restaurant in a readable terminal card."""

    output(f"\n--- Restaurant #{index + 1} ---")
    for key, value in restaurant.model_dump().items():
        output(f"{key}: {value}")


def _read_index(
    prompt: str,
    record_count: int,
    input_fn: InputFunction,
) -> int:
    """Read a user-facing one-based index and return its zero-based value."""

    value = int(input_fn(prompt))
    index = value - 1
    if not 0 <= index < record_count:
        raise IndexError("Restaurant number is out of range")
    return index


def _confirm_write(
    input_fn: InputFunction,
    output: OutputFunction,
) -> bool:
    output("\nSECURITY WARNING: you are entering write mode.")
    output("Confirmed changes are saved immediately and the old file is backed up.")
    return input_fn("Type 'yes' to proceed: ").strip().lower() == "yes"


def run_console(
    database: RestaurantDatabase,
    input_fn: InputFunction = input,
    output: OutputFunction = print,
) -> None:
    """Run the interactive restaurant database until the user exits."""

    while True:
        records = database.all()
        output(f"\nRESTAURANT DATABASE | Records: {len(records)}")
        output("1. Browse all names")
        output("2. View detailed record")
        output("3. Add new restaurant")
        output("4. Edit restaurant")
        output("5. Delete restaurant")
        output("6. Exit")

        choice = input_fn("\nAction: ").strip()

        try:
            if choice == "1":
                output("\n--- Current listings ---")
                for index, record in enumerate(records, start=1):
                    output(f"{index}. {record.name}")

            elif choice == "2":
                index = _read_index(
                    "Enter restaurant number: ",
                    len(records),
                    input_fn,
                )
                show_restaurant_card(records[index], index, output)

            elif choice in {"3", "4", "5"}:
                if not _confirm_write(input_fn, output):
                    output("Operation cancelled.")
                    continue

                if choice == "3":
                    description = input_fn("Enter a restaurant description: ")
                    added = database.add_from_description(description)
                    output(f"Restaurant added: {added.name} (ID {added.itemId})")

                elif choice == "4":
                    index = _read_index(
                        "Enter restaurant number to edit: ",
                        len(records),
                        input_fn,
                    )
                    current = records[index]
                    changes: dict[str, Any] = {}
                    for field in EDITABLE_FIELDS:
                        existing = getattr(current, field)
                        raw_value = input_fn(
                            f"{field} [{existing}] (blank keeps current): "
                        )
                        if raw_value != "":
                            changes[field] = parse_edit_value(field, raw_value)

                    updated = database.update(index, changes)
                    output(f"Restaurant updated: {updated.name}")

                else:
                    index = _read_index(
                        "Enter restaurant number to delete: ",
                        len(records),
                        input_fn,
                    )
                    deleted = database.delete(index)
                    output(f"Restaurant deleted: {deleted.name}")

            elif choice == "6":
                output("Goodbye.")
                return

            else:
                output("Invalid action.")

        except (IndexError, ValueError, ValidationError) as error:
            output(f"Could not complete operation: {error}")


def create_default_extractor() -> RestaurantExtraction:
    """Create the bounded Lab 01 extraction pipeline for new records."""

    settings = WatsonxSettings.from_env()
    extractor = RestaurantExtractor(
        create_watsonx_llm(settings),
        max_repair_attempts=3,
    )
    return extractor.extract


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Browse and safely update the structured restaurant dataset."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("data/processed/structured_restaurant_data.json"),
        help="Restaurant JSON file to manage.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    extractor: RestaurantExtraction | None = None

    def lazy_extract(description: str) -> Restaurant:
        nonlocal extractor
        if extractor is None:
            extractor = create_default_extractor()
        return extractor(description)

    database = RestaurantDatabase(args.file, extractor=lazy_extract)
    run_console(database)


if __name__ == "__main__":
    main()
