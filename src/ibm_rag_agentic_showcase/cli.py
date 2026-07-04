"""Command-line interface for the restaurant extraction project."""

from __future__ import annotations

import argparse
from pathlib import Path

from .restaurant_extraction import (
    RestaurantExtractor,
    WatsonxSettings,
    create_watsonx_llm,
    download_dataset,
    load_descriptions,
    save_records,
)

DEFAULT_INPUT = Path("data/raw/California-Culinary-Map.txt")
DEFAULT_OUTPUT = Path("data/processed/structured_restaurant_data.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract validated restaurant JSON with IBM Granite."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N records (useful for a low-cost smoke test).",
    )
    parser.add_argument("--max-repair-attempts", type=int, default=3)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Fail instead of downloading the course dataset when input is missing.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    if not args.input.exists():
        if args.no_download:
            raise SystemExit(f"Input file does not exist: {args.input}")
        print(f"Downloading course dataset to {args.input}...")
        download_dataset(args.input)

    descriptions = load_descriptions(args.input)
    if args.limit is not None:
        descriptions = descriptions[: args.limit]

    settings = WatsonxSettings.from_env()
    extractor = RestaurantExtractor(
        create_watsonx_llm(settings),
        max_repair_attempts=args.max_repair_attempts,
    )
    total = len(descriptions)
    records = extractor.extract_many(
        descriptions,
        progress=lambda completed: print(f"Processed {completed}/{total}"),
    )
    save_records(records, args.output)
    print(f"Saved {len(records)} validated records to {args.output}")


if __name__ == "__main__":
    main()
