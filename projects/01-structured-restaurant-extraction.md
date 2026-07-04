# Project 01 — Structured Restaurant Extraction

## Problem

Natural-language restaurant profiles are useful to read but difficult to search,
filter, or pass into a retrieval system. This project turns each profile into a
consistent JSON record suitable for downstream indexing.

## Schema

Each validated record contains:

- `name`, `location`, `type`, and `food_style`
- optional `rating`, `price_range`, and `vibe`
- list-valued `signatures` and `shortcomings`
- a required `environment` summary
- a deterministic `itemId` added after validation

## Approach

1. Split the source document into individual restaurant descriptions.
2. Prompt IBM Granite with field rules and a one-shot example.
3. Validate the response against a Pydantic model.
4. On failure, send the candidate JSON and validation errors to a repair prompt.
5. Retry up to the configured limit, then fail explicitly.
6. Add stable item IDs and save the complete collection as formatted JSON.

## Engineering decisions

### Validation is a pipeline stage

JSON syntax alone is not enough. Pydantic also checks required fields and value
types, giving the repair prompt precise feedback.

### Repair attempts are bounded

The original learning exercise repairs until a valid result appears. The
showcase implementation caps retries so an unexpected model behavior cannot
create an infinite loop or an unbounded bill.

### Model calls are injected

The extraction pipeline accepts any callable with the same two-message
interface. Production uses watsonx.ai; tests use deterministic fake responses.
This keeps business logic testable without network access.

## Run it

```bash
pip install -e ".[dev]"
export WATSONX_APIKEY="..."
export WATSONX_PROJECT_ID="..."
restaurant-extract
```

Useful options:

```bash
restaurant-extract --limit 5
restaurant-extract --input path/to/data.txt --output path/to/results.json
restaurant-extract --max-repair-attempts 2
```

Use `restaurant-extract --help` for the full command reference.

## Next step

The structured records are a natural source for the next project: embedding and
indexing restaurant data for retrieval-augmented recommendations with cited,
grounded answers.
