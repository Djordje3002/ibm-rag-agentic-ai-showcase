"""Validated restaurant extraction with IBM Granite."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

from pydantic import BaseModel, Field, ValidationError

DATASET_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "1r_mM6ZPYNxcFv65QkzubA/California-Culinary-Map.txt"
)

EXAMPLE_RESTAURANT = """
Mar de Cortez in Santa Monica is a casual taqueria serving Baja-style seafood.
The restaurant has a 4.2 rating and an affordable price level of 1. Guests
recommend the beer-battered snapper tacos and zesty octopus ceviche. Its
salt-air energy makes it a premier sun-drenched spot for open-air dining near
the pier.
""".strip()

EXAMPLE_OUTPUT = {
    "name": "Mar de Cortez",
    "location": "Santa Monica",
    "type": "casual taqueria",
    "food_style": "Baja-style seafood",
    "rating": 4.2,
    "price_range": 1,
    "signatures": [
        "beer-battered snapper tacos",
        "zesty octopus ceviche",
    ],
    "vibe": "salt-air energy",
    "environment": ("a premier sun-drenched spot for open-air dining near the pier."),
    "shortcomings": [],
}

LLMCall = Callable[[str, str], str]


class Restaurant(BaseModel):
    """Structured representation of one restaurant description."""

    name: str
    location: str
    type: str
    food_style: str
    rating: float | None = None
    price_range: int | None = None
    signatures: list[str] = Field(default_factory=list)
    vibe: str | None = None
    environment: str
    shortcomings: list[str] = Field(default_factory=list)


class ExtractionError(RuntimeError):
    """Raised when model output remains invalid after all repair attempts."""


@dataclass(frozen=True)
class WatsonxSettings:
    """Configuration used to create a watsonx.ai model client."""

    project_id: str
    api_key: str | None = None
    url: str = "https://us-south.ml.cloud.ibm.com"
    model_id: str = "ibm/granite-4-h-small"
    max_new_tokens: int = 300

    @classmethod
    def from_env(cls) -> WatsonxSettings:
        """Load settings from environment variables."""

        project_id = os.getenv("WATSONX_PROJECT_ID", "skills-network")
        return cls(
            project_id=project_id,
            api_key=os.getenv("WATSONX_APIKEY"),
            url=os.getenv("WATSONX_URL", cls.url),
            model_id=os.getenv("WATSONX_MODEL_ID", cls.model_id),
        )


def build_extraction_prompt(description: str) -> tuple[str, str]:
    """Build the one-shot prompt used to structure one description."""

    system = (
        "You are an expert information extraction assistant. Extract restaurant "
        "information from the provided description. Return only valid JSON, "
        "without markdown, explanations, or additional text."
    )
    example_json = json.dumps(EXAMPLE_OUTPUT, indent=2)
    user = f"""
Extract the restaurant information into exactly these fields:
name, location, type, food_style, rating, price_range, signatures, vibe,
environment, shortcomings.

Rules:
- Use null for a missing rating, price_range, or vibe.
- signatures and shortcomings must always be lists.
- Use an empty list when there are no shortcomings.
- Return only one valid JSON object.

Restaurant description:
{description}

One-shot example:
Input:
{EXAMPLE_RESTAURANT}

Output:
{example_json}
""".strip()
    return system, user


def build_repair_prompt(candidate: str, error: ValidationError) -> tuple[str, str]:
    """Build a targeted prompt from a failed Pydantic validation."""

    system = (
        "You repair invalid JSON to match a required schema. Return only the "
        "corrected JSON object, with no markdown or explanation."
    )
    user = f"""
The candidate below failed validation.

Validation errors:
{error.json()}

Candidate:
{candidate}

Required schema:
- name, location, type, food_style, environment: string
- rating: float or null
- price_range: integer or null
- vibe: string or null
- signatures, shortcomings: lists of strings

Repair the candidate and return only valid JSON.
""".strip()
    return system, user


class RestaurantExtractor:
    """Extract and validate restaurant records through an injected LLM."""

    def __init__(self, llm: LLMCall, max_repair_attempts: int = 3) -> None:
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts cannot be negative")
        self.llm = llm
        self.max_repair_attempts = max_repair_attempts

    def extract(self, description: str) -> Restaurant:
        """Extract one restaurant, repairing invalid output when possible."""

        candidate = self.llm(*build_extraction_prompt(description))
        last_error: ValidationError | None = None

        for attempt in range(self.max_repair_attempts + 1):
            try:
                return Restaurant.model_validate_json(candidate)
            except ValidationError as error:
                last_error = error
                if attempt == self.max_repair_attempts:
                    break
                candidate = self.llm(*build_repair_prompt(candidate, error))

        raise ExtractionError(
            "Model output remained invalid after "
            f"{self.max_repair_attempts} repair attempt(s): {last_error}"
        )

    def extract_many(
        self,
        descriptions: Iterable[str],
        progress: Callable[[int], None] | None = None,
    ) -> list[Restaurant]:
        """Extract a sequence of descriptions in source order."""

        records = []
        for index, description in enumerate(descriptions, start=1):
            records.append(self.extract(description))
            if progress is not None:
                progress(index)
        return records


def create_watsonx_llm(
    settings: WatsonxSettings,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> LLMCall:
    """Create a retrying two-message chat callable backed by watsonx.ai."""

    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

    credential_args: dict[str, Any] = {"url": settings.url}
    if settings.api_key:
        credential_args["api_key"] = settings.api_key

    model = ModelInference(
        model_id=settings.model_id,
        credentials=Credentials(**credential_args),
        project_id=settings.project_id,
        params={
            GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
            GenParams.MAX_NEW_TOKENS: settings.max_new_tokens,
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.TEMPERATURE: 0,
        },
    )

    def call(system: str, prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                response = model.chat(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ]
                )
                return response["choices"][0]["message"]["content"]
            except Exception as error:
                last_error = error
                if attempt < retries - 1:
                    time.sleep(retry_delay)
        raise RuntimeError(
            f"watsonx.ai call failed after {retries} tries"
        ) from last_error

    return call


def download_dataset(destination: Path, url: str = DATASET_URL) -> Path:
    """Download the course dataset when it is not already present."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        urlretrieve(url, destination)
    return destination


def load_descriptions(path: Path) -> list[str]:
    """Load restaurant paragraphs, excluding the dataset title block."""

    blocks = [
        block.strip()
        for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip())
        if block.strip()
    ]
    if len(blocks) < 2:
        raise ValueError("Expected a title followed by restaurant descriptions")
    return blocks[1:]


def serialize_records(
    records: Iterable[Restaurant],
    first_item_id: int = 1_000_001,
) -> list[dict[str, Any]]:
    """Convert validated records to dictionaries with deterministic IDs."""

    return [
        {"itemId": first_item_id + index, **record.model_dump()}
        for index, record in enumerate(records)
    ]


def save_records(records: Iterable[Restaurant], destination: Path) -> None:
    """Write validated records as formatted UTF-8 JSON."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(serialize_records(records), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
