"""Multimodal captioning and dataset enrichment for Lab 02."""

from __future__ import annotations

import ast
import base64
import json
import mimetypes
import os
import zipfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

RECIPES_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "hpTjb6liKBLVHQK0UgMi5A/Recipes.json"
)
REVIEWS_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "fQUs9wQ6aB6ts6fmkD2V2w/Synthetic-User-Reviews.json"
)
RECIPE_IMAGES_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "5_Rr6ohviItzucyWk6nkrw/synthetic-recipe-images.zip"
)

VisionCall = Callable[[str, str, bytes, str], str]
ImageDownloader = Callable[[str], tuple[bytes, str]]
ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class VisionSettings:
    """Configuration for the watsonx.ai vision-language model."""

    project_id: str
    api_key: str | None = None
    url: str = "https://us-south.ml.cloud.ibm.com"
    model_id: str = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    max_tokens: int = 300

    @classmethod
    def from_env(cls) -> VisionSettings:
        """Load cloud settings while retaining Skills Network defaults."""

        return cls(
            project_id=os.getenv("WATSONX_PROJECT_ID", "skills-network"),
            api_key=os.getenv("WATSONX_APIKEY"),
            url=os.getenv("WATSONX_URL", cls.url),
            model_id=os.getenv("WATSONX_VISION_MODEL_ID", cls.model_id),
        )


def create_vision_llm(settings: VisionSettings) -> VisionCall:
    """Create one reusable multimodal model client for all images."""

    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference

    credential_args: dict[str, str] = {"url": settings.url}
    if settings.api_key:
        credential_args["api_key"] = settings.api_key

    model = ModelInference(
        model_id=settings.model_id,
        credentials=Credentials(**credential_args),
        project_id=settings.project_id,
        params={"max_tokens": settings.max_tokens},
    )

    def call(
        system_message: str,
        prompt: str,
        image_bytes: bytes,
        media_type: str,
    ) -> str:
        encoded_image = base64.b64encode(image_bytes).decode("ascii")
        response = model.chat(
            messages=[
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (f"data:{media_type};base64,{encoded_image}")
                            },
                        },
                    ],
                },
            ]
        )
        return response["choices"][0]["message"]["content"]

    return call


def recipe_caption_prompt(food_name: str) -> tuple[str, str]:
    """Create a focused prompt for a recipe image."""

    system = (
        "You are a helpful vision-language assistant. Describe food images "
        "accurately and concisely. Return only a clear caption without markdown."
    )
    prompt = f"""
Write a concise caption for this image of {food_name}.

Describe the visible ingredients, appearance, plating, colors, and style.
Make the caption useful for a recipe or restaurant recommendation system.
""".strip()
    return system, prompt


def review_caption_prompt(review_text: str) -> tuple[str, str]:
    """Create an image prompt grounded in the associated written review."""

    system = (
        "You describe restaurant review images using the written review as "
        "context. Return one concise, informative caption without markdown."
    )
    prompt = f"""
Caption this review image using the written review as context:

{review_text}

Focus on visible food or drinks, atmosphere, presentation, and details that
help explain the reviewer's experience.
""".strip()
    return system, prompt


def media_type_for_path(path: Path) -> str:
    """Infer an image MIME type instead of assuming every file is JPEG."""

    media_type, _ = mimetypes.guess_type(path.name)
    if not media_type or not media_type.startswith("image/"):
        return "application/octet-stream"
    return media_type


def download_file(
    url: str,
    destination: Path,
    *,
    timeout: tuple[float, float] = (10.0, 180.0),
) -> Path:
    """Stream a course asset to disk only when it is not already present."""

    if destination.exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
    return destination


def safe_extract_zip(archive: Path, destination: Path) -> None:
    """Extract a ZIP while rejecting path traversal and symbolic links."""

    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()

    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe ZIP path: {member.filename}")

            unix_mode = member.external_attr >> 16
            if unix_mode & 0o170000 == 0o120000:
                raise ValueError(f"ZIP contains a symbolic link: {member.filename}")

        zip_file.extractall(destination)


def load_json_records(path: Path) -> list[dict[str, Any]]:
    """Load a JSON array and verify that every item is an object."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"Expected a JSON array of objects in {path}")
    return value


def recipe_image_path(
    recipe: Mapping[str, Any],
    image_directory: Path,
) -> Path:
    """Resolve the archive's recipe<ID>.png convention."""

    recipe_id = recipe.get("id")
    if not isinstance(recipe_id, int):
        raise ValueError("Recipe is missing an integer id")

    path = image_directory / f"recipe{recipe_id}.png"
    if not path.is_file():
        raise FileNotFoundError(f"Image for recipe {recipe_id} not found: {path}")
    return path


def parse_image_urls(value: Any) -> list[str]:
    """Normalize the review dataset's stringified list of image URLs."""

    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(parsed, list) or not all(isinstance(url, str) for url in parsed):
        raise ValueError("Review images must be a list of URL strings")
    return parsed


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def download_image_with_retry(url: str) -> tuple[bytes, str]:
    """Download one review image with bounded exponential backoff."""

    response = requests.get(url, timeout=(5.0, 30.0))
    response.raise_for_status()
    media_type = response.headers.get("Content-Type", "application/octet-stream")
    media_type = media_type.partition(";")[0]
    return response.content, media_type


def augment_recipes(
    recipes: Iterable[Mapping[str, Any]],
    image_directory: Path,
    vision_llm: VisionCall,
    progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Add an image description to every recipe record."""

    source_records = list(recipes)
    enriched_records: list[dict[str, Any]] = []

    for index, recipe in enumerate(source_records, start=1):
        image_path = recipe_image_path(recipe, image_directory)
        system, prompt = recipe_caption_prompt(str(recipe.get("name", "dish")))
        caption = vision_llm(
            system,
            prompt,
            image_path.read_bytes(),
            media_type_for_path(image_path),
        )
        enriched_records.append({**recipe, "image_description": caption})
        if progress:
            progress(index, len(source_records))

    return enriched_records


def augment_reviews(
    reviews: Iterable[Mapping[str, Any]],
    vision_llm: VisionCall,
    *,
    image_downloader: ImageDownloader = download_image_with_retry,
    progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Add contextual captions for every downloadable review image."""

    source_records = list(reviews)
    enriched_records: list[dict[str, Any]] = []

    for index, review in enumerate(source_records, start=1):
        review_text = str(review.get("text", ""))
        system, prompt = review_caption_prompt(review_text)
        captions = []

        for image_url in parse_image_urls(review.get("images", [])):
            try:
                image_bytes, media_type = image_downloader(image_url)
            except requests.RequestException:
                continue
            captions.append(vision_llm(system, prompt, image_bytes, media_type))

        enriched_records.append({**review, "image_captions": captions})
        if progress:
            progress(index, len(source_records))

    return enriched_records


def save_json_records(records: Iterable[Mapping[str, Any]], path: Path) -> None:
    """Save enriched records as readable UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(list(records), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
