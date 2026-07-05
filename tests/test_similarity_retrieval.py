from pathlib import Path

import pytest

from ibm_rag_agentic_showcase.multimodal_vector_index import (
    IMAGE_EMBEDDING_DIMENSION,
    TEXT_EMBEDDING_DIMENSION,
    build_recipe_image_documents,
    build_restaurant_documents,
    construct_multimodal_index,
)
from ibm_rag_agentic_showcase.similarity_retrieval import (
    existing_metadata_filter,
    format_hits,
    image_record_at_index,
    open_multimodal_stores,
    retrieve_articles,
    retrieve_images_by_image,
)

RESTAURANTS = [
    {
        "itemId": 1,
        "name": "Pasadena Pasta",
        "location": "Pasadena",
        "type": "trattoria",
        "food_style": "Italian",
        "rating": 4.7,
        "price_range": 3,
        "signatures": ["handmade pasta"],
        "vibe": "romantic",
        "environment": "A candlelit dining room.",
        "shortcomings": [],
    },
    {
        "itemId": 2,
        "name": "Coastal Noodles",
        "location": "Santa Monica",
        "type": "noodle house",
        "food_style": "Asian",
        "rating": 4.4,
        "price_range": 2,
        "signatures": ["broth noodles"],
        "vibe": "cozy",
        "environment": "A warm neighborhood room.",
        "shortcomings": [],
    },
]

RECIPES = [
    {
        "id": 1,
        "name": "Garden Bowl",
        "cuisine": "Californian",
        "ingredients": ["tomato"],
        "image_description": "A colorful vegetable bowl.",
    },
    {
        "id": 2,
        "name": "Pasta Plate",
        "cuisine": "Italian",
        "ingredients": ["pasta"],
        "image_description": "Fresh pasta on a white plate.",
    },
]


class FakeTextEmbedder:
    dimension = TEXT_EMBEDDING_DIMENSION

    def embed_texts(self, texts):
        return [[1.0] + [0.0] * (self.dimension - 1) for _ in texts]


class FakeImageEmbedder:
    dimension = IMAGE_EMBEDDING_DIMENSION

    def embed_images(self, paths):
        return [[1.0] + [0.0] * (self.dimension - 1) for _ in paths]


@pytest.fixture
def retrieval_setup(tmp_path: Path):
    for recipe in RECIPES:
        (tmp_path / f"recipe{recipe['id']}.png").write_bytes(b"image")

    restaurant_documents = build_restaurant_documents(RESTAURANTS)
    image_documents = build_recipe_image_documents(RECIPES, tmp_path)
    persist_directory = tmp_path / "index"
    construct_multimodal_index(
        restaurant_documents,
        image_documents,
        FakeTextEmbedder(),
        FakeImageEmbedder(),
        persist_directory,
    )
    return (
        open_multimodal_stores(persist_directory),
        persist_directory,
    )


def test_opens_and_counts_existing_collections(retrieval_setup):
    stores, _ = retrieval_setup

    assert stores.article_count == 2
    assert stores.image_count == 2


def test_article_retrieval_applies_location_filter(retrieval_setup):
    stores, _ = retrieval_setup

    hits = retrieve_articles(
        stores.articles,
        FakeTextEmbedder(),
        "romantic handmade pasta",
        where={"location": "Pasadena"},
    )

    assert len(hits) == 1
    assert hits[0].metadata["location"] == "Pasadena"


def test_existing_filter_falls_back_to_real_metadata(retrieval_setup):
    stores, _ = retrieval_setup

    selected = existing_metadata_filter(
        stores.articles,
        "location",
        preferred_value="Not A Real Place",
    )

    assert selected == {"location": "Pasadena"}


def test_image_retrieval_applies_cuisine_filter(retrieval_setup):
    stores, _ = retrieval_setup
    image_path, _ = image_record_at_index(stores.images, 0)

    hits = retrieve_images_by_image(
        stores.images,
        FakeImageEmbedder(),
        image_path,
        where={"cuisine": "Italian"},
    )

    assert len(hits) == 1
    assert hits[0].metadata["cuisine"] == "Italian"


def test_image_index_validation_and_empty_format(retrieval_setup):
    stores, _ = retrieval_setup

    with pytest.raises(IndexError, match="outside"):
        image_record_at_index(stores.images, 99)

    output = format_hits([], title="Empty demo")
    assert "No results found" in output
