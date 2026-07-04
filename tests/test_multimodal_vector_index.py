from pathlib import Path

import numpy as np
import pytest

from ibm_rag_agentic_showcase.multimodal_vector_index import (
    IMAGE_COLLECTION,
    IMAGE_EMBEDDING_DIMENSION,
    RESTAURANT_COLLECTION,
    TEXT_EMBEDDING_DIMENSION,
    build_recipe_image_documents,
    build_restaurant_documents,
    construct_multimodal_index,
    query_collection,
    validate_embeddings,
)

RESTAURANT = {
    "itemId": 1_000_001,
    "name": "Coastal Table",
    "location": "Santa Monica",
    "type": "bistro",
    "food_style": "Californian seafood",
    "rating": 4.6,
    "price_range": 3,
    "signatures": ["grilled snapper"],
    "vibe": "bright",
    "environment": "An airy dining room near the beach.",
    "shortcomings": [],
}

RECIPE = {
    "id": 7,
    "name": "Garden Bowl",
    "cuisine": "Californian",
    "ingredients": ["tomato", "avocado"],
    "image_description": "A colorful bowl of fresh vegetables.",
}


class FakeTextEmbedder:
    dimension = TEXT_EMBEDDING_DIMENSION

    def embed_texts(self, texts):
        return [[1.0] + [0.0] * (self.dimension - 1) for _ in texts]


class FakeImageEmbedder:
    dimension = IMAGE_EMBEDDING_DIMENSION

    def embed_images(self, paths):
        return [[1.0] + [0.0] * (self.dimension - 1) for _ in paths]


def test_builds_restaurant_document_with_required_metadata():
    document = build_restaurant_documents([RESTAURANT])[0]

    assert "Signature dishes: grilled snapper" in document.page_content
    assert document.metadata["location"] == "Santa Monica"
    assert document.metadata["source"] == "restaurant_article"


def test_builds_recipe_image_document_with_required_metadata(
    tmp_path: Path,
):
    image_path = tmp_path / "recipe7.png"
    image_path.write_bytes(b"image")

    document = build_recipe_image_documents([RECIPE], tmp_path)[0]

    assert document.metadata["cuisine"] == "Californian"
    assert document.metadata["image_path"] == str(image_path.resolve())
    assert document.metadata["source"] == "recipe_image"


def test_embedding_validation_accepts_normalized_expected_shape():
    vectors = np.zeros((2, TEXT_EMBEDDING_DIMENSION), dtype=np.float32)
    vectors[:, 0] = 1.0

    validate_embeddings(
        vectors.tolist(),
        expected_count=2,
        expected_dimension=TEXT_EMBEDDING_DIMENSION,
    )


def test_embedding_validation_rejects_wrong_dimension():
    with pytest.raises(ValueError, match="embedding shape"):
        validate_embeddings(
            [[1.0, 0.0]],
            expected_count=1,
            expected_dimension=TEXT_EMBEDDING_DIMENSION,
        )


def test_embedding_validation_rejects_non_normalized_vectors():
    vector = [2.0] + [0.0] * (IMAGE_EMBEDDING_DIMENSION - 1)

    with pytest.raises(ValueError, match="L2-normalized"):
        validate_embeddings(
            [vector],
            expected_count=1,
            expected_dimension=IMAGE_EMBEDDING_DIMENSION,
        )


def test_constructs_and_queries_both_persistent_collections(
    tmp_path: Path,
):
    image_path = tmp_path / "recipe7.png"
    image_path.write_bytes(b"image")
    restaurants = build_restaurant_documents([RESTAURANT])
    images = build_recipe_image_documents([RECIPE], tmp_path)
    persist_directory = tmp_path / "index"

    summary = construct_multimodal_index(
        restaurants,
        images,
        FakeTextEmbedder(),
        FakeImageEmbedder(),
        persist_directory,
    )

    assert summary.restaurant_count == 1
    assert summary.image_count == 1

    restaurant_result = query_collection(
        persist_directory,
        RESTAURANT_COLLECTION,
        [1.0] + [0.0] * (TEXT_EMBEDDING_DIMENSION - 1),
    )
    image_result = query_collection(
        persist_directory,
        IMAGE_COLLECTION,
        [1.0] + [0.0] * (IMAGE_EMBEDDING_DIMENSION - 1),
    )

    assert restaurant_result["ids"] == [["restaurant-1000001"]]
    assert image_result["ids"] == [["recipe-7"]]
