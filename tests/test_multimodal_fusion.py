import pytest

from ibm_rag_agentic_showcase.multimodal_fusion import (
    cosine_distances_to_similarities,
    format_fused_results,
    fuse_hits,
    minmax_normalize,
)
from ibm_rag_agentic_showcase.similarity_retrieval import RetrievalHit


def make_hit(identifier: str, distance: float, source: str) -> RetrievalHit:
    return RetrievalHit(
        id=identifier,
        content=f"Content for {identifier}",
        metadata={"source": source},
        distance=distance,
    )


def test_distance_conversion_and_minmax_normalization():
    similarities = cosine_distances_to_similarities([0.1, 0.4, 0.7])

    assert similarities == pytest.approx([0.9, 0.6, 0.3])
    assert minmax_normalize(similarities) == pytest.approx([1.0, 0.5, 0.0])


def test_constant_pool_retains_valid_candidate_scores():
    assert minmax_normalize([0.75]) == [1.0]
    assert minmax_normalize([0.5, 0.5]) == [1.0, 1.0]
    assert minmax_normalize([]) == []


def test_weight_tuning_changes_top_modality():
    articles = [make_hit("article-1", 0.1, "restaurant_article")]
    images = [make_hit("image-1", 0.1, "recipe_image")]

    article_heavy = fuse_hits(
        articles,
        images,
        text_weight=0.8,
        image_weight=0.2,
    )
    image_heavy = fuse_hits(
        articles,
        images,
        text_weight=0.2,
        image_weight=0.8,
    )

    assert article_heavy[0].modality == "article"
    assert image_heavy[0].modality == "image"


def test_invalid_weights_and_top_n_are_rejected():
    with pytest.raises(ValueError, match="negative"):
        fuse_hits([], [], text_weight=-1, image_weight=1)
    with pytest.raises(ValueError, match="positive"):
        fuse_hits([], [], text_weight=0, image_weight=0)
    with pytest.raises(ValueError, match="top_n"):
        fuse_hits([], [], top_n=-1)


def test_empty_fusion_formats_actionable_message():
    assert "relaxing" in format_fused_results([], title="Empty")
