"""Weighted fusion and reranking across text and image retrieval pools."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_chroma import Chroma

from .multimodal_vector_index import TextEmbeddingModel
from .similarity_retrieval import (
    RetrievalHit,
    retrieve_articles,
    retrieve_images_by_text,
)


@dataclass(frozen=True)
class FusedResult:
    """A candidate with modality-specific and weighted normalized scores."""

    modality: str
    id: str
    metadata: dict[str, Any]
    text_score: float
    image_score: float
    fused_score: float
    snippet: str


def cosine_distances_to_similarities(
    distances: Sequence[float],
) -> list[float]:
    """Convert cosine distance to similarity before score normalization."""

    return [1.0 - float(distance) for distance in distances]


def minmax_normalize(values: Sequence[float]) -> list[float]:
    """Scale a modality to 0..1, preserving constant pools as valid matches."""

    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    spread = maximum - minimum
    if spread <= 1e-12:
        return [1.0] * len(values)
    return [(float(value) - minimum) / spread for value in values]


def _normalized_weights(text_weight: float, image_weight: float) -> tuple[float, float]:
    if text_weight < 0 or image_weight < 0:
        raise ValueError("Fusion weights cannot be negative")
    total = text_weight + image_weight
    if total <= 0:
        raise ValueError("At least one fusion weight must be positive")
    return text_weight / total, image_weight / total


def fuse_hits(
    article_hits: Sequence[RetrievalHit],
    image_hits: Sequence[RetrievalHit],
    *,
    text_weight: float = 0.6,
    image_weight: float = 0.4,
    top_n: int | None = 5,
) -> list[FusedResult]:
    """Normalize two candidate pools, weight them, and produce one ranking."""

    text_weight, image_weight = _normalized_weights(
        text_weight,
        image_weight,
    )
    text_scores = minmax_normalize(
        cosine_distances_to_similarities([hit.distance for hit in article_hits])
    )
    image_scores = minmax_normalize(
        cosine_distances_to_similarities([hit.distance for hit in image_hits])
    )

    rows = [
        FusedResult(
            modality="article",
            id=hit.id,
            metadata=hit.metadata,
            text_score=text_scores[index],
            image_score=0.0,
            fused_score=text_weight * text_scores[index],
            snippet=" ".join(hit.content.split()),
        )
        for index, hit in enumerate(article_hits)
    ]
    rows.extend(
        FusedResult(
            modality="image",
            id=hit.id,
            metadata=hit.metadata,
            text_score=0.0,
            image_score=image_scores[index],
            fused_score=image_weight * image_scores[index],
            snippet=" ".join(hit.content.split()),
        )
        for index, hit in enumerate(image_hits)
    )
    rows.sort(key=lambda row: (-row.fused_score, row.modality, row.id))

    if top_n is None:
        return rows
    if top_n < 0:
        raise ValueError("top_n cannot be negative")
    return rows[:top_n]


def fuse_rank(
    query: str,
    article_store: Chroma,
    image_store: Chroma,
    text_embedder: TextEmbeddingModel,
    clip_embedder: TextEmbeddingModel,
    *,
    k_text: int = 5,
    k_image: int = 5,
    text_weight: float = 0.6,
    image_weight: float = 0.4,
    where_text: Mapping[str, Any] | None = None,
    where_image: Mapping[str, Any] | None = None,
    top_n: int | None = 5,
) -> list[FusedResult]:
    """Retrieve from both collections and rerank the mixed candidate pool."""

    article_hits = retrieve_articles(
        article_store,
        text_embedder,
        query,
        k=k_text,
        where=where_text,
    )
    image_hits = retrieve_images_by_text(
        image_store,
        clip_embedder,
        query,
        k=k_image,
        where=where_image,
    )
    return fuse_hits(
        article_hits,
        image_hits,
        text_weight=text_weight,
        image_weight=image_weight,
        top_n=top_n,
    )


def format_fused_results(
    rows: Sequence[FusedResult],
    *,
    title: str,
    max_chars: int = 90,
) -> str:
    """Format fused candidates with transparent component scores."""

    lines = [f"\n=== {title} ==="]
    if not rows:
        lines.append("No results found. Try relaxing the metadata filters.")
        return "\n".join(lines)

    for index, row in enumerate(rows, start=1):
        snippet = row.snippet
        if len(snippet) > max_chars:
            snippet = f"{snippet[:max_chars].rstrip()}..."
        lines.append(
            f"[{index}] {row.modality} | id={row.id} | "
            f"cuisine={row.metadata.get('cuisine', 'N/A')} | "
            f"location={row.metadata.get('location', 'N/A')} | "
            f"fused={row.fused_score:.4f} "
            f"(text={row.text_score:.4f}, image={row.image_score:.4f})"
        )
        lines.append(snippet)
    return "\n".join(lines)
