"""Similarity retrieval and metadata filtering for Lab 05."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma

from .multimodal_vector_index import (
    IMAGE_COLLECTION,
    RESTAURANT_COLLECTION,
    ImageEmbeddingModel,
    TextEmbeddingModel,
)


@dataclass(frozen=True)
class RetrievalHit:
    """One search result with content, metadata, and cosine distance."""

    id: str
    content: str
    metadata: dict[str, Any]
    distance: float


@dataclass(frozen=True)
class MultimodalStores:
    """The two persistent vector stores created by Lab 04."""

    articles: Chroma
    images: Chroma
    article_count: int
    image_count: int


def _store_count(store: Chroma) -> int:
    """Count records through the public LangChain Chroma API."""

    return len(store.get(include=["metadatas"])["ids"])


def open_multimodal_stores(persist_directory: Path) -> MultimodalStores:
    """Open and verify both Lab 04 collections without creating new ones."""

    if not persist_directory.is_dir():
        raise FileNotFoundError(
            f"Vector database directory not found: {persist_directory}"
        )

    articles = Chroma(
        collection_name=RESTAURANT_COLLECTION,
        persist_directory=str(persist_directory),
        embedding_function=None,
        create_collection_if_not_exists=False,
    )
    images = Chroma(
        collection_name=IMAGE_COLLECTION,
        persist_directory=str(persist_directory),
        embedding_function=None,
        create_collection_if_not_exists=False,
    )
    article_count = _store_count(articles)
    image_count = _store_count(images)
    if article_count == 0 or image_count == 0:
        raise ValueError("One or more multimodal collections are empty")

    return MultimodalStores(
        articles=articles,
        images=images,
        article_count=article_count,
        image_count=image_count,
    )


def _to_hits(results: Sequence[tuple[Any, float]]) -> list[RetrievalHit]:
    hits = []
    for document, distance in results:
        metadata = dict(document.metadata)
        fallback_id = metadata.get("item_id") or metadata.get("recipe_id") or "unknown"
        hits.append(
            RetrievalHit(
                id=str(document.id or fallback_id),
                content=document.page_content,
                metadata=metadata,
                distance=float(distance),
            )
        )
    return hits


def retrieve_articles(
    store: Chroma,
    text_embedder: TextEmbeddingModel,
    query: str,
    *,
    k: int = 5,
    where: Mapping[str, Any] | None = None,
) -> list[RetrievalHit]:
    """Search restaurant text with an optional Chroma metadata constraint."""

    if not query.strip():
        raise ValueError("Article query cannot be blank")
    if k < 1:
        raise ValueError("k must be at least 1")

    query_vector = text_embedder.embed_texts([query])[0]
    results = store.similarity_search_by_vector_with_relevance_scores(
        embedding=query_vector,
        k=k,
        filter=dict(where) if where else None,
    )
    return _to_hits(results)


def retrieve_images_by_image(
    store: Chroma,
    image_embedder: ImageEmbeddingModel,
    query_image_path: Path,
    *,
    k: int = 5,
    where: Mapping[str, Any] | None = None,
) -> list[RetrievalHit]:
    """Search the CLIP collection using one image and optional metadata."""

    if not query_image_path.is_file():
        raise FileNotFoundError(f"Query image not found: {query_image_path}")
    if k < 1:
        raise ValueError("k must be at least 1")

    query_vector = image_embedder.embed_images([query_image_path])[0]
    results = store.similarity_search_by_vector_with_relevance_scores(
        embedding=query_vector,
        k=k,
        filter=dict(where) if where else None,
    )
    return _to_hits(results)


def metadata_values(store: Chroma, field: str) -> list[Any]:
    """Return sorted distinct scalar values available for a metadata field."""

    result = store.get(include=["metadatas"])
    values = {
        metadata[field]
        for metadata in result["metadatas"]
        if metadata and field in metadata
    }
    return sorted(values, key=str)


def existing_metadata_filter(
    store: Chroma,
    field: str,
    preferred_value: Any | None = None,
) -> dict[str, Any] | None:
    """Select a real filter value, preferring the requested value if present."""

    values = metadata_values(store, field)
    if not values:
        return None
    selected = preferred_value if preferred_value in values else values[0]
    return {field: selected}


def image_record_at_index(
    store: Chroma,
    index: int,
) -> tuple[Path, dict[str, Any]]:
    """Return the image path and metadata for an interactive query index."""

    result = store.get(include=["metadatas"])
    metadata_rows = result["metadatas"]
    if not 0 <= index < len(metadata_rows):
        raise IndexError(
            f"Image query index {index} is outside 0..{len(metadata_rows) - 1}"
        )

    metadata = dict(metadata_rows[index] or {})
    raw_path = metadata.get("image_path")
    if not isinstance(raw_path, str):
        raise ValueError("Selected image record has no image_path metadata")
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"Indexed image is missing: {path}")
    return path, metadata


def format_hits(
    hits: Sequence[RetrievalHit],
    *,
    title: str,
    max_chars: int = 180,
) -> str:
    """Format mixed article/image results for terminal or notebook output."""

    lines = [f"\n=== {title} ==="]
    if not hits:
        lines.append("No results found with the current query and filter.")
        return "\n".join(lines)

    for index, hit in enumerate(hits, start=1):
        snippet = " ".join(hit.content.split())
        if len(snippet) > max_chars:
            snippet = f"{snippet[:max_chars].rstrip()}..."

        cuisine = hit.metadata.get("cuisine", "N/A")
        location = hit.metadata.get("location", "N/A")
        source = hit.metadata.get("source", "N/A")
        lines.append(
            f"[{index}] id={hit.id} | cuisine={cuisine} | "
            f"location={location} | source={source} | "
            f"distance={hit.distance:.4f}"
        )
        lines.append(snippet)
    return "\n".join(lines)
