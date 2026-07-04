"""Build persistent text and image vector collections for Lab 04."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

import numpy as np
from langchain_core.documents import Document

from .multimodal_augmentation import load_json_records, recipe_image_path

TEXT_EMBEDDING_DIMENSION = 384
IMAGE_EMBEDDING_DIMENSION = 512
RESTAURANT_COLLECTION = "restaurant_articles"
IMAGE_COLLECTION = "food_images"
T = TypeVar("T")


class TextEmbeddingModel(Protocol):
    """Minimal interface required by the restaurant indexing pipeline."""

    dimension: int

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one unit-normalized vector for each text."""


class ImageEmbeddingModel(Protocol):
    """Minimal interface required by the recipe-image indexing pipeline."""

    dimension: int

    def embed_images(self, paths: Sequence[Path]) -> list[list[float]]:
        """Return one unit-normalized vector for each image."""


@dataclass(frozen=True)
class IndexBuildSummary:
    """Observable result of one complete persistent-index build."""

    persist_directory: Path
    restaurant_count: int
    image_count: int
    text_dimension: int
    image_dimension: int


class SentenceTransformerEmbedder:
    """CPU Sentence-Transformers adapter producing normalized 384-D vectors."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        if self.dimension != TEXT_EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected a {TEXT_EMBEDDING_DIMENSION}-D text model, "
                f"received {self.dimension}-D"
            )

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > self.batch_size,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()


class CLIPEmbedder:
    """CPU CLIP adapter for normalized image and cross-modal text vectors."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        *,
        device: str = "cpu",
        batch_size: int = 8,
    ) -> None:
        import torch
        from transformers import AutoProcessor, CLIPModel

        self.torch = torch
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.dimension = int(self.model.config.projection_dim)
        if self.dimension != IMAGE_EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected a {IMAGE_EMBEDDING_DIMENSION}-D CLIP model, "
                f"received {self.dimension}-D"
            )

    @staticmethod
    def _feature_tensor(value: Any) -> Any:
        """Support tensor and model-output return styles across Transformers."""

        for attribute in ("image_embeds", "text_embeds", "pooler_output"):
            if hasattr(value, attribute):
                return getattr(value, attribute)
        return value

    def embed_images(self, paths: Sequence[Path]) -> list[list[float]]:
        from PIL import Image

        all_vectors = []
        for start in range(0, len(paths), self.batch_size):
            batch_paths = paths[start : start + self.batch_size]
            images = []
            try:
                for path in batch_paths:
                    images.append(Image.open(path).convert("RGB"))
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                with self.torch.inference_mode():
                    features = self._feature_tensor(
                        self.model.get_image_features(**inputs)
                    )
                    if features.shape[-1] != self.dimension:
                        raise ValueError("CLIP returned an unexpected image dimension")
                    features = self.torch.nn.functional.normalize(
                        features,
                        p=2,
                        dim=1,
                    )
                all_vectors.extend(features.cpu().numpy().tolist())
            finally:
                for image in images:
                    image.close()
        return all_vectors

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed text in the same 512-D space for text-to-image retrieval."""

        all_vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            inputs = self.processor(
                text=batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                features = self._feature_tensor(self.model.get_text_features(**inputs))
                if features.shape[-1] != self.dimension:
                    raise ValueError("CLIP returned an unexpected text dimension")
                features = self.torch.nn.functional.normalize(
                    features,
                    p=2,
                    dim=1,
                )
            all_vectors.extend(features.cpu().numpy().tolist())
        return all_vectors


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _present_lines(fields: Iterable[tuple[str, Any]]) -> str:
    lines = []
    for label, value in fields:
        if value is None or value == "" or value == []:
            continue
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        lines.append(f"{label}: {value}")
    return "\n".join(lines)


def _scalar_metadata(values: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only scalar values supported by Chroma metadata."""

    allowed = (str, int, float, bool)
    return {key: value for key, value in values.items() if isinstance(value, allowed)}


def build_restaurant_documents(
    records: Iterable[Mapping[str, Any]],
) -> list[Document]:
    """Convert structured restaurant records to retrieval-ready documents."""

    documents = []
    for record in records:
        item_id = record.get("itemId")
        name = record.get("name")
        location = record.get("location")
        if not isinstance(item_id, int) or not isinstance(name, str):
            raise ValueError("Each restaurant needs an integer itemId and name")
        if not isinstance(location, str):
            raise ValueError(f"Restaurant {item_id} is missing its location")

        content = _present_lines(
            [
                ("Name", name),
                ("Location", location),
                ("Type", record.get("type")),
                ("Food style", record.get("food_style")),
                ("Rating", record.get("rating")),
                ("Price range", record.get("price_range")),
                ("Signature dishes", _text_list(record.get("signatures"))),
                ("Vibe", record.get("vibe")),
                ("Environment", record.get("environment")),
                ("Shortcomings", _text_list(record.get("shortcomings"))),
            ]
        )
        metadata = _scalar_metadata(
            {
                "item_id": item_id,
                "name": name,
                "location": location,
                "restaurant_type": record.get("type"),
                "food_style": record.get("food_style"),
                "source": "restaurant_article",
            }
        )
        documents.append(Document(page_content=content, metadata=metadata))
    return documents


def build_recipe_image_documents(
    records: Iterable[Mapping[str, Any]],
    image_directory: Path,
) -> list[Document]:
    """Create recipe documents whose metadata points to source images."""

    documents = []
    for record in records:
        recipe_id = record.get("id")
        name = record.get("name")
        cuisine = record.get("cuisine")
        if not isinstance(recipe_id, int) or not isinstance(name, str):
            raise ValueError("Each recipe needs an integer id and name")
        if not isinstance(cuisine, str):
            raise ValueError(f"Recipe {recipe_id} is missing its cuisine")

        image_path = recipe_image_path(record, image_directory).resolve()
        content = _present_lines(
            [
                ("Recipe", name),
                ("Cuisine", cuisine),
                ("Image description", record.get("image_description")),
                ("Ingredients", _text_list(record.get("ingredients"))),
            ]
        )
        metadata = {
            "recipe_id": recipe_id,
            "name": name,
            "cuisine": cuisine,
            "image_path": str(image_path),
            "source": "recipe_image",
        }
        documents.append(Document(page_content=content, metadata=metadata))
    return documents


def validate_embeddings(
    vectors: Sequence[Sequence[float]],
    *,
    expected_count: int,
    expected_dimension: int,
    tolerance: float = 1e-4,
) -> None:
    """Reject missing, wrongly shaped, non-finite, or non-normalized vectors."""

    array = np.asarray(vectors, dtype=np.float32)
    expected_shape = (expected_count, expected_dimension)
    if array.shape != expected_shape:
        raise ValueError(
            f"Expected embedding shape {expected_shape}, received {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("Embeddings contain NaN or infinite values")
    norms = np.linalg.norm(array, axis=1)
    if not np.allclose(norms, 1.0, atol=tolerance):
        raise ValueError("Embeddings must be L2-normalized")


def _chunks(items: Sequence[T], batch_size: int) -> Iterable[Sequence[T]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _document_id(document: Document, *, kind: str) -> str:
    identifier_key = "item_id" if kind == "restaurant" else "recipe_id"
    return f"{kind}-{document.metadata[identifier_key]}"


def construct_multimodal_index(
    restaurant_documents: Sequence[Document],
    image_documents: Sequence[Document],
    text_embedder: TextEmbeddingModel,
    image_embedder: ImageEmbeddingModel,
    persist_directory: Path,
    *,
    reset: bool = True,
    batch_size: int = 32,
) -> IndexBuildSummary:
    """Build separate persistent collections for incompatible vector spaces."""

    import chromadb

    if text_embedder.dimension != TEXT_EMBEDDING_DIMENSION:
        raise ValueError("Text embedder must produce 384-D vectors")
    if image_embedder.dimension != IMAGE_EMBEDDING_DIMENSION:
        raise ValueError("Image embedder must produce 512-D vectors")
    if not restaurant_documents or not image_documents:
        raise ValueError("Both restaurant and image documents are required")

    persist_directory.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_directory))
    collection_names = {
        item if isinstance(item, str) else item.name
        for item in client.list_collections()
    }
    if reset:
        for name in (RESTAURANT_COLLECTION, IMAGE_COLLECTION):
            if name in collection_names:
                client.delete_collection(name)

    restaurant_collection = client.get_or_create_collection(
        name=RESTAURANT_COLLECTION,
        embedding_function=None,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dimension": TEXT_EMBEDDING_DIMENSION,
        },
    )
    image_collection = client.get_or_create_collection(
        name=IMAGE_COLLECTION,
        embedding_function=None,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": "openai/clip-vit-base-patch32",
            "embedding_dimension": IMAGE_EMBEDDING_DIMENSION,
        },
    )

    restaurant_ids = [
        _document_id(document, kind="restaurant") for document in restaurant_documents
    ]
    image_ids = [_document_id(document, kind="recipe") for document in image_documents]
    if len(set(restaurant_ids)) != len(restaurant_ids):
        raise ValueError("Restaurant document IDs must be unique")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Recipe document IDs must be unique")

    for batch in _chunks(restaurant_documents, batch_size):
        texts = [document.page_content for document in batch]
        vectors = text_embedder.embed_texts(texts)
        validate_embeddings(
            vectors,
            expected_count=len(batch),
            expected_dimension=TEXT_EMBEDDING_DIMENSION,
        )
        restaurant_collection.upsert(
            ids=[_document_id(document, kind="restaurant") for document in batch],
            documents=texts,
            metadatas=[document.metadata for document in batch],
            embeddings=vectors,
        )

    for batch in _chunks(image_documents, batch_size):
        paths = [Path(document.metadata["image_path"]) for document in batch]
        vectors = image_embedder.embed_images(paths)
        validate_embeddings(
            vectors,
            expected_count=len(batch),
            expected_dimension=IMAGE_EMBEDDING_DIMENSION,
        )
        image_collection.upsert(
            ids=[_document_id(document, kind="recipe") for document in batch],
            documents=[document.page_content for document in batch],
            metadatas=[document.metadata for document in batch],
            embeddings=vectors,
        )

    return IndexBuildSummary(
        persist_directory=persist_directory,
        restaurant_count=restaurant_collection.count(),
        image_count=image_collection.count(),
        text_dimension=text_embedder.dimension,
        image_dimension=image_embedder.dimension,
    )


def query_collection(
    persist_directory: Path,
    collection_name: str,
    query_vector: Sequence[float],
    *,
    limit: int = 3,
) -> dict[str, Any]:
    """Query one collection with an embedding from its matching model."""

    import chromadb

    client = chromadb.PersistentClient(path=str(persist_directory))
    collection = client.get_collection(
        name=collection_name,
        embedding_function=None,
    )
    return collection.query(
        query_embeddings=[list(query_vector)],
        n_results=min(limit, collection.count()),
        include=["documents", "metadatas", "distances"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construct persistent restaurant-text and recipe-image indexes."
    )
    parser.add_argument(
        "--restaurants",
        type=Path,
        default=Path("data/processed/structured_restaurant_data.json"),
    )
    parser.add_argument(
        "--recipes",
        type=Path,
        default=Path("data/processed/lab02/augmented_food_recipe.json"),
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("data/raw/lab02/images/synthetic_recipe_images"),
    )
    parser.add_argument(
        "--persist-directory",
        type=Path,
        default=Path("data/indexes/multimodal"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Index only the first N records from each dataset.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Upsert into existing collections instead of rebuilding them.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    restaurant_records = load_json_records(args.restaurants)
    recipe_records = load_json_records(args.recipes)
    if args.limit:
        restaurant_records = restaurant_records[: args.limit]
        recipe_records = recipe_records[: args.limit]

    print(
        f"Loaded {len(restaurant_records)} restaurants and "
        f"{len(recipe_records)} recipes"
    )
    restaurant_documents = build_restaurant_documents(restaurant_records)
    image_documents = build_recipe_image_documents(
        recipe_records,
        args.images,
    )
    print(
        f"Built {len(restaurant_documents)} restaurant Documents and "
        f"{len(image_documents)} image Documents"
    )

    print("Loading CPU embedding models...")
    text_embedder = SentenceTransformerEmbedder()
    image_embedder = CLIPEmbedder()
    print(
        f"Text dimension: {text_embedder.dimension}; "
        f"image dimension: {image_embedder.dimension}"
    )

    summary = construct_multimodal_index(
        restaurant_documents,
        image_documents,
        text_embedder,
        image_embedder,
        args.persist_directory,
        reset=not args.keep_existing,
    )
    print(
        f"Indexed {summary.restaurant_count} restaurant articles and "
        f"{summary.image_count} food images in {summary.persist_directory}"
    )


if __name__ == "__main__":
    main()
