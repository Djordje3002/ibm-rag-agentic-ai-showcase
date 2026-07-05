# %% [markdown]
# # Lab 05 — Similarity retrieval with metadata filtering
#
# ## What this lab does
#
# Lab 04 built two persistent Chroma collections. This lab opens those existing
# collections and demonstrates three retrieval patterns:
#
# 1. Restaurant article similarity search without a filter.
# 2. Restaurant article similarity search constrained by location metadata.
# 3. Recipe image-to-image similarity search, optionally constrained by cuisine.
#
# Text queries use the same normalized 384-D Sentence-Transformers model used to
# index restaurant articles. Image queries use the same normalized 512-D CLIP
# image encoder used to index recipes. A query must always use the embedding
# model associated with its collection.
#
# ## Portfolio-level safety improvements
#
# - Uses public LangChain Chroma methods instead of private `._collection`.
# - Opens collections with `create_collection_if_not_exists=False`, preventing
#   a typo from silently creating an empty database.
# - Chooses filter values that actually exist in stored metadata.
# - Handles empty filtered searches and missing image paths clearly.
# - Closes image files after creating safe thumbnail copies.
#
# Complete Lab 04 before running this file. The first model initialization may
# download weights into the Hugging Face cache.

# %%
from pathlib import Path

from PIL import Image

from ibm_rag_agentic_showcase.multimodal_vector_index import (
    CLIPEmbedder,
    SentenceTransformerEmbedder,
)
from ibm_rag_agentic_showcase.similarity_retrieval import (
    existing_metadata_filter,
    format_hits,
    image_record_at_index,
    open_multimodal_stores,
    retrieve_articles,
    retrieve_images_by_image,
)

try:
    from IPython.display import display as display_image
except ImportError:

    def display_image(image: Image.Image) -> None:
        image.show()


def show_thumbnail(path: Path, size: tuple[int, int] = (300, 300)) -> None:
    """Display a detached thumbnail while closing the source image promptly."""

    with Image.open(path) as source:
        thumbnail = source.copy()
    thumbnail.thumbnail(size)
    display_image(thumbnail)


# %% [markdown]
# ## 1. Open and verify the persistent vector database

# %%
database_directory = Path("data/indexes/multimodal").resolve()
stores = open_multimodal_stores(database_directory)

print(f"Article vectors: {stores.article_count}")
print(f"Image vectors:   {stores.image_count}")

# %% [markdown]
# ## 2. Initialize the matching embedding models

# %%
text_embedder = SentenceTransformerEmbedder(device="cpu")
image_embedder = CLIPEmbedder(device="cpu")

print("Text embedder ready: 384-D normalized vectors")
print("Image embedder ready: 512-D normalized vectors")

# %% [markdown]
# ## Demo 1 — Article similarity search without a filter

# %%
demo_1_hits = retrieve_articles(
    stores.articles,
    text_embedder,
    "cozy restaurant with noodles and warm atmosphere",
    k=5,
)
print(
    format_hits(
        demo_1_hits,
        title="Demo 1 — Article similarity search (no filter)",
    )
)
print("Demo 1 complete")

# %% [markdown]
# ## Demo 2 — Article similarity search with metadata filtering
#
# Pasadena is preferred when present. If the current dataset has no Pasadena
# record, the helper selects the first real location value instead of running a
# misleading filter that can never match.

# %%
location_filter = existing_metadata_filter(
    stores.articles,
    "location",
    preferred_value="Pasadena",
)
print(f"Using article filter: {location_filter}")

demo_2_hits = retrieve_articles(
    stores.articles,
    text_embedder,
    "handmade pasta and romantic dinner",
    k=5,
    where=location_filter,
)
print(
    format_hits(
        demo_2_hits,
        title="Demo 2 — Article similarity search + metadata filter",
    )
)
print("Demo 2 complete")

# %% [markdown]
# ## Demo 3 — Image similarity search (image to image)
#
# Change `QUERY_INDEX` to explore a different indexed image. The optional
# cuisine filter uses the selected image's own cuisine, guaranteeing that the
# constraint exists in the collection.

# %%
QUERY_INDEX = 0
query_image, query_metadata = image_record_at_index(
    stores.images,
    QUERY_INDEX,
)
print(f"Query image: {query_image}")
show_thumbnail(query_image)

image_filter = None
if "cuisine" in query_metadata:
    image_filter = {"cuisine": query_metadata["cuisine"]}
print(f"Using image filter: {image_filter}")

demo_3_hits = retrieve_images_by_image(
    stores.images,
    image_embedder,
    query_image,
    k=5,
    where=image_filter,
)
print(
    format_hits(
        demo_3_hits,
        title="Demo 3 — Image similarity search (image→image)",
    )
)

for hit in demo_3_hits:
    result_path = hit.metadata.get("image_path")
    if isinstance(result_path, str) and Path(result_path).is_file():
        show_thumbnail(Path(result_path), size=(220, 220))

print("Demo 3 complete")
print("Similarity Retrieval with Metadata Filtering COMPLETE")
