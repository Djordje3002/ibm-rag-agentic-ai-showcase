# %% [markdown]
# # Lab 06 — Multimodal similarity fusion and retrieval ranking
#
# ## What this lab does
#
# Restaurant articles and recipe images live in different embedding spaces, so
# their raw distances are not directly comparable. This lab:
#
# 1. Retrieves text candidates with Sentence-Transformers.
# 2. Retrieves image candidates with a CLIP text query.
# 3. Converts cosine distance to similarity.
# 4. Min-max normalizes scores separately inside each modality.
# 5. Applies configurable text/image weights.
# 6. Reranks the combined candidate pool by weighted score.
#
# This is weighted candidate-pool fusion, not entity-level feature fusion:
# restaurant and recipe records do not share a common entity ID. Keeping that
# distinction explicit makes the resulting ranking honest and debuggable.
#
# A single-result or constant-score pool receives normalized scores of 1.0.
# Otherwise a strict filter could turn its only valid candidate into zero.

# %%
from pathlib import Path

from ibm_rag_agentic_showcase.multimodal_fusion import (
    format_fused_results,
    fuse_rank,
)
from ibm_rag_agentic_showcase.multimodal_vector_index import (
    CLIPEmbedder,
    SentenceTransformerEmbedder,
)
from ibm_rag_agentic_showcase.similarity_retrieval import (
    existing_metadata_filter,
    open_multimodal_stores,
)

# %% [markdown]
# ## 1. Open the Lab 04 collections and matching embedding models

# %%
stores = open_multimodal_stores(Path("data/indexes/multimodal").resolve())
text_embedder = SentenceTransformerEmbedder(device="cpu")
clip_embedder = CLIPEmbedder(device="cpu")

print(f"Article candidates available: {stores.article_count}")
print(f"Image candidates available:   {stores.image_count}")

# %% [markdown]
# ## Demo 1 — Multimodal fusion without filters

# %%
query = "cozy noodles with a warm restaurant atmosphere"
demo_1 = fuse_rank(
    query,
    stores.articles,
    stores.images,
    text_embedder,
    clip_embedder,
    k_text=5,
    k_image=5,
    text_weight=0.6,
    image_weight=0.4,
    top_n=5,
)
print(
    format_fused_results(
        demo_1,
        title="Demo 1 — Multimodal fusion (no filters)",
    )
)
print("Demo 1 complete")

# %% [markdown]
# ## Demo 2 — Multimodal fusion with metadata filters
#
# The article filter prefers Pasadena but falls back to a location that exists.
# The image filter uses the `recipe_image` source written by Lab 04.

# %%
query = "handmade pasta and romantic dinner"
article_filter = existing_metadata_filter(
    stores.articles,
    "location",
    preferred_value="Pasadena",
)
image_filter = existing_metadata_filter(
    stores.images,
    "source",
    preferred_value="recipe_image",
)
print(f"Article filter: {article_filter}")
print(f"Image filter:   {image_filter}")

demo_2 = fuse_rank(
    query,
    stores.articles,
    stores.images,
    text_embedder,
    clip_embedder,
    k_text=5,
    k_image=5,
    text_weight=0.6,
    image_weight=0.4,
    where_text=article_filter,
    where_image=image_filter,
    top_n=5,
)
print(
    format_fused_results(
        demo_2,
        title="Demo 2 — Multimodal fusion (metadata filters)",
    )
)
print("Demo 2 complete")

# %% [markdown]
# ## Demo 3 — Weight tuning and reranking behavior
#
# Higher text weight favors article candidates. Higher image weight favors
# recipe-image candidates. Equal weights let normalized rank quality dominate.

# %%
weight_settings = [
    (0.8, 0.2, "Article-heavy ranking"),
    (0.5, 0.5, "Balanced ranking"),
    (0.2, 0.8, "Image-heavy ranking"),
]

for text_weight, image_weight, label in weight_settings:
    rows = fuse_rank(
        query,
        stores.articles,
        stores.images,
        text_embedder,
        clip_embedder,
        k_text=5,
        k_image=5,
        text_weight=text_weight,
        image_weight=image_weight,
        top_n=5,
    )
    print(
        format_fused_results(
            rows,
            title=(f"Demo 3 — {label} (text={text_weight}, image={image_weight})"),
        )
    )

print("Demo 3 complete")
print("Multimodal Similarity Fusion and Retrieval Ranking COMPLETE")
