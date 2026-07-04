# %% [markdown]
# # Lab 04 — Construct a multimodal vector index
#
# ## What this lab does
#
# This lab converts the structured and visually enriched data from earlier labs
# into two persistent semantic-search indexes:
#
# 1. `restaurant_articles` stores 384-dimensional Sentence-Transformers
#    embeddings of restaurant descriptions.
# 2. `food_images` stores 512-dimensional CLIP embeddings of recipe images.
#
# Every vector is L2-normalized before it enters Chroma. The two collections
# stay separate because their dimensions and embedding spaces are incompatible.
# LangChain `Document` objects carry searchable text plus metadata such as
# cuisine, location, image path, and source.
#
# CLIP embeds both images and text into one shared 512-D space. That allows a
# phrase such as "colorful vegetarian bowl" to retrieve visually similar recipe
# images even though the collection itself was built from pixels.
#
# ## Prerequisites
#
# Complete Labs 01 and 02 first so these paths exist:
#
# - `data/processed/structured_restaurant_data.json`
# - `data/processed/lab02/augmented_food_recipe.json`
# - `data/raw/lab02/images/synthetic_recipe_images/`
#
# On Linux, install CPU-only PyTorch before the vector dependencies:
#
#     pip install torch --index-url https://download.pytorch.org/whl/cpu
#     pip install -e ".[vector]"
#
# On macOS:
#
#     pip install -e ".[vector]"
#
# Model files are downloaded on first use. `LAB04_LIMIT=10` is the default
# preview; set `LAB04_LIMIT=0` to build the complete index.

# %%
import os
from pathlib import Path

import chromadb
import langchain_core
import numpy as np
import PIL
import sentence_transformers
import torch
import transformers

from ibm_rag_agentic_showcase.multimodal_augmentation import load_json_records
from ibm_rag_agentic_showcase.multimodal_vector_index import (
    IMAGE_COLLECTION,
    RESTAURANT_COLLECTION,
    CLIPEmbedder,
    SentenceTransformerEmbedder,
    build_recipe_image_documents,
    build_restaurant_documents,
    construct_multimodal_index,
    query_collection,
)

# %% [markdown]
# ## 1. Verify the dependency environment
#
# This prints the principal package versions and confirms that model execution
# is pinned to CPU for a reproducible course environment.

# %%
print(f"PyTorch: {torch.__version__}")
print(f"LangChain Core: {langchain_core.__version__}")
print(f"Chroma: {chromadb.__version__}")
print(f"Sentence-Transformers: {sentence_transformers.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Pillow: {PIL.__version__}")
print("Embedding device: CPU")

# %% [markdown]
# ## 2. Load and verify the source records

# %%
restaurant_path = Path("data/processed/structured_restaurant_data.json")
recipe_path = Path("data/processed/lab02/augmented_food_recipe.json")
image_directory = Path("data/raw/lab02/images/synthetic_recipe_images")

restaurant_records = load_json_records(restaurant_path)
recipe_records = load_json_records(recipe_path)
print(f"Restaurant records: {len(restaurant_records)}")
print(f"Recipe records: {len(recipe_records)}")

limit = int(os.getenv("LAB04_LIMIT", "10"))
if limit > 0:
    restaurant_records = restaurant_records[:limit]
    recipe_records = recipe_records[:limit]
    print(f"Preview mode: indexing at most {limit} records per collection")

# %% [markdown]
# ## 3. Build LangChain documents with retrieval metadata
#
# Restaurant documents contain the complete structured article and metadata
# including location and source. Recipe-image documents contain recipe context
# plus cuisine, image path, and source metadata.

# %%
restaurant_documents = build_restaurant_documents(restaurant_records)
image_documents = build_recipe_image_documents(
    recipe_records,
    image_directory,
)

print(restaurant_documents[0])
print(image_documents[0])

# %% [markdown]
# ## 4. Initialize and verify both embedding models
#
# `all-MiniLM-L6-v2` produces normalized 384-D vectors for restaurant text.
# `clip-vit-base-patch32` produces normalized 512-D vectors for recipe images
# and cross-modal text queries.

# %%
text_embedder = SentenceTransformerEmbedder(device="cpu")
clip_embedder = CLIPEmbedder(device="cpu")

assert text_embedder.dimension == 384
assert clip_embedder.dimension == 512
print("Text vectors: 384-D, L2 normalization enabled")
print("Image vectors: 512-D, L2 normalization enabled")

# %% [markdown]
# ## 5. Create two persistent Chroma collections
#
# Existing Lab 04 collections are rebuilt by default so their counts exactly
# match the current inputs. Other Chroma collections in the directory remain
# untouched.

# %%
persist_directory = Path("data/indexes/multimodal")
summary = construct_multimodal_index(
    restaurant_documents,
    image_documents,
    text_embedder,
    clip_embedder,
    persist_directory,
    reset=True,
)

print(summary)
assert summary.restaurant_count == len(restaurant_documents)
assert summary.image_count == len(image_documents)

# %% [markdown]
# ## 6. Verify retrieval
#
# The restaurant query uses the Sentence-Transformers space. The food-image
# query uses CLIP's text encoder, so text can retrieve indexed images.

# %%
restaurant_query = text_embedder.embed_texts(
    ["cozy seafood restaurant near the coast"]
)[0]
restaurant_results = query_collection(
    persist_directory,
    RESTAURANT_COLLECTION,
    restaurant_query,
)
print(restaurant_results["metadatas"][0])

image_query = clip_embedder.embed_texts(
    ["a colorful vegetarian dish with fresh vegetables"]
)[0]
image_results = query_collection(
    persist_directory,
    IMAGE_COLLECTION,
    image_query,
)
print(image_results["metadatas"][0])
