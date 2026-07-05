# %% [markdown]
# # Lab 02 — Multimodal food-data augmentation
#
# ## What this lab does
#
# This lab turns images into searchable text for a future multimodal RAG
# system. It enriches two datasets:
#
# 1. Recipe records receive a visual description of the finished dish.
# 2. Restaurant reviews receive captions for their attached images. The
#    written review is supplied to the model as context, helping it connect
#    visible details with the user's experience.
#
# A vision-language model hosted on IBM watsonx.ai reads each image. The final
# JSON keeps the original fields and adds `image_description` for recipes or
# `image_captions` for reviews.
#
# ## Reliability and cost notes
#
# - The watsonx.ai model is created once and reused for every image.
# - Review-image downloads use bounded exponential retries.
# - ZIP extraction rejects unsafe paths and symbolic links.
# - Image MIME types are detected rather than hard-coded as JPEG.
# - The live review dataset uses the field `text` (not `review`).
# - The recipe archive maps recipe ID `n` to `recipe<n>.png`.
# - The 215 MB image archive is downloaded only if it is missing.
# - The default preview processes three records. Set `LAB02_LIMIT=0` to run all
#   records, which makes many model calls and can consume inference credits.
#
# Install the lab dependencies from the repository root with:
#
#     pip install -e ".[labs]"
#
# Then configure `WATSONX_APIKEY` and `WATSONX_PROJECT_ID`, unless this runs in
# the IBM Skills Network environment.

# %%
import os
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from ibm_rag_agentic_showcase.multimodal_augmentation import (
    RECIPE_IMAGES_URL,
    RECIPES_URL,
    REVIEWS_URL,
    VisionSettings,
    augment_recipes,
    augment_reviews,
    create_vision_llm,
    download_file,
    download_image_with_retry,
    load_json_records,
    parse_image_urls,
    recipe_caption_prompt,
    recipe_image_path,
    safe_extract_zip,
    save_json_records,
)

# %% [markdown]
# ## 1. Download and load the course assets
#
# Downloads are cached under `data/raw/lab02`, so rerunning this cell does not
# fetch the same files again.

# %%
raw_directory = Path("data/raw/lab02")
processed_directory = Path("data/processed/lab02")

recipes_path = download_file(RECIPES_URL, raw_directory / "Recipes.json")
reviews_path = download_file(
    REVIEWS_URL,
    raw_directory / "Synthetic-User-Reviews.json",
)
images_zip_path = download_file(
    RECIPE_IMAGES_URL,
    raw_directory / "synthetic-recipe-images.zip",
)

image_extract_directory = raw_directory / "images"
image_directory = image_extract_directory / "synthetic_recipe_images"
if not image_directory.exists():
    safe_extract_zip(images_zip_path, image_extract_directory)

recipes = load_json_records(recipes_path)
reviews = load_json_records(reviews_path)
print(f"Loaded {len(recipes)} recipes and {len(reviews)} reviews")

# %% [markdown]
# ## 2. Inspect one recipe and its image
#
# The JSON currently contains recipe metadata but no `image_path`. The helper
# derives the correct path from the archive's `recipe<ID>.png` convention.

# %%
first_recipe = recipes[0]
for key, value in first_recipe.items():
    print(f"{key} ({type(value).__name__}): {value}")

first_recipe_image = recipe_image_path(first_recipe, image_directory)
with Image.open(first_recipe_image) as image:
    plt.imshow(image)
    plt.title(first_recipe["name"])
    plt.axis("off")
    plt.show()

# %% [markdown]
# ## 3. Caption one recipe image

# %%
vision_llm = create_vision_llm(VisionSettings.from_env())
system_message, prompt = recipe_caption_prompt(first_recipe["name"])
first_caption = vision_llm(
    system_message,
    prompt,
    first_recipe_image.read_bytes(),
    "image/png",
)
print(first_caption)

# %% [markdown]
# ## 4. Enrich recipe records
#
# Three records are processed by default. Use `LAB02_LIMIT=0` for the complete
# dataset after checking model access and expected inference cost.

# %%
limit = int(os.getenv("LAB02_LIMIT", "3"))
selected_recipes = recipes if limit == 0 else recipes[:limit]

augmented_recipes = augment_recipes(
    selected_recipes,
    image_directory,
    vision_llm,
    progress=lambda completed, total: print(f"Recipes: {completed}/{total}"),
)
recipe_output_path = processed_directory / "augmented_food_recipe.json"
save_json_records(augmented_recipes, recipe_output_path)
print(f"Saved recipe data to {recipe_output_path}")

# %% [markdown]
# ## 5. Inspect a review image
#
# In the course data, `images` is a string representation of a Python list.
# `parse_image_urls` converts it into a validated list without using `eval`.

# %%
first_review = reviews[0]
for key, value in first_review.items():
    print(f"{key} ({type(value).__name__}): {value}")

first_review_image_url = parse_image_urls(first_review["images"])[0]
review_image_bytes, _ = download_image_with_retry(first_review_image_url)
with Image.open(BytesIO(review_image_bytes)) as image:
    plt.imshow(image)
    plt.title(first_review["title"])
    plt.axis("off")
    plt.show()

# %% [markdown]
# ## 6. Enrich review records
#
# Each image caption is grounded in the review's `text` field. An unavailable
# image is skipped after bounded retries rather than stopping the whole batch.

# %%
selected_reviews = reviews if limit == 0 else reviews[:limit]

augmented_reviews = augment_reviews(
    selected_reviews,
    vision_llm,
    progress=lambda completed, total: print(f"Reviews: {completed}/{total}"),
)
review_output_path = processed_directory / "augmented_user_review.json"
save_json_records(augmented_reviews, review_output_path)
print(f"Saved review data to {review_output_path}")
