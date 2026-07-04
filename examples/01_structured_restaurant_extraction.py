# %% [markdown]
# # Structured restaurant extraction with IBM Granite
#
# This cell-friendly script preserves the course workflow while importing the
# tested pipeline from the package. Open it in VS Code or a Jupyter environment
# that supports Python percent cells.

# %%
from pathlib import Path

from ibm_rag_agentic_showcase.restaurant_extraction import (
    RestaurantExtractor,
    WatsonxSettings,
    create_watsonx_llm,
    download_dataset,
    load_descriptions,
    save_records,
)

# %% [markdown]
# ## Load the course data

# %%
input_path = download_dataset(Path("data/raw/California-Culinary-Map.txt"))
restaurant_descriptions = load_descriptions(input_path)

print(f"Loaded {len(restaurant_descriptions)} restaurant descriptions")
print(restaurant_descriptions[0][:300])

# %% [markdown]
# ## Configure Granite and test one extraction

# %%
settings = WatsonxSettings.from_env()
extractor = RestaurantExtractor(
    create_watsonx_llm(settings),
    max_repair_attempts=3,
)

first_restaurant = extractor.extract(restaurant_descriptions[0])
print(first_restaurant.model_dump_json(indent=2))

# %% [markdown]
# ## Process and save the full collection
#
# Each response is validated before it enters the output dataset. Invalid
# responses receive a targeted repair prompt, up to the configured limit.

# %%
structured_restaurants = extractor.extract_many(
    restaurant_descriptions,
    progress=lambda completed: print(
        f"Processed {completed}/{len(restaurant_descriptions)}"
    ),
)

output_path = Path("data/processed/structured_restaurant_data.json")
save_records(structured_restaurants, output_path)
print(f"Saved {len(structured_restaurants)} records to {output_path}")
