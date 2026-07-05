# %% [markdown]
# # Lab 03 — Safe restaurant database
#
# ## What this lab does
#
# Lab 03 turns the structured JSON created in Lab 01 into a small interactive
# application. The terminal menu supports:
#
# 1. Browsing restaurant names.
# 2. Viewing a complete restaurant record.
# 3. Adding a restaurant from a natural-language paragraph through IBM Granite.
# 4. Editing fields while preserving their Pydantic types.
# 5. Deleting a record after explicit confirmation.
#
# Every write creates a backup of the previous JSON file. The replacement file
# is written atomically, and the complete record is validated before it reaches
# disk.
#
# ## Improvements over the initial course draft
#
# - New IDs use the highest existing ID plus one, so deleting a record cannot
#   cause a duplicate ID later.
# - Rating, price range, list fields, and null values are parsed into their
#   proper types instead of being stored as arbitrary strings.
# - New restaurant extraction reuses Lab 01's bounded repair pipeline.
# - Browse-only actions do not initialize watsonx.ai or require credentials.
# - Menu indexes are one-based and invalid input is reported without crashing.
# - Storage and UI behavior are separated, making offline unit tests possible.
#
# Run the installed command from the repository root:
#
#     restaurant-db --file data/processed/structured_restaurant_data.json
#
# Adding a record requires `WATSONX_APIKEY` and `WATSONX_PROJECT_ID` outside the
# IBM Skills Network environment. Browsing, editing, and deleting local records
# do not call the model.

# %%
from pathlib import Path

from ibm_rag_agentic_showcase.restaurant_database import (
    RestaurantDatabase,
    create_default_extractor,
    run_console,
)
from ibm_rag_agentic_showcase.restaurant_extraction import Restaurant

# %% [markdown]
# ## Configure the database
#
# The extractor is created lazily: opening the menu or browsing data does not
# contact watsonx.ai. It is initialized only if the user chooses "Add".

# %%
restaurant_path = Path("data/processed/structured_restaurant_data.json")
extractor = None


def lazy_extract(description: str) -> Restaurant:
    global extractor
    if extractor is None:
        extractor = create_default_extractor()
    return extractor(description)


database = RestaurantDatabase(
    restaurant_path,
    backup_path=restaurant_path.with_suffix(".json.bak"),
    extractor=lazy_extract,
)

# %% [markdown]
# ## Start the terminal menu
#
# Comment out this call when importing the file only to inspect its cells.

# %%
run_console(database)
