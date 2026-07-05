# %% [markdown]
# # Lab 09 — Build a chatbot interface for the recommendation system
#
# This lab turns the Lab 08 multi-agent workflow into an interactive Gradio app.
# It covers:
#
# 1. a basic echo chatbot;
# 2. intent classification;
# 3. structured preference extraction;
# 4. restaurant and recipe recommendation requests;
# 5. per-browser session memory with `gr.State`; and
# 6. validated add, update, and delete forms for contributed catalog records.
#
# Install the compatible dependency set from the repository:
#
#     pip install -e ".[agents,openai,ui,dev]"
#
# This replaces the assignment's separate old pins, which would downgrade the
# modern LangChain/LangGraph environment used by Lab 08.

# %%
import sys
from pathlib import Path
from typing import Any

from ibm_rag_agentic_showcase.chatbot_interface import (
    CatalogStore,
    RecommendationChatService,
    build_gradio_app,
    classify_intent,
    echo_chatbot,
    extract_preferences,
)
from ibm_rag_agentic_showcase.recommendation_workflow import (
    WorkflowServices,
    build_recommendation_workflow,
)

# %% [markdown]
# ## Task 1 — Build and test a basic echo chatbot
#
# Gradio chat callbacks receive the new message and prior message history. The
# smallest useful callback simply returns the input.

# %%
print(echo_chatbot("Hello!", []))


# %% [markdown]
# ## Classify intent
#
# The local interpreter supports the five assignment labels: `restaurant`,
# `recipe`, `both`, `clarification`, and `database`. In production,
# `LangChainMessageInterpreter` can accept a `ChatOpenAI` model instead.

# %%
for test_message in (
    "I'm looking for Italian restaurants",
    "How do I make pad thai?",
    "Give me dinner ideas",
    "What can you do?",
    "Delete a database entry",
):
    print(f"{test_message!r} -> {classify_intent(test_message)}")


# %% [markdown]
# ## Extract preferences
#
# This is the completed version of the assignment's first blank cell.

# %%
test_message = (
    "I love spicy Thai food, I am vegetarian, and want a date night under $$"
)
preferences = extract_preferences(test_message)
print("Extracted preferences:")
for key, value in preferences.items():
    print(f"  {key}: {value}")


# %% [markdown]
# ## Connect the Lab 08 workflow
#
# The deterministic specialist below keeps this example free to run. It is
# passed into the real LangGraph builder, so the UI exercises the same
# sequential → parallel → synthesis architecture as Lab 08. A production app
# replaces it with `LangChainJSONAgentCaller(ChatOpenAI(...))` and the real
# multimodal retriever.


# %%
class DemoAgentCaller:
    """Predictable specialist output for a local Gradio demonstration."""

    def __init__(self) -> None:
        self.user_text = ""

    def __call__(self, agent_id: str, user_message: str) -> dict[str, Any]:
        lowered = user_message.casefold()
        if agent_id == "user_profile_generator":
            self.user_text = lowered
            cuisines = [
                cuisine
                for cuisine in ("Thai", "Italian", "Mediterranean")
                if cuisine.casefold() in lowered
            ]
            restrictions = [
                restriction
                for restriction in ("vegetarian", "vegan", "gluten-free")
                if restriction in lowered
            ]
            return {
                "favorite_cuisines": cuisines,
                "dietary_restrictions": restrictions,
                "dining_occasions": (
                    ["date night"] if "date night" in lowered else ["dinner"]
                ),
                "price_range": "$$" if "$$" in user_message else None,
                "adventurousness_score": 7,
                "flavor_preferences": ["spicy"] if "spicy" in lowered else [],
                "summary": "A diner seeking recommendations from the chat request.",
            }
        if agent_id == "food_trend_analyst":
            return {
                "trends": [
                    {
                        "name": "plant-forward menus",
                        "description": "Vegetables lead flexible modern menus.",
                        "relevance": "high",
                        "confidence": "high",
                    }
                ]
            }
        if agent_id == "food_style_expert":
            return {
                "matches": [
                    {"name": "Saffron Garden"},
                    {"name": "Spicy Thai Basil Tofu"},
                ],
                "mismatches": [],
                "flavor_summary": "Bright herbs and adjustable heat match the request.",
            }
        if agent_id == "nutrition_expert":
            return {
                "compliant_items": ["Saffron Garden", "Spicy Thai Basil Tofu"],
                "flagged_items": [],
                "nutritional_highlights": ["vegetable-rich", "plant protein"],
            }
        if agent_id == "recommendation_expert":
            return {
                "restaurants": [
                    {
                        "name": "Saffron Garden",
                        "reasoning": (
                            "Its plant-forward Mediterranean menu suits a relaxed "
                            "date night and offers clearly labeled vegetarian dishes."
                        ),
                        "cuisine": "Mediterranean",
                        "dietary_notes": ["confirm cross-contact with the restaurant"],
                        "source_ids": ["restaurant-demo-1"],
                    }
                ],
                "recipes": [
                    {
                        "name": "Spicy Thai Basil Tofu",
                        "reasoning": (
                            "This quick recipe combines Thai basil, adjustable heat, "
                            "and tofu for a satisfying vegetarian dinner."
                        ),
                        "cuisine": "Thai",
                        "dietary_notes": ["vegetarian", "check sauce ingredients"],
                        "source_ids": ["recipe-demo-1"],
                    }
                ],
            }
        raise KeyError(agent_id)


def demo_retriever(_profile):
    """Return grounded-looking local records in the Lab 08 retriever format."""

    return (
        [
            {
                "id": "restaurant-demo-1",
                "name": "Saffron Garden",
                "cuisine": "Mediterranean",
                "description": "Plant-forward neighborhood dining.",
            }
        ],
        [
            {
                "id": "recipe-demo-1",
                "name": "Spicy Thai Basil Tofu",
                "cuisine": "Thai",
                "description": "Tofu, Thai basil, vegetables, and chili.",
            }
        ],
    )


workflow = build_recommendation_workflow(
    WorkflowServices(
        agent_caller=DemoAgentCaller(),
        candidate_retriever=demo_retriever,
    )
)
chat_service = RecommendationChatService(workflow)


# %% [markdown]
# ## Test restaurant and recipe requests
#
# The second assignment blank is completed here with a recipe request.

# %%
restaurant_message = (
    "I'm looking for healthy vegetarian restaurants for a date night"
)
restaurant_response, session = chat_service.respond(restaurant_message)
print("\nRestaurant request:")
print(restaurant_response)

test_recipe_message = "How do I make a spicy vegetarian Thai dinner?"
recipe_response, session = chat_service.respond(test_recipe_message, session)
print("\nRecipe request:")
print(recipe_response)


# %% [markdown]
# ## Build the full Gradio interface
#
# Catalog writes go to a generated data file and use validation plus atomic
# replacement. A catalog change is not silently inserted into Chroma: the UI
# tells the operator to rebuild the multimodal index so embeddings stay
# consistent.

# %%
catalog = CatalogStore(Path("data/processed/chatbot_catalog.json"))
demo = build_gradio_app(chat_service, catalog)
print("\nGradio interface created.")
print("Run this file with --launch to open http://127.0.0.1:7860")


# %%
if __name__ == "__main__" and "--launch" in sys.argv:
    # Local-only by default. Set share=True yourself only when public exposure
    # is intentional and the app has appropriate authentication.
    demo.launch(server_name="127.0.0.1", share=False)
