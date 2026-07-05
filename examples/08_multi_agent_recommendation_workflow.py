# %% [markdown]
# # Lab 08 — Implement and test a multi-agent recommendation workflow
#
# ## Architecture
#
# The workflow combines sequential and parallel phases:
#
# 1. User Profile Generator — sequential.
# 2. RAG Retriever — sequential and grounded in the vector database.
# 3. Trend, Style, and Nutrition specialists — parallel LangGraph branches.
# 4. Recommendation Expert — sequential synthesis after all branches finish.
#
# Nodes return partial state updates rather than mutating one shared dictionary.
# `workflow_step` records the last completed phase for progress UI, debugging,
# routing, and future checkpoint/resume support.
#
# Install current dependencies with:
#
#     pip install -e ".[agents,openai,dev]"
#
# This example uses deterministic local fakes so it runs without an API key.
# Production setup with `ChatOpenAI` is shown at the bottom.

# %%
from typing import Any

from ibm_rag_agentic_showcase.recommendation_workflow import (
    WorkflowServices,
    build_recommendation_workflow,
    evaluate_recommendations,
    format_evaluation,
    initial_state,
)


class DemoAgentCaller:
    """Deterministic specialist responses for a zero-cost workflow test."""

    def __init__(self) -> None:
        self.adventurous = False

    def __call__(self, agent_id: str, user_message: str) -> dict[str, Any]:
        if agent_id == "user_profile_generator":
            self.adventurous = "omakase" in user_message.lower()
            if self.adventurous:
                return {
                    "favorite_cuisines": ["Japanese", "global fusion"],
                    "dietary_restrictions": [],
                    "dining_occasions": ["tasting menu", "food exploration"],
                    "price_range": "$$$$",
                    "adventurousness_score": 10,
                    "flavor_preferences": ["umami", "experimental"],
                    "summary": "Adventurous diner seeking novel techniques.",
                }
            return {
                "favorite_cuisines": ["Mediterranean"],
                "dietary_restrictions": ["vegan", "gluten-free"],
                "dining_occasions": ["casual lunch"],
                "price_range": "$$",
                "adventurousness_score": 6,
                "flavor_preferences": ["fresh", "herb-forward"],
                "summary": "Plant-based diner who prefers fresh Mediterranean food.",
            }
        if agent_id == "food_trend_analyst":
            return {
                "trends": [
                    {
                        "name": "plant-forward dining",
                        "description": "Vegetables lead the plate.",
                        "relevance": "high",
                        "confidence": "high",
                    }
                ]
            }
        if agent_id == "food_style_expert":
            assert "User profile:" in user_message
            return {
                "matches": [{"name": "Herb Garden Cafe"}],
                "mismatches": [],
                "flavor_summary": "Fresh herbs and bright acidity fit the profile.",
            }
        if agent_id == "nutrition_expert":
            return {
                "compliant_items": ["Herb Garden Cafe", "Quinoa Mezze Bowl"],
                "flagged_items": [],
                "nutritional_highlights": ["fiber-rich", "plant protein"],
            }
        if agent_id == "recommendation_expert":
            if self.adventurous:
                return {
                    "restaurants": [
                        {
                            "name": "Omakase Lab",
                            "reasoning": (
                                "Its chef-led tasting format and experimental "
                                "Japanese techniques fit a highly adventurous diner."
                            ),
                            "cuisine": "Japanese",
                            "dietary_notes": ["confirm allergens course by course"],
                            "source_ids": ["restaurant-2"],
                        }
                    ],
                    "recipes": [
                        {
                            "name": "Molecular Ramen",
                            "reasoning": (
                                "It combines familiar umami flavors with novel "
                                "presentation and modern technique."
                            ),
                            "cuisine": "Japanese fusion",
                            "dietary_notes": ["review sodium and allergens"],
                            "source_ids": ["recipe-2"],
                        }
                    ],
                }
            return {
                "restaurants": [
                    {
                        "name": "Herb Garden Cafe",
                        "reasoning": (
                            "Its Mediterranean menu and fresh herb profile match "
                            "the user while supporting vegan dining."
                        ),
                        "cuisine": "Mediterranean",
                        "dietary_notes": ["confirm gluten-free preparation"],
                        "source_ids": ["restaurant-1"],
                    }
                ],
                "recipes": [
                    {
                        "name": "Quinoa Mezze Bowl",
                        "reasoning": (
                            "It offers plant protein, bright flavors, and an "
                            "easy gluten-free preparation."
                        ),
                        "cuisine": "Mediterranean",
                        "dietary_notes": ["vegan", "gluten-free"],
                        "source_ids": ["recipe-1"],
                    }
                ],
            }
        raise KeyError(agent_id)


def demo_retriever(profile):
    """Stand-in for Lab 04 retrieval; production uses MultimodalCandidateRetriever."""

    if profile.get("adventurousness_score", 0) >= 9:
        return (
            [
                {
                    "id": "restaurant-2",
                    "name": "Omakase Lab",
                    "cuisine": "Japanese",
                    "description": "Experimental chef-led tasting menu.",
                }
            ],
            [
                {
                    "id": "recipe-2",
                    "name": "Molecular Ramen",
                    "cuisine": "Japanese fusion",
                    "description": "Ramen flavors with modernist presentation.",
                }
            ],
        )
    return (
        [
            {
                "id": "restaurant-1",
                "name": "Herb Garden Cafe",
                "cuisine": "Mediterranean",
                "description": "Plant-forward bowls and herb sauces.",
            }
        ],
        [
            {
                "id": "recipe-1",
                "name": "Quinoa Mezze Bowl",
                "cuisine": "Mediterranean",
                "description": "Quinoa, chickpeas, herbs, and lemon.",
            }
        ],
    )


# %% [markdown]
# ## Run test case 1 — Health-conscious diner

# %%
workflow = build_recommendation_workflow(
    WorkflowServices(
        agent_caller=DemoAgentCaller(),
        candidate_retriever=demo_retriever,
    )
)

test_user_1 = """
Restaurant visits: Green Bowl eight times; Mediterranean Grill five times.
Posts: Loving my plant-based journey and gluten-free Mediterranean bowls.
Dietary restrictions: vegan and gluten-free.
""".strip()

result_1 = workflow.invoke(initial_state(test_user_1))
print(result_1["final_recommendations"])
print(format_evaluation(evaluate_recommendations(result_1)))

# %% [markdown]
# ## Inspect phase updates
#
# Streaming updates exposes `workflow_step` transitions and shows the three
# analysis branches completing before synthesis.

# %%
for update in workflow.stream(initial_state(test_user_1), stream_mode="updates"):
    print(update)

# %% [markdown]
# ## Run test case 2 — Adventurous foodie

# %%
test_user_2 = """
Restaurant visits: Omakase Sushi four times; international street-food market
six times; Molecular Gastronomy Lab twice.
Posts: The 12-course tasting menu was mind-blowing. I enjoy trying unfamiliar
ingredients and experimental takes on traditional ramen.
Dietary restrictions: none.
""".strip()

workflow_2 = build_recommendation_workflow(
    WorkflowServices(
        agent_caller=DemoAgentCaller(),
        candidate_retriever=demo_retriever,
    )
)
result_2 = workflow_2.invoke(initial_state(test_user_2))
print(result_2["final_recommendations"])
print(format_evaluation(evaluate_recommendations(result_2)))

# %% [markdown]
# ## Production model configuration
#
# Uncomment after setting `OPENAI_API_KEY`. The multimodal candidate retriever
# is created from the Lab 04 stores and embedding models.
#
# ```python
# from langchain_openai import ChatOpenAI
# from ibm_rag_agentic_showcase.recommendation_workflow import (
#     LangChainJSONAgentCaller,
# )
#
# model = ChatOpenAI(model="gpt-5", temperature=0.2)
# caller = LangChainJSONAgentCaller(model)
# ```
