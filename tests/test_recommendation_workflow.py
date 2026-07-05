from typing import Any

from ibm_rag_agentic_showcase.recommendation_workflow import (
    WorkflowServices,
    build_recommendation_workflow,
    evaluate_recommendations,
    initial_state,
    profile_to_query,
)


class FakeAgentCaller:
    def __init__(self):
        self.calls = []
        self.style_prompt = ""

    def __call__(self, agent_id: str, user_message: str) -> dict[str, Any]:
        self.calls.append(agent_id)
        if agent_id == "user_profile_generator":
            return {
                "favorite_cuisines": ["Mediterranean"],
                "dietary_restrictions": ["vegan"],
                "dining_occasions": ["lunch"],
                "price_range": "$$",
                "adventurousness_score": 7,
                "flavor_preferences": ["fresh herbs"],
                "summary": "Plant-forward Mediterranean diner.",
            }
        if agent_id == "food_trend_analyst":
            return {"trends": [{"name": "plant-forward"}]}
        if agent_id == "food_style_expert":
            self.style_prompt = user_message
            return {
                "matches": [{"name": "Cafe"}],
                "mismatches": [],
                "flavor_summary": "Fresh and herb-forward.",
            }
        if agent_id == "nutrition_expert":
            return {
                "compliant_items": ["Cafe", "Bowl"],
                "flagged_items": [],
                "nutritional_highlights": ["fiber"],
            }
        if agent_id == "recommendation_expert":
            return {
                "restaurants": [
                    {
                        "name": "Cafe",
                        "reasoning": "Matches the profile.",
                        "cuisine": "Mediterranean",
                        "dietary_notes": ["vegan"],
                        "source_ids": ["restaurant-1"],
                    }
                ],
                "recipes": [
                    {
                        "name": "Bowl",
                        "reasoning": "Matches the profile.",
                        "cuisine": "Mediterranean",
                        "dietary_notes": ["vegan"],
                        "source_ids": ["recipe-1"],
                    }
                ],
            }
        raise KeyError(agent_id)


def fake_retriever(_profile):
    return (
        [{"id": "restaurant-1", "name": "Cafe"}],
        [{"id": "recipe-1", "name": "Bowl"}],
    )


def test_hybrid_workflow_completes_and_merges_parallel_outputs():
    caller = FakeAgentCaller()
    graph = build_recommendation_workflow(WorkflowServices(caller, fake_retriever))

    result = graph.invoke(initial_state("I like vegan Mediterranean food."))

    assert result["workflow_step"] == "complete"
    assert result["trend_analysis"]["trends"]
    assert result["style_analysis"]["matches"]
    assert result["nutrition_analysis"]["compliant_items"]
    assert result["final_recommendations"]["restaurants"]
    assert result["errors"] == []
    assert "User profile:" in caller.style_prompt


def test_stream_exposes_workflow_step_phase_markers():
    graph = build_recommendation_workflow(
        WorkflowServices(FakeAgentCaller(), fake_retriever)
    )

    updates = list(graph.stream(initial_state("vegan"), stream_mode="updates"))
    steps = {
        update["workflow_step"]
        for event in updates
        for update in event.values()
        if "workflow_step" in update
    }

    assert steps == {
        "profile_generated",
        "candidates_retrieved",
        "analysis_complete",
        "complete",
    }


def test_evaluation_reports_complete_explainable_output():
    graph = build_recommendation_workflow(
        WorkflowServices(FakeAgentCaller(), fake_retriever)
    )
    result = graph.invoke(initial_state("vegan"))

    report = evaluate_recommendations(result)

    assert report.restaurant_count == 1
    assert report.recipe_count == 1
    assert report.reasoning_coverage == 1.0
    assert report.dietary_note_coverage == 1.0
    assert report.quality_score == 100.0
    assert report.first_restaurant.startswith("Cafe:")
    assert report.first_recipe.startswith("Bowl:")
    assert report.warnings == ()


def test_agent_error_is_recorded_without_stopping_graph():
    caller = FakeAgentCaller()

    def failing_caller(agent_id: str, message: str):
        if agent_id == "food_trend_analyst":
            raise RuntimeError("trend service unavailable")
        return caller(agent_id, message)

    graph = build_recommendation_workflow(
        WorkflowServices(failing_caller, fake_retriever)
    )
    result = graph.invoke(initial_state("vegan"))

    assert result["workflow_step"] == "complete"
    assert result["trend_analysis"] == {"trends": []}
    assert any("food_trend_analyst" in error for error in result["errors"])


def test_profile_query_contains_retrieval_signals():
    query = profile_to_query(
        {
            "favorite_cuisines": ["Thai"],
            "flavor_preferences": ["spicy"],
            "dietary_restrictions": ["vegan"],
            "summary": "Adventurous diner",
        }
    )

    assert all(term in query for term in ("Thai", "spicy", "vegan"))
