import json

import pytest

from ibm_rag_agentic_showcase.chatbot_interface import (
    CatalogStore,
    LangChainMessageInterpreter,
    RecommendationChatService,
    RuleBasedInterpreter,
    build_gradio_app,
    classify_intent,
    echo_chatbot,
    extract_preferences,
    format_recommendations,
    new_session_state,
)


class FakeWorkflow:
    def __init__(self) -> None:
        self.inputs = []

    def invoke(self, state):
        self.inputs.append(state["user_input"])
        return {
            "final_recommendations": {
                "restaurants": [
                    {
                        "name": "Green Table",
                        "cuisine": "Mediterranean",
                        "reasoning": "Plant-forward food suits the stated preferences.",
                        "dietary_notes": ["vegetarian options"],
                    }
                ],
                "recipes": [
                    {
                        "name": "Thai Basil Tofu",
                        "cuisine": "Thai",
                        "reasoning": "It is spicy, quick, and vegetarian.",
                        "dietary_notes": ["vegetarian"],
                    }
                ],
            },
            "errors": [],
        }


class FakeMessage:
    def __init__(self, content):
        self.content = content


class SequenceModel:
    def __init__(self, responses):
        self.responses = iter(responses)

    def invoke(self, _messages):
        return FakeMessage(next(self.responses))


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Find an Italian restaurant", "restaurant"),
        ("How do I make pad thai?", "recipe"),
        ("Give me dinner ideas", "both"),
        ("Please delete a database entry", "database"),
        ("What can you help with?", "clarification"),
    ],
)
def test_rule_based_intent_classification(message, expected):
    assert classify_intent(message) == expected


def test_preference_extraction_completes_assignment_cell():
    message = "I love spicy Thai food, I am vegetarian, and want a date night under $$"

    preferences = extract_preferences(message)

    assert preferences["favorite_cuisines"] == ["Thai"]
    assert preferences["dietary_restrictions"] == ["vegetarian"]
    assert preferences["dining_occasion"] == "date night"
    assert preferences["price_range"] == "$$"
    assert preferences["flavor_preferences"] == ["spicy"]


def test_langchain_interpreter_validates_intent_and_json():
    model = SequenceModel(
        [
            "recipe",
            json.dumps(
                {
                    "favorite_cuisines": ["Thai"],
                    "dietary_restrictions": ["vegan"],
                    "dining_occasion": "casual",
                    "price_range": "$$",
                    "flavor_preferences": ["spicy"],
                    "other_preferences": "quick",
                }
            ),
        ]
    )
    interpreter = LangChainMessageInterpreter(model)

    assert interpreter.classify("recipe please") == "recipe"
    assert interpreter.extract("recipe please").favorite_cuisines == ["Thai"]


def test_chat_service_calls_lab08_contract_and_filters_recipe_output():
    workflow = FakeWorkflow()
    service = RecommendationChatService(workflow, RuleBasedInterpreter())

    response, session = service.respond(
        "How do I make a spicy vegetarian Thai dinner?", new_session_state()
    )

    assert "Thai Basil Tofu" in response
    assert "Green Table" not in response
    assert session["preferences"]["favorite_cuisines"] == ["Thai"]
    assert session["preferences"]["dietary_restrictions"] == ["vegetarian"]
    assert "Known session preferences" in workflow.inputs[0]


def test_chat_service_retains_preferences_across_turns():
    workflow = FakeWorkflow()
    service = RecommendationChatService(workflow)

    _, session = service.respond("Find a vegetarian Italian restaurant")
    _, session = service.respond("Find a spicy Thai restaurant", session)

    assert session["preferences"]["favorite_cuisines"] == ["Italian", "Thai"]
    assert session["preferences"]["dietary_restrictions"] == ["vegetarian"]


def test_clarification_does_not_invoke_workflow():
    workflow = FakeWorkflow()
    service = RecommendationChatService(workflow)

    response, _ = service.respond("What can you do?")

    assert "restaurants" in response
    assert workflow.inputs == []


def test_recommendation_formatter_handles_empty_results():
    assert "could not find" in format_recommendations({}, "both")


def test_echo_chatbot():
    assert echo_chatbot("Hello!", []) == "You said: Hello!"


def test_catalog_store_add_update_and_confirmed_delete(tmp_path):
    store = CatalogStore(tmp_path / "catalog.json")
    restaurant = store.add_restaurant(
        "Green Table",
        "Mediterranean",
        "$$",
        "Pasadena",
        "Plant-forward neighborhood restaurant.",
    )
    recipe = store.add_recipe(
        "Thai Basil Tofu",
        "Thai",
        "Easy",
        "30 minutes",
        "tofu\nbasil",
        "Fry tofu\nAdd basil",
    )

    updated = store.update_description(
        "restaurants",
        restaurant.id,
        "Seasonal plant-forward menu.",
        confirmed=True,
    )

    assert updated["description"] == "Seasonal plant-forward menu."
    assert store.all()["recipes"][0]["ingredients"] == ["tofu", "basil"]
    with pytest.raises(PermissionError):
        store.delete("recipes", recipe.id, confirmed=False)
    deleted = store.delete("recipes", recipe.id, confirmed=True)
    assert deleted["id"] == recipe.id
    assert store.all()["recipes"] == []


def test_gradio_app_builds_chat_state_and_catalog_tabs(tmp_path):
    app = build_gradio_app(
        RecommendationChatService(FakeWorkflow()),
        CatalogStore(tmp_path / "catalog.json"),
    )
    component_types = {
        component["type"] for component in app.config.get("components", [])
    }

    assert {"chatbot", "state", "tabs"}.issubset(component_types)
    assert len(app.config.get("dependencies", [])) >= 8
