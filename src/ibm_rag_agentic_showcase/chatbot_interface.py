"""Gradio-facing chat service for the recommendation workflow in Lab 09."""

from __future__ import annotations

import json
import re
import threading
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from .recommendation_workflow import initial_state

Intent = Literal["restaurant", "recipe", "both", "clarification", "database"]
CatalogKind = Literal["restaurants", "recipes"]
ChatMessage = dict[str, str]

VALID_INTENTS: tuple[Intent, ...] = (
    "restaurant",
    "recipe",
    "both",
    "clarification",
    "database",
)


class PreferenceProfile(BaseModel):
    """Preferences accumulated during one browser session."""

    favorite_cuisines: list[str] = Field(default_factory=list)
    dietary_restrictions: list[str] = Field(default_factory=list)
    dining_occasion: str = "not specified"
    price_range: str = "not specified"
    flavor_preferences: list[str] = Field(default_factory=list)
    other_preferences: str = ""


class MessageInterpreter(Protocol):
    """Convert natural-language chat messages into structured routing data."""

    def classify(self, message: str) -> Intent:
        """Return one supported intent."""

    def extract(self, message: str) -> PreferenceProfile:
        """Return preferences explicitly present in a message."""


class RecommendationWorkflow(Protocol):
    """The small part of a compiled LangGraph used by the UI."""

    def invoke(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Execute the workflow and return its final state."""


def _unique(values: Sequence[str]) -> list[str]:
    """Deduplicate strings case-insensitively while preserving their order."""

    seen: set[str] = set()
    result = []
    for value in values:
        cleaned = value.strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


class RuleBasedInterpreter:
    """Deterministic fallback used for local demos and offline tests."""

    cuisines = (
        "Italian",
        "Thai",
        "Indian",
        "Japanese",
        "Chinese",
        "Mexican",
        "Mediterranean",
        "French",
        "Korean",
        "Greek",
        "Vietnamese",
    )
    restrictions = (
        "vegan",
        "vegetarian",
        "gluten-free",
        "dairy-free",
        "nut-free",
        "halal",
        "kosher",
    )
    flavors = ("spicy", "sweet", "savory", "sour", "smoky", "fresh", "umami")

    def classify(self, message: str) -> Intent:
        lowered = message.casefold()
        database_words = ("database", "add ", "edit ", "update ", "delete ")
        restaurant_words = (
            "restaurant",
            "where should i eat",
            "place to eat",
            "dining out",
        )
        recipe_words = ("recipe", "cook", "make ", "ingredients")

        if any(word in lowered for word in database_words):
            return "database"

        wants_restaurant = any(word in lowered for word in restaurant_words)
        wants_recipe = any(word in lowered for word in recipe_words)
        if wants_restaurant and wants_recipe:
            return "both"
        if wants_restaurant:
            return "restaurant"
        if wants_recipe:
            return "recipe"
        if any(
            phrase in lowered
            for phrase in ("dinner ideas", "food ideas", "recommend something")
        ):
            return "both"
        return "clarification"

    def extract(self, message: str) -> PreferenceProfile:
        lowered = message.casefold()
        cuisines = [
            cuisine for cuisine in self.cuisines if cuisine.casefold() in lowered
        ]
        restrictions = [
            restriction
            for restriction in self.restrictions
            if restriction in lowered
        ]
        flavors = [flavor for flavor in self.flavors if flavor in lowered]

        occasion = "not specified"
        occasion_patterns = {
            "date night": ("date night", "romantic"),
            "quick bite": ("quick bite", "quick lunch", "in a hurry"),
            "fine dining": ("fine dining", "tasting menu"),
            "casual": ("casual", "laid-back"),
            "family meal": ("family meal", "family dinner"),
        }
        for label, patterns in occasion_patterns.items():
            if any(pattern in lowered for pattern in patterns):
                occasion = label
                break

        price_range = "not specified"
        explicit_price = re.search(r"(?<!\$)(\${1,4})(?!\$)", message)
        if explicit_price:
            price_range = explicit_price.group(1)
        elif any(word in lowered for word in ("cheap", "budget", "affordable")):
            price_range = "$"
        elif any(word in lowered for word in ("luxury", "expensive", "splurge")):
            price_range = "$$$$"

        return PreferenceProfile(
            favorite_cuisines=cuisines,
            dietary_restrictions=restrictions,
            dining_occasion=occasion,
            price_range=price_range,
            flavor_preferences=flavors,
        )


def _response_text(response: Any) -> str:
    """Extract plain text from a LangChain message-like response."""

    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(content)


class LangChainMessageInterpreter:
    """Use a LangChain chat model for intent and preference extraction."""

    def __init__(self, model: Runnable) -> None:
        self.model = model

    def classify(self, message: str) -> Intent:
        prompt = """
Classify the user's food-assistant request as exactly one label:
restaurant, recipe, both, clarification, or database.
Database means adding, editing, or deleting catalog entries.
Return only the lowercase label.
""".strip()
        response = self.model.invoke(
            [SystemMessage(content=prompt), HumanMessage(content=message)]
        )
        intent = _response_text(response).strip().casefold()
        return intent if intent in VALID_INTENTS else "clarification"  # type: ignore[return-value]

    def extract(self, message: str) -> PreferenceProfile:
        schema = json.dumps(PreferenceProfile.model_json_schema(), indent=2)
        prompt = f"""
Extract only preferences explicitly stated by the user.
Return one valid JSON object matching this schema:
{schema}
Use empty lists, "not specified", or an empty string for missing values.
""".strip()
        response = self.model.invoke(
            [SystemMessage(content=prompt), HumanMessage(content=message)]
        )
        text = _response_text(response).strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        return PreferenceProfile.model_validate_json(text)


def classify_intent(user_message: str, llm: Runnable | None = None) -> Intent:
    """Classify a request with an LLM or the deterministic local fallback."""

    interpreter: MessageInterpreter = (
        LangChainMessageInterpreter(llm) if llm else RuleBasedInterpreter()
    )
    return interpreter.classify(user_message)


def extract_preferences(
    user_message: str,
    llm: Runnable | None = None,
) -> dict[str, Any]:
    """Extract preferences with an LLM or the deterministic local fallback."""

    interpreter: MessageInterpreter = (
        LangChainMessageInterpreter(llm) if llm else RuleBasedInterpreter()
    )
    return interpreter.extract(user_message).model_dump()


def merge_preferences(
    current: Mapping[str, Any],
    update: PreferenceProfile,
) -> dict[str, Any]:
    """Merge newly stated preferences into session state."""

    merged = PreferenceProfile.model_validate(current or {}).model_dump()
    incoming = update.model_dump()
    for key in (
        "favorite_cuisines",
        "dietary_restrictions",
        "flavor_preferences",
    ):
        merged[key] = _unique([*merged[key], *incoming[key]])
    for key in ("dining_occasion", "price_range"):
        if incoming[key] != "not specified":
            merged[key] = incoming[key]
    if incoming["other_preferences"]:
        merged["other_preferences"] = incoming["other_preferences"]
    return merged


def format_recommendations(
    recommendations: Mapping[str, Any],
    recommendation_type: Literal["restaurant", "recipe", "both"] = "both",
) -> str:
    """Format safe, readable Markdown for the chat window."""

    sections = []
    if recommendation_type in {"restaurant", "both"}:
        restaurants = recommendations.get("restaurants", [])
        if restaurants:
            lines = ["🍽️ **Restaurant recommendations**"]
            for index, item in enumerate(restaurants, 1):
                name = item.get("name", "Unnamed restaurant")
                cuisine = item.get("cuisine") or "Not specified"
                reasoning = item.get("reasoning", "No explanation supplied.")
                notes = ", ".join(item.get("dietary_notes", [])) or "Confirm with venue"
                lines.extend(
                    [
                        f"\n**{index}. {name}**",
                        f"- Cuisine: {cuisine}",
                        f"- Why it fits: {reasoning}",
                        f"- Dietary notes: {notes}",
                    ]
                )
            sections.append("\n".join(lines))

    if recommendation_type in {"recipe", "both"}:
        recipes = recommendations.get("recipes", [])
        if recipes:
            lines = ["👨‍🍳 **Recipe recommendations**"]
            for index, item in enumerate(recipes, 1):
                name = item.get("name", "Unnamed recipe")
                cuisine = item.get("cuisine") or "Not specified"
                reasoning = item.get("reasoning", "No explanation supplied.")
                notes = ", ".join(item.get("dietary_notes", [])) or "Check ingredients"
                lines.extend(
                    [
                        f"\n**{index}. {name}**",
                        f"- Cuisine: {cuisine}",
                        f"- Why it fits: {reasoning}",
                        f"- Dietary notes: {notes}",
                    ]
                )
            sections.append("\n".join(lines))

    return "\n\n".join(sections) or (
        "I could not find a grounded match. Try adding a cuisine, dietary need, "
        "budget, or dining occasion."
    )


def new_session_state() -> dict[str, Any]:
    """Return an independent initial value suitable for ``gr.State``."""

    return {
        "preferences": PreferenceProfile().model_dump(),
        "last_intent": "clarification",
    }


class RecommendationChatService:
    """Route chat messages and invoke the Lab 08 workflow."""

    def __init__(
        self,
        workflow: RecommendationWorkflow,
        interpreter: MessageInterpreter | None = None,
    ) -> None:
        self.workflow = workflow
        self.interpreter = interpreter or RuleBasedInterpreter()

    def respond(
        self,
        message: str,
        session: Mapping[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Return the assistant response and updated per-session state."""

        session_data = dict(session or new_session_state())
        intent = self.interpreter.classify(message)

        if intent == "clarification":
            session_data["last_intent"] = intent
            return (
                "I can recommend **restaurants**, **recipes**, or both. Tell me "
                "a cuisine, dietary restriction, budget, flavor, or occasion—for "
                "example: “Find a spicy vegetarian Thai recipe.”",
                session_data,
            )

        if intent == "database":
            session_data["last_intent"] = intent
            return (
                "Use the **Catalog manager** tabs to add validated restaurant or "
                "recipe records. Updates and deletions require an exact record ID "
                "and explicit confirmation.",
                session_data,
            )

        extracted = self.interpreter.extract(message)
        preferences = merge_preferences(session_data.get("preferences", {}), extracted)
        session_data.update({"preferences": preferences, "last_intent": intent})

        workflow_input = (
            f"Current request: {message}\n"
            f"Known session preferences: {json.dumps(preferences, ensure_ascii=False)}"
        )
        result = self.workflow.invoke(initial_state(workflow_input))
        recommendations = result.get("final_recommendations", {})
        answer = format_recommendations(recommendations, intent)

        errors = result.get("errors", [])
        if errors:
            answer += (
                "\n\n_The workflow completed with partial specialist failures; "
                "treat these suggestions as provisional._"
            )
        return answer, session_data


def echo_chatbot(message: str, history: list[ChatMessage]) -> str:
    """Small Task 1 echo function used to introduce Gradio chat callbacks."""

    del history
    return f"You said: {message}"


class RestaurantContribution(BaseModel):
    id: str = Field(default_factory=lambda: f"restaurant-{uuid.uuid4().hex[:10]}")
    name: str = Field(min_length=1)
    cuisine: str = Field(min_length=1)
    price_range: str = Field(min_length=1)
    location: str = Field(min_length=1)
    description: str = Field(min_length=1)


class RecipeContribution(BaseModel):
    id: str = Field(default_factory=lambda: f"recipe-{uuid.uuid4().hex[:10]}")
    name: str = Field(min_length=1)
    cuisine: str = Field(min_length=1)
    difficulty: str = Field(min_length=1)
    prep_time: str = Field(min_length=1)
    ingredients: list[str] = Field(min_length=1)
    instructions: list[str] = Field(min_length=1)


class CatalogStore:
    """Validated JSON contribution store with atomic writes and explicit deletes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _load(self) -> dict[str, list[dict[str, Any]]]:
        if not self.path.exists():
            return {"restaurants": [], "recipes": []}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Catalog must contain one JSON object")
        restaurants = [
            RestaurantContribution.model_validate(item).model_dump()
            for item in value.get("restaurants", [])
        ]
        recipes = [
            RecipeContribution.model_validate(item).model_dump()
            for item in value.get("recipes", [])
        ]
        return {"restaurants": restaurants, "recipes": recipes}

    def _save(self, data: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def all(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            return self._load()

    def add_restaurant(
        self,
        name: str,
        cuisine: str,
        price_range: str,
        location: str,
        description: str,
    ) -> RestaurantContribution:
        entry = RestaurantContribution(
            name=name,
            cuisine=cuisine,
            price_range=price_range,
            location=location,
            description=description,
        )
        with self._lock:
            data = self._load()
            data["restaurants"].append(entry.model_dump())
            self._save(data)
        return entry

    def add_recipe(
        self,
        name: str,
        cuisine: str,
        difficulty: str,
        prep_time: str,
        ingredients: str,
        instructions: str,
    ) -> RecipeContribution:
        entry = RecipeContribution(
            name=name,
            cuisine=cuisine,
            difficulty=difficulty,
            prep_time=prep_time,
            ingredients=_split_lines(ingredients),
            instructions=_split_lines(instructions),
        )
        with self._lock:
            data = self._load()
            data["recipes"].append(entry.model_dump())
            self._save(data)
        return entry

    def update_description(
        self,
        kind: CatalogKind,
        record_id: str,
        description: str,
        confirmed: bool,
    ) -> dict[str, Any]:
        """Update only the narrative field after explicit confirmation."""

        if not confirmed:
            raise PermissionError("Confirm the update before saving")
        field = "description" if kind == "restaurants" else "instructions"
        with self._lock:
            data = self._load()
            for index, item in enumerate(data[kind]):
                if item["id"] != record_id:
                    continue
                item[field] = (
                    description if kind == "restaurants" else _split_lines(description)
                )
                model = (
                    RestaurantContribution
                    if kind == "restaurants"
                    else RecipeContribution
                )
                updated = model.model_validate(item).model_dump()
                data[kind][index] = updated
                self._save(data)
                return updated
        raise KeyError(f"Unknown {kind[:-1]} ID: {record_id}")

    def delete(
        self,
        kind: CatalogKind,
        record_id: str,
        confirmed: bool,
    ) -> dict[str, Any]:
        """Delete one exact ID after explicit confirmation."""

        if not confirmed:
            raise PermissionError("Confirm the deletion before saving")
        with self._lock:
            data = self._load()
            for index, item in enumerate(data[kind]):
                if item["id"] == record_id:
                    deleted = data[kind].pop(index)
                    self._save(data)
                    return deleted
        raise KeyError(f"Unknown {kind[:-1]} ID: {record_id}")


def _split_lines(value: str) -> list[str]:
    return [
        item.strip(" \t-0123456789.")
        for item in value.splitlines()
        if item.strip(" \t-0123456789.")
    ]


def _catalog_snapshot(store: CatalogStore) -> str:
    return "```json\n" + json.dumps(store.all(), indent=2) + "\n```"


def build_gradio_app(
    service: RecommendationChatService,
    catalog: CatalogStore,
) -> Any:
    """Build the full chat and catalog UI without starting a public server."""

    try:
        import gradio as gr
    except ImportError as error:  # pragma: no cover - exercised without UI extra
        raise RuntimeError(
            'Gradio is optional. Install it with: pip install -e ".[ui]"'
        ) from error

    def submit_chat(
        message: str,
        history: list[ChatMessage],
        session: Mapping[str, Any],
    ) -> tuple[str, list[ChatMessage], dict[str, Any]]:
        if not message.strip():
            return "", history, dict(session)
        answer, updated_session = service.respond(message, session)
        updated_history = [
            *history,
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        return "", updated_history, updated_session

    def add_restaurant(
        name: str,
        cuisine: str,
        price: str,
        location: str,
        description: str,
    ) -> tuple[str, str]:
        try:
            entry = catalog.add_restaurant(
                name, cuisine, price, location, description
            )
            return (
                f"✅ Added **{entry.name}** with ID `{entry.id}`. Rebuild the "
                "multimodal index before retrieval.",
                _catalog_snapshot(catalog),
            )
        except Exception as error:
            return f"⚠️ {error}", _catalog_snapshot(catalog)

    def add_recipe(
        name: str,
        cuisine: str,
        difficulty: str,
        prep_time: str,
        ingredients: str,
        instructions: str,
    ) -> tuple[str, str]:
        try:
            entry = catalog.add_recipe(
                name,
                cuisine,
                difficulty,
                prep_time,
                ingredients,
                instructions,
            )
            return (
                f"✅ Added **{entry.name}** with ID `{entry.id}`. Rebuild the "
                "multimodal index before retrieval.",
                _catalog_snapshot(catalog),
            )
        except Exception as error:
            return f"⚠️ {error}", _catalog_snapshot(catalog)

    def update_record(
        kind: CatalogKind,
        record_id: str,
        value: str,
        confirmed: bool,
    ) -> tuple[str, str]:
        try:
            updated = catalog.update_description(
                kind, record_id.strip(), value, confirmed
            )
            return f"✅ Updated `{updated['id']}`.", _catalog_snapshot(catalog)
        except Exception as error:
            return f"⚠️ {error}", _catalog_snapshot(catalog)

    def delete_record(
        kind: CatalogKind,
        record_id: str,
        confirmed: bool,
    ) -> tuple[str, str]:
        try:
            deleted = catalog.delete(kind, record_id.strip(), confirmed)
            return f"✅ Deleted `{deleted['id']}`.", _catalog_snapshot(catalog)
        except Exception as error:
            return f"⚠️ {error}", _catalog_snapshot(catalog)

    with gr.Blocks(title="Agentic Food Guide") as demo:
        gr.Markdown(
            "# 🍜 Agentic Food Guide\n"
            "A conversational interface over the Lab 08 multi-agent RAG workflow."
        )
        session = gr.State(new_session_state)

        with gr.Tab("Recommendations"):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=480,
                placeholder=(
                    "Ask for a restaurant, a recipe, or both. Preferences are "
                    "remembered only for this browser session."
                ),
            )
            message = gr.Textbox(
                label="Your request",
                placeholder="Find a spicy vegetarian Thai recipe for date night",
            )
            with gr.Row():
                submit = gr.Button("Recommend", variant="primary")
                clear = gr.Button("Clear session")
            gr.Examples(
                examples=[
                    "Find healthy vegetarian restaurants for a date night",
                    "How do I make a spicy gluten-free Thai dinner?",
                    "Give me both restaurant and recipe ideas under $$",
                ],
                inputs=message,
            )
            submit.click(
                submit_chat,
                inputs=[message, chatbot, session],
                outputs=[message, chatbot, session],
            )
            message.submit(
                submit_chat,
                inputs=[message, chatbot, session],
                outputs=[message, chatbot, session],
            )
            clear.click(
                lambda: ([], new_session_state()),
                outputs=[chatbot, session],
            )

        with gr.Tab("Add restaurant"):
            restaurant_name = gr.Textbox(label="Name")
            restaurant_cuisine = gr.Textbox(label="Cuisine")
            restaurant_price = gr.Dropdown(
                ["$", "$$", "$$$", "$$$$"], value="$$", label="Price"
            )
            restaurant_location = gr.Textbox(label="Location")
            restaurant_description = gr.Textbox(label="Description", lines=4)
            restaurant_status = gr.Markdown()
            restaurant_add = gr.Button("Add validated restaurant", variant="primary")

        with gr.Tab("Add recipe"):
            recipe_name = gr.Textbox(label="Name")
            recipe_cuisine = gr.Textbox(label="Cuisine")
            recipe_difficulty = gr.Dropdown(
                ["Easy", "Medium", "Hard"], value="Easy", label="Difficulty"
            )
            recipe_time = gr.Textbox(label="Preparation time", value="30 minutes")
            recipe_ingredients = gr.Textbox(
                label="Ingredients (one per line)", lines=6
            )
            recipe_instructions = gr.Textbox(
                label="Instructions (one step per line)", lines=6
            )
            recipe_status = gr.Markdown()
            recipe_add = gr.Button("Add validated recipe", variant="primary")

        with gr.Tab("Manage catalog"):
            snapshot = gr.Markdown(_catalog_snapshot(catalog))
            refresh = gr.Button("Refresh records")
            record_kind = gr.Radio(
                ["restaurants", "recipes"],
                value="restaurants",
                label="Record type",
            )
            record_id = gr.Textbox(label="Exact record ID")
            replacement = gr.Textbox(
                label="New description or recipe steps",
                lines=4,
            )
            confirm = gr.Checkbox(label="I confirm this write operation")
            with gr.Row():
                update = gr.Button("Update narrative field")
                delete = gr.Button("Delete record", variant="stop")
            management_status = gr.Markdown()

        restaurant_add.click(
            add_restaurant,
            inputs=[
                restaurant_name,
                restaurant_cuisine,
                restaurant_price,
                restaurant_location,
                restaurant_description,
            ],
            outputs=[restaurant_status, snapshot],
        )
        recipe_add.click(
            add_recipe,
            inputs=[
                recipe_name,
                recipe_cuisine,
                recipe_difficulty,
                recipe_time,
                recipe_ingredients,
                recipe_instructions,
            ],
            outputs=[recipe_status, snapshot],
        )
        refresh.click(lambda: _catalog_snapshot(catalog), outputs=snapshot)
        update.click(
            update_record,
            inputs=[record_kind, record_id, replacement, confirm],
            outputs=[management_status, snapshot],
        )
        delete.click(
            delete_record,
            inputs=[record_kind, record_id, confirm],
            outputs=[management_status, snapshot],
        )

    return demo
