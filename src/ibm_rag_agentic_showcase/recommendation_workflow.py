"""Stateful hybrid multi-agent recommendation workflow for Lab 08."""

from __future__ import annotations

import json
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .multimodal_vector_index import TextEmbeddingModel
from .similarity_retrieval import (
    MultimodalStores,
    retrieve_articles,
    retrieve_images_by_text,
)
from .specialized_agents import SPECIALIST_AGENTS

WorkflowStep = Literal[
    "start",
    "profile_generated",
    "candidates_retrieved",
    "analysis_complete",
    "complete",
]


class AgentState(TypedDict):
    """Shared state whose fields are updated by specialized graph nodes."""

    user_input: str
    user_profile: dict[str, Any]
    retrieved_restaurants: list[dict[str, Any]]
    retrieved_recipes: list[dict[str, Any]]
    trend_analysis: dict[str, Any]
    style_analysis: dict[str, Any]
    nutrition_analysis: dict[str, Any]
    final_recommendations: dict[str, Any]
    workflow_step: WorkflowStep
    errors: Annotated[list[str], operator.add]


class UserProfile(BaseModel):
    favorite_cuisines: list[str] = Field(default_factory=list)
    dietary_restrictions: list[str] = Field(default_factory=list)
    dining_occasions: list[str] = Field(default_factory=list)
    price_range: str | None = None
    adventurousness_score: int = Field(default=5, ge=1, le=10)
    flavor_preferences: list[str] = Field(default_factory=list)
    summary: str


class TrendAnalysis(BaseModel):
    trends: list[dict[str, Any]] = Field(default_factory=list)


class StyleAnalysis(BaseModel):
    matches: list[dict[str, Any]] = Field(default_factory=list)
    mismatches: list[dict[str, Any]] = Field(default_factory=list)
    flavor_summary: str = ""


class NutritionAnalysis(BaseModel):
    compliant_items: list[str] = Field(default_factory=list)
    flagged_items: list[dict[str, Any]] = Field(default_factory=list)
    nutritional_highlights: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    name: str
    reasoning: str
    cuisine: str | None = None
    dietary_notes: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class FinalRecommendations(BaseModel):
    restaurants: list[Recommendation] = Field(default_factory=list)
    recipes: list[Recommendation] = Field(default_factory=list)


class AgentCaller(Protocol):
    """Return structured output from one specialist."""

    def __call__(
        self,
        agent_id: str,
        user_message: str,
    ) -> Mapping[str, Any]:
        """Call a specialist and return its parsed JSON object."""


class CandidateRetriever(Protocol):
    """Retrieve grounded candidates from the multimodal database."""

    def __call__(
        self,
        profile: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return restaurant and recipe candidates."""


@dataclass(frozen=True)
class WorkflowServices:
    """Runtime dependencies injected into graph nodes for testability."""

    agent_caller: AgentCaller
    candidate_retriever: CandidateRetriever


class LangChainJSONAgentCaller:
    """Call specialists through any LangChain chat model and parse JSON."""

    def __init__(self, model: Runnable) -> None:
        self.model = model
        self.parser = JsonOutputParser()
        self.definitions = {
            definition.id: definition for definition in SPECIALIST_AGENTS
        }

    def __call__(
        self,
        agent_id: str,
        user_message: str,
    ) -> Mapping[str, Any]:
        definition = self.definitions[agent_id]
        response = self.model.invoke(
            [
                SystemMessage(
                    content=(
                        f"{definition.system_prompt()}\n\n"
                        "Return only one valid JSON object matching the "
                        "requested schema."
                    )
                ),
                HumanMessage(content=user_message),
            ]
        )
        parsed = self.parser.invoke(response)
        if not isinstance(parsed, dict):
            raise ValueError(f"{agent_id} returned a non-object JSON value")
        return parsed


@dataclass(frozen=True)
class MultimodalCandidateRetriever:
    """Ground the RAG phase in the persistent Lab 04 collections."""

    stores: MultimodalStores
    text_embedder: TextEmbeddingModel
    clip_embedder: TextEmbeddingModel
    limit: int = 20

    def __call__(
        self,
        profile: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        query = profile_to_query(profile)
        article_hits = retrieve_articles(
            self.stores.articles,
            self.text_embedder,
            query,
            k=self.limit,
        )
        recipe_hits = retrieve_images_by_text(
            self.stores.images,
            self.clip_embedder,
            query,
            k=self.limit,
        )

        restaurants = [
            {
                **hit.metadata,
                "id": hit.id,
                "description": hit.content,
                "distance": hit.distance,
            }
            for hit in article_hits
        ]
        recipes = [
            {
                **hit.metadata,
                "id": hit.id,
                "description": hit.content,
                "distance": hit.distance,
            }
            for hit in recipe_hits
        ]
        return restaurants, recipes


def profile_to_query(profile: Mapping[str, Any]) -> str:
    """Turn structured preferences into a grounded retrieval query."""

    values = []
    for field in (
        "favorite_cuisines",
        "flavor_preferences",
        "dining_occasions",
        "dietary_restrictions",
    ):
        value = profile.get(field, [])
        if isinstance(value, list):
            values.extend(str(item) for item in value)
    if profile.get("price_range"):
        values.append(str(profile["price_range"]))
    if profile.get("summary"):
        values.append(str(profile["summary"]))
    return ", ".join(values) or "diverse restaurant and recipe recommendations"


def initial_state(user_input: str) -> AgentState:
    """Create a complete, observable starting state."""

    return {
        "user_input": user_input,
        "user_profile": {},
        "retrieved_restaurants": [],
        "retrieved_recipes": [],
        "trend_analysis": {},
        "style_analysis": {},
        "nutrition_analysis": {},
        "final_recommendations": {},
        "workflow_step": "start",
        "errors": [],
    }


def _json_context(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def build_recommendation_workflow(services: WorkflowServices) -> Any:
    """Compile sequential, parallel, then sequential recommendation phases."""

    def generate_profile(state: AgentState) -> dict[str, Any]:
        prompt = f"""
Analyze this user data:

{state["user_input"]}

Return these fields:
favorite_cuisines, dietary_restrictions, dining_occasions, price_range,
adventurousness_score (1-10), flavor_preferences, summary.
""".strip()
        try:
            raw = services.agent_caller("user_profile_generator", prompt)
            profile = UserProfile.model_validate(raw).model_dump()
            return {
                "user_profile": profile,
                "workflow_step": "profile_generated",
            }
        except Exception as error:
            return {
                "user_profile": {
                    "summary": "Profile generation failed.",
                    "favorite_cuisines": [],
                    "dietary_restrictions": [],
                    "dining_occasions": [],
                    "price_range": None,
                    "adventurousness_score": 5,
                    "flavor_preferences": [],
                },
                "workflow_step": "profile_generated",
                "errors": [f"user_profile_generator: {error}"],
            }

    def retrieve_candidates(state: AgentState) -> dict[str, Any]:
        try:
            restaurants, recipes = services.candidate_retriever(state["user_profile"])
            return {
                "retrieved_restaurants": restaurants,
                "retrieved_recipes": recipes,
                "workflow_step": "candidates_retrieved",
            }
        except Exception as error:
            return {
                "retrieved_restaurants": [],
                "retrieved_recipes": [],
                "workflow_step": "candidates_retrieved",
                "errors": [f"rag_retriever: {error}"],
            }

    def analyze_trends(state: AgentState) -> dict[str, Any]:
        prompt = f"""
Identify 3-5 current or emerging food trends in these grounded candidates.
Explain relevance and confidence.

Restaurants:
{_json_context(state["retrieved_restaurants"][:10])}

Recipes:
{_json_context(state["retrieved_recipes"][:10])}

Return: {{"trends": [{{"name": str, "description": str,
"relevance": str, "confidence": str}}]}}
""".strip()
        try:
            raw = services.agent_caller("food_trend_analyst", prompt)
            return {"trend_analysis": TrendAnalysis.model_validate(raw).model_dump()}
        except Exception as error:
            return {
                "trend_analysis": {"trends": []},
                "errors": [f"food_trend_analyst: {error}"],
            }

    def analyze_styles(state: AgentState) -> dict[str, Any]:
        prompt = f"""
Compare the cuisine types, cooking methods, textures, and flavor profiles in
these candidates with the user profile.

User profile:
{_json_context(state["user_profile"])}

Restaurants:
{_json_context(state["retrieved_restaurants"][:10])}

Recipes:
{_json_context(state["retrieved_recipes"][:10])}

Return: {{"matches": [object], "mismatches": [object],
"flavor_summary": str}}
""".strip()
        try:
            raw = services.agent_caller("food_style_expert", prompt)
            return {"style_analysis": StyleAnalysis.model_validate(raw).model_dump()}
        except Exception as error:
            return {
                "style_analysis": {
                    "matches": [],
                    "mismatches": [],
                    "flavor_summary": "",
                },
                "errors": [f"food_style_expert: {error}"],
            }

    def evaluate_nutrition(state: AgentState) -> dict[str, Any]:
        prompt = f"""
Evaluate dietary compliance, allergens, unknown ingredients, and nutritional
balance for the candidates. Never infer safety from missing information.

User profile:
{_json_context(state["user_profile"])}

Restaurants:
{_json_context(state["retrieved_restaurants"][:10])}

Recipes:
{_json_context(state["retrieved_recipes"][:10])}

Return: {{"compliant_items": [str], "flagged_items": [object],
"nutritional_highlights": [str]}}
""".strip()
        try:
            raw = services.agent_caller("nutrition_expert", prompt)
            return {
                "nutrition_analysis": NutritionAnalysis.model_validate(raw).model_dump()
            }
        except Exception as error:
            return {
                "nutrition_analysis": {
                    "compliant_items": [],
                    "flagged_items": [],
                    "nutritional_highlights": [],
                },
                "errors": [f"nutrition_expert: {error}"],
            }

    def mark_analysis_complete(_: AgentState) -> dict[str, Any]:
        return {"workflow_step": "analysis_complete"}

    def generate_recommendations(state: AgentState) -> dict[str, Any]:
        prompt = f"""
Create up to five restaurant and five recipe recommendations by synthesizing
all specialist evidence below. Respect dietary restrictions and cite source IDs.

User profile:
{_json_context(state["user_profile"])}

Restaurant candidates:
{_json_context(state["retrieved_restaurants"][:10])}

Recipe candidates:
{_json_context(state["retrieved_recipes"][:10])}

Trend analysis:
{_json_context(state["trend_analysis"])}

Food style analysis:
{_json_context(state["style_analysis"])}

Nutrition analysis:
{_json_context(state["nutrition_analysis"])}

Return: {{"restaurants": [{{"name": str, "reasoning": str,
"cuisine": str|null, "dietary_notes": [str], "source_ids": [str]}}],
"recipes": [same schema]}}
""".strip()
        try:
            raw = services.agent_caller("recommendation_expert", prompt)
            recommendations = FinalRecommendations.model_validate(raw).model_dump()
            return {
                "final_recommendations": recommendations,
                "workflow_step": "complete",
            }
        except Exception as error:
            return {
                "final_recommendations": {
                    "restaurants": [],
                    "recipes": [],
                },
                "workflow_step": "complete",
                "errors": [f"recommendation_expert: {error}"],
            }

    builder = StateGraph(AgentState)
    builder.add_node("generate_profile", generate_profile)
    builder.add_node("retrieve_candidates", retrieve_candidates)
    builder.add_node("analyze_trends", analyze_trends)
    builder.add_node("analyze_styles", analyze_styles)
    builder.add_node("evaluate_nutrition", evaluate_nutrition)
    builder.add_node("mark_analysis_complete", mark_analysis_complete)
    builder.add_node("generate_recommendations", generate_recommendations)

    builder.add_edge(START, "generate_profile")
    builder.add_edge("generate_profile", "retrieve_candidates")
    builder.add_edge("retrieve_candidates", "analyze_trends")
    builder.add_edge("retrieve_candidates", "analyze_styles")
    builder.add_edge("retrieve_candidates", "evaluate_nutrition")
    builder.add_edge(
        ["analyze_trends", "analyze_styles", "evaluate_nutrition"],
        "mark_analysis_complete",
    )
    builder.add_edge("mark_analysis_complete", "generate_recommendations")
    builder.add_edge("generate_recommendations", END)
    return builder.compile()


@dataclass(frozen=True)
class EvaluationReport:
    """Machine-readable recommendation quality smoke test."""

    restaurant_count: int
    recipe_count: int
    dietary_restrictions: tuple[str, ...]
    favorite_cuisines: tuple[str, ...]
    reasoning_coverage: float
    dietary_note_coverage: float
    quality_score: float
    first_restaurant: str | None
    first_recipe: str | None
    warnings: tuple[str, ...]


def evaluate_recommendations(result: Mapping[str, Any]) -> EvaluationReport:
    """Assess completeness, explanation coverage, and dietary observability."""

    profile = result.get("user_profile", {})
    recommendations = result.get("final_recommendations", {})
    restaurants = list(recommendations.get("restaurants", []))
    recipes = list(recommendations.get("recipes", []))
    all_items = restaurants + recipes
    restrictions = tuple(profile.get("dietary_restrictions", []))
    cuisines = tuple(profile.get("favorite_cuisines", []))

    reasoning_count = sum(
        bool(str(item.get("reasoning", "")).strip()) for item in all_items
    )
    note_count = sum(bool(item.get("dietary_notes")) for item in all_items)
    item_count = len(all_items)
    reasoning_coverage = reasoning_count / item_count if item_count else 0.0
    dietary_note_coverage = note_count / item_count if item_count else 0.0

    warnings = []
    if not restaurants:
        warnings.append("No restaurant recommendations were produced.")
    if not recipes:
        warnings.append("No recipe recommendations were produced.")
    if restrictions and dietary_note_coverage < 1.0:
        warnings.append("Some recommendations do not expose dietary compliance notes.")
    if reasoning_coverage < 1.0:
        warnings.append("Some recommendations do not include reasoning.")
    if result.get("errors"):
        warnings.append("The workflow recorded one or more agent errors.")

    quality_score = 0.0
    quality_score += 25.0 if restaurants else 0.0
    quality_score += 25.0 if recipes else 0.0
    quality_score += 30.0 * reasoning_coverage
    quality_score += 20.0 * (dietary_note_coverage if restrictions else 1.0)

    def first_summary(items: list[dict[str, Any]]) -> str | None:
        if not items:
            return None
        return f"{items[0].get('name', 'N/A')}: {items[0].get('reasoning', 'N/A')}"

    return EvaluationReport(
        restaurant_count=len(restaurants),
        recipe_count=len(recipes),
        dietary_restrictions=restrictions,
        favorite_cuisines=cuisines,
        reasoning_coverage=reasoning_coverage,
        dietary_note_coverage=dietary_note_coverage,
        quality_score=quality_score,
        first_restaurant=first_summary(restaurants),
        first_recipe=first_summary(recipes),
        warnings=tuple(warnings),
    )


def format_evaluation(report: EvaluationReport) -> str:
    """Render an evaluation report for a notebook or terminal."""

    lines = [
        "=" * 80,
        "RECOMMENDATION EVALUATION",
        "=" * 80,
        f"Restaurant recommendations: {report.restaurant_count}",
        f"Recipe recommendations: {report.recipe_count}",
        f"Dietary restrictions: {', '.join(report.dietary_restrictions) or 'None'}",
        f"Favorite cuisines: {', '.join(report.favorite_cuisines) or 'None'}",
        f"Reasoning coverage: {report.reasoning_coverage:.0%}",
        f"Dietary-note coverage: {report.dietary_note_coverage:.0%}",
        f"Quality smoke-test score: {report.quality_score:.1f}/100",
    ]
    if report.first_restaurant:
        lines.append(f"First restaurant: {report.first_restaurant}")
    if report.first_recipe:
        lines.append(f"First recipe: {report.first_recipe}")
    lines.extend(f"Warning: {warning}" for warning in report.warnings)
    lines.append("=" * 80)
    return "\n".join(lines)
