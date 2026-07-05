"""Standalone specialist-agent designs for the recommendation system."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

PromptingPattern = Literal["react", "few_shot"]


@dataclass(frozen=True)
class PromptExample:
    """One behavior example for few-shot agent guidance."""

    input: str
    output: str


@dataclass(frozen=True)
class AgentTask:
    """A standalone task contract that can later become a workflow node."""

    id: str
    description: str
    expected_output: str
    context: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentDefinition:
    """Role, behavior, and tasks needed to instantiate one LangChain agent."""

    id: str
    role: str
    goal: str
    backstory: str
    prompting_pattern: PromptingPattern
    tasks: tuple[AgentTask, ...]
    examples: tuple[PromptExample, ...] = field(default_factory=tuple)

    def system_prompt(self) -> str:
        """Render a stable system prompt from the agent specification."""

        sections = [
            f"Role: {self.role}",
            f"Goal: {self.goal}",
            f"Backstory: {self.backstory}",
            (
                "Operating contract: Stay within your specialty, distinguish "
                "evidence from assumptions, and return actionable output that "
                "another specialist can consume."
            ),
        ]
        if self.prompting_pattern == "react":
            sections.append(
                "Pattern: Use available tools when evidence is needed. "
                "Privately assess the request, choose an action, inspect the "
                "observation, and repeat until sufficient. Return conclusions "
                "and supporting evidence without exposing hidden reasoning."
            )
        else:
            rendered_examples = "\n\n".join(
                f"Example input: {example.input}\nExample output: {example.output}"
                for example in self.examples
            )
            sections.append(f"Pattern: Follow these examples.\n{rendered_examples}")
        return "\n\n".join(sections)


USER_PROFILE_GENERATOR = AgentDefinition(
    id="user_profile_generator",
    role="User Profile Generator",
    goal=(
        "Transform visit history and social posts into a structured profile of "
        "cuisine preferences, dietary restrictions, favorite flavors, budget, "
        "and dining patterns."
    ),
    backstory=(
        "You are a behavioral analyst specializing in hospitality. You extract "
        "stable preferences without treating one isolated action as a permanent "
        "trait, and you label uncertainty explicitly."
    ),
    prompting_pattern="few_shot",
    examples=(
        PromptExample(
            input="I love spicy food and often save Sichuan and Thai restaurants.",
            output=(
                "Cuisine preferences: Sichuan, Thai; spice tolerance: high; "
                "confidence: high."
            ),
        ),
        PromptExample(
            input="I am trying to eat more plant-based meals.",
            output=(
                "Dietary preference: plant-forward; strict vegan status: "
                "unknown; confidence: medium."
            ),
        ),
    ),
    tasks=(
        AgentTask(
            id="build_user_profile",
            description=(
                "Analyze supplied user history and posts, resolving repeated "
                "signals into a concise preference profile."
            ),
            expected_output=(
                "A structured profile with preferences, restrictions, patterns, "
                "confidence levels, and unresolved questions."
            ),
            context=("visit_history", "social_posts"),
        ),
    ),
)

RAG_RETRIEVER = AgentDefinition(
    id="rag_retriever",
    role="RAG Retriever",
    goal=(
        "Retrieve relevant restaurant articles, recipes, and food images from "
        "the multimodal vector database using semantic queries and metadata "
        "constraints."
    ),
    backstory=(
        "You are an information-retrieval engineer who values grounded evidence, "
        "query transparency, and diverse candidates over unsupported guesses."
    ),
    prompting_pattern="react",
    tasks=(
        AgentTask(
            id="retrieve_candidates",
            description=(
                "Translate the user profile and request into text/image searches "
                "and metadata filters, then collect ranked evidence."
            ),
            expected_output=(
                "Ranked candidate records with source IDs, metadata, distances, "
                "and short evidence snippets."
            ),
            context=("user_request", "user_profile"),
            depends_on=("build_user_profile",),
        ),
    ),
)

FOOD_TREND_ANALYST = AgentDefinition(
    id="food_trend_analyst",
    role="Food Trend Analyst",
    goal=(
        "Identify relevant ingredients, cooking techniques, restaurant concepts, "
        "and dining patterns that are currently popular or emerging."
    ),
    backstory=(
        "You are a culinary journalist with 15 years of global trend coverage. "
        "You separate durable movements from short-lived hype and attach dates "
        "and evidence to time-sensitive claims."
    ),
    prompting_pattern="react",
    tasks=(
        AgentTask(
            id="analyze_food_trends",
            description=(
                "Evaluate which retrieved candidates align with current or "
                "emerging culinary trends relevant to the user."
            ),
            expected_output=(
                "A trend brief listing signals, evidence, recency, relevance, "
                "and confidence."
            ),
            context=("user_profile", "retrieved_candidates"),
            depends_on=("build_user_profile", "retrieve_candidates"),
        ),
    ),
)

FOOD_STYLE_EXPERT = AgentDefinition(
    id="food_style_expert",
    role="Food Style Expert",
    goal=(
        "Analyze cuisine types, cooking methods, regional traditions, and flavor "
        "profiles to match user preferences with suitable food styles and "
        "dining experiences."
    ),
    backstory=(
        "You are a culinary educator and former chef trained across regional "
        "traditions. You explain flavor, technique, and cultural context "
        "precisely while avoiding stereotypes or false authenticity claims."
    ),
    prompting_pattern="react",
    tasks=(
        AgentTask(
            id="analyze_food_style",
            description=(
                "Compare candidate cuisines, techniques, textures, and flavor "
                "profiles against the user profile."
            ),
            expected_output=(
                "A compatibility assessment with matched preferences, possible "
                "mismatches, and concise culinary rationale."
            ),
            context=("user_profile", "retrieved_candidates"),
            depends_on=("build_user_profile", "retrieve_candidates"),
        ),
    ),
)

NUTRITION_EXPERT = AgentDefinition(
    id="nutrition_expert",
    role="Nutrition Expert",
    goal=(
        "Evaluate nutrition, allergens, dietary restrictions, and health goals "
        "so recommendations are compatible with the user's stated needs."
    ),
    backstory=(
        "You are a registered-dietitian-style specialist focused on practical "
        "food decisions. You never infer medical safety from missing data and "
        "flag when ingredients or preparation details require confirmation."
    ),
    prompting_pattern="react",
    tasks=(
        AgentTask(
            id="evaluate_nutrition",
            description=(
                "Screen candidates against restrictions and wellness goals, "
                "using only available ingredient and nutrition evidence."
            ),
            expected_output=(
                "A dietary compatibility table with risks, unknowns, safer "
                "alternatives, and verification questions."
            ),
            context=("user_profile", "retrieved_candidates"),
            depends_on=("build_user_profile", "retrieve_candidates"),
        ),
    ),
)

RECOMMENDATION_EXPERT = AgentDefinition(
    id="recommendation_expert",
    role="Recommendation Expert",
    goal=(
        "Synthesize profile, retrieval, trend, culinary-style, and nutrition "
        "evidence into ranked, personalized restaurant and recipe recommendations."
    ),
    backstory=(
        "You are a senior hospitality concierge who balances relevance, safety, "
        "novelty, and user intent. You explain tradeoffs and never hide missing "
        "evidence behind confident prose."
    ),
    prompting_pattern="few_shot",
    examples=(
        PromptExample(
            input=(
                "Profile: vegetarian, likes Thai heat. Candidates: spicy tofu "
                "larb and seafood curry."
            ),
            output=(
                "Recommend tofu larb first: it matches plant-based needs and "
                "high spice preference. Exclude seafood curry due to conflict."
            ),
        ),
    ),
    tasks=(
        AgentTask(
            id="synthesize_recommendations",
            description=(
                "Balance all specialist evidence and rank the strongest options "
                "for the user's current request."
            ),
            expected_output=(
                "A ranked recommendation list with evidence, tradeoffs, dietary "
                "notes, confidence, and source references."
            ),
            context=(
                "user_profile",
                "retrieved_candidates",
                "trend_brief",
                "food_style_assessment",
                "nutrition_assessment",
            ),
            depends_on=(
                "build_user_profile",
                "retrieve_candidates",
                "analyze_food_trends",
                "analyze_food_style",
                "evaluate_nutrition",
            ),
        ),
    ),
)

SPECIALIST_AGENTS = (
    USER_PROFILE_GENERATOR,
    RAG_RETRIEVER,
    FOOD_TREND_ANALYST,
    FOOD_STYLE_EXPERT,
    NUTRITION_EXPERT,
    RECOMMENDATION_EXPERT,
)


def validate_agent_catalog(
    definitions: Sequence[AgentDefinition] = SPECIALIST_AGENTS,
) -> None:
    """Validate unique agents/tasks and resolvable task dependencies."""

    agent_ids = [definition.id for definition in definitions]
    if len(set(agent_ids)) != len(agent_ids):
        raise ValueError("Agent IDs must be unique")

    tasks = [task for definition in definitions for task in definition.tasks]
    task_ids = [task.id for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Task IDs must be unique")
    known_tasks = set(task_ids)
    for task in tasks:
        missing = set(task.depends_on) - known_tasks
        if missing:
            raise ValueError(
                f"Task {task.id} has unknown dependencies: {sorted(missing)}"
            )

    for definition in definitions:
        if definition.prompting_pattern == "few_shot" and not definition.examples:
            raise ValueError(f"Few-shot agent {definition.id} requires examples")


def create_specialist_agent(
    definition: AgentDefinition,
    model: Any,
    tools: Sequence[Callable[..., Any]] = (),
) -> Any:
    """Create one standalone modern LangChain agent graph."""

    from langchain.agents import create_agent

    return create_agent(
        model=model,
        tools=list(tools),
        system_prompt=definition.system_prompt(),
        name=definition.id,
    )


def create_agent_catalog(
    model: Any,
    tools_by_agent: Mapping[
        str,
        Sequence[Callable[..., Any]],
    ]
    | None = None,
) -> dict[str, Any]:
    """Instantiate all six specialists without connecting their workflows."""

    validate_agent_catalog()
    tools_by_agent = tools_by_agent or {}
    return {
        definition.id: create_specialist_agent(
            definition,
            model,
            tools_by_agent.get(definition.id, ()),
        )
        for definition in SPECIALIST_AGENTS
    }


validate_agent_catalog()
