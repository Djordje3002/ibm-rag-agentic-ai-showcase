from ibm_rag_agentic_showcase.specialized_agents import (
    FOOD_STYLE_EXPERT,
    SPECIALIST_AGENTS,
    create_specialist_agent,
    validate_agent_catalog,
)


def test_catalog_contains_six_unique_specialists():
    validate_agent_catalog()

    assert len(SPECIALIST_AGENTS) == 6
    assert len({agent.id for agent in SPECIALIST_AGENTS}) == 6


def test_every_agent_has_complete_actionable_design():
    for agent in SPECIALIST_AGENTS:
        assert agent.role
        assert agent.goal
        assert agent.backstory
        assert agent.tasks
        assert all(task.expected_output for task in agent.tasks)


def test_prompt_patterns_render_expected_guidance():
    prompts = {
        agent.prompting_pattern: agent.system_prompt() for agent in SPECIALIST_AGENTS
    }

    assert "available tools" in prompts["react"]
    assert "Example input:" in prompts["few_shot"]


def test_food_style_goal_covers_required_specialty():
    goal = FOOD_STYLE_EXPERT.goal.lower()

    assert "cuisine" in goal
    assert "cooking methods" in goal
    assert "flavor profiles" in goal
    assert "user preferences" in goal


def test_create_specialist_agent_uses_langchain_factory(monkeypatch):
    captured = {}

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return "agent-graph"

    monkeypatch.setattr("langchain.agents.create_agent", fake_create_agent)

    result = create_specialist_agent(
        FOOD_STYLE_EXPERT,
        model="fake-model",
    )

    assert result == "agent-graph"
    assert captured["name"] == "food_style_expert"
    assert "Food Style Expert" in captured["system_prompt"]
