# %% [markdown]
# # Lab 07 — Design specialized recommendation agents
#
# ## What this lab does
#
# This lab defines six standalone specialists:
#
# 1. User Profile Generator
# 2. RAG Retriever
# 3. Food Trend Analyst
# 4. Food Style Expert
# 5. Nutrition Expert
# 6. Recommendation Expert
#
# Every definition has a role, goal, backstory, prompting pattern, and task
# contract. ReAct-style agents may later receive tools; few-shot agents include
# examples that demonstrate the desired output. The specialists are deliberately
# not connected into a workflow yet—that belongs to the next lesson.

# %%
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage

from ibm_rag_agentic_showcase.specialized_agents import (
    FOOD_STYLE_EXPERT,
    SPECIALIST_AGENTS,
    create_specialist_agent,
    validate_agent_catalog,
)

# %% [markdown]
# ## 1. Validate and inspect every standalone specification

# %%
validate_agent_catalog()

for definition in SPECIALIST_AGENTS:
    print(f"\n{definition.role}")
    print(f"Goal: {definition.goal}")
    print(f"Pattern: {definition.prompting_pattern}")
    for task in definition.tasks:
        print(f"Task: {task.id} -> {task.expected_output}")

# %% [markdown]
# ## 2. Inspect the Food Style Expert goal
#
# This exact implementation is also captured in the required
# `M3L1_food_style_expert_goal.jpg` artifact.

# %%
print(FOOD_STYLE_EXPERT.goal)

# %% [markdown]
# ## 3. Create and smoke-test a standalone LangChain agent
#
# A deterministic fake chat model proves the agent graph and prompt work without
# an API key. Replace it with a tool-capable production chat model later.

# %%
fake_model = FakeMessagesListChatModel(
    responses=[
        AIMessage(
            content=(
                "The dish matches the requested bright, herb-forward flavor "
                "profile; verify heat tolerance before recommending."
            )
        )
    ]
)
food_style_agent = create_specialist_agent(
    FOOD_STYLE_EXPERT,
    fake_model,
)
result = food_style_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Assess whether a Thai basil dish suits someone who likes "
                    "fresh herbs but has unknown spice tolerance."
                ),
            }
        ]
    }
)
print(result["messages"][-1].content)

print("Six specialized recommendation agents designed successfully")
