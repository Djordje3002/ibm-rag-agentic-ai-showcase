# Project 07 — Specialized Recommendation Agents

## Goal

This lab defines six standalone LangChain agents that will later collaborate in
a recommendation workflow. Each agent owns one responsibility and exposes an
actionable task contract.

## Agent catalog

| Agent | Primary responsibility | Pattern |
|---|---|---|
| User Profile Generator | Extract preferences, restrictions, and behavior | Few-shot |
| RAG Retriever | Retrieve grounded multimodal evidence | ReAct |
| Food Trend Analyst | Assess current and emerging culinary trends | ReAct |
| Food Style Expert | Match cuisine, technique, and flavor profiles | ReAct |
| Nutrition Expert | Screen dietary compatibility and uncertainty | ReAct |
| Recommendation Expert | Synthesize and rank final recommendations | Few-shot |

## Design components

Each `AgentDefinition` includes:

- a concise role;
- a specific goal;
- a backstory that guides expertise and tone;
- a ReAct or few-shot prompting pattern;
- one or more `AgentTask` contracts;
- expected output, context keys, and task dependencies.

Task dependencies are declared but not executed in this lab. The catalog
validator checks that agent IDs and task IDs are unique and every dependency
refers to a known task.

## LangChain integration

`create_specialist_agent` uses modern LangChain `create_agent` with the rendered
specialist system prompt and an optional tool list. The function returns a
standalone graph. `create_agent_catalog` creates all six graphs but intentionally
does not connect them.

The example uses a deterministic fake chat model to smoke-test the Food Style
Expert without credentials or external calls.

## Run

```bash
pip install -e ".[agents,dev]"
python examples/07_design_specialized_agents.py
```

## Prompting choices

ReAct is used where a specialist will eventually need tools or iterative
evidence gathering: retrieval, trends, culinary analysis, and nutrition.
Instructions request concise conclusions and evidence without exposing hidden
chain-of-thought.

Few-shot prompts guide tasks where stable output patterns matter most: user
profile extraction and final recommendation synthesis.

## Screenshot artifact

The assignment screenshot is stored at:

`docs/screenshots/M3L1_food_style_expert_goal.jpg`

It captures the Food Style Expert role and goal implementation.

## Next step

The next lesson can connect these task dependencies in a stateful graph, attach
the multimodal retriever as a tool, run compatible specialists in parallel, and
route their outputs to the Recommendation Expert.
