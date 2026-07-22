<div align="center">

# IBM RAG & Agentic AI Showcase

**Reproducible AI engineering patterns built from IBM coursework: structured generation today, retrieval and agentic workflows next.**

[![CI](https://github.com/Djordje3002/ibm-rag-agentic-ai-showcase/actions/workflows/ci.yml/badge.svg)](https://github.com/Djordje3002/ibm-rag-agentic-ai-showcase/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-validated-E92063?logo=pydantic&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-16A34A)

<img src="assets/validated-extraction-showcase.png" alt="Unstructured restaurant descriptions flowing through a language model, schema validation, and a bounded repair loop" width="100%">

</div>

> **Status:** Project 01 is complete and tested. Additional RAG and agentic workflows will be added as the coursework progresses.

## Project portfolio

| # | Project | Concepts | Status |
| ---: | --- | --- | --- |
| 01 | [Structured restaurant extraction](projects/01-structured-restaurant-extraction.md) | IBM Granite, one-shot prompting, Pydantic validation, JSON self-repair | Complete |
| 02 | Retrieval-augmented generation | Chunking, embeddings, vector search, grounded answers | Planned |
| 03 | Agentic AI workflow | Tools, planning, memory, multi-step execution | Planned |

## Project 01: from unstructured text to reliable JSON

Natural-language restaurant descriptions are easy to read but difficult to search, filter, or index. The first project converts each description into a typed record, validates every field, and gives malformed model output a bounded opportunity to repair itself.

~~~mermaid
flowchart LR
    A[Restaurant descriptions] --> B[One-shot extraction prompt]
    B --> C[IBM Granite on watsonx.ai]
    C --> D{Pydantic validation}
    D -->|Valid| E[Structured JSON dataset]
    D -->|Invalid| F[Targeted repair prompt]
    F --> C
~~~

### What this demonstrates

- Constrained generation with an explicit JSON contract
- One-shot prompting with a representative example
- Runtime validation instead of trusting model output
- Repair prompts informed by exact validation errors
- Bounded retry and clear terminal-failure behavior
- Dependency injection for deterministic offline tests
- Separation between provider access, extraction logic, and file I/O

### Reliability contract

| Boundary | Behavior |
| --- | --- |
| Model response | Treated as untrusted text |
| Schema | Pydantic checks required fields and value types |
| Repair | Invalid candidates receive exact validation errors |
| Retry policy | Repair attempts are bounded and configurable |
| Failure | <code>ExtractionError</code> is raised instead of saving partial data |
| Tests | An injected fake LLM keeps checks offline and deterministic |

## Quick start

### Prerequisites

- Python 3.11 or newer
- An IBM watsonx.ai project and API key, or the IBM Skills Network lab runtime

~~~bash
git clone https://github.com/Djordje3002/ibm-rag-agentic-ai-showcase.git
cd ibm-rag-agentic-ai-showcase

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

cp .env.example .env
# Edit .env, then load it into this shell:
set -a
source .env
set +a

restaurant-extract --limit 3
~~~

The command downloads the course dataset when it is missing and writes validated records to <code>data/processed/structured_restaurant_data.json</code>. Downloaded and generated data is ignored by Git.

### Useful CLI options

~~~bash
restaurant-extract --limit 5
restaurant-extract --input path/to/data.txt --output path/to/results.json
restaurant-extract --max-repair-attempts 2
restaurant-extract --no-download
restaurant-extract --help
~~~

Use a small <code>--limit</code> for the first live run so provider configuration and cost behavior can be checked before processing the whole dataset.

## Output contract

Every accepted restaurant record contains:

- required <code>name</code>, <code>location</code>, <code>type</code>, <code>food_style</code>, and <code>environment</code>
- nullable <code>rating</code>, <code>price_range</code>, and <code>vibe</code>
- list-valued <code>signatures</code> and <code>shortcomings</code>
- a deterministic <code>itemId</code> assigned after schema validation

The complete design reasoning lives in the [Project 01 write-up](projects/01-structured-restaurant-extraction.md).

## Configuration

| Variable | Required | Purpose |
| --- | --- | --- |
| <code>WATSONX_APIKEY</code> | IBM Cloud | watsonx.ai authentication |
| <code>WATSONX_PROJECT_ID</code> | IBM Cloud | Project used for model inference |
| <code>WATSONX_URL</code> | Optional | Regional service URL |
| <code>WATSONX_MODEL_ID</code> | Optional | Granite model identifier |

The IBM Skills Network runtime can provide its own credentials and project context. Never commit a populated <code>.env</code> file.

## Run the checks

~~~bash
ruff check .
pytest
~~~

The tests do not call watsonx.ai, so they are fast, deterministic, and consume no inference credits. GitHub Actions runs linting and tests on every push and pull request to <code>main</code>.

## Repository layout

~~~text
.
├── assets/                     README showcase artwork
├── examples/                   cell-friendly course workflow
├── projects/                   project write-ups and design notes
├── src/ibm_rag_agentic_showcase/
│   ├── restaurant_extraction.py
│   └── cli.py
├── tests/                      offline unit tests with a fake LLM
├── .env.example
└── pyproject.toml
~~~

## Engineering notes

- The extractor accepts any callable with the same system-and-user-message interface, keeping provider code outside the core validation loop.
- watsonx.ai calls use greedy decoding and a retry limit for provider failures.
- Output is written only after the complete collection has been validated.
- Source data is downloaded at runtime rather than redistributed.

## Responsible use

Schema validation proves that output has the expected shape; it does not prove that every extracted claim is factually correct. Review or ground model-created records before using them for recommendations, analytics, or user-facing decisions. Keep credentials in environment variables, use small smoke tests before bulk runs, and monitor inference cost.

## Acknowledgements

Built while completing IBM coursework in retrieval-augmented generation and agentic AI. The restaurant descriptions come from the IBM Skills Network course dataset and are downloaded at runtime.

## License

Code in this repository is available under the [MIT License](LICENSE).

## Showcase artwork

The custom hero illustrates the repository’s real extraction, validation, and bounded-repair flow. It contains no real restaurant data and makes no model-accuracy claim.
