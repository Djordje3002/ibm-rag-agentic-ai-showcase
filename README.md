# IBM RAG & Agentic AI Showcase

Hands-on projects from IBM's RAG and Agentic AI coursework, rebuilt as a
portfolio-ready collection of reproducible AI engineering patterns.

> **Status:** Labs 01 through 04 are ready. More RAG and agentic workflows will
> be added as the course progresses.

## Projects

| # | Project | Concepts | Status |
|---|---|---|---|
| 01 | [Structured restaurant extraction](projects/01-structured-restaurant-extraction.md) | IBM Granite, one-shot prompting, Pydantic validation, JSON self-repair | Complete |
| 02 | [Multimodal food-data augmentation](projects/02-multimodal-food-data-augmentation.md) | Vision-language models, image captioning, contextual enrichment, resilient downloads | Complete |
| 03 | [Safe restaurant database](projects/03-safe-restaurant-database.md) | CRUD, LLM-assisted entry, typed edits, backups, unit testing | Complete |
| 04 | [Multimodal vector index](projects/04-multimodal-vector-index.md) | Sentence-Transformers, CLIP, LangChain documents, persistent Chroma collections | Complete |
| 05 | Retrieval-augmented generation | Chunking, vector search, grounded answers | Planned |
| 06 | Agentic AI workflow | Tools, planning, memory, multi-step execution | Planned |

A printable, chapter-by-chapter explanation is maintained in
[the LaTeX lab guide](docs/lab-guide.tex).

## Project 01: From unstructured text to reliable JSON

The first project converts free-form restaurant descriptions into typed,
machine-readable records. It uses an IBM Granite model for extraction, validates
every response with Pydantic, and asks the model to repair malformed output
within a bounded retry loop.

```mermaid
flowchart LR
    A["Restaurant descriptions"] --> B["One-shot extraction prompt"]
    B --> C["IBM Granite on watsonx.ai"]
    C --> D{"Pydantic validation"}
    D -->|Valid| E["Structured JSON dataset"]
    D -->|Invalid| F["Targeted repair prompt"]
    F --> C
```

### What this demonstrates

- Constrained generation with an explicit JSON schema
- One-shot prompting with a representative example
- Runtime validation rather than trusting model output
- Automatic repair informed by exact validation errors
- Bounded retries and clear failure behavior
- Separation between model access, extraction logic, and file I/O

## Quick start

Prerequisites:

- Python 3.11+
- An IBM watsonx.ai project and API key, or the IBM Skills Network lab runtime

```bash
git clone https://github.com/Djordje3002/ibm-rag-agentic-ai-showcase.git
cd ibm-rag-agentic-ai-showcase

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,labs]"

cp .env.example .env
# Add your watsonx.ai values, then export them into your shell.

restaurant-extract --limit 3
```

By default, the command downloads the course dataset and writes the validated
records to `data/processed/structured_restaurant_data.json`. Generated data is
ignored by Git so that the repository stays small and reproducible.

## Repository layout

```text
.
├── examples/                    # Course concepts in a step-by-step script
├── docs/lab-guide.tex           # One evolving explanation of every lab
├── projects/                    # Project write-ups and design notes
├── src/ibm_rag_agentic_showcase/
│   ├── restaurant_extraction.py # Schema, prompts, validation, repair pipeline
│   ├── multimodal_augmentation.py # Vision captioning and data enrichment
│   ├── restaurant_database.py   # Safe CRUD and terminal interface
│   ├── multimodal_vector_index.py # Text/image embeddings and Chroma
│   └── cli.py                   # Reproducible command-line entry point
└── tests/                       # Offline unit tests with a fake LLM
```

## Run the checks

```bash
pytest
```

The test suite does not call watsonx.ai, so it is fast, deterministic, and does
not consume inference credits.

Lab 04 uses larger local ML dependencies. Install a CPU build of PyTorch first
on Linux, then install the vector extra:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[vector,dev]"
```

On macOS, the regular `pip install -e ".[vector,dev]"` command installs the
native PyTorch build.

To build the lab guide when a LaTeX distribution is installed:

```bash
pdflatex -output-directory docs docs/lab-guide.tex
pdflatex -output-directory docs docs/lab-guide.tex
```

## Responsible use

Model output is treated as untrusted input. The pipeline validates all records
before saving them, caps repair attempts, and raises a clear error if a response
still does not satisfy the schema. API keys belong in environment variables and
must never be committed.

## Acknowledgements

Built while completing IBM coursework in retrieval-augmented generation and
agentic AI. The restaurant descriptions are provided by the IBM Skills Network
course dataset and are downloaded at runtime rather than redistributed here.

## License

Code in this repository is available under the [MIT License](LICENSE).
