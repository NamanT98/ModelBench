# ModelBench

**LLM experimentation and evaluation platform.**

ModelBench is a framework for systematically comparing LLM approaches to structured tasks. It provides reproducible experiment pipelines, automated evaluation metrics, and clear comparisons across model configurations.

## Why Text-to-SQL?

Version 1 focuses exclusively on **Text-to-SQL** — translating natural language questions into executable SQL queries. This is an ideal first task because:

- **Objectively measurable**: SQL queries can be executed against databases, and results can be compared directly. No subjective evaluation needed.
- **Rich design space**: Prompting strategies (zero-shot, few-shot), retrieval-augmented generation (schema retrieval), and fine-tuning (QLoRA/PEFT) all apply, making it a strong testbed for comparing approaches.
- **Well-established benchmarks**: Datasets like Spider and BIRD provide standardized evaluation with known baselines.
- **Practical relevance**: Text-to-SQL has clear real-world applications in data analytics and business intelligence.

## Research Questions

ModelBench V1 is designed to answer:

| Question | Approach |
|---|---|
| Does fine-tuning improve Text-to-SQL accuracy? | Compare zero-shot vs. QLoRA fine-tuned models |
| Does schema retrieval help? | Compare with/without schema context retrieval |
| Does fine-tuning + retrieval outperform either alone? | Factorial comparison |
| How well do models generalize to unseen databases? | Cross-database evaluation splits |
| What are the practical trade-offs? | Measure latency, token usage, and context size alongside accuracy |

## Evaluation Metrics

- **Execution accuracy** (primary) — does the generated SQL produce the correct result?
- SQL validity — is the generated SQL syntactically correct?
- Exact match — does the SQL string match the gold standard?
- Component matching — are individual SQL clauses (SELECT, WHERE, etc.) correct?
- Schema retrieval recall — does the retriever find the relevant tables/columns?
- Latency and token usage

## Development Roadmap

| Milestone | Description | Status |
|---|---|---|
| **M0** | Project foundation, packaging, CLI, config, logging | ✅ Current |
| M1 | Dataset loading (Spider/BIRD), schema parsing, SQLite execution | Planned |
| M2 | Evaluation engine with execution accuracy | Planned |
| M3 | Model adapter for local inference (e.g., CodeLlama, StarCoder) | Planned |
| M4 | Prompt strategies (zero-shot, few-shot) | Planned |
| M5 | Schema retrieval pipeline | Planned |
| M6 | QLoRA/PEFT fine-tuning integration | Planned |
| M7 | Experiment runner and MLflow tracking | Planned |
| M8 | FastAPI backend + React frontend | Planned |

> **Note:** M0 is the project foundation only. It does not include any ML, evaluation, or inference code. Each milestone is designed to be independently testable.

## Quick Start

```bash
# Install in development mode
pip install -e ".[dev]"

# Run the CLI
modelbench --version
modelbench info

# Run tests
pytest

# Lint and format
ruff check src/ tests/
ruff format src/ tests/
```

## Project Structure

```
modelbench/
├── src/
│   └── modelbench/        # Core package
│       ├── __init__.py
│       ├── cli.py          # CLI entry point
│       ├── config.py       # Configuration management
│       └── logging.py      # Logging setup
├── tests/                  # Test suite
├── configs/                # Configuration files
├── datasets/               # Benchmark data (not tracked)
├── scripts/                # Utility scripts
├── pyproject.toml          # Project metadata and tooling config
└── README.md
```

## License

MIT
