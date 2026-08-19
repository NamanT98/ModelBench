# ModelBench

**LLM experimentation and evaluation platform.**

ModelBench is a framework for systematically comparing LLM approaches to structured tasks. It provides reproducible experiment pipelines, automated evaluation metrics, and clear comparisons across model configurations.

## Why Text-to-SQL?

Version 1 focuses exclusively on **Text-to-SQL** — translating natural language questions into executable SQL queries. This is an ideal first task because:

- **Objectively measurable**: SQL queries can be executed against databases, and results can be compared directly. No subjective evaluation needed.
- **Rich design space**: Prompting strategies (zero-shot, few-shot), retrieval-augmented generation (schema retrieval), and fine-tuning (QLoRA/PEFT) all apply, making it a strong testbed for comparing approaches.
- **Well-established benchmarks**: Datasets like Spider and BIRD provide standardized evaluation with known baselines.
- **Practical relevance**: Text-to-SQL has clear real-world applications in data analytics and business intelligence.

## Evaluation: Why Execution Accuracy?

ModelBench uses **execution accuracy** as its primary evaluation metric. This means: we execute both the predicted SQL and the gold-standard SQL against the same database, then compare the result sets.

This is more meaningful than **exact match** (comparing SQL strings) because many different SQL queries can produce the same correct result:

```
Gold:      SELECT DISTINCT c.name FROM customers c JOIN orders o ON c.customer_id = o.customer_id
Predicted: SELECT DISTINCT name FROM customers WHERE customer_id IN (SELECT customer_id FROM orders)
```

These two queries are textually different but semantically equivalent — both correctly answer "which customers have placed orders?" Exact match would score this as a failure; execution accuracy correctly scores it as a success.

### Text-to-SQL Evaluation Flow

```
Predicted SQL ──→ Execute against SQLite ──→ Predicted Result
                                                    │
                                                    ▼
                                              Compare Results ──→ Execution Accuracy
                                                    ▲
                                                    │
Gold SQL ────────→ Execute against SQLite ──→ Gold Result
```

Additional metrics are also computed:
- **SQL validity** — does the predicted SQL execute without error?
- **Exact match** — does the normalized SQL string match the gold standard?

## Research Questions

ModelBench V1 is designed to answer:

| Question | Approach |
|---|---|
| Does fine-tuning improve Text-to-SQL accuracy? | Compare zero-shot vs. QLoRA fine-tuned models |
| Does schema retrieval help? | Compare with/without schema context retrieval |
| Does fine-tuning + retrieval outperform either alone? | Factorial comparison |
| How well do models generalize to unseen databases? | Cross-database evaluation splits |
| What are the practical trade-offs? | Measure latency, token usage, and context size alongside accuracy |

## Development Roadmap

| Milestone | Description | Status |
|---|---|---|
| **M0** | Project foundation, packaging, CLI, config, logging | ✅ Complete |
| **M1** | SQLite execution, SQL normalization, evaluation engine, fixture benchmark | ✅ Current |
| M2 | Dataset loading (Spider/BIRD), schema parsing | Planned |
| M3 | Model adapter for local inference (e.g., CodeLlama, StarCoder) | Planned |
| M4 | Prompt strategies (zero-shot, few-shot) | Planned |
| M5 | Schema retrieval pipeline | Planned |
| M6 | QLoRA/PEFT fine-tuning integration | Planned |
| M7 | Experiment runner and MLflow tracking | Planned |
| M8 | FastAPI backend + React frontend | Planned |

> **Note:** The fixture evaluation included in M1 uses a small, internal e-commerce database. It is **not** a real benchmark like Spider or BIRD — it exists to verify the evaluation pipeline works correctly during development.

## Quick Start

```bash
# Install in development mode
pip install -e ".[dev]"

# Run the CLI
modelbench --version
modelbench info

# Run the fixture evaluation
modelbench evaluate-fixture

# Run tests
pytest

# Lint and format
ruff check src/ tests/
ruff format src/ tests/
```

### Example: Fixture Evaluation Output

```
ModelBench Text-to-SQL Fixture Evaluation
============================================

  [✓] 1. How many customers are there?
  [✓] 2. What is the total revenue from all orders?
  [✓] 3. Which customers have placed at least one order?
  [✓] 4. How many orders has each customer placed?
  [✓] 5. List all product names and prices.

Samples:            5
SQL Validity:       5/5 (100%)
Exact Match:        2/5 (40%)
Execution Accuracy: 5/5 (100%)
```

Notice that exact match is only 40% while execution accuracy is 100% — three of the predicted queries use different SQL syntax but produce the same correct result. This is precisely why execution accuracy is the primary metric.

## Project Structure

```
modelbench/
├── src/
│   └── modelbench/
│       ├── __init__.py        # Package init, version
│       ├── cli.py             # CLI entry point (click)
│       ├── config.py          # YAML configuration management
│       ├── logging.py         # Logging setup
│       ├── types.py           # Domain types (TextToSQLSample, EvaluationResult, etc.)
│       ├── db.py              # SQLite query execution
│       ├── sql.py             # SQL normalization for exact match
│       ├── evaluation.py      # Evaluation engine (result comparison, evaluate_sample)
│       └── fixture.py         # Fixture database and sample definitions
├── tests/
│   ├── conftest.py            # Shared test fixtures
│   ├── test_cli.py
│   ├── test_config.py
│   ├── test_logging.py
│   ├── test_db.py             # Database executor tests
│   ├── test_sql.py            # SQL normalization tests
│   └── test_evaluation.py     # Evaluation pipeline tests
├── configs/
├── datasets/
├── scripts/
├── pyproject.toml
└── README.md
```

## License

MIT
