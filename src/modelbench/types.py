"""Domain types for Text-to-SQL evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextToSQLSample:
    """A single Text-to-SQL evaluation sample.

    This is independent of any specific benchmark (Spider, BIRD, etc.)
    and represents the minimal information needed to evaluate a
    Text-to-SQL prediction.

    Attributes:
        question: Natural-language question.
        db_id: Identifier for the database (e.g., "ecommerce").
        db_path: Path to the SQLite database file.
        gold_sql: Reference SQL query that produces the correct result.
    """

    question: str
    db_id: str
    db_path: str
    gold_sql: str


@dataclass(frozen=True)
class QueryResult:
    """Result of executing a SQL query against a database.

    Attributes:
        columns: Column names from the result set.
        rows: Row tuples in retrieval order.
    """

    columns: tuple[str, ...]
    rows: list[tuple]


@dataclass(frozen=True)
class EvaluationResult:
    """Result of evaluating a predicted SQL query against a gold standard.

    Attributes:
        sql_valid: Whether the predicted SQL executed without error.
        exact_match: Whether the predicted SQL matches the gold SQL
            after normalization (whitespace, casing).
        execution_accuracy: Whether the predicted SQL produces the
            same result set as the gold SQL.
        execution_error: Error message if the predicted SQL failed
            to execute, None otherwise.
    """

    sql_valid: bool
    exact_match: bool
    execution_accuracy: bool
    execution_error: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    """Result of a single model generation call.

    Attributes:
        text: The generated text (raw model output before SQL extraction).
        latency_seconds: Wall-clock time for generation in seconds.
        input_tokens: Number of input tokens, if available.
        output_tokens: Number of generated tokens, if available.
    """

    text: str
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class ExperimentMetadata:
    """Metadata describing an experiment run."""

    experiment_name: str
    dataset: str
    split: str
    limit: int | None
    seed: int | None
    model_id: str
    model_revision: str
    schema_strategy: str
    prompting_strategy: str
    generation_config: dict[str, str | int | float | bool | None]


@dataclass(frozen=True)
class SampleResult:
    """Detailed result for a single sample in an experiment."""

    sample_id: str
    db_id: str
    question: str
    gold_sql: str
    generated_text: str
    extracted_sql: str | None
    sql_valid: bool
    exact_match: bool
    execution_accuracy: bool
    execution_error: str | None
    latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True)
class ExperimentResult:
    """The aggregate result of an experiment."""

    metadata: ExperimentMetadata
    total_samples: int
    valid_sql_count: int
    exact_match_count: int
    execution_correct_count: int
    sql_validity_rate: float
    exact_match_rate: float
    execution_accuracy: float
    avg_latency_seconds: float
    samples: list[SampleResult]
