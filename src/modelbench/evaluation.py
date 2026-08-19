"""Text-to-SQL evaluation logic.

This module implements deterministic evaluation of predicted SQL queries
against gold-standard references.  No LLM judge or external API is used —
evaluation is purely based on SQL execution and result comparison.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from modelbench.db import DatabaseExecutionError, execute_query
from modelbench.sql import normalize_sql
from modelbench.types import EvaluationResult, QueryResult

logger = logging.getLogger(__name__)


def results_match(
    gold: QueryResult,
    predicted: QueryResult,
    *,
    ordered: bool = False,
) -> bool:
    """Compare two SQL query results for equivalence.

    Comparison semantics:
      - Column count must be equal.
      - Column names are compared case-insensitively.
      - Row values are compared exactly (type and value).
      - If ``ordered=False`` (default), rows are compared as multisets:
        both row lists are sorted before comparison so retrieval order
        is ignored.
      - If ``ordered=True``, rows are compared in their original order.

    Use ``ordered=True`` when the gold SQL contains ORDER BY and the
    row ordering is semantically meaningful.

    Args:
        gold: Expected query result.
        predicted: Actual query result from the predicted SQL.
        ordered: Whether row order matters for this comparison.

    Returns:
        True if the results are equivalent under the chosen semantics.
    """
    # Column count
    if len(gold.columns) != len(predicted.columns):
        return False

    # Column names (case-insensitive)
    gold_cols = tuple(c.lower() for c in gold.columns)
    pred_cols = tuple(c.lower() for c in predicted.columns)
    if gold_cols != pred_cols:
        return False

    # Row values
    if ordered:
        return gold.rows == predicted.rows

    return sorted(gold.rows) == sorted(predicted.rows)


def _has_order_by(sql: str) -> bool:
    r"""Check whether a SQL string contains an ORDER BY clause.

    Uses a simple regex heuristic — looks for the token sequence
    ``ORDER BY`` with word boundaries.  This is not a full SQL parser
    and may produce false positives in rare edge cases (e.g. ORDER BY
    inside a string literal), but it is sufficient for benchmark
    evaluation.
    """
    return bool(re.search(r"\border\s+by\b", sql, re.IGNORECASE))


def evaluate_sample(
    predicted_sql: str,
    gold_sql: str,
    db_path: str | Path,
) -> EvaluationResult:
    """Evaluate a single predicted SQL query against a gold standard.

    Evaluation pipeline:
      1. **Exact match** — compare the normalized SQL strings.
      2. **SQL validity** — attempt to execute the predicted SQL.
      3. **Execution accuracy** — execute both queries against the
         same database and compare the result sets.

    If the gold SQL contains ``ORDER BY``, row ordering is considered
    significant during result comparison; otherwise rows are compared
    as unordered multisets.

    Args:
        predicted_sql: The SQL query produced by a model (or fixture).
        gold_sql: The reference SQL query.
        db_path: Path to the SQLite database to execute against.

    Returns:
        An :class:`EvaluationResult` with all metrics populated.

    Raises:
        ValueError: If the gold SQL itself fails to execute, which
            indicates a problem with the benchmark data rather than
            the prediction.
    """
    db_path = Path(db_path)

    # 1. Exact match (on normalised strings)
    exact_match = normalize_sql(predicted_sql) == normalize_sql(gold_sql)

    # 2. Execute gold SQL — must succeed
    try:
        gold_result = execute_query(db_path, gold_sql)
    except (DatabaseExecutionError, FileNotFoundError) as e:
        raise ValueError(f"Gold SQL failed to execute: {e}") from e

    # 3. Execute predicted SQL
    try:
        predicted_result = execute_query(db_path, predicted_sql)
    except DatabaseExecutionError as e:
        return EvaluationResult(
            sql_valid=False,
            exact_match=exact_match,
            execution_accuracy=False,
            execution_error=str(e),
        )

    # 4. Compare results
    ordered = _has_order_by(gold_sql)
    match = results_match(gold_result, predicted_result, ordered=ordered)

    return EvaluationResult(
        sql_valid=True,
        exact_match=exact_match,
        execution_accuracy=match,
        execution_error=None,
    )
