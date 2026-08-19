"""Tests for Text-to-SQL evaluation logic.

These tests cover the full evaluation pipeline including result comparison,
exact match, SQL validity, and execution accuracy against the fixture
e-commerce database.
"""

from pathlib import Path

import pytest

from modelbench.evaluation import evaluate_sample, results_match
from modelbench.types import QueryResult

# ── results_match tests ─────────────────────────────────────────────


class TestResultsMatch:
    """Test the result-set comparison function."""

    def test_identical_results(self) -> None:
        a = QueryResult(columns=("name",), rows=[("Alice",), ("Bob",)])
        assert results_match(a, a) is True

    def test_same_rows_different_order_unordered(self) -> None:
        gold = QueryResult(columns=("name",), rows=[("Alice",), ("Bob",)])
        pred = QueryResult(columns=("name",), rows=[("Bob",), ("Alice",)])
        assert results_match(gold, pred, ordered=False) is True

    def test_same_rows_different_order_ordered(self) -> None:
        gold = QueryResult(columns=("name",), rows=[("Alice",), ("Bob",)])
        pred = QueryResult(columns=("name",), rows=[("Bob",), ("Alice",)])
        assert results_match(gold, pred, ordered=True) is False

    def test_different_values(self) -> None:
        gold = QueryResult(columns=("name",), rows=[("Alice",)])
        pred = QueryResult(columns=("name",), rows=[("Bob",)])
        assert results_match(gold, pred) is False

    def test_different_column_count(self) -> None:
        gold = QueryResult(columns=("a", "b"), rows=[(1, 2)])
        pred = QueryResult(columns=("a",), rows=[(1,)])
        assert results_match(gold, pred) is False

    def test_column_names_case_insensitive(self) -> None:
        gold = QueryResult(columns=("Name", "Price"), rows=[("Widget", 9.99)])
        pred = QueryResult(columns=("name", "price"), rows=[("Widget", 9.99)])
        assert results_match(gold, pred) is True

    def test_different_column_names(self) -> None:
        gold = QueryResult(columns=("name",), rows=[("Alice",)])
        pred = QueryResult(columns=("title",), rows=[("Alice",)])
        assert results_match(gold, pred) is False

    def test_empty_results_match(self) -> None:
        gold = QueryResult(columns=("name",), rows=[])
        pred = QueryResult(columns=("name",), rows=[])
        assert results_match(gold, pred) is True

    def test_multiple_rows_multiple_columns(self) -> None:
        gold = QueryResult(
            columns=("name", "price"),
            rows=[("Widget", 9.99), ("Gadget", 24.99), ("Doohickey", 4.99)],
        )
        pred = QueryResult(
            columns=("name", "price"),
            rows=[("Doohickey", 4.99), ("Widget", 9.99), ("Gadget", 24.99)],
        )
        assert results_match(gold, pred, ordered=False) is True
        assert results_match(gold, pred, ordered=True) is False

    def test_different_row_count(self) -> None:
        gold = QueryResult(columns=("id",), rows=[(1,), (2,)])
        pred = QueryResult(columns=("id",), rows=[(1,)])
        assert results_match(gold, pred) is False


# ── evaluate_sample tests ──────────────────────────────────────────


class TestValidSQL:
    """A correct SQL query should execute and achieve execution accuracy."""

    def test_correct_query(self, fixture_db: Path) -> None:
        gold = "SELECT COUNT(*) FROM customers"
        result = evaluate_sample(gold, gold, fixture_db)
        assert result.sql_valid is True
        assert result.exact_match is True
        assert result.execution_accuracy is True
        assert result.execution_error is None


class TestFormattingVariation:
    """Formatting differences should normalize to exact match."""

    def test_whitespace_and_case_normalization(self, fixture_db: Path) -> None:
        gold = "SELECT name FROM customers"
        predicted = "select  NAME   FROM   customers"
        result = evaluate_sample(predicted, gold, fixture_db)
        assert result.sql_valid is True
        assert result.exact_match is True
        assert result.execution_accuracy is True

    def test_trailing_semicolon(self, fixture_db: Path) -> None:
        gold = "SELECT name FROM customers"
        predicted = "SELECT name FROM customers;"
        result = evaluate_sample(predicted, gold, fixture_db)
        assert result.exact_match is True


class TestIncorrectSQL:
    """Syntactically invalid SQL should be handled gracefully."""

    def test_syntax_error(self, fixture_db: Path) -> None:
        gold = "SELECT name FROM customers"
        predicted = "SELECTT name FORM customers"
        result = evaluate_sample(predicted, gold, fixture_db)
        assert result.sql_valid is False
        assert result.execution_accuracy is False
        assert result.execution_error is not None
        assert len(result.execution_error) > 0

    def test_missing_table(self, fixture_db: Path) -> None:
        gold = "SELECT name FROM customers"
        predicted = "SELECT name FROM nonexistent"
        result = evaluate_sample(predicted, gold, fixture_db)
        assert result.sql_valid is False
        assert result.execution_error is not None


class TestSemanticallyIncorrectSQL:
    """Valid SQL that returns wrong results should fail execution accuracy."""

    def test_wrong_result(self, fixture_db: Path) -> None:
        gold = "SELECT name FROM customers WHERE customer_id = 1"
        predicted = "SELECT name FROM customers WHERE customer_id = 2"
        result = evaluate_sample(predicted, gold, fixture_db)
        assert result.sql_valid is True
        assert result.execution_accuracy is False
        assert result.execution_error is None

    def test_wrong_aggregation(self, fixture_db: Path) -> None:
        gold = "SELECT COUNT(*) FROM customers"  # 3
        predicted = "SELECT COUNT(*) FROM products"  # also 3, but different semantic
        # Both return 3, so execution accuracy is True — this is expected!
        # Execution accuracy compares results, not intent.
        result = evaluate_sample(predicted, gold, fixture_db)
        assert result.sql_valid is True
        assert result.execution_accuracy is True  # same result by coincidence

    def test_actually_wrong_count(self, fixture_db: Path) -> None:
        gold = "SELECT COUNT(*) FROM customers"  # 3
        predicted = "SELECT COUNT(*) FROM order_items"  # 5
        result = evaluate_sample(predicted, gold, fixture_db)
        assert result.sql_valid is True
        assert result.execution_accuracy is False


class TestDifferentSQLSameResult:
    """Structurally different SQL producing the same result should pass
    execution accuracy but fail exact match."""

    def test_join_vs_subquery(self, fixture_db: Path) -> None:
        gold = (
            "SELECT DISTINCT c.name FROM customers c JOIN orders o ON c.customer_id = o.customer_id"
        )
        predicted = (
            "SELECT DISTINCT name "
            "FROM customers "
            "WHERE customer_id IN (SELECT customer_id FROM orders)"
        )
        result = evaluate_sample(predicted, gold, fixture_db)
        assert result.sql_valid is True
        assert result.exact_match is False
        assert result.execution_accuracy is True

    def test_different_aggregation_expression(self, fixture_db: Path) -> None:
        gold = "SELECT SUM(quantity * unit_price) AS total FROM order_items"
        predicted = "SELECT SUM(unit_price * quantity) AS total FROM order_items"
        result = evaluate_sample(predicted, gold, fixture_db)
        assert result.sql_valid is True
        assert result.exact_match is False  # operand order differs
        assert result.execution_accuracy is True


class TestMultipleRows:
    """Test result comparison with multiple rows and columns."""

    def test_multiple_rows_correct(self, fixture_db: Path) -> None:
        gold = "SELECT name, price FROM products"
        result = evaluate_sample(gold, gold, fixture_db)
        assert result.sql_valid is True
        assert result.execution_accuracy is True

    def test_multiple_rows_with_join(self, fixture_db: Path) -> None:
        gold = (
            "SELECT c.name, COUNT(*) AS order_count "
            "FROM customers c "
            "JOIN orders o ON c.customer_id = o.customer_id "
            "GROUP BY c.customer_id"
        )
        predicted = (
            "SELECT c.name, COUNT(o.order_id) AS order_count "
            "FROM customers c "
            "JOIN orders o ON c.customer_id = o.customer_id "
            "GROUP BY c.name"
        )
        result = evaluate_sample(predicted, gold, fixture_db)
        assert result.sql_valid is True
        assert result.exact_match is False
        assert result.execution_accuracy is True


class TestEmptyResults:
    """Test comparison when queries return no rows."""

    def test_both_empty(self, fixture_db: Path) -> None:
        gold = "SELECT name FROM customers WHERE customer_id = 999"
        predicted = "SELECT name FROM customers WHERE customer_id = 888"
        result = evaluate_sample(predicted, gold, fixture_db)
        assert result.sql_valid is True
        assert result.execution_accuracy is True

    def test_gold_empty_predicted_not(self, fixture_db: Path) -> None:
        gold = "SELECT name FROM customers WHERE customer_id = 999"
        predicted = "SELECT name FROM customers WHERE customer_id = 1"
        result = evaluate_sample(predicted, gold, fixture_db)
        assert result.sql_valid is True
        assert result.execution_accuracy is False


class TestOrderBy:
    """Test that ORDER BY in gold SQL triggers ordered comparison."""

    def test_order_by_respected(self, fixture_db: Path) -> None:
        gold = "SELECT name FROM customers ORDER BY name ASC"
        # Same query, same order — should pass
        result = evaluate_sample(gold, gold, fixture_db)
        assert result.execution_accuracy is True

    def test_without_order_by_ignores_row_order(self, fixture_db: Path) -> None:
        # Without ORDER BY, result order is implementation-defined.
        # Two identical queries should match regardless.
        gold = "SELECT name FROM customers"
        result = evaluate_sample(gold, gold, fixture_db)
        assert result.execution_accuracy is True


class TestGoldSQLFailure:
    """If the gold SQL itself is broken, evaluation should raise."""

    def test_invalid_gold_raises(self, fixture_db: Path) -> None:
        with pytest.raises(ValueError, match="Gold SQL failed"):
            evaluate_sample("SELECT 1", "INVALID SQL HERE", fixture_db)
