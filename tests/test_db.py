"""Tests for the SQLite database executor."""

from pathlib import Path

import pytest

from modelbench.db import DatabaseExecutionError, execute_query


class TestExecuteQuery:
    """Test SQL query execution against the fixture database."""

    def test_simple_count(self, fixture_db: Path) -> None:
        result = execute_query(fixture_db, "SELECT COUNT(*) FROM customers")
        assert result.columns == ("COUNT(*)",)
        assert result.rows == [(3,)]

    def test_select_with_columns(self, fixture_db: Path) -> None:
        result = execute_query(fixture_db, "SELECT name, price FROM products ORDER BY product_id")
        assert result.columns == ("name", "price")
        assert result.rows == [
            ("Widget", 9.99),
            ("Gadget", 24.99),
            ("Doohickey", 4.99),
        ]

    def test_join_query(self, fixture_db: Path) -> None:
        sql = (
            "SELECT c.name, o.order_date "
            "FROM customers c "
            "JOIN orders o ON c.customer_id = o.customer_id "
            "ORDER BY o.order_id"
        )
        result = execute_query(fixture_db, sql)
        assert result.columns == ("name", "order_date")
        assert len(result.rows) == 3
        assert result.rows[0] == ("Alice", "2024-01-15")

    def test_aggregation(self, fixture_db: Path) -> None:
        sql = "SELECT SUM(quantity * unit_price) AS total FROM order_items"
        result = execute_query(fixture_db, sql)
        assert result.columns == ("total",)
        assert len(result.rows) == 1
        # 2*9.99 + 1*4.99 + 1*24.99 + 3*9.99 + 1*24.99 = 104.92
        assert result.rows[0][0] == pytest.approx(104.92)

    def test_empty_result(self, fixture_db: Path) -> None:
        result = execute_query(fixture_db, "SELECT * FROM customers WHERE customer_id = 999")
        assert result.rows == []
        assert len(result.columns) > 0  # columns still present

    def test_invalid_sql_raises(self, fixture_db: Path) -> None:
        with pytest.raises(DatabaseExecutionError, match="syntax error"):
            execute_query(fixture_db, "SELECTT name FORM customers")

    def test_missing_table_raises(self, fixture_db: Path) -> None:
        with pytest.raises(DatabaseExecutionError, match="no such table"):
            execute_query(fixture_db, "SELECT * FROM nonexistent_table")

    def test_missing_database_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Database not found"):
            execute_query(tmp_path / "missing.db", "SELECT 1")

    def test_read_only_prevents_writes(self, fixture_db: Path) -> None:
        with pytest.raises(DatabaseExecutionError):
            execute_query(fixture_db, "INSERT INTO customers VALUES (99, 'Mallory')")
