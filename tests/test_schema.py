"""Tests for schema extraction from SQLite databases."""

from pathlib import Path

from modelbench.schema import extract_schema_from_db


class TestExtractSchema:
    """Test schema extraction against the fixture database."""

    def test_contains_all_tables(self, fixture_db: Path) -> None:
        schema = extract_schema_from_db(fixture_db)
        assert "Table customers:" in schema
        assert "Table products:" in schema
        assert "Table orders:" in schema
        assert "Table order_items:" in schema

    def test_contains_columns(self, fixture_db: Path) -> None:
        schema = extract_schema_from_db(fixture_db)
        assert "customer_id INTEGER" in schema
        assert "name TEXT" in schema
        assert "price REAL" in schema

    def test_marks_primary_keys(self, fixture_db: Path) -> None:
        schema = extract_schema_from_db(fixture_db)
        assert "customer_id INTEGER (PRIMARY KEY)" in schema
        assert "product_id INTEGER (PRIMARY KEY)" in schema

    def test_contains_foreign_keys(self, fixture_db: Path) -> None:
        schema = extract_schema_from_db(fixture_db)
        assert "FOREIGN KEY (customer_id) REFERENCES customers(customer_id)" in schema
        assert "FOREIGN KEY (order_id) REFERENCES orders(order_id)" in schema

    def test_deterministic(self, fixture_db: Path) -> None:
        """Same database should always produce the same schema text."""
        s1 = extract_schema_from_db(fixture_db)
        s2 = extract_schema_from_db(fixture_db)
        assert s1 == s2

    def test_missing_database(self, tmp_path: Path) -> None:
        import pytest

        with pytest.raises(FileNotFoundError):
            extract_schema_from_db(tmp_path / "missing.db")

    def test_returns_string(self, fixture_db: Path) -> None:
        schema = extract_schema_from_db(fixture_db)
        assert isinstance(schema, str)
        assert len(schema) > 0
