"""Tests for SQL extraction from model output."""

import pytest

from modelbench.extract import SQLExtractionError, extract_sql


class TestExtractSql:
    """Test SQL extraction from various model output formats."""

    def test_plain_sql(self) -> None:
        assert extract_sql("SELECT * FROM users") == "SELECT * FROM users"

    def test_sql_fenced_block(self) -> None:
        text = "Here is the query:\n```sql\nSELECT COUNT(*) FROM users\n```"
        assert extract_sql(text) == "SELECT COUNT(*) FROM users"

    def test_sql_fenced_block_case_insensitive(self) -> None:
        text = "```SQL\nSELECT 1\n```"
        assert extract_sql(text) == "SELECT 1"

    def test_generic_fenced_block(self) -> None:
        text = "```\nSELECT name FROM products\n```"
        assert extract_sql(text) == "SELECT name FROM products"

    def test_sql_fence_preferred_over_generic(self) -> None:
        text = "```\nwrong\n```\n```sql\nSELECT 1\n```"
        assert extract_sql(text) == "SELECT 1"

    def test_strips_whitespace(self) -> None:
        assert extract_sql("  SELECT 1  ") == "SELECT 1"

    def test_preserves_trailing_semicolon(self) -> None:
        """Semicolons are preserved here; normalization is a separate step."""
        assert extract_sql("SELECT 1;") == "SELECT 1;"

    def test_multiline_sql_in_fence(self) -> None:
        text = "```sql\nSELECT\n  name,\n  price\nFROM products\n```"
        expected = "SELECT\n  name,\n  price\nFROM products"
        assert extract_sql(text) == expected

    def test_text_around_fenced_block(self) -> None:
        text = "I think the answer is:\n```sql\nSELECT 1\n```\nHope this helps!"
        assert extract_sql(text) == "SELECT 1"

    def test_empty_raises(self) -> None:
        with pytest.raises(SQLExtractionError, match="Empty"):
            extract_sql("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(SQLExtractionError, match="Empty"):
            extract_sql("   \n\t  ")

    def test_raw_multiline_no_fence(self) -> None:
        """Without fences, the full text is returned."""
        text = "SELECT name\nFROM users\nWHERE id = 1"
        assert extract_sql(text) == text
