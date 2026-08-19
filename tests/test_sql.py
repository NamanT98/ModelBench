"""Tests for SQL normalization."""

from modelbench.sql import normalize_sql


class TestNormalizeSql:
    """Test the normalize_sql function."""

    def test_strips_whitespace(self) -> None:
        assert normalize_sql("  SELECT 1  ") == "select 1"

    def test_collapses_internal_whitespace(self) -> None:
        assert normalize_sql("SELECT   name   FROM   customers") == "select name from customers"

    def test_lowercases(self) -> None:
        assert normalize_sql("SELECT Name FROM Customers") == "select name from customers"

    def test_removes_trailing_semicolon(self) -> None:
        assert normalize_sql("SELECT 1;") == "select 1"

    def test_removes_trailing_semicolon_with_whitespace(self) -> None:
        assert normalize_sql("SELECT 1 ;  ") == "select 1"

    def test_handles_tabs_and_newlines(self) -> None:
        sql = "SELECT\n\tname,\n\tprice\nFROM\n\tproducts"
        assert normalize_sql(sql) == "select name, price from products"

    def test_combined_normalization(self) -> None:
        raw = "  SELECT   Name,  Price  FROM  Products  ; "
        expected = "select name, price from products"
        assert normalize_sql(raw) == expected

    def test_identical_after_normalization(self) -> None:
        a = "SELECT name FROM customers"
        b = "select  NAME   FROM   customers"
        assert normalize_sql(a) == normalize_sql(b)

    def test_different_queries_stay_different(self) -> None:
        a = "SELECT name FROM customers"
        b = "SELECT name FROM products"
        assert normalize_sql(a) != normalize_sql(b)

    def test_empty_string(self) -> None:
        assert normalize_sql("") == ""

    def test_only_whitespace(self) -> None:
        assert normalize_sql("   ") == ""
