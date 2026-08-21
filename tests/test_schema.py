"""Tests for schema extraction, domain models, and strategy hierarchy."""

from pathlib import Path

import pytest

from modelbench.schema import (
    Column,
    DatabaseSchema,
    FKExpandedSchemaLinkingStrategy,
    ForeignKey,
    FullSchemaStrategy,
    SchemaLinkingStrategy,
    StructuredFullSchemaStrategy,
    Table,
    create_schema_strategy,
    extract_schema_from_db,
    introspect_database,
)

# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture()
def ecommerce_schema(fixture_db: Path) -> DatabaseSchema:
    """Introspect the fixture e-commerce database."""
    return introspect_database(fixture_db)


@pytest.fixture()
def multi_table_schema() -> DatabaseSchema:
    """A synthetic schema with a clear FK graph for linking tests.

    Schema:
        countries(country_id PK, country_name)
        cities(city_id PK, city_name, country_id FK->countries)
        airports(airport_id PK, airport_name, city_id FK->cities)
        airlines(airline_id PK, airline_name, country_id FK->countries)
        flights(flight_id PK, airline_id FK->airlines,
                source_airport FK->airports, dest_airport FK->airports)
    """
    return DatabaseSchema(
        tables=(
            Table(
                name="countries",
                columns=(
                    Column(name="country_id", col_type="INTEGER", is_primary_key=True),
                    Column(name="country_name", col_type="TEXT"),
                ),
            ),
            Table(
                name="cities",
                columns=(
                    Column(name="city_id", col_type="INTEGER", is_primary_key=True),
                    Column(name="city_name", col_type="TEXT"),
                    Column(name="country_id", col_type="INTEGER"),
                ),
                foreign_keys=(
                    ForeignKey(
                        from_table="cities",
                        from_column="country_id",
                        to_table="countries",
                        to_column="country_id",
                    ),
                ),
            ),
            Table(
                name="airports",
                columns=(
                    Column(name="airport_id", col_type="INTEGER", is_primary_key=True),
                    Column(name="airport_name", col_type="TEXT"),
                    Column(name="city_id", col_type="INTEGER"),
                ),
                foreign_keys=(
                    ForeignKey(
                        from_table="airports",
                        from_column="city_id",
                        to_table="cities",
                        to_column="city_id",
                    ),
                ),
            ),
            Table(
                name="airlines",
                columns=(
                    Column(name="airline_id", col_type="INTEGER", is_primary_key=True),
                    Column(name="airline_name", col_type="TEXT"),
                    Column(name="country_id", col_type="INTEGER"),
                ),
                foreign_keys=(
                    ForeignKey(
                        from_table="airlines",
                        from_column="country_id",
                        to_table="countries",
                        to_column="country_id",
                    ),
                ),
            ),
            Table(
                name="flights",
                columns=(
                    Column(name="flight_id", col_type="INTEGER", is_primary_key=True),
                    Column(name="airline_id", col_type="INTEGER"),
                    Column(name="source_airport", col_type="INTEGER"),
                    Column(name="dest_airport", col_type="INTEGER"),
                ),
                foreign_keys=(
                    ForeignKey(
                        from_table="flights",
                        from_column="airline_id",
                        to_table="airlines",
                        to_column="airline_id",
                    ),
                    ForeignKey(
                        from_table="flights",
                        from_column="source_airport",
                        to_table="airports",
                        to_column="airport_id",
                    ),
                    ForeignKey(
                        from_table="flights",
                        from_column="dest_airport",
                        to_table="airports",
                        to_column="airport_id",
                    ),
                ),
            ),
        )
    )


# ── Legacy Function ────────────────────────────────────────────────


class TestExtractSchema:
    """Test the legacy extract_schema_from_db function still works."""

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
        s1 = extract_schema_from_db(fixture_db)
        s2 = extract_schema_from_db(fixture_db)
        assert s1 == s2

    def test_missing_database(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            extract_schema_from_db(tmp_path / "missing.db")

    def test_returns_string(self, fixture_db: Path) -> None:
        schema = extract_schema_from_db(fixture_db)
        assert isinstance(schema, str)
        assert len(schema) > 0


# ── Domain Models ──────────────────────────────────────────────────


class TestIntrospection:
    """Test introspect_database produces correct domain objects."""

    def test_table_count(self, ecommerce_schema: DatabaseSchema) -> None:
        assert len(ecommerce_schema.tables) == 4

    def test_table_names_sorted(self, ecommerce_schema: DatabaseSchema) -> None:
        names = ecommerce_schema.table_names
        assert names == sorted(names)

    def test_column_types(self, ecommerce_schema: DatabaseSchema) -> None:
        customers = ecommerce_schema.get_table("customers")
        assert customers is not None
        col_names = [c.name for c in customers.columns]
        assert "customer_id" in col_names
        assert "name" in col_names

    def test_primary_keys(self, ecommerce_schema: DatabaseSchema) -> None:
        customers = ecommerce_schema.get_table("customers")
        assert customers is not None
        pks = [c for c in customers.columns if c.is_primary_key]
        assert len(pks) == 1
        assert pks[0].name == "customer_id"

    def test_foreign_keys(self, ecommerce_schema: DatabaseSchema) -> None:
        orders = ecommerce_schema.get_table("orders")
        assert orders is not None
        assert len(orders.foreign_keys) == 1
        fk = orders.foreign_keys[0]
        assert fk.from_column == "customer_id"
        assert fk.to_table == "customers"

    def test_missing_database(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            introspect_database(tmp_path / "missing.db")

    def test_total_column_count(self, ecommerce_schema: DatabaseSchema) -> None:
        total = ecommerce_schema.total_column_count
        assert total > 0

    def test_get_table_none_for_missing(self, ecommerce_schema: DatabaseSchema) -> None:
        assert ecommerce_schema.get_table("nonexistent") is None


# ── Strategy: Full ─────────────────────────────────────────────────


class TestFullSchemaStrategy:
    """Test the legacy full schema strategy."""

    def test_contains_all_tables(self, ecommerce_schema: DatabaseSchema) -> None:
        strategy = FullSchemaStrategy()
        result = strategy.get_schema_string(ecommerce_schema, "anything")
        assert "Table customers:" in result
        assert "Table products:" in result
        assert "Table orders:" in result
        assert "Table order_items:" in result

    def test_deterministic(self, ecommerce_schema: DatabaseSchema) -> None:
        strategy = FullSchemaStrategy()
        r1 = strategy.get_schema_string(ecommerce_schema, "q1")
        r2 = strategy.get_schema_string(ecommerce_schema, "q2")
        assert r1 == r2

    def test_diagnostics_no_reduction(self, ecommerce_schema: DatabaseSchema) -> None:
        strategy = FullSchemaStrategy()
        strategy.get_schema_string(ecommerce_schema, "test")
        diag = strategy.get_diagnostics()
        assert diag.schema_reduction_ratio == 0.0
        assert diag.linking_success is True
        assert diag.fallback_used is False
        assert diag.selected_table_count == diag.original_table_count


# ── Strategy: Structured Full (M4-A) ──────────────────────────────


class TestStructuredFullSchemaStrategy:
    """Test the structured full schema representation."""

    def test_contains_database_schema_header(self, ecommerce_schema: DatabaseSchema) -> None:
        strategy = StructuredFullSchemaStrategy()
        result = strategy.get_schema_string(ecommerce_schema, "test")
        assert result.startswith("DATABASE SCHEMA")

    def test_contains_table_prefix(self, ecommerce_schema: DatabaseSchema) -> None:
        strategy = StructuredFullSchemaStrategy()
        result = strategy.get_schema_string(ecommerce_schema, "test")
        assert "TABLE: customers" in result
        assert "TABLE: products" in result

    def test_contains_column_dashes(self, ecommerce_schema: DatabaseSchema) -> None:
        strategy = StructuredFullSchemaStrategy()
        result = strategy.get_schema_string(ecommerce_schema, "test")
        assert "  - customer_id INTEGER [PRIMARY KEY]" in result

    def test_inline_fk_annotation(self, ecommerce_schema: DatabaseSchema) -> None:
        strategy = StructuredFullSchemaStrategy()
        result = strategy.get_schema_string(ecommerce_schema, "test")
        assert "FOREIGN KEY → customers.customer_id" in result

    def test_deterministic(self, ecommerce_schema: DatabaseSchema) -> None:
        strategy = StructuredFullSchemaStrategy()
        r1 = strategy.get_schema_string(ecommerce_schema, "q1")
        r2 = strategy.get_schema_string(ecommerce_schema, "q2")
        assert r1 == r2

    def test_diagnostics_no_reduction(self, ecommerce_schema: DatabaseSchema) -> None:
        strategy = StructuredFullSchemaStrategy()
        strategy.get_schema_string(ecommerce_schema, "test")
        diag = strategy.get_diagnostics()
        assert diag.schema_reduction_ratio == 0.0
        assert diag.linking_success is True
        assert diag.fallback_used is False


# ── Strategy: Schema Linking (M4-B) ───────────────────────────────


class TestSchemaLinkingStrategy:
    """Test the deterministic lexical schema linker."""

    def test_selects_relevant_tables(self, ecommerce_schema: DatabaseSchema) -> None:
        strategy = SchemaLinkingStrategy()
        result = strategy.get_schema_string(ecommerce_schema, "How many customers are there?")
        assert "TABLE: customers" in result

    def test_excludes_irrelevant_tables(self, ecommerce_schema: DatabaseSchema) -> None:
        strategy = SchemaLinkingStrategy()
        result = strategy.get_schema_string(ecommerce_schema, "How many customers are there?")
        assert "TABLE: products" not in result

    def test_column_name_matching(self, ecommerce_schema: DatabaseSchema) -> None:
        strategy = SchemaLinkingStrategy()
        result = strategy.get_schema_string(ecommerce_schema, "What is the total price?")
        assert "TABLE: products" in result
        assert "price" in result

    def test_no_match_returns_empty_schema(self, ecommerce_schema: DatabaseSchema) -> None:
        strategy = SchemaLinkingStrategy()
        result = strategy.get_schema_string(ecommerce_schema, "What is the weather like today?")
        assert "No schema elements matched" in result

    def test_no_match_linking_success_false(self, ecommerce_schema: DatabaseSchema) -> None:
        strategy = SchemaLinkingStrategy()
        strategy.get_schema_string(ecommerce_schema, "What is the weather like today?")
        diag = strategy.get_diagnostics()
        assert diag.linking_success is False
        assert diag.fallback_used is False

    def test_no_silent_fallback(self, ecommerce_schema: DatabaseSchema) -> None:
        """Strict: never silently falls back to full schema."""
        strategy = SchemaLinkingStrategy()
        strategy.get_schema_string(ecommerce_schema, "What is the weather like today?")
        diag = strategy.get_diagnostics()
        assert diag.fallback_used is False
        assert diag.selected_table_count == 0

    def test_schema_reduction_ratio(self, ecommerce_schema: DatabaseSchema) -> None:
        strategy = SchemaLinkingStrategy()
        strategy.get_schema_string(ecommerce_schema, "How many customers are there?")
        diag = strategy.get_diagnostics()
        assert diag.schema_reduction_ratio > 0.0
        assert diag.selected_table_count < diag.original_table_count

    def test_deterministic(self, ecommerce_schema: DatabaseSchema) -> None:
        strategy = SchemaLinkingStrategy()
        r1 = strategy.get_schema_string(ecommerce_schema, "How many customers?")
        r2 = strategy.get_schema_string(ecommerce_schema, "How many customers?")
        assert r1 == r2

    def test_fk_preserved_between_selected(self, ecommerce_schema: DatabaseSchema) -> None:
        """FK annotations should be present when both tables are selected."""
        strategy = SchemaLinkingStrategy()
        result = strategy.get_schema_string(ecommerce_schema, "Show customer orders")
        # Both customers and orders should be selected
        assert "TABLE: customers" in result
        assert "TABLE: orders" in result


# ── Strategy: FK Expansion (M4-C) ─────────────────────────────────


class TestFKExpandedSchemaLinkingStrategy:
    """Test schema linking with FK-graph expansion."""

    def test_expands_intermediate_tables(self, multi_table_schema: DatabaseSchema) -> None:
        """If we mention 'flights', FK expansion should include airlines and airports."""
        strategy = FKExpandedSchemaLinkingStrategy(max_fk_depth=1)
        result = strategy.get_schema_string(multi_table_schema, "How many flights are there?")
        assert "TABLE: flights" in result
        # FK neighbors at depth 1
        assert "TABLE: airlines" in result
        assert "TABLE: airports" in result

    def test_records_fk_expanded_tables(self, multi_table_schema: DatabaseSchema) -> None:
        strategy = FKExpandedSchemaLinkingStrategy(max_fk_depth=1)
        strategy.get_schema_string(multi_table_schema, "How many flights are there?")
        diag = strategy.get_diagnostics()
        assert len(diag.fk_expanded_tables) > 0
        assert "airlines" in diag.fk_expanded_tables
        assert "airports" in diag.fk_expanded_tables

    def test_respects_max_depth(self, multi_table_schema: DatabaseSchema) -> None:
        """Depth 1 from 'flights' should NOT reach 'countries' (which is 2 hops away)."""
        strategy = FKExpandedSchemaLinkingStrategy(max_fk_depth=1)
        result = strategy.get_schema_string(multi_table_schema, "How many flights are there?")
        # countries is 2 hops from flights (flights->airlines->countries)
        assert "TABLE: countries" not in result

    def test_depth_2_reaches_further(self, multi_table_schema: DatabaseSchema) -> None:
        """Depth 2 from 'flights' should reach 'countries' via airlines."""
        strategy = FKExpandedSchemaLinkingStrategy(max_fk_depth=2)
        result = strategy.get_schema_string(multi_table_schema, "How many flights are there?")
        assert "TABLE: countries" in result

    def test_no_match_no_expansion(self, multi_table_schema: DatabaseSchema) -> None:
        strategy = FKExpandedSchemaLinkingStrategy(max_fk_depth=1)
        strategy.get_schema_string(multi_table_schema, "weather forecast today")
        diag = strategy.get_diagnostics()
        assert diag.linking_success is False
        assert diag.fallback_used is False
        assert len(diag.fk_expanded_tables) == 0

    def test_diagnostics_reduction(self, multi_table_schema: DatabaseSchema) -> None:
        strategy = FKExpandedSchemaLinkingStrategy(max_fk_depth=1)
        strategy.get_schema_string(multi_table_schema, "How many flights are there?")
        diag = strategy.get_diagnostics()
        assert diag.schema_reduction_ratio >= 0.0
        assert diag.selected_table_count <= diag.original_table_count

    def test_deterministic(self, multi_table_schema: DatabaseSchema) -> None:
        strategy = FKExpandedSchemaLinkingStrategy(max_fk_depth=1)
        r1 = strategy.get_schema_string(multi_table_schema, "flights count")
        r2 = strategy.get_schema_string(multi_table_schema, "flights count")
        assert r1 == r2

    def test_expanded_tables_get_full_columns(self, multi_table_schema: DatabaseSchema) -> None:
        """FK-expanded tables should include all their columns, not just matched ones."""
        strategy = FKExpandedSchemaLinkingStrategy(max_fk_depth=1)
        result = strategy.get_schema_string(multi_table_schema, "How many flights are there?")
        # airlines is FK-expanded, should have all its columns
        assert "airline_name" in result


# ── Strategy Factory ───────────────────────────────────────────────


class TestCreateSchemaStrategy:
    """Test the factory function."""

    def test_creates_full(self) -> None:
        s = create_schema_strategy("full")
        assert isinstance(s, FullSchemaStrategy)

    def test_creates_structured_full(self) -> None:
        s = create_schema_strategy("structured_full")
        assert isinstance(s, StructuredFullSchemaStrategy)

    def test_creates_schema_linking(self) -> None:
        s = create_schema_strategy("schema_linking")
        assert isinstance(s, SchemaLinkingStrategy)

    def test_creates_schema_linking_fk(self) -> None:
        s = create_schema_strategy("schema_linking_fk", max_fk_depth=2)
        assert isinstance(s, FKExpandedSchemaLinkingStrategy)
        assert s.max_fk_depth == 2

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown schema strategy"):
            create_schema_strategy("nonexistent")

    def test_default_fk_depth(self) -> None:
        s = create_schema_strategy("schema_linking_fk")
        assert isinstance(s, FKExpandedSchemaLinkingStrategy)
        assert s.max_fk_depth == 1
