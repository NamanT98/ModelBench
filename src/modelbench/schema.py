"""Schema extraction, domain models, and strategy hierarchy.

Provides:
- Domain models for representing database schemas as Python objects.
- An introspection function that reads SQLite PRAGMA data into domain objects.
- A hierarchy of SchemaStrategy implementations that convert domain objects
  into LLM-ready prompt strings.
- The legacy ``extract_schema_from_db`` function for backward compatibility.
"""

from __future__ import annotations

import re
import sqlite3
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except ImportError:
    nltk = None
    stopwords = None
    WordNetLemmatizer = None

# ── Domain Models ───────────────────────────────────────────────────


@dataclass(frozen=True)
class Column:
    """A single column in a database table."""

    name: str
    col_type: str
    is_primary_key: bool = False


@dataclass(frozen=True)
class ForeignKey:
    """A foreign-key relationship."""

    from_table: str
    from_column: str
    to_table: str
    to_column: str


@dataclass(frozen=True)
class Table:
    """A database table with columns and outgoing foreign keys."""

    name: str
    columns: tuple[Column, ...]
    foreign_keys: tuple[ForeignKey, ...] = ()


@dataclass(frozen=True)
class DatabaseSchema:
    """Complete schema for a database."""

    tables: tuple[Table, ...]

    @property
    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]

    def get_table(self, name: str) -> Table | None:
        for t in self.tables:
            if t.name == name:
                return t
        return None

    @property
    def all_foreign_keys(self) -> list[ForeignKey]:
        fks: list[ForeignKey] = []
        for t in self.tables:
            fks.extend(t.foreign_keys)
        return fks

    @property
    def total_column_count(self) -> int:
        return sum(len(t.columns) for t in self.tables)


# ── Introspection ──────────────────────────────────────────────────


def introspect_database(db_path: str | Path) -> DatabaseSchema:
    """Read a SQLite database and return its schema as domain objects.

    Args:
        db_path: Path to a SQLite database file.

    Returns:
        A DatabaseSchema containing all tables, columns, and foreign keys.

    Raises:
        FileNotFoundError: If the database file does not exist.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        table_names = _get_table_names(conn)
        tables: list[Table] = []
        for tname in table_names:
            columns = _get_columns(conn, tname)
            fks = _get_foreign_keys(conn, tname)
            tables.append(Table(name=tname, columns=tuple(columns), foreign_keys=tuple(fks)))
        return DatabaseSchema(tables=tuple(tables))
    finally:
        conn.close()


def _get_table_names(conn: sqlite3.Connection) -> list[str]:
    """Return sorted table names, excluding SQLite internal tables."""
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name"
    )
    return [row[0] for row in cursor.fetchall()]


def _get_columns(conn: sqlite3.Connection, table: str) -> list[Column]:
    """Return columns for a table via PRAGMA table_info."""
    cursor = conn.execute(f"PRAGMA table_info({table})")
    cols = []
    for row in cursor.fetchall():
        # row: (cid, name, type, notnull, dflt_value, pk)
        cols.append(
            Column(
                name=row[1],
                col_type=row[2] or "ANY",
                is_primary_key=bool(row[5]),
            )
        )
    return cols


def _get_foreign_keys(conn: sqlite3.Connection, table: str) -> list[ForeignKey]:
    """Return foreign keys for a table via PRAGMA foreign_key_list."""
    cursor = conn.execute(f"PRAGMA foreign_key_list({table})")
    fks = []
    for fk in cursor.fetchall():
        # fk: (id, seq, ref_table, from_col, to_col, ...)
        fks.append(
            ForeignKey(
                from_table=table,
                from_column=fk[3],
                to_table=fk[2],
                to_column=fk[4],
            )
        )
    return fks


# ── Legacy function (backward compatibility) ───────────────────────


def extract_schema_from_db(db_path: str | Path) -> str:
    """Extract a text description of the database schema.

    This is the original M2 function preserved for backward compatibility.
    It returns the same unstructured format used by FullSchemaStrategy.

    Args:
        db_path: Path to a SQLite database file.

    Returns:
        A multi-line string describing every table, column, and
        foreign-key relationship in the database.

    Raises:
        FileNotFoundError: If the database file does not exist.
    """
    schema = introspect_database(db_path)
    strategy = FullSchemaStrategy()
    return strategy.get_schema_string(schema, question="")


# ── Schema Diagnostics ─────────────────────────────────────────────


@dataclass(frozen=True)
class SchemaDiagnostics:
    """Diagnostic information about schema strategy execution."""

    original_table_count: int
    selected_table_count: int
    selected_column_count: int
    schema_string_length: int
    schema_reduction_ratio: float
    linking_success: bool
    fallback_used: bool
    fk_expanded_tables: tuple[str, ...] = ()


# ── Strategy Hierarchy ─────────────────────────────────────────────


class SchemaStrategy(ABC):
    """Abstract base class for schema-to-string strategies."""

    @abstractmethod
    def get_schema_string(self, db_schema: DatabaseSchema, question: str) -> str:
        """Convert a DatabaseSchema into a prompt-ready string.

        Args:
            db_schema: The full database schema as domain objects.
            question: The natural-language question (used by linking strategies).

        Returns:
            A string representation of the schema for the LLM prompt.
        """

    @abstractmethod
    def get_diagnostics(self) -> SchemaDiagnostics:
        """Return diagnostics from the most recent get_schema_string call."""


class FullSchemaStrategy(SchemaStrategy):
    """Reproduce the original M2/M3 unstructured schema format."""

    def __init__(self) -> None:
        self._diagnostics: SchemaDiagnostics | None = None

    def get_schema_string(self, db_schema: DatabaseSchema, question: str) -> str:
        sections: list[str] = []
        for table in db_schema.tables:
            lines = [f"Table {table.name}:"]
            for col in table.columns:
                pk_suffix = " (PRIMARY KEY)" if col.is_primary_key else ""
                lines.append(f"  {col.name} {col.col_type}{pk_suffix}")
            for fk in table.foreign_keys:
                fk_str = (
                    f"  FOREIGN KEY ({fk.from_column}) REFERENCES {fk.to_table}({fk.to_column})"
                )
                lines.append(fk_str)
            sections.append("\n".join(lines))

        result = "\n\n".join(sections)
        total_cols = db_schema.total_column_count
        self._diagnostics = SchemaDiagnostics(
            original_table_count=len(db_schema.tables),
            selected_table_count=len(db_schema.tables),
            selected_column_count=total_cols,
            schema_string_length=len(result),
            schema_reduction_ratio=0.0,
            linking_success=True,
            fallback_used=False,
        )
        return result

    def get_diagnostics(self) -> SchemaDiagnostics:
        if self._diagnostics is None:
            raise RuntimeError("get_schema_string must be called before get_diagnostics")
        return self._diagnostics


class StructuredFullSchemaStrategy(SchemaStrategy):
    """M4-A: Structured representation of the full database schema.

    Produces a deterministic, structured format:

        DATABASE SCHEMA

        TABLE: <table_name>
          - <column_name> <type> [PRIMARY KEY]
          - <column_name> <type> [FOREIGN KEY → other_table.other_column]
    """

    def __init__(self) -> None:
        self._diagnostics: SchemaDiagnostics | None = None

    def get_schema_string(self, db_schema: DatabaseSchema, question: str) -> str:
        # Build a lookup: (table, column) -> FK target for inline annotation
        fk_map: dict[tuple[str, str], str] = {}
        for table in db_schema.tables:
            for fk in table.foreign_keys:
                fk_map[(table.name, fk.from_column)] = f"{fk.to_table}.{fk.to_column}"

        sections: list[str] = ["DATABASE SCHEMA", ""]
        for table in db_schema.tables:
            sections.append(f"TABLE: {table.name}")
            for col in table.columns:
                annotations: list[str] = []
                if col.is_primary_key:
                    annotations.append("PRIMARY KEY")
                fk_target = fk_map.get((table.name, col.name))
                if fk_target:
                    annotations.append(f"FOREIGN KEY → {fk_target}")
                suffix = f" [{', '.join(annotations)}]" if annotations else ""
                sections.append(f"  - {col.name} {col.col_type}{suffix}")
            sections.append("")

        result = "\n".join(sections).rstrip()
        total_cols = db_schema.total_column_count
        self._diagnostics = SchemaDiagnostics(
            original_table_count=len(db_schema.tables),
            selected_table_count=len(db_schema.tables),
            selected_column_count=total_cols,
            schema_string_length=len(result),
            schema_reduction_ratio=0.0,
            linking_success=True,
            fallback_used=False,
        )
        return result

    def get_diagnostics(self) -> SchemaDiagnostics:
        if self._diagnostics is None:
            raise RuntimeError("get_schema_string must be called before get_diagnostics")
        return self._diagnostics


# ── Lexical Linking Utilities ──────────────────────────────────────


def _tokenize_question(question: str) -> set[str]:
    """Tokenize and normalize a natural-language question.

    Splits on non-alphanumeric boundaries, lowercases, and removes
    common stop words that would cause false-positive schema matches.
    """
    tokens = set(re.findall(r"[a-zA-Z0-9]+", question.lower()))
    # Minimal stop-word set to avoid over-matching
    stop_words = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "will",
        "would",
        "can",
        "could",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "by",
        "with",
        "from",
        "and",
        "or",
        "not",
        "no",
        "if",
        "but",
        "so",
        "than",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "where",
        "when",
        "why",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "i",
        "me",
        "my",
        "we",
        "us",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "they",
        "them",
        "their",
        "list",
        "find",
        "show",
        "give",
        "get",
        "return",
        "display",
        "many",
        "much",
        "any",
    }
    return tokens - stop_words


def _tokenize_name(name: str) -> set[str]:
    """Tokenize a schema identifier (table or column name).

    Splits on underscores and lowercases.
    """
    return set(name.lower().split("_"))


def _name_matches_tokens(name: str, question_tokens: set[str]) -> bool:
    """Check whether a schema name has overlap with question tokens.

    Matches if:
    - The full lowered name appears in question tokens, OR
    - Any individual token from the name appears in question tokens.
    """
    lowered = name.lower()
    if lowered in question_tokens:
        return True
    name_tokens = _tokenize_name(name)
    return bool(name_tokens & question_tokens)


class SchemaLinkingStrategy(SchemaStrategy):
    """M4-B: Deterministic lexical schema linking.

    Selects only tables and columns whose names have lexical overlap
    with the question. Strict: no silent fallback to full schema.
    Records linking_success=False if nothing is matched.
    """

    def __init__(self) -> None:
        self._diagnostics: SchemaDiagnostics | None = None

    def get_schema_string(self, db_schema: DatabaseSchema, question: str) -> str:
        question_tokens = _tokenize_question(question)
        selected_tables: list[Table] = []

        for table in db_schema.tables:
            table_matches = _name_matches_tokens(table.name, question_tokens)

            # Select columns that match the question
            matched_cols: list[Column] = []
            for col in table.columns:
                if _name_matches_tokens(col.name, question_tokens):
                    matched_cols.append(col)

            if table_matches:
                # Include all columns for matched tables
                selected_tables.append(table)
            elif matched_cols:
                # Include the table but only with matched columns + PKs
                pk_cols = [c for c in table.columns if c.is_primary_key]
                combined = {c.name: c for c in pk_cols}
                for c in matched_cols:
                    combined[c.name] = c
                selected_tables.append(
                    Table(
                        name=table.name,
                        columns=tuple(combined.values()),
                        foreign_keys=table.foreign_keys,
                    )
                )

        linking_success = len(selected_tables) > 0

        # Build the schema string from selected tables
        result = self._format_linked_schema(selected_tables, db_schema)

        total_original_cols = db_schema.total_column_count
        selected_col_count = sum(len(t.columns) for t in selected_tables)
        if total_original_cols > 0:
            reduction = 1.0 - (selected_col_count / total_original_cols)
        else:
            reduction = 0.0

        self._diagnostics = SchemaDiagnostics(
            original_table_count=len(db_schema.tables),
            selected_table_count=len(selected_tables),
            selected_column_count=selected_col_count,
            schema_string_length=len(result),
            schema_reduction_ratio=round(reduction, 4),
            linking_success=linking_success,
            fallback_used=False,
        )
        return result

    def _format_linked_schema(self, selected_tables: list[Table], db_schema: DatabaseSchema) -> str:
        """Format selected tables using the structured template."""
        if not selected_tables:
            return "DATABASE SCHEMA\n\n(No schema elements matched the question.)"

        selected_names = {t.name for t in selected_tables}

        # Build FK map for inline annotation
        fk_map: dict[tuple[str, str], str] = {}
        for table in selected_tables:
            for fk in table.foreign_keys:
                if fk.to_table in selected_names:
                    fk_map[(table.name, fk.from_column)] = f"{fk.to_table}.{fk.to_column}"

        sections: list[str] = ["DATABASE SCHEMA", ""]
        for table in selected_tables:
            sections.append(f"TABLE: {table.name}")
            for col in table.columns:
                annotations: list[str] = []
                if col.is_primary_key:
                    annotations.append("PRIMARY KEY")
                fk_target = fk_map.get((table.name, col.name))
                if fk_target:
                    annotations.append(f"FOREIGN KEY → {fk_target}")
                suffix = f" [{', '.join(annotations)}]" if annotations else ""
                sections.append(f"  - {col.name} {col.col_type}{suffix}")
            sections.append("")

        return "\n".join(sections).rstrip()

    def get_diagnostics(self) -> SchemaDiagnostics:
        if self._diagnostics is None:
            raise RuntimeError("get_schema_string must be called before get_diagnostics")
        return self._diagnostics


# ── NLTK Normalized Linking ─────────────────────────────────────────

def _ensure_nltk_resources() -> None:
    """Ensure required NLTK data is available."""
    if nltk is None:
        raise ImportError("NLTK is not installed.")
    try:
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        # also download omw-1.4 which is often required for wordnet lemmatizer
        nltk.download('omw-1.4', quiet=True)

import functools

@functools.lru_cache(maxsize=10000)
def _nltk_tokenize_question(question: str) -> frozenset[str]:
    """Tokenize and normalize a natural-language question using NLTK."""
    _ensure_nltk_resources()
    
    tokens = nltk.word_tokenize(question)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    normalized_tokens = set()
    for token in tokens:
        token = token.lower()
        if not token.isalnum():
            continue
        if token in stop_words:
            continue
        normalized_tokens.add(lemmatizer.lemmatize(token))
        
    return frozenset(normalized_tokens)


@functools.lru_cache(maxsize=10000)
def _nltk_tokenize_name(name: str) -> frozenset[str]:
    """Tokenize and normalize a schema identifier using NLTK.
    
    Handles snake_case, camelCase, and hyphens by substituting them
    with spaces before passing to NLTK's word_tokenize.
    """
    _ensure_nltk_resources()
        
    # Split camelCase
    spaced_name = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name)
    # Replace underscores and hyphens
    spaced_name = spaced_name.replace('_', ' ').replace('-', ' ')
    
    tokens = nltk.word_tokenize(spaced_name)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    normalized = set()
    for token in tokens:
        token = token.lower()
        if not token.isalnum():
            continue
        if token in stop_words:
            continue
        normalized.add(lemmatizer.lemmatize(token))
        
    return frozenset(normalized)


def _nltk_name_matches_tokens(name: str, question_tokens: frozenset[str]) -> bool:
    """Check whether a schema name has an NLTK normalized overlap with question tokens."""
    name_tokens = _nltk_tokenize_name(name)
    return bool(name_tokens & question_tokens)


class NormalizedSchemaLinkingStrategy(SchemaStrategy):
    """M4-B.1: NLTK-Normalized deterministic schema linking.
    
    Uses standard NLP tokenization, stopword removal, and lemmatization
    to increase recall over simple lexical matching.
    """

    def __init__(self) -> None:
        self._diagnostics: SchemaDiagnostics | None = None

    def get_schema_string(self, db_schema: DatabaseSchema, question: str) -> str:
        question_tokens = _nltk_tokenize_question(question)
        selected_tables: list[Table] = []

        for table in db_schema.tables:
            table_matches = _nltk_name_matches_tokens(table.name, question_tokens)

            # Select columns that match the question
            matched_cols: list[Column] = []
            for col in table.columns:
                if _nltk_name_matches_tokens(col.name, question_tokens):
                    matched_cols.append(col)

            if table_matches:
                # Include all columns for matched tables
                selected_tables.append(table)
            elif matched_cols:
                # Include the table but only with matched columns + PKs
                pk_cols = [c for c in table.columns if c.is_primary_key]
                combined = {c.name: c for c in pk_cols}
                for c in matched_cols:
                    combined[c.name] = c
                selected_tables.append(
                    Table(
                        name=table.name,
                        columns=tuple(combined.values()),
                        foreign_keys=table.foreign_keys,
                    )
                )

        linking_success = len(selected_tables) > 0

        # Build the schema string using the same format logic as M4-B
        result = SchemaLinkingStrategy._format_linked_schema(self, selected_tables, db_schema)

        total_original_cols = db_schema.total_column_count
        selected_col_count = sum(len(t.columns) for t in selected_tables)
        if total_original_cols > 0:
            reduction = 1.0 - (selected_col_count / total_original_cols)
        else:
            reduction = 0.0

        self._diagnostics = SchemaDiagnostics(
            original_table_count=len(db_schema.tables),
            selected_table_count=len(selected_tables),
            selected_column_count=selected_col_count,
            schema_string_length=len(result),
            schema_reduction_ratio=round(reduction, 4),
            linking_success=linking_success,
            fallback_used=False,
        )
        return result

    def get_diagnostics(self) -> SchemaDiagnostics:
        if self._diagnostics is None:
            raise RuntimeError("get_schema_string must be called before get_diagnostics")
        return self._diagnostics


class FKExpandedSchemaLinkingStrategy(SchemaStrategy):
    """M4-C: Schema linking with FK-graph expansion.

    First performs lexical schema linking (same as M4-B).
    Then expands the selected tables by traversing foreign-key edges
    via BFS up to ``max_fk_depth`` hops. This adds intermediate tables
    needed for JOIN paths.
    """

    def __init__(self, max_fk_depth: int = 1) -> None:
        self.max_fk_depth = max_fk_depth
        self._diagnostics: SchemaDiagnostics | None = None

    def get_schema_string(self, db_schema: DatabaseSchema, question: str) -> str:
        # Phase 1: Lexical linking (reuse the same logic as M4-B)
        question_tokens = _tokenize_question(question)
        linked_table_names: set[str] = set()
        linked_tables_map: dict[str, Table] = {}

        for table in db_schema.tables:
            table_matches = _name_matches_tokens(table.name, question_tokens)
            matched_cols = [
                c for c in table.columns if _name_matches_tokens(c.name, question_tokens)
            ]

            if table_matches:
                linked_table_names.add(table.name)
                linked_tables_map[table.name] = table
            elif matched_cols:
                linked_table_names.add(table.name)
                pk_cols = [c for c in table.columns if c.is_primary_key]
                combined = {c.name: c for c in pk_cols}
                for c in matched_cols:
                    combined[c.name] = c
                linked_tables_map[table.name] = Table(
                    name=table.name,
                    columns=tuple(combined.values()),
                    foreign_keys=table.foreign_keys,
                )

        linking_success = len(linked_table_names) > 0

        # Phase 2: FK expansion via BFS
        all_fks = db_schema.all_foreign_keys
        # Build undirected adjacency list from FK graph
        adjacency: dict[str, set[str]] = {}
        for fk in all_fks:
            adjacency.setdefault(fk.from_table, set()).add(fk.to_table)
            adjacency.setdefault(fk.to_table, set()).add(fk.from_table)

        expanded_names: set[str] = set()
        fk_expanded_table_names: list[str] = []

        if linking_success:
            # BFS from each linked table up to max_fk_depth
            visited: set[str] = set(linked_table_names)
            queue: deque[tuple[str, int]] = deque()
            for tname in linked_table_names:
                queue.append((tname, 0))

            while queue:
                current, depth = queue.popleft()
                if depth >= self.max_fk_depth:
                    continue
                for neighbor in adjacency.get(current, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        expanded_names.add(neighbor)
                        fk_expanded_table_names.append(neighbor)
                        queue.append((neighbor, depth + 1))

        # Build final table list: linked + expanded (all with full columns)
        final_table_names = linked_table_names | expanded_names
        final_tables: list[Table] = []
        for table in db_schema.tables:
            if table.name in final_table_names:
                if table.name in linked_tables_map:
                    # Use the potentially column-reduced version from linking
                    # But for FK-expanded tables we want full columns
                    if table.name in expanded_names:
                        final_tables.append(table)
                    else:
                        final_tables.append(linked_tables_map[table.name])
                else:
                    # FK-expanded table: include with all columns
                    final_tables.append(table)

        # Format
        result = self._format_linked_schema(final_tables, db_schema, final_table_names)

        total_original_cols = db_schema.total_column_count
        selected_col_count = sum(len(t.columns) for t in final_tables)
        if total_original_cols > 0:
            reduction = 1.0 - (selected_col_count / total_original_cols)
        else:
            reduction = 0.0

        self._diagnostics = SchemaDiagnostics(
            original_table_count=len(db_schema.tables),
            selected_table_count=len(final_tables),
            selected_column_count=selected_col_count,
            schema_string_length=len(result),
            schema_reduction_ratio=round(reduction, 4),
            linking_success=linking_success,
            fallback_used=False,
            fk_expanded_tables=tuple(sorted(fk_expanded_table_names)),
        )
        return result

    def _format_linked_schema(
        self,
        selected_tables: list[Table],
        db_schema: DatabaseSchema,
        selected_names: set[str],
    ) -> str:
        """Format selected tables using the structured template."""
        if not selected_tables:
            return "DATABASE SCHEMA\n\n(No schema elements matched the question.)"

        # Build FK map for inline annotation
        fk_map: dict[tuple[str, str], str] = {}
        for table in selected_tables:
            for fk in table.foreign_keys:
                if fk.to_table in selected_names:
                    fk_map[(table.name, fk.from_column)] = f"{fk.to_table}.{fk.to_column}"

        sections: list[str] = ["DATABASE SCHEMA", ""]
        for table in selected_tables:
            sections.append(f"TABLE: {table.name}")
            for col in table.columns:
                annotations: list[str] = []
                if col.is_primary_key:
                    annotations.append("PRIMARY KEY")
                fk_target = fk_map.get((table.name, col.name))
                if fk_target:
                    annotations.append(f"FOREIGN KEY → {fk_target}")
                suffix = f" [{', '.join(annotations)}]" if annotations else ""
                sections.append(f"  - {col.name} {col.col_type}{suffix}")
            sections.append("")

        return "\n".join(sections).rstrip()

    def get_diagnostics(self) -> SchemaDiagnostics:
        if self._diagnostics is None:
            raise RuntimeError("get_schema_string must be called before get_diagnostics")
        return self._diagnostics


# ── Strategy Factory ───────────────────────────────────────────────


def create_schema_strategy(strategy_name: str, **kwargs: Any) -> SchemaStrategy:
    """Instantiate a SchemaStrategy by name.

    Args:
        strategy_name: One of 'full', 'structured_full', 'schema_linking',
            'schema_linking_fk'.
        **kwargs: Strategy-specific parameters (e.g., max_fk_depth).

    Returns:
        A SchemaStrategy instance.

    Raises:
        ValueError: If the strategy name is not recognized.
    """
    strategies: dict[str, type[SchemaStrategy]] = {
        "full": FullSchemaStrategy,
        "structured_full": StructuredFullSchemaStrategy,
        "schema_linking": SchemaLinkingStrategy,
        "schema_linking_normalized": NormalizedSchemaLinkingStrategy,
        "schema_linking_fk": FKExpandedSchemaLinkingStrategy,
    }
    cls = strategies.get(strategy_name)
    if cls is None:
        raise ValueError(
            f"Unknown schema strategy: {strategy_name!r}. "
            f"Must be one of: {sorted(strategies.keys())}"
        )
    if strategy_name == "schema_linking_fk":
        return cls(max_fk_depth=kwargs.get("max_fk_depth", 1))
    return cls()
