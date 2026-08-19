"""Schema extraction from SQLite databases.

Produces a human-readable text representation of a database schema
suitable for inclusion in LLM prompts.  Includes table names, column
names and types, primary keys, and foreign-key relationships.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def extract_schema_from_db(db_path: str | Path) -> str:
    """Extract a text description of the database schema.

    The output format is designed to be clear and compact when embedded
    in a prompt:

        Table customers:
          customer_id INTEGER (PRIMARY KEY)
          name TEXT
          ...

    Args:
        db_path: Path to a SQLite database file.

    Returns:
        A multi-line string describing every table, column, and
        foreign-key relationship in the database.

    Raises:
        FileNotFoundError: If the database file does not exist.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        tables = _get_table_names(conn)
        sections: list[str] = []
        for table in tables:
            sections.append(_describe_table(conn, table))
        return "\n\n".join(sections)
    finally:
        conn.close()


def _get_table_names(conn: sqlite3.Connection) -> list[str]:
    """Return sorted table names, excluding SQLite internal tables."""
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name"
    )
    return [row[0] for row in cursor.fetchall()]


def _describe_table(conn: sqlite3.Connection, table: str) -> str:
    """Produce a text description of a single table."""
    lines = [f"Table {table}:"]

    # Columns via PRAGMA table_info
    cursor = conn.execute(f"PRAGMA table_info({table})")
    for row in cursor.fetchall():
        # row: (cid, name, type, notnull, dflt_value, pk)
        col_name = row[1]
        col_type = row[2] or "ANY"
        pk_suffix = " (PRIMARY KEY)" if row[5] else ""
        lines.append(f"  {col_name} {col_type}{pk_suffix}")

    # Foreign keys via PRAGMA foreign_key_list
    cursor = conn.execute(f"PRAGMA foreign_key_list({table})")
    fks = cursor.fetchall()
    for fk in fks:
        # fk: (id, seq, ref_table, from_col, to_col, on_update, on_delete, match)
        lines.append(f"  FOREIGN KEY ({fk[3]}) REFERENCES {fk[2]}({fk[4]})")

    return "\n".join(lines)
