"""SQLite database execution for Text-to-SQL evaluation."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from modelbench.types import QueryResult

logger = logging.getLogger(__name__)


class DatabaseExecutionError(Exception):
    """Raised when a SQL query fails to execute against the database."""


def execute_query(db_path: str | Path, sql: str) -> QueryResult:
    """Execute a read-only SQL query against a SQLite database.

    The connection is opened in read-only mode to prevent any
    accidental modifications to the benchmark database.

    Args:
        db_path: Path to the SQLite database file.
        sql: SQL query to execute.

    Returns:
        A QueryResult containing column names and rows.

    Raises:
        DatabaseExecutionError: If the SQL query fails to execute
            (syntax error, missing table, etc.).
        FileNotFoundError: If the database file does not exist.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    uri = f"file:{db_path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        try:
            cursor = conn.execute(sql)
            columns = tuple(desc[0] for desc in cursor.description) if cursor.description else ()
            rows = cursor.fetchall()
            return QueryResult(columns=columns, rows=rows)
        finally:
            conn.close()
    except sqlite3.Error as e:
        raise DatabaseExecutionError(str(e)) from e
