"""SQL normalization utilities for exact-match comparison."""

from __future__ import annotations

import re


def normalize_sql(sql: str) -> str:
    """Normalize a SQL string for exact-match comparison.

    Applies the following normalizations:
      - Strip leading and trailing whitespace.
      - Collapse consecutive whitespace characters into a single space.
      - Convert to lowercase.
      - Remove trailing semicolons.

    This is deliberately simple and does NOT:
      - Parse SQL into an AST.
      - Normalize aliases or table references.
      - Reorder clauses, columns, or operands.
      - Handle SQL dialect differences.

    Args:
        sql: Raw SQL string.

    Returns:
        Normalized SQL string suitable for string comparison.
    """
    result = sql.strip()
    result = re.sub(r"\s+", " ", result)
    result = result.lower()
    result = result.rstrip(";").strip()
    return result
