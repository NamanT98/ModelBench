"""SQL extraction from LLM output.

Model responses may contain markdown code fences, explanatory text, or
other formatting around the actual SQL query.  This module provides a
small deterministic extractor for the common output patterns.
"""

from __future__ import annotations

import re


class SQLExtractionError(Exception):
    """Raised when SQL cannot be extracted from model output."""


def extract_sql(text: str) -> str:
    """Extract a SQL query from raw model output.

    Extraction strategy (in order of precedence):

    1. If the text contains a ``sql`` fenced code block, extract its
       contents.
    2. If the text contains a generic fenced code block, extract its
       contents.
    3. Otherwise, return the full text stripped of leading/trailing
       whitespace (treating the entire output as raw SQL).

    Trailing semicolons are preserved — normalization is handled
    separately by :func:`modelbench.sql.normalize_sql`.

    Args:
        text: Raw model output.

    Returns:
        The extracted SQL string.

    Raises:
        SQLExtractionError: If the input is empty or whitespace-only,
            indicating no SQL could possibly be extracted.
    """
    text = text.strip()
    if not text:
        raise SQLExtractionError("Empty model output — no SQL to extract")

    # 1. ```sql ... ``` blocks
    match = re.search(r"```sql\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 2. Generic ``` ... ``` blocks
    match = re.search(r"```\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 3. Raw text — assume the whole output is SQL
    return text
