"""Text-to-SQL prompt construction.

Builds prompts for LLMs that include the database schema and the
natural-language question.  The prompt format is model-agnostic — it
produces plain text that the model adapter wraps with any
model-specific chat template.
"""

from __future__ import annotations

_SYSTEM_INSTRUCTION = (
    "You are a SQL expert. Given the database schema below and a "
    "natural-language question, write a single SQL query that answers "
    "the question.\n"
    "Return ONLY the SQL query. Do not include any explanation."
)


def build_text_to_sql_prompt(question: str, schema: str) -> str:
    """Build a Text-to-SQL prompt from a question and schema.

    The prompt structure is deterministic and designed for zero-shot
    generation.  Few-shot examples will be added in a later milestone.

    Args:
        question: The natural-language question to translate.
        schema: A text description of the database schema (as produced
            by :func:`modelbench.schema.extract_schema_from_db`).

    Returns:
        A fully formed prompt string ready for the model.
    """
    return f"{_SYSTEM_INSTRUCTION}\n\nDatabase schema:\n{schema}\n\nQuestion: {question}\n\nSQL:"
