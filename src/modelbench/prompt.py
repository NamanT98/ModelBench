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


from modelbench.types import TextToSQLSample

def build_text_to_sql_prompt(
    question: str, 
    schema: str, 
    examples: list[tuple[TextToSQLSample, str]] | None = None
) -> str:
    """Build a Text-to-SQL prompt from a question and schema.

    Args:
        question: The natural-language question to translate.
        schema: A text description of the database schema (as produced
            by :func:`modelbench.schema.extract_schema_from_db`).
        examples: Optional list of tuples containing (demonstration sample, 
            demonstration schema).

    Returns:
        A fully formed prompt string ready for the model.
    """
    prompt = f"{_SYSTEM_INSTRUCTION}\n\n"
    
    if examples:
        for i, (ex_sample, ex_schema) in enumerate(examples, 1):
            prompt += f"EXAMPLE {i}:\n"
            prompt += f"Database schema:\n{ex_schema}\n\n"
            prompt += f"Question: {ex_sample.question}\n\n"
            prompt += f"SQL: {ex_sample.gold_sql}\n\n"
    
    prompt += f"Database schema:\n{schema}\n\n"
    prompt += f"TARGET QUESTION: {question}\n\nSQL:"
    return prompt
