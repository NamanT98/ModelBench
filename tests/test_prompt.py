"""Tests for the Text-to-SQL prompt builder."""

from modelbench.prompt import build_text_to_sql_prompt


class TestBuildPrompt:
    """Test prompt construction."""

    def test_contains_question(self) -> None:
        prompt = build_text_to_sql_prompt("How many users?", "Table users: id INTEGER")
        assert "How many users?" in prompt

    def test_contains_schema(self) -> None:
        schema = "Table users:\n  id INTEGER\n  name TEXT"
        prompt = build_text_to_sql_prompt("Count users", schema)
        assert schema in prompt

    def test_contains_sql_instruction(self) -> None:
        prompt = build_text_to_sql_prompt("Q?", "schema")
        # Should instruct the model to output SQL
        assert "SQL" in prompt

    def test_deterministic(self) -> None:
        p1 = build_text_to_sql_prompt("Q?", "S")
        p2 = build_text_to_sql_prompt("Q?", "S")
        assert p1 == p2

    def test_different_questions_produce_different_prompts(self) -> None:
        p1 = build_text_to_sql_prompt("How many?", "schema")
        p2 = build_text_to_sql_prompt("What is the total?", "schema")
        assert p1 != p2

    def test_returns_string(self) -> None:
        prompt = build_text_to_sql_prompt("Q?", "S")
        assert isinstance(prompt, str)
        assert len(prompt) > 0
