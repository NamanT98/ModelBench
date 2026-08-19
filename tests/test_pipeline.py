"""Integration tests: fake model -> prompt -> SQL extraction -> M1 evaluator.

Verifies the full pipeline works end-to-end without requiring a real
Hugging Face model, GPU, or internet access.
"""

from __future__ import annotations

from pathlib import Path

from modelbench.evaluation import evaluate_sample
from modelbench.extract import extract_sql
from modelbench.fixture import get_fixture_samples
from modelbench.prompt import build_text_to_sql_prompt
from modelbench.schema import extract_schema_from_db
from modelbench.types import GenerationResult


class FakeModel:
    """Deterministic fake model that returns pre-set SQL responses."""

    def __init__(self, responses: dict[str, str]) -> None:
        self._responses = responses

    @property
    def model_id(self) -> str:
        return "fake/test-model"

    def generate(self, prompt: str) -> GenerationResult:
        # Find the matching response by checking which question is in the prompt
        for question, sql in self._responses.items():
            if question in prompt:
                return GenerationResult(text=sql, latency_seconds=0.001)
        return GenerationResult(text="SELECT 1", latency_seconds=0.001)


class TestFullPipeline:
    """End-to-end pipeline: question -> prompt -> model -> extract -> evaluate."""

    def test_correct_prediction_pipeline(self, fixture_db: Path) -> None:
        """A fake model returning the gold SQL should get execution_accuracy=True."""
        schema = extract_schema_from_db(fixture_db)
        samples = get_fixture_samples(fixture_db)
        sample = samples[0]  # "How many customers are there?"

        fake = FakeModel({"How many customers": "SELECT COUNT(*) FROM customers"})
        prompt = build_text_to_sql_prompt(sample.question, schema)
        gen_result = fake.generate(prompt)
        predicted_sql = extract_sql(gen_result.text)
        eval_result = evaluate_sample(predicted_sql, sample.gold_sql, sample.db_path)

        assert eval_result.sql_valid is True
        assert eval_result.execution_accuracy is True

    def test_wrong_prediction_pipeline(self, fixture_db: Path) -> None:
        """A fake model returning wrong SQL should get execution_accuracy=False."""
        schema = extract_schema_from_db(fixture_db)
        samples = get_fixture_samples(fixture_db)
        sample = samples[0]  # "How many customers are there?" -> 3

        # Return wrong count (order_items has 5 rows, customers has 3)
        fake = FakeModel({"How many customers": "SELECT COUNT(*) FROM order_items"})
        prompt = build_text_to_sql_prompt(sample.question, schema)
        gen_result = fake.generate(prompt)
        predicted_sql = extract_sql(gen_result.text)
        eval_result = evaluate_sample(predicted_sql, sample.gold_sql, sample.db_path)

        assert eval_result.sql_valid is True
        assert eval_result.execution_accuracy is False

    def test_invalid_sql_pipeline(self, fixture_db: Path) -> None:
        """A fake model returning invalid SQL should get sql_valid=False."""
        schema = extract_schema_from_db(fixture_db)
        samples = get_fixture_samples(fixture_db)
        sample = samples[0]

        fake = FakeModel({"How many customers": "INVALID SQL GIBBERISH"})
        prompt = build_text_to_sql_prompt(sample.question, schema)
        gen_result = fake.generate(prompt)
        predicted_sql = extract_sql(gen_result.text)
        eval_result = evaluate_sample(predicted_sql, sample.gold_sql, sample.db_path)

        assert eval_result.sql_valid is False
        assert eval_result.execution_accuracy is False
        assert eval_result.execution_error is not None

    def test_fenced_sql_extraction_pipeline(self, fixture_db: Path) -> None:
        """Model output wrapped in markdown fences should be extracted correctly."""
        schema = extract_schema_from_db(fixture_db)
        samples = get_fixture_samples(fixture_db)
        sample = samples[0]

        fenced = "```sql\nSELECT COUNT(*) FROM customers\n```"
        fake = FakeModel({"How many customers": fenced})
        prompt = build_text_to_sql_prompt(sample.question, schema)
        gen_result = fake.generate(prompt)
        predicted_sql = extract_sql(gen_result.text)
        eval_result = evaluate_sample(predicted_sql, sample.gold_sql, sample.db_path)

        assert eval_result.sql_valid is True
        assert eval_result.execution_accuracy is True

    def test_all_fixture_samples(self, fixture_db: Path) -> None:
        """Run the full pipeline on all 5 fixture samples with correct predictions."""
        schema = extract_schema_from_db(fixture_db)
        samples = get_fixture_samples(fixture_db)

        # Build a fake that returns the gold SQL for each question
        responses = {s.question[:20]: s.gold_sql for s in samples}
        fake = FakeModel(responses)

        for sample in samples:
            prompt = build_text_to_sql_prompt(sample.question, schema)
            gen_result = fake.generate(prompt)
            predicted_sql = extract_sql(gen_result.text)
            eval_result = evaluate_sample(predicted_sql, sample.gold_sql, sample.db_path)

            assert eval_result.sql_valid is True, f"Failed for: {sample.question}"
            assert eval_result.execution_accuracy is True, f"Failed for: {sample.question}"


class TestPromptContainsSchema:
    """Verify that the prompt includes schema when built from a real DB."""

    def test_schema_in_prompt(self, fixture_db: Path) -> None:
        schema = extract_schema_from_db(fixture_db)
        prompt = build_text_to_sql_prompt("How many customers?", schema)
        assert "customers" in prompt
        assert "products" in prompt
        assert "customer_id" in prompt

    def test_question_in_prompt(self, fixture_db: Path) -> None:
        schema = extract_schema_from_db(fixture_db)
        prompt = build_text_to_sql_prompt("How many customers?", schema)
        assert "How many customers?" in prompt
