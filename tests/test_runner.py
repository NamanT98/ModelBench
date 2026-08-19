import tempfile

from modelbench.config import Config, DatasetConfig, ExperimentConfig, ModelConfig
from modelbench.runner import ExperimentRunner
from modelbench.types import GenerationResult


class FakeModel:
    def __init__(self, *args, **kwargs):
        self.model_id = "fake-model"

    def generate(self, prompt: str) -> GenerationResult:
        # We know the fixture asks about customers
        if "How many customers" in prompt:
            sql = "SELECT COUNT(*) FROM customers"
        else:
            sql = "SELECT name, price FROM products"

        return GenerationResult(
            text=f"```sql\n{sql}\n```", latency_seconds=0.1, input_tokens=10, output_tokens=5
        )


def test_experiment_runner(monkeypatch):
    # Mock create_model to return our FakeModel
    monkeypatch.setattr("modelbench.runner.create_model", lambda *args: FakeModel())

    config = Config(
        experiment=ExperimentConfig(name="test_exp"),
        dataset=DatasetConfig(name="spider", path="tests/data/spider_fixture", split="dev"),
        model=ModelConfig(model_id="fake-model"),
    )

    runner = ExperimentRunner(config)
    result = runner.run()

    assert result.total_samples == 2
    assert result.valid_sql_count == 2
    assert result.execution_correct_count == 2
    assert result.exact_match_count == 2
    assert result.avg_latency_seconds == 0.1

    # Test save_result
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_path = runner.save_result(result, output_dir=tmpdir)
        assert saved_path.exists()
        assert saved_path.name == "test_exp.json"

        import json

        with open(saved_path) as f:
            data = json.load(f)

        assert data["total_samples"] == 2
        assert data["metadata"]["dataset"] == "spider"
