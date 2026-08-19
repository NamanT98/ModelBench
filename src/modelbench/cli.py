"""CLI entry point for ModelBench."""

from __future__ import annotations

import tempfile
from pathlib import Path

import click

from modelbench import __version__
from modelbench.logging import setup_logging


@click.group()
@click.version_option(version=__version__, prog_name="modelbench")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose (DEBUG) logging.")
def main(verbose: bool) -> None:
    """ModelBench -- LLM experimentation and evaluation platform for Text-to-SQL."""
    setup_logging(verbose=verbose)


@main.command()
def info() -> None:
    """Display project information and current configuration."""
    from modelbench.config import load_config

    config = load_config()
    click.echo(f"ModelBench v{__version__}")
    click.echo(f"Project: {config.project_name}")
    click.echo(f"Log level: {config.log_level}")
    click.echo(f"Model:   {config.model.model_id} ({config.model.provider})")
    click.echo(f"Device:  {config.model.device}")
    click.echo(f"Dtype:   {config.model.dtype}")


@main.command("evaluate-fixture")
def evaluate_fixture() -> None:
    """Run the built-in fixture evaluation to verify the pipeline.

    Creates a temporary SQLite database, evaluates 5 predefined
    Text-to-SQL samples, and prints aggregate metrics.

    This is an internal development fixture, NOT a real benchmark.
    """
    from modelbench.evaluation import evaluate_sample
    from modelbench.fixture import FIXTURE_PREDICTIONS, create_fixture_db, get_fixture_samples

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = create_fixture_db(Path(tmpdir) / "fixture_ecommerce.db")
        samples = get_fixture_samples(db_path)

        click.echo("ModelBench Text-to-SQL Fixture Evaluation")
        click.echo("=" * 44)
        click.echo()

        results = []
        for i, (sample, predicted) in enumerate(
            zip(samples, FIXTURE_PREDICTIONS, strict=True), 1
        ):
            result = evaluate_sample(predicted, sample.gold_sql, sample.db_path)
            results.append(result)

            status = "pass" if result.execution_accuracy else "FAIL"
            click.echo(f"  [{status}] {i}. {sample.question}")
            if result.execution_error:
                click.echo(f"       Error: {result.execution_error}")

        click.echo()
        _print_summary(results)


@main.command("evaluate-model")
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to a YAML configuration file.",
)
def evaluate_model(config_path: str | None) -> None:
    """Run a model against the fixture benchmark.

    Loads the configured Hugging Face model, generates SQL for each
    fixture sample, evaluates with the M1 engine, and prints results.

    \b
    Example:
        modelbench evaluate-model --config configs/qwen_fixture.yaml
    """
    from modelbench.config import load_config
    from modelbench.evaluation import evaluate_sample
    from modelbench.extract import SQLExtractionError, extract_sql
    from modelbench.fixture import create_fixture_db, get_fixture_samples
    from modelbench.model import create_model
    from modelbench.prompt import build_text_to_sql_prompt
    from modelbench.schema import extract_schema_from_db
    from modelbench.types import EvaluationResult

    config = load_config(config_path)
    model = create_model(config.model, config.generation)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = create_fixture_db(Path(tmpdir) / "fixture_ecommerce.db")
        samples = get_fixture_samples(db_path)
        schema = extract_schema_from_db(db_path)

        click.echo("ModelBench Text-to-SQL Model Evaluation")
        click.echo("=" * 44)
        click.echo(f"Model:          {model.model_id}")
        click.echo(f"Device:         {config.model.device}")
        click.echo(f"Dtype:          {config.model.dtype}")
        click.echo(f"Max new tokens: {config.generation.max_new_tokens}")
        click.echo(f"Samples:        {len(samples)}")
        click.echo()

        results: list[EvaluationResult] = []
        latencies: list[float] = []

        for i, sample in enumerate(samples, 1):
            prompt = build_text_to_sql_prompt(sample.question, schema)
            gen_result = model.generate(prompt)
            latencies.append(gen_result.latency_seconds)

            try:
                predicted_sql = extract_sql(gen_result.text)
            except SQLExtractionError as e:
                click.echo(f"  [FAIL] {i}. {sample.question}")
                click.echo(f"         Extraction error: {e}")
                click.echo(f"         Raw output: {gen_result.text!r}")
                click.echo()
                results.append(
                    EvaluationResult(
                        sql_valid=False,
                        exact_match=False,
                        execution_accuracy=False,
                        execution_error=f"SQL extraction failed: {e}",
                    )
                )
                continue

            eval_result = evaluate_sample(predicted_sql, sample.gold_sql, sample.db_path)
            results.append(eval_result)

            status = "pass" if eval_result.execution_accuracy else "FAIL"
            click.echo(f"  [{status}] {i}. {sample.question}")
            click.echo(f"         SQL: {predicted_sql}")
            click.echo(f"         Latency: {gen_result.latency_seconds:.2f}s")
            if eval_result.execution_error:
                click.echo(f"         Error: {eval_result.execution_error}")
            click.echo()

        click.echo("-" * 44)
        _print_summary(results, latencies=latencies)


def _print_summary(
    results: list,
    *,
    latencies: list[float] | None = None,
) -> None:
    """Print aggregate evaluation metrics."""
    total = len(results)
    if total == 0:
        click.echo("No results to summarise.")
        return

    valid = sum(1 for r in results if r.sql_valid)
    exact = sum(1 for r in results if r.exact_match)
    exec_acc = sum(1 for r in results if r.execution_accuracy)

    click.echo(f"Samples:            {total}")
    click.echo(f"SQL Validity:       {valid}/{total} ({100 * valid // total}%)")
    click.echo(f"Exact Match:        {exact}/{total} ({100 * exact // total}%)")
    click.echo(f"Execution Accuracy: {exec_acc}/{total} ({100 * exec_acc // total}%)")

    if latencies:
        avg = sum(latencies) / len(latencies)
        click.echo(f"Avg Latency:        {avg:.2f}s")
