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
    """ModelBench — LLM experimentation and evaluation platform for Text-to-SQL."""
    setup_logging(verbose=verbose)


@main.command()
def info() -> None:
    """Display project information and current configuration."""
    from modelbench.config import load_config

    config = load_config()
    click.echo(f"ModelBench v{__version__}")
    click.echo(f"Project: {config.project_name}")
    click.echo(f"Log level: {config.log_level}")


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
        for i, (sample, predicted) in enumerate(zip(samples, FIXTURE_PREDICTIONS, strict=True), 1):
            result = evaluate_sample(predicted, sample.gold_sql, sample.db_path)
            results.append(result)

            status = "✓" if result.execution_accuracy else "✗"
            click.echo(f"  [{status}] {i}. {sample.question}")
            if result.execution_error:
                click.echo(f"       Error: {result.execution_error}")

        click.echo()

        total = len(results)
        valid = sum(1 for r in results if r.sql_valid)
        exact = sum(1 for r in results if r.exact_match)
        exec_acc = sum(1 for r in results if r.execution_accuracy)

        click.echo(f"Samples:            {total}")
        click.echo(f"SQL Validity:       {valid}/{total} ({100 * valid // total}%)")
        click.echo(f"Exact Match:        {exact}/{total} ({100 * exact // total}%)")
        click.echo(f"Execution Accuracy: {exec_acc}/{total} ({100 * exec_acc // total}%)")
