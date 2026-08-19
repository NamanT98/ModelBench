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

            status = "pass" if result.execution_accuracy else "FAIL"
            click.echo(f"  [{status}] {i}. {sample.question}")
            if result.execution_error:
                click.echo(f"       Error: {result.execution_error}")

        click.echo()
        _print_summary(results)


@main.command("run")
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to a YAML configuration file.",
)
def run_experiment(config_path: str | None) -> None:
    """Run a ModelBench experiment based on the provided configuration.

    Example:
        modelbench run --config configs/qwen_spider_baseline.yaml
    """
    from modelbench.config import load_config
    from modelbench.runner import ExperimentRunner

    # Load configuration
    try:
        config = load_config(config_path)
    except Exception as e:
        click.echo(f"Failed to load configuration: {e}", err=True)
        raise click.Abort()

    click.echo("ModelBench Experiment")
    click.echo("=====================")
    click.echo()
    click.echo(f"Experiment: {config.experiment.name}")
    click.echo(f"Dataset:    {config.dataset.name}")
    click.echo(f"Split:      {config.dataset.split}")
    click.echo(f"Limit:      {config.dataset.limit or 'all'}")
    click.echo(f"Seed:       {config.dataset.seed or config.experiment.seed or 'none'}")
    click.echo(f"Model:      {config.model.model_id}")
    click.echo(f"Schema:     {config.schema.strategy}")
    click.echo(f"Strategy:   {config.strategy.name}")
    click.echo()

    try:
        runner = ExperimentRunner(config)
    except Exception as e:
        click.echo(f"Failed to initialize runner: {e}", err=True)
        raise click.Abort()

    click.echo("Progress:")
    try:
        result = runner.run()
    except Exception as e:
        click.echo(f"Experiment failed during execution: {e}", err=True)
        raise click.Abort()

    click.echo()
    click.echo("Results")
    click.echo("-------")
    click.echo(
        f"SQL Validity:       {result.valid_sql_count}/{result.total_samples} ({result.sql_validity_rate * 100:.1f}%)"
    )
    click.echo(
        f"Exact Match:        {result.exact_match_count}/{result.total_samples} ({result.exact_match_rate * 100:.1f}%)"
    )
    click.echo(
        f"Execution Accuracy: {result.execution_correct_count}/{result.total_samples} ({result.execution_accuracy * 100:.1f}%)"
    )
    click.echo(f"Avg Latency:        {result.avg_latency_seconds:.2f}s")
    click.echo()

    try:
        saved_path = runner.save_result(result)
        click.echo(f"Results:\n{saved_path}")
    except Exception as e:
        click.echo(f"Failed to save results: {e}", err=True)
        raise click.Abort()


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
