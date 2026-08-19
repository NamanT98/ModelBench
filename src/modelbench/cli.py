"""CLI entry point for ModelBench."""

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
