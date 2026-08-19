"""Tests for the ModelBench CLI."""

from click.testing import CliRunner

from modelbench.cli import main


class TestCli:
    """Test CLI commands."""

    def test_version_flag(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "modelbench" in result.output
        assert "0.1.0" in result.output

    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Text-to-SQL" in result.output

    def test_info_command(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["info"])
        assert result.exit_code == 0
        assert "ModelBench v0.1.0" in result.output
