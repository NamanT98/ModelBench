"""Tests for ModelBench configuration management."""

from pathlib import Path

from modelbench.config import Config, load_config


class TestConfigDefaults:
    """Test that default configuration values are sensible."""

    def test_default_project_name(self) -> None:
        config = Config()
        assert config.project_name == "modelbench"

    def test_default_log_level(self) -> None:
        config = Config()
        assert config.log_level == "INFO"

    def test_default_data_dir(self) -> None:
        config = Config()
        assert config.data_dir == "datasets"

    def test_extra_defaults_to_empty(self) -> None:
        config = Config()
        assert config.extra == {}


class TestLoadConfig:
    """Test YAML config file loading."""

    def test_load_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        config = load_config(tmp_path / "nonexistent.yaml")
        assert config.project_name == "modelbench"

    def test_load_explicit_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "test.yaml"
        config_file.write_text("project_name: my_project\nlog_level: DEBUG\n")
        config = load_config(config_file)
        assert config.project_name == "my_project"
        assert config.log_level == "DEBUG"

    def test_unknown_keys_go_to_extra(self, tmp_path: Path) -> None:
        config_file = tmp_path / "test.yaml"
        config_file.write_text("project_name: test\ncustom_key: custom_value\n")
        config = load_config(config_file)
        assert config.project_name == "test"
        assert config.extra == {"custom_key": "custom_value"}

    def test_empty_yaml_returns_defaults(self, tmp_path: Path) -> None:
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        config = load_config(config_file)
        assert config.project_name == "modelbench"
