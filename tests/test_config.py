"""Tests for ModelBench configuration management."""

from pathlib import Path

import pytest

from modelbench.config import Config, GenerationConfig, ModelConfig, load_config


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

    def test_default_model_config(self) -> None:
        config = Config()
        assert config.model.provider == "huggingface"
        assert config.model.model_id == "Qwen/Qwen2.5-Coder-3B-Instruct"

    def test_default_generation_config(self) -> None:
        config = Config()
        assert config.generation.max_new_tokens == 256
        assert config.generation.do_sample is False


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

    def test_load_model_section(self, tmp_path: Path) -> None:
        config_file = tmp_path / "test.yaml"
        config_file.write_text(
            "model:\n"
            "  model_id: meta-llama/Llama-3-8B\n"
            "  device: cpu\n"
            "  dtype: float32\n"
        )
        config = load_config(config_file)
        assert config.model.model_id == "meta-llama/Llama-3-8B"
        assert config.model.device == "cpu"
        assert config.model.dtype == "float32"
        # Unspecified fields should get defaults
        assert config.model.provider == "huggingface"

    def test_load_generation_section(self, tmp_path: Path) -> None:
        config_file = tmp_path / "test.yaml"
        config_file.write_text(
            "generation:\n"
            "  max_new_tokens: 512\n"
            "  temperature: 0.7\n"
            "  do_sample: true\n"
        )
        config = load_config(config_file)
        assert config.generation.max_new_tokens == 512
        assert config.generation.temperature == 0.7
        assert config.generation.do_sample is True

    def test_load_full_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / "full.yaml"
        config_file.write_text(
            "project_name: test_project\n"
            "model:\n"
            "  model_id: test/model\n"
            "  device: cuda\n"
            "generation:\n"
            "  max_new_tokens: 128\n"
        )
        config = load_config(config_file)
        assert config.project_name == "test_project"
        assert config.model.model_id == "test/model"
        assert config.model.device == "cuda"
        assert config.generation.max_new_tokens == 128

    def test_missing_model_section_uses_defaults(self, tmp_path: Path) -> None:
        config_file = tmp_path / "test.yaml"
        config_file.write_text("project_name: test\n")
        config = load_config(config_file)
        assert config.model.model_id == "Qwen/Qwen2.5-Coder-3B-Instruct"
        assert config.generation.max_new_tokens == 256

    def test_invalid_model_config_raises(self, tmp_path: Path) -> None:
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("model:\n  provider: openai\n")
        with pytest.raises(ValueError, match="Unsupported provider"):
            load_config(config_file)

    def test_invalid_generation_config_raises(self, tmp_path: Path) -> None:
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("generation:\n  max_new_tokens: 0\n")
        with pytest.raises(ValueError, match="max_new_tokens must be >= 1"):
            load_config(config_file)
