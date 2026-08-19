"""Tests for the model abstraction layer.

All tests run without torch, transformers, a GPU, or internet access.
They verify configuration, lazy loading, and protocol compliance
using only the public API and test fakes.
"""

from __future__ import annotations

import pytest

from modelbench.config import GenerationConfig, ModelConfig
from modelbench.model import HuggingFaceCausalLM, Model, create_model
from modelbench.types import GenerationResult


# ── Fake model for protocol tests ───────────────────────────────────


class FakeModel:
    """A fake model satisfying the Model protocol."""

    def __init__(self, responses: list[str], model_id: str = "fake/model") -> None:
        self._responses = responses
        self._model_id = model_id
        self._call_count = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(self, prompt: str) -> GenerationResult:
        response = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return GenerationResult(
            text=response,
            latency_seconds=0.001,
            input_tokens=len(prompt.split()),
            output_tokens=len(response.split()),
        )


# ── Model protocol tests ───────────────────────────────────────────


class TestModelProtocol:
    """Verify that FakeModel satisfies the Model protocol."""

    def test_fake_model_is_model(self) -> None:
        fake = FakeModel(["SELECT 1"])
        assert isinstance(fake, Model)

    def test_fake_model_generate(self) -> None:
        fake = FakeModel(["SELECT COUNT(*) FROM users"])
        result = fake.generate("How many users?")
        assert result.text == "SELECT COUNT(*) FROM users"
        assert result.latency_seconds > 0

    def test_fake_model_id(self) -> None:
        fake = FakeModel(["x"], model_id="test/my-model")
        assert fake.model_id == "test/my-model"

    def test_fake_model_cycles_responses(self) -> None:
        fake = FakeModel(["A", "B"])
        assert fake.generate("p").text == "A"
        assert fake.generate("p").text == "B"
        assert fake.generate("p").text == "A"


# ── ModelConfig tests ───────────────────────────────────────────────


class TestModelConfig:
    """Test model configuration and validation."""

    def test_defaults(self) -> None:
        cfg = ModelConfig()
        assert cfg.provider == "huggingface"
        assert cfg.model_id == "Qwen/Qwen2.5-Coder-3B-Instruct"
        assert cfg.revision == "main"
        assert cfg.device == "auto"
        assert cfg.dtype == "auto"

    def test_custom_model_id(self) -> None:
        cfg = ModelConfig(model_id="meta-llama/Llama-3-8B")
        assert cfg.model_id == "meta-llama/Llama-3-8B"

    def test_custom_revision(self) -> None:
        cfg = ModelConfig(revision="v2.0")
        assert cfg.revision == "v2.0"

    def test_custom_device(self) -> None:
        cfg = ModelConfig(device="cpu")
        assert cfg.device == "cpu"

    def test_custom_dtype(self) -> None:
        cfg = ModelConfig(dtype="bfloat16")
        assert cfg.dtype == "bfloat16"

    def test_invalid_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported provider"):
            ModelConfig(provider="openai")

    def test_empty_model_id_raises(self) -> None:
        with pytest.raises(ValueError, match="model_id must not be empty"):
            ModelConfig(model_id="")

    def test_invalid_device_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported device"):
            ModelConfig(device="tpu")

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported dtype"):
            ModelConfig(dtype="int8")


# ── GenerationConfig tests ──────────────────────────────────────────


class TestGenerationConfig:
    """Test generation configuration and validation."""

    def test_defaults(self) -> None:
        cfg = GenerationConfig()
        assert cfg.max_new_tokens == 256
        assert cfg.temperature == 0.0
        assert cfg.do_sample is False

    def test_custom_values(self) -> None:
        cfg = GenerationConfig(max_new_tokens=512, temperature=0.7, do_sample=True)
        assert cfg.max_new_tokens == 512
        assert cfg.temperature == 0.7
        assert cfg.do_sample is True

    def test_invalid_max_tokens_raises(self) -> None:
        with pytest.raises(ValueError, match="max_new_tokens must be >= 1"):
            GenerationConfig(max_new_tokens=0)

    def test_negative_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature must be >= 0"):
            GenerationConfig(temperature=-1.0)


# ── HuggingFaceCausalLM construction tests ──────────────────────────


class TestHuggingFaceCausalLM:
    """Test HuggingFaceCausalLM without loading any real model."""

    def test_lazy_loading(self) -> None:
        """Constructing the adapter must NOT load the model."""
        model = HuggingFaceCausalLM(ModelConfig(), GenerationConfig())
        assert model._hf_model is None
        assert model._tokenizer is None

    def test_model_id_property(self) -> None:
        model = HuggingFaceCausalLM(
            ModelConfig(model_id="custom/model"),
            GenerationConfig(),
        )
        assert model.model_id == "custom/model"

    def test_different_model_ids(self) -> None:
        m1 = HuggingFaceCausalLM(ModelConfig(model_id="a/b"), GenerationConfig())
        m2 = HuggingFaceCausalLM(ModelConfig(model_id="c/d"), GenerationConfig())
        assert m1.model_id != m2.model_id

    def test_stores_generation_config(self) -> None:
        gen = GenerationConfig(max_new_tokens=512)
        model = HuggingFaceCausalLM(ModelConfig(), gen)
        assert model._gen_config.max_new_tokens == 512


# ── create_model factory tests ──────────────────────────────────────


class TestCreateModel:
    """Test the model factory function."""

    def test_creates_huggingface_model(self) -> None:
        model = create_model(ModelConfig(), GenerationConfig())
        assert isinstance(model, HuggingFaceCausalLM)
        assert model.model_id == "Qwen/Qwen2.5-Coder-3B-Instruct"

    def test_unsupported_provider_raises(self) -> None:
        # Bypass ModelConfig.__post_init__ validation to test the factory
        bad_config = ModelConfig.__new__(ModelConfig)
        bad_config.provider = "unsupported_provider"
        bad_config.model_id = "test/model"
        bad_config.revision = "main"
        bad_config.device = "auto"
        bad_config.dtype = "auto"
        with pytest.raises(ValueError, match="Unsupported model provider"):
            create_model(bad_config, GenerationConfig())
