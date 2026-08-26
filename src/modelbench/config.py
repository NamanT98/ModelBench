"""Configuration management for ModelBench.

Supports flat application settings and nested sections for model
inference and generation parameters.  Unknown YAML keys are collected
into ``extra`` rather than raising, so configs remain forward-compatible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_FILENAME = "modelbench.yaml"

# ── Nested config sections ──────────────────────────────────────────

_VALID_PROVIDERS = {"huggingface", "vllm"}
_VALID_DEVICES = {"auto", "cpu", "cuda"}
_VALID_DTYPES = {"auto", "float16", "float32", "bfloat16"}


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""

    name: str = "spider"
    split: str = "dev"
    limit: int | None = None
    seed: int | None = None
    path: str | None = None


@dataclass
class SchemaConfig:
    """Configuration for schema extraction/retrieval."""

    strategy: str = "full"
    max_fk_depth: int = 1


@dataclass
class StrategyConfig:
    """Configuration for the overall prompting strategy."""

    name: str = "zero_shot"
    retriever: str | None = None
    k: int = 3
    train_split: str = "train"
    embedding_model: str | None = None
    hybrid_alpha: float | None = None
    hybrid_rrf_constant: int = 60
    hybrid_union_n: int = 10
    hybrid_candidate_n: int | None = None


@dataclass
class ExperimentConfig:
    """Configuration identifying an experiment run."""

    name: str = "default_experiment"
    seed: int | None = 42


@dataclass
class ModelConfig:
    """Configuration for the inference model.

    Attributes:
        provider: Model backend. Currently only ``"huggingface"``.
        model_id: Hugging Face model identifier (e.g. ``"Qwen/Qwen2.5-Coder-3B-Instruct"``).
        revision: Model revision / git branch.
        device: Device to run the model on (``"auto"``, ``"cuda"``, ``"cpu"``).
        dtype: Data type for model weights (``"auto"``, ``"float16"``, ``"bfloat16"``, etc.).
    """

    provider: str = "huggingface"
    model_id: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    revision: str = "main"
    device: str = "auto"
    dtype: str = "auto"

    def __post_init__(self) -> None:
        if self.provider not in _VALID_PROVIDERS:
            raise ValueError(
                f"Unsupported provider: {self.provider!r}. "
                f"Must be one of: {sorted(_VALID_PROVIDERS)}"
            )
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if self.device not in _VALID_DEVICES:
            raise ValueError(
                f"Unsupported device: {self.device!r}. Must be one of: {sorted(_VALID_DEVICES)}"
            )
        if self.dtype not in _VALID_DTYPES:
            raise ValueError(
                f"Unsupported dtype: {self.dtype!r}. Must be one of: {sorted(_VALID_DTYPES)}"
            )


@dataclass
class GenerationConfig:
    """Parameters for text generation.

    Attributes:
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (only used when ``do_sample=True``).
        do_sample: Whether to use sampling. ``False`` means greedy decoding.
    """

    max_new_tokens: int = 256
    temperature: float = 0.0
    do_sample: bool = False
    batch_size: int = 1

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")


# ── Top-level config ────────────────────────────────────────────────


@dataclass
class Config:
    """Application configuration."""

    project_name: str = "modelbench"
    log_level: str = "INFO"
    data_dir: str = "datasets"

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    schema: SchemaConfig = field(default_factory=SchemaConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

    # Allow arbitrary extra keys from config files without breaking
    extra: dict[str, Any] = field(default_factory=dict)


# ── Loading ─────────────────────────────────────────────────────────

# Fields that are handled as nested dataclass sections
_NESTED_SECTIONS: dict[str, type] = {
    "experiment": ExperimentConfig,
    "dataset": DatasetConfig,
    "model": ModelConfig,
    "generation": GenerationConfig,
    "schema": SchemaConfig,
    "strategy": StrategyConfig,
}

# Flat fields on Config (excluding nested sections and extra)
_FLAT_FIELDS = {"project_name", "log_level", "data_dir"}


def _build_nested(cls: type, raw: dict[str, Any] | None) -> Any:
    """Construct a nested config dataclass from a raw dict."""
    if not raw or not isinstance(raw, dict):
        return cls()
    known_fields = {f.name for f in cls.__dataclass_fields__.values()}
    known = {k: v for k, v in raw.items() if k in known_fields}
    unknown = {k for k in raw if k not in known_fields}
    if unknown:
        logger.warning("Unknown config keys for %s: %s", cls.__name__, sorted(unknown))
    return cls(**known)


def load_config(path: Path | str | None = None) -> Config:
    """Load configuration from a YAML file.

    Resolution order:
      1. Explicit ``path`` argument.
      2. ``modelbench.yaml`` in the current working directory.
      3. Fall back to defaults.

    Nested sections (``model``, ``generation``) are parsed into their
    respective dataclass types with validation.

    Args:
        path: Optional explicit path to a YAML config file.

    Returns:
        A populated :class:`Config` instance.
    """
    config_path = Path(path) if path is not None else Path.cwd() / _DEFAULT_CONFIG_FILENAME

    if config_path.is_file():
        logger.debug("Loading config from %s", config_path)
        raw = yaml.safe_load(config_path.read_text()) or {}
    else:
        logger.debug("No config file found at %s, using defaults", config_path)
        raw = {}

    # Pop and build nested sections
    nested = {}
    for key, cls in _NESTED_SECTIONS.items():
        nested[key] = _build_nested(cls, raw.pop(key, None))

    # Build flat fields + extras
    known = {k: v for k, v in raw.items() if k in _FLAT_FIELDS}
    extra = {k: v for k, v in raw.items() if k not in _FLAT_FIELDS}

    return Config(**nested, **known, extra=extra)
