"""Configuration management for ModelBench."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_FILENAME = "modelbench.yaml"


@dataclass
class Config:
    """Application configuration."""

    project_name: str = "modelbench"
    log_level: str = "INFO"
    data_dir: str = "datasets"

    # Allow arbitrary extra keys from config files without breaking
    extra: dict[str, Any] = field(default_factory=dict)


def load_config(path: Path | str | None = None) -> Config:
    """Load configuration from a YAML file.

    Resolution order:
      1. Explicit ``path`` argument.
      2. ``modelbench.yaml`` in the current working directory.
      3. Fall back to defaults.

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

    known_fields = {f.name for f in Config.__dataclass_fields__.values() if f.name != "extra"}
    known = {k: v for k, v in raw.items() if k in known_fields}
    extra = {k: v for k, v in raw.items() if k not in known_fields}

    return Config(**known, extra=extra)
