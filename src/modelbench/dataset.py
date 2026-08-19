"""Dataset abstraction and loading for ModelBench.

Currently supports loading the official Spider dataset JSON format.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Iterator
from pathlib import Path

from modelbench.config import DatasetConfig
from modelbench.types import TextToSQLSample

logger = logging.getLogger(__name__)


class SpiderDataset:
    """Adapter for the official Spider Text-to-SQL dataset.

    Expects a directory structure:
        <dataset_path>/
            dev.json
            train_spider.json
            database/
                <db_id>/
                    <db_id>.sqlite
    """

    def __init__(self, config: DatasetConfig):
        """Initialize the Spider dataset adapter.

        Args:
            config: Dataset configuration specifying path, split, limit, and seed.
        """
        self.config = config

        if not self.config.path:
            raise ValueError("Dataset path must be configured for the Spider dataset.")

        self.base_path = Path(self.config.path)
        if not self.base_path.is_dir():
            raise FileNotFoundError(f"Spider dataset directory not found: {self.base_path}")

        # Standard spider files are usually named based on the split
        split_file = f"{self.config.split}.json"
        if self.config.split == "train":
            split_file = "train_spider.json"

        self.data_file = self.base_path / split_file
        if not self.data_file.is_file():
            raise FileNotFoundError(
                f"Spider split file not found: {self.data_file}. "
                "Ensure the dataset split is correct and files exist."
            )

        self.db_dir = self.base_path / "database"
        if not self.db_dir.is_dir():
            raise FileNotFoundError(f"Spider databases directory not found: {self.db_dir}.")

    def load(self) -> Iterator[TextToSQLSample]:
        """Load and yield deterministic samples from the dataset.

        Yields:
            TextToSQLSample objects ready for evaluation.
        """
        logger.info("Loading Spider dataset from %s", self.data_file)
        with self.data_file.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            raise ValueError(f"Expected a JSON list in {self.data_file}")

        # Deterministic sub-sampling
        if self.config.seed is not None:
            # We seed a local Random instance to avoid global state side-effects
            rng = random.Random(self.config.seed)
            # Shuffle deterministically
            rng.shuffle(raw_data)

        if self.config.limit is not None:
            raw_data = raw_data[: self.config.limit]

        logger.info("Yielding %d samples from Spider dataset", len(raw_data))

        for item in raw_data:
            db_id = item.get("db_id")
            question = item.get("question")
            gold_sql = item.get("query")

            if not db_id or not question or not gold_sql:
                logger.warning("Skipping invalid sample missing required fields: %s", item)
                continue

            db_path = self.db_dir / db_id / f"{db_id}.sqlite"
            if not db_path.is_file():
                logger.warning("Database not found for db_id %r: %s", db_id, db_path)
                # For robustness, we still yield it, the evaluator will fail correctly

            yield TextToSQLSample(
                question=question,
                db_id=db_id,
                db_path=str(db_path),
                gold_sql=gold_sql,
            )
