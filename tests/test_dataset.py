from pathlib import Path

from modelbench.config import DatasetConfig
from modelbench.dataset import SpiderDataset


def test_spider_dataset_load():
    config = DatasetConfig(
        name="spider", path="tests/data/spider_fixture", split="dev", limit=None, seed=None
    )
    dataset = SpiderDataset(config)
    samples = list(dataset.load())

    assert len(samples) == 2
    assert Path(samples[0].db_path).name == "fixture_ecommerce.sqlite"
    assert samples[0].gold_sql == "SELECT COUNT(*) FROM customers"


def test_spider_dataset_limit():
    config = DatasetConfig(
        name="spider", path="tests/data/spider_fixture", split="dev", limit=1, seed=None
    )
    dataset = SpiderDataset(config)
    samples = list(dataset.load())

    assert len(samples) == 1
    assert samples[0].question == "How many customers are there?"


def test_spider_dataset_seed():
    # If seed is provided, ordering should be deterministic.
    # We only have 2 samples so it might just swap them.
    config1 = DatasetConfig(name="spider", path="tests/data/spider_fixture", split="dev", seed=42)
    config2 = DatasetConfig(name="spider", path="tests/data/spider_fixture", split="dev", seed=42)

    ds1 = list(SpiderDataset(config1).load())
    ds2 = list(SpiderDataset(config2).load())

    assert ds1[0].question == ds2[0].question
