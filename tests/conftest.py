"""Shared pytest fixtures."""

from pathlib import Path

import pytest

from modelbench.fixture import create_fixture_db


@pytest.fixture()
def fixture_db(tmp_path: Path) -> Path:
    """Create the fixture e-commerce database in a temporary directory."""
    db_path = tmp_path / "fixture_ecommerce.db"
    create_fixture_db(db_path)
    return db_path
