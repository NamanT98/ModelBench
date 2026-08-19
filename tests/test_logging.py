"""Tests for ModelBench logging setup."""

import logging

from modelbench.logging import setup_logging


class TestLogging:
    """Test logging configuration."""

    def test_default_level_is_info(self) -> None:
        setup_logging(verbose=False)
        logger = logging.getLogger("modelbench")
        assert logger.level == logging.INFO

    def test_verbose_sets_debug(self) -> None:
        setup_logging(verbose=True)
        logger = logging.getLogger("modelbench")
        assert logger.level == logging.DEBUG

    def test_has_exactly_one_handler(self) -> None:
        setup_logging()
        setup_logging()  # Call twice to verify deduplication
        logger = logging.getLogger("modelbench")
        assert len(logger.handlers) == 1
