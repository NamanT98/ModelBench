"""Application logging configuration."""

import logging
import sys


def setup_logging(verbose: bool = False) -> None:
    """Configure application-wide logging.

    Args:
        verbose: If True, set log level to DEBUG. Otherwise use INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root_logger = logging.getLogger("modelbench")
    root_logger.setLevel(level)
    root_logger.addHandler(handler)

    # Prevent duplicate handlers on repeated calls
    root_logger.handlers = [handler]
