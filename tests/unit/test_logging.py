import logging

from agentic_bi.config.logging import get_logger


def test_get_logger_returns_named_logger():
    logger = get_logger("test")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "test"