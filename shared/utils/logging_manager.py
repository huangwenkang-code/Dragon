"""Minimal logging stub — mirrors tradingagents.utils.logging_manager interface."""

import logging
import sys


def get_logger(name: str = "default") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
        ))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


class LoggerManager:
    """Drop-in replacement for the original LoggerManager."""
    def __init__(self):
        self._loggers: dict[str, logging.Logger] = {}

    def get_logger(self, name: str = "default") -> logging.Logger:
        if name not in self._loggers:
            self._loggers[name] = get_logger(name)
        return self._loggers[name]


def get_logger_manager() -> LoggerManager:
    return LoggerManager()
