"""Minimal logging init stub — mirrors tradingagents.utils.logging_init interface."""

from shared.utils.logging_manager import get_logger


def setup_dataflow_logging():
    """Stub — returns a logger configured for dataflow modules."""
    return get_logger("dataflow")


def setup_llm_logging():
    """Stub — returns a logger configured for LLM modules."""
    return get_logger("llm")
