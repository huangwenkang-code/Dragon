"""
ChromaDB unified configuration module for dragon-engine.
Supports Windows 10/11 and other OS auto-detection.

Adapted from TradingAgents-CN chromadb_config.py.
"""
import os
import platform
from typing import Optional

from shared.utils.logging import get_logger

logger = get_logger(__name__)

# Conditional import — chromadb may not be available in all environments.
_chromadb_available = False
_chromadb_Settings = None
_chromadb_Client = None
try:
    import chromadb
    from chromadb.config import Settings

    _chromadb_available = True
    _chromadb_Settings = Settings
    _chromadb_Client = chromadb.Client
except ImportError:
    logger.warning("ChromaDB not installed. Memory persistence disabled.")


def is_windows_11() -> bool:
    """
    Detect whether the host is Windows 11.

    Returns:
        bool: True if Windows 11, False otherwise.
    """
    if platform.system() != "Windows":
        return False

    # Windows 11 build number starts at 22000.
    version = platform.version()
    try:
        version_parts = version.split('.')
        if len(version_parts) >= 3:
            build_number = int(version_parts[2])
            return build_number >= 22000
    except (ValueError, IndexError):
        pass

    return False


def get_win10_chromadb_client() -> Optional[object]:
    """
    Return a Windows 10-compatible ChromaDB client.

    Returns:
        chromadb.Client or None if ChromaDB is unavailable.
    """
    if not _chromadb_available:
        return None

    settings = _chromadb_Settings(
        allow_reset=True,
        anonymized_telemetry=False,
        is_persistent=False,
        chroma_db_impl="duckdb+parquet",
        chroma_api_impl="chromadb.api.segment.SegmentAPI",
        persist_directory=None,
    )

    try:
        return _chromadb_Client(settings)
    except Exception:
        basic_settings = _chromadb_Settings(
            allow_reset=True,
            is_persistent=False,
        )
        return _chromadb_Client(basic_settings)


def get_win11_chromadb_client() -> Optional[object]:
    """
    Return a Windows 11-optimized ChromaDB client.

    Returns:
        chromadb.Client or None if ChromaDB is unavailable.
    """
    if not _chromadb_available:
        return None

    settings = _chromadb_Settings(
        allow_reset=True,
        anonymized_telemetry=False,
        is_persistent=False,
        chroma_db_impl="duckdb+parquet",
        chroma_api_impl="chromadb.api.segment.SegmentAPI",
    )

    try:
        return _chromadb_Client(settings)
    except Exception:
        minimal_settings = _chromadb_Settings(
            allow_reset=True,
            anonymized_telemetry=False,
            is_persistent=False,
        )
        return _chromadb_Client(minimal_settings)


def get_optimal_chromadb_client() -> Optional[object]:
    """
    Auto-select the best ChromaDB configuration for the current OS.

    Returns:
        chromadb.Client or None if ChromaDB is unavailable.
    """
    if not _chromadb_available:
        logger.warning("ChromaDB not available — returning None client.")
        return None

    system = platform.system()

    if system == "Windows":
        if is_windows_11():
            logger.info(
                "ChromaDB Windows 11 optimized config (build %s)",
                platform.version(),
            )
            return get_win11_chromadb_client()
        else:
            logger.info("ChromaDB Windows 10 compatible config")
            return get_win10_chromadb_client()
    else:
        settings = _chromadb_Settings(
            allow_reset=True,
            anonymized_telemetry=False,
            is_persistent=False,
        )
        logger.info("ChromaDB %s standard config", system)
        return _chromadb_Client(settings)


__all__ = [
    'get_optimal_chromadb_client',
    'get_win10_chromadb_client',
    'get_win11_chromadb_client',
    'is_windows_11',
]
