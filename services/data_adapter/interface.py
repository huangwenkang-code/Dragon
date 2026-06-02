"""Unified data interface — adapter over TradingAgents-CN dataflows.

All service code calls this interface; the implementation delegates to
TradingAgents-CN providers via the adapter layer.  This keeps the
dependency unidirectional and replaceable.
"""

from typing import Optional

from shared.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Adapter — lazy import of TradingAgents-CN providers
# ---------------------------------------------------------------------------

_providers_loaded = False
_china_providers: dict = {}


def _ensure_providers():
    """Lazy-load TradingAgents-CN data providers on first access."""
    global _providers_loaded, _china_providers
    if _providers_loaded:
        return

    # Attempt to import from the sibling TradingAgents-CN-main repo.
    # The repo is NOT copied — we reference it at runtime via sys.path
    # or PYTHONPATH.  If unavailable, the adapter degrades gracefully.
    try:
        import sys
        from pathlib import Path

        _ta_path = Path(__file__).resolve().parents[3] / "TradingAgents-CN-main"
        if str(_ta_path) not in sys.path:
            sys.path.insert(0, str(_ta_path))

        from tradingagents.dataflows.interface import (  # type: ignore[import-untyped]
            get_stock_market_data_unified,
            get_stock_news_unified,
            get_stock_sentiment_unified,
            get_stock_fundamentals_unified,
        )

        _china_providers["market"] = get_stock_market_data_unified
        _china_providers["news"] = get_stock_news_unified
        _china_providers["sentiment"] = get_stock_sentiment_unified
        _china_providers["fundamentals"] = get_stock_fundamentals_unified

        _providers_loaded = True
        logger.info("data-adapter: TradingAgents-CN providers loaded")
    except Exception:
        logger.warning(
            "data-adapter: TradingAgents-CN providers unavailable — running in stub mode"
        )
        _providers_loaded = True  # mark as done so we don't retry every call


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def get_market_data(
    stock_code: str,
    trade_date: str,
    *,
    days: int = 60,
) -> Optional[dict]:
    """Fetch OHLCV + technical indicators for *stock_code*."""
    _ensure_providers()
    fn = _china_providers.get("market")
    if fn is None:
        logger.warning("[data-adapter] market data unavailable (stub)")
        return None
    try:
        return fn(stock_code, trade_date, days=days)
    except Exception as exc:
        logger.error("[data-adapter] get_market_data failed: %s", exc)
        return None


async def get_news(
    stock_code: str,
    trade_date: str,
    *,
    limit: int = 20,
) -> Optional[list]:
    """Fetch recent news for *stock_code*."""
    _ensure_providers()
    fn = _china_providers.get("news")
    if fn is None:
        logger.warning("[data-adapter] news unavailable (stub)")
        return None
    try:
        return fn(stock_code, trade_date, limit=limit)
    except Exception as exc:
        logger.error("[data-adapter] get_news failed: %s", exc)
        return None


async def get_sentiment(
    stock_code: str,
    trade_date: str,
) -> Optional[dict]:
    """Fetch social-media sentiment for *stock_code*."""
    _ensure_providers()
    fn = _china_providers.get("sentiment")
    if fn is None:
        logger.warning("[data-adapter] sentiment unavailable (stub)")
        return None
    try:
        return fn(stock_code, trade_date)
    except Exception as exc:
        logger.error("[data-adapter] get_sentiment failed: %s", exc)
        return None


async def get_fundamentals(
    stock_code: str,
    trade_date: str,
) -> Optional[dict]:
    """Fetch fundamental data for *stock_code*."""
    _ensure_providers()
    fn = _china_providers.get("fundamentals")
    if fn is None:
        logger.warning("[data-adapter] fundamentals unavailable (stub)")
        return None
    try:
        return fn(stock_code, trade_date)
    except Exception as exc:
        logger.error("[data-adapter] get_fundamentals failed: %s", exc)
        return None
