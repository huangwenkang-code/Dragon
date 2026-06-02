"""Adapter: dragon-engine → TradingAgents-CN dataflows (copied to integrations/).

This module wraps the copied TradingAgents-CN data providers behind the
dragon-engine data-adapter interface.  All service code calls this adapter;
never the integration code directly.
"""

from __future__ import annotations

from typing import Optional

from shared.utils.logging import get_logger

logger = get_logger(__name__)


class TradingAgentsDataAdapter:
    """Lazy-loading adapter for TradingAgents-CN data providers."""

    def __init__(self):
        self._loaded = False
        self._market_fn = None
        self._news_fn = None
        self._sentiment_fn = None
        self._fundamentals_fn = None

    def _ensure_loaded(self):
        if self._loaded:
            return
        try:
            from services.data_adapter.dataflows.interface import (  # type: ignore
                get_stock_market_data_unified,
                get_stock_news_unified,
                get_stock_sentiment_unified,
                get_stock_fundamentals_unified,
            )

            self._market_fn = get_stock_market_data_unified
            self._news_fn = get_stock_news_unified
            self._sentiment_fn = get_stock_sentiment_unified
            self._fundamentals_fn = get_stock_fundamentals_unified
            self._loaded = True
            logger.info("TradingAgentsDataAdapter: providers loaded")
        except Exception as exc:
            logger.warning("TradingAgentsDataAdapter: unavailable — %s", exc)
            self._loaded = True  # don't retry

    # ------------------------------------------------------------------
    # Public API (mirrors data-adapter interface)
    # ------------------------------------------------------------------

    async def get_market_data(
        self, stock_code: str, trade_date: str, *, days: int = 60
    ) -> Optional[dict]:
        self._ensure_loaded()
        if self._market_fn is None:
            return None
        try:
            return self._market_fn(stock_code, trade_date, days=days)
        except Exception as exc:
            logger.error("get_market_data failed for %s: %s", stock_code, exc)
            return None

    async def get_news(
        self, stock_code: str, trade_date: str, *, limit: int = 20
    ) -> Optional[list]:
        self._ensure_loaded()
        if self._news_fn is None:
            return None
        try:
            return self._news_fn(stock_code, trade_date, limit=limit)
        except Exception as exc:
            logger.error("get_news failed for %s: %s", stock_code, exc)
            return None

    async def get_sentiment(
        self, stock_code: str, trade_date: str
    ) -> Optional[dict]:
        self._ensure_loaded()
        if self._sentiment_fn is None:
            return None
        try:
            return self._sentiment_fn(stock_code, trade_date)
        except Exception as exc:
            logger.error("get_sentiment failed for %s: %s", stock_code, exc)
            return None

    async def get_fundamentals(
        self, stock_code: str, trade_date: str
    ) -> Optional[dict]:
        self._ensure_loaded()
        if self._fundamentals_fn is None:
            return None
        try:
            return self._fundamentals_fn(stock_code, trade_date)
        except Exception as exc:
            logger.error("get_fundamentals failed for %s: %s", stock_code, exc)
            return None


# Singleton
_adapter: Optional[TradingAgentsDataAdapter] = None


def get_tradingagents_adapter() -> TradingAgentsDataAdapter:
    global _adapter
    if _adapter is None:
        _adapter = TradingAgentsDataAdapter()
    return _adapter
