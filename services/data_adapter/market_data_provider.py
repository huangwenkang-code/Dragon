"""market_data_provider — the canonical entry point for all data access.

Every service imports from here, never from integrations/ directly.
"""

from __future__ import annotations

from typing import Optional

from services.data_adapter.adapters.tradingagents_adapter import (
    get_tradingagents_adapter,
)
from shared.utils.logging import get_logger

logger = get_logger(__name__)


async def get_market_data(
    stock_code: str,
    trade_date: str,
    *,
    days: int = 60,
) -> Optional[dict]:
    return await get_tradingagents_adapter().get_market_data(
        stock_code, trade_date, days=days
    )


async def get_news(
    stock_code: str,
    trade_date: str,
    *,
    limit: int = 20,
) -> Optional[list]:
    return await get_tradingagents_adapter().get_news(
        stock_code, trade_date, limit=limit
    )


async def get_sentiment(stock_code: str, trade_date: str) -> Optional[dict]:
    return await get_tradingagents_adapter().get_sentiment(stock_code, trade_date)


async def get_fundamentals(stock_code: str, trade_date: str) -> Optional[dict]:
    return await get_tradingagents_adapter().get_fundamentals(stock_code, trade_date)
