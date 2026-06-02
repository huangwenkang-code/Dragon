"""Price-based sentiment calibration for FinBERT.

FinBERT classifies the *tone of news text*, not the actual health of a stock.
When news reports "X stock hits limit-up" it's 99% positive — even if the stock
just crashed 50% in 12 consecutive limit-downs.

This module provides price-based penalty scores that temper FinBERT's optimism:
  - Fetches recent 20 daily bars per stock (AKShare Sina, LRU-cached)
  - Computes a "distress penalty" (0-1) from returns, limit-downs, consecutive drops
  - The penalty is applied AFTER FinBERT inference to pull down positive scores

Usage in analyzer.py:
    penalty = get_price_penalty(symbol)
    finbert_pos_adj = finbert_pos * (1 - penalty)
    sentiment = finbert_pos_adj - finbert_neg
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

from shared.utils.logging import get_logger

logger = get_logger(__name__)


def _to_sina_symbol(symbol: str) -> str:
    """Convert 6-digit A-share code to Sina API format (sh600000 / sz000001)."""
    if symbol.startswith(("6", "68")):
        return f"sh{symbol}"
    return f"sz{symbol}"


@lru_cache(maxsize=128)
def _fetch_recent_bars(symbol: str, days: int = 20):
    """Fetch recent daily bars for a single stock (LRU-cached per process)."""
    import akshare as ak

    end = date.today()
    start = end - timedelta(days=max(60, days * 3))

    df = ak.stock_zh_a_daily(
        symbol=_to_sina_symbol(symbol),
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        adjust="qfq",
    )
    return df.tail(days)


def get_price_penalty(symbol: str) -> float:
    """Compute a distress penalty (0-1) from recent price action.

    0   = stock is healthy, no penalty applied
    0.5 = moderate distress (e.g. -10% in 10 days)
    1.0 = severe distress (e.g. multiple limit-downs, -30% in 10 days)

    Returns 0 if price data is unavailable (no penalty → trust FinBERT raw).
    """
    try:
        df = _fetch_recent_bars(symbol, days=20)
        if df.empty or len(df) < 3:
            return 0.0

        closes = df["close"].astype(float).values
        current = closes[-1]

        # Compute 10-day return
        ret10 = 0.0
        if len(closes) > 10:
            ret10 = (current / closes[-11] - 1) * 100

        # Count limit-downs and consecutive drops in the window
        limit_downs = 0
        max_consecutive_drops = 0
        consecutive_drops = 0
        for i in range(1, len(closes)):
            chg = (closes[i] / closes[i - 1] - 1) * 100
            if chg <= -9.5:
                limit_downs += 1
            if chg < 0:
                consecutive_drops += 1
                max_consecutive_drops = max(max_consecutive_drops, consecutive_drops)
            else:
                consecutive_drops = 0
        max_consecutive_drops = max(max_consecutive_drops, consecutive_drops)

        # --- Penalty scoring rules ---
        penalty = 0.0

        # 10-day return component (most important)
        if ret10 <= -30:
            penalty += 0.6
        elif ret10 <= -20:
            penalty += 0.45
        elif ret10 <= -10:
            penalty += 0.25
        elif ret10 <= -5:
            penalty += 0.1

        # Limit-down component
        if limit_downs >= 3:
            penalty += 0.3
        elif limit_downs >= 1:
            penalty += 0.2

        # Consecutive drops component
        if max_consecutive_drops >= 8:
            penalty += 0.2
        elif max_consecutive_drops >= 5:
            penalty += 0.1

        return min(penalty, 0.85)  # cap: even worst case keeps some signal

    except Exception:
        return 0.0


def get_price_context(symbol: str) -> str:
    """Build a human-readable price context string (for logging/debug)."""
    try:
        df = _fetch_recent_bars(symbol, days=20)
        if df.empty or len(df) < 3:
            return ""

        closes = df["close"].astype(float).values
        current = closes[-1]

        def ret(n: int) -> float:
            if len(closes) <= n:
                return 0.0
            return round((current / closes[-(n + 1)] - 1) * 100, 1)

        penalty = get_price_penalty(symbol)
        parts = [f"近5日:{ret(5):+.1f}%", f"近10日:{ret(10):+.1f}%"]
        if abs(ret(20)) > 0.1:
            parts.append(f"近20日:{ret(20):+.1f}%")

        return f"[{symbol}] " + ", ".join(parts) + f" | penalty={penalty:.2f}"

    except Exception:
        return ""
