"""
Capital flow data fetcher for dragon-engine cognitive engine.

Wraps AKShare native APIs to retrieve:
  - Individual stock fund flows (big/small order net flows)
  - Market-wide fund flows
  - Northbound (沪港通/深港通) capital flows
  - Sector-level fund flows

All functions use defensive imports so the module is importable even
when akshare is not installed.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from shared.utils.logging import get_logger

logger = get_logger(__name__)

# Conditional import — the module stays importable without akshare.
_ak_available = False
_ak = None
try:
    import akshare as _ak  # type: ignore[no-redef]
    _ak_available = True
except ImportError:
    logger.warning("akshare not installed — capital flow fetcher will return empty DataFrames")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _determine_market(code: str) -> str:
    """Map a 6-digit stock code to 'sh' or 'sz'."""
    code = str(code).zfill(6)
    if code.startswith(("60", "68", "90")):
        return "sh"
    elif code.startswith(("00", "30", "20")):
        return "sz"
    elif code.startswith(("8", "4")):
        return "bj"
    return "sh"


def _empty_df(columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _safe_call(fn, *args: Any, **kwargs: Any) -> pd.DataFrame:
    """Call *fn* with fallback to an empty DataFrame on any error."""
    if not _ak_available:
        return pd.DataFrame()
    try:
        result = fn(*args, **kwargs)
        if result is None:
            return pd.DataFrame()
        if isinstance(result, pd.DataFrame):
            return result
        return pd.DataFrame(result)
    except Exception as e:
        logger.warning("AKShare call %s failed: %s", getattr(fn, "__name__", str(fn)), e)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_individual_fund_flow(stock_code: str) -> pd.DataFrame:
    """Fetch individual stock fund flow data.

    Uses ``ak.stock_individual_fund_flow(stock=code, market="sh"/"sz")``.

    Args:
        stock_code: 6-digit stock code, e.g. "600519".

    Returns:
        DataFrame with columns typically including: 日期, 主力净流入, etc.
        Empty DataFrame on failure.
    """
    if not _ak_available:
        logger.warning("AKShare unavailable — cannot fetch individual fund flow for %s", stock_code)
        return _empty_df(["date", "main_force_net", "close"])

    market = _determine_market(stock_code)
    logger.info("Fetching individual fund flow for %s (market=%s)", stock_code, market)

    try:
        df = _ak.stock_individual_fund_flow(stock=stock_code, market=market)
    except Exception as e:
        logger.error("stock_individual_fund_flow(%s, %s) failed: %s", stock_code, market, e)
        return _empty_df(["date", "main_force_net", "close"])

    if df is None or df.empty:
        logger.warning("Empty individual fund flow for %s", stock_code)
        return _empty_df(["date", "main_force_net", "close"])

    logger.info("Fetched %d rows of individual fund flow for %s", len(df), stock_code)
    return df


def fetch_market_fund_flow() -> pd.DataFrame:
    """Fetch market-wide fund flow data.

    Uses ``ak.stock_market_fund_flow()``.

    Returns:
        DataFrame with market-level inflow/outflow data.
    """
    if not _ak_available:
        logger.warning("AKShare unavailable — cannot fetch market fund flow")
        return _empty_df(["date", "main_force_net"])

    logger.info("Fetching market-wide fund flow")
    df = _safe_call(_ak.stock_market_fund_flow)
    if not df.empty:
        logger.info("Fetched %d rows of market fund flow", len(df))
    return df


def fetch_northbound_flow(indicator: str = "north_flow") -> pd.DataFrame:
    """Fetch northbound (沪港通/深港通) capital flow data.

    Uses ``ak.stock_hsgt_north_flow(indicator=...)``.

    Args:
        indicator: One of "north_flow" (default), "south_flow",
            "north_net", "south_net", "north_balance", "south_balance".

    Returns:
        DataFrame with northbound flow data.
    """
    if not _ak_available:
        logger.warning("AKShare unavailable — cannot fetch northbound flow")
        return _empty_df(["date", "net_flow"])

    valid_indicators = {
        "north_flow", "south_flow", "north_net",
        "south_net", "north_balance", "south_balance",
    }
    if indicator not in valid_indicators:
        logger.warning(
            "Unknown indicator '%s' — falling back to 'north_flow'. Valid: %s",
            indicator, sorted(valid_indicators),
        )
        indicator = "north_flow"

    logger.info("Fetching northbound flow (indicator=%s)", indicator)
    df = _safe_call(_ak.stock_hsgt_north_flow, indicator=indicator)
    if not df.empty:
        logger.info("Fetched %d rows of northbound flow", len(df))
    return df


def fetch_sector_fund_flow() -> pd.DataFrame:
    """Fetch sector-level fund flow data.

    Uses ``ak.stock_sector_fund_flow()`` (or
    ``stock_sector_fund_flow_rank`` as a fallback).

    Returns:
        DataFrame with sector-level inflow/outflow data.
    """
    if not _ak_available:
        logger.warning("AKShare unavailable — cannot fetch sector fund flow")
        return _empty_df(["sector", "main_force_net"])

    logger.info("Fetching sector fund flow")

    # Try the standard function first.
    df = _safe_call(_ak.stock_sector_fund_flow)
    if df is not None and not df.empty:
        logger.info("Fetched %d rows of sector fund flow", len(df))
        return df

    # Fallback: try _rank variant.
    try:
        df = _ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="industry")
        if df is not None and not df.empty:
            logger.info("Fetched %d rows via _rank fallback", len(df))
            return df
    except Exception:
        pass

    logger.warning("Sector fund flow returned empty")
    return _empty_df(["sector", "main_force_net"])


def fetch_all_flows(watchlist: List[str]) -> Dict[str, Any]:
    """Batch-fetch all capital flow types for a watchlist.

    Args:
        watchlist: List of 6-digit stock codes.

    Returns:
        Dict with keys:
            - "individual": Dict[str, pd.DataFrame] — code -> flow DataFrame
            - "market": pd.DataFrame
            - "northbound": pd.DataFrame
            - "sector": pd.DataFrame
            - "fetched_at": ISO-format timestamp
    """
    logger.info("Batch fetching capital flows for %d symbols", len(watchlist))

    individual: Dict[str, pd.DataFrame] = {}
    for code in watchlist:
        individual[code] = fetch_individual_fund_flow(code)

    result: Dict[str, Any] = {
        "individual": individual,
        "market": fetch_market_fund_flow(),
        "northbound": fetch_northbound_flow(),
        "sector": fetch_sector_fund_flow(),
        "fetched_at": datetime.now().isoformat(),
    }

    individual_count = sum(1 for df in individual.values() if not df.empty)
    logger.info(
        "Capital flow batch complete: %d/%d individual, market=%s, north=%s, sector=%s",
        individual_count,
        len(watchlist),
        "OK" if not result["market"].empty else "empty",
        "OK" if not result["northbound"].empty else "empty",
        "OK" if not result["sector"].empty else "empty",
    )

    return result


# ---------------------------------------------------------------------------
# Test block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=== Capital Flow Fetcher — Self Test ===\n")

    if not _ak_available:
        print("AKShare is NOT available. Tests will return empty DataFrames.\n")
    else:
        print(f"AKShare available (version: {getattr(_ak, '__version__', 'unknown')})\n")

    # --- 1. Individual fund flow ---
    print("--- Test 1: Individual Fund Flow (600519) ---")
    ind_df = fetch_individual_fund_flow("600519")
    if not ind_df.empty:
        print(ind_df.tail(3).to_string())
    else:
        print("[empty]")
    print()

    # --- 2. Market fund flow ---
    print("--- Test 2: Market Fund Flow ---")
    mkt_df = fetch_market_fund_flow()
    if not mkt_df.empty:
        print(mkt_df.head(3).to_string())
    else:
        print("[empty]")
    print()

    # --- 3. Northbound flow ---
    print("--- Test 3: Northbound Flow ---")
    nb_df = fetch_northbound_flow()
    if not nb_df.empty:
        print(nb_df.head(3).to_string())
    else:
        print("[empty]")
    print()

    # --- 4. Sector fund flow ---
    print("--- Test 4: Sector Fund Flow ---")
    sec_df = fetch_sector_fund_flow()
    if not sec_df.empty:
        print(sec_df.head(3).to_string())
    else:
        print("[empty]")
    print()

    # --- 5. Batch fetch ---
    print("--- Test 5: Batch fetch_all_flows ---")
    batch = fetch_all_flows(["600519", "000858"])
    print(f"  Individual: {len(batch['individual'])} symbols")
    for code, df in batch["individual"].items():
        print(f"    {code}: {len(df)} rows")
    print(f"  Market:     {len(batch['market'])} rows")
    print(f"  Northbound: {len(batch['northbound'])} rows")
    print(f"  Sector:     {len(batch['sector'])} rows")
    print(f"  Fetched at: {batch['fetched_at']}")

    print("\n=== Test complete ===")
