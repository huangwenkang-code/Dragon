"""
Dragon-Tiger Board (龙虎榜) data fetcher for dragon-engine cognitive engine.

Wraps AKShare / Tushare APIs to retrieve:
  - Daily dragon-tiger board list
  - Individual stock LHB detail
  - Institutional LHB activity
  - Famous trader seat identification (游资席位识别)

All functions use defensive imports so the module is importable even
when the underlying data providers are not installed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from shared.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------
_ak_available = False
_ak = None
try:
    import akshare as _ak  # type: ignore[no-redef]
    _ak_available = True
except ImportError:
    logger.warning("akshare not installed — LHB fetcher will return empty DataFrames")

_tushare_available = False
_ts = None
try:
    import tushare as _ts  # type: ignore[no-redef]
    _tushare_available = True
except ImportError:
    pass  # Tushare is optional


# ---------------------------------------------------------------------------
# Famous trader seat mapping (游资席位 -> 知名游资)
# ---------------------------------------------------------------------------

FAMOUS_TRADER_SEATS: Dict[str, str] = {
    "中信证券上海分公司": "章盟主",
    "华泰证券总部": "量化/机构混合",
    "国泰君安上海分公司": "赵老哥",
    "中信建投上海分公司": "作手新一",
    "东方证券上海分公司": "炒股养家",
    "招商证券深圳分公司": "活跃游资",
    # Extended (commonly observed aliases / variants)
    "中信证券股份有限公司上海分公司": "章盟主",
    "国泰君安证券股份有限公司上海分公司": "赵老哥",
    "中信建投证券股份有限公司上海分公司": "作手新一",
    "东方证券股份有限公司上海分公司": "炒股养家",
    "招商证券股份有限公司深圳分公司": "活跃游资",
    "华鑫证券上海分公司": "量化打板",
    "华鑫证券有限责任公司上海分公司": "量化打板",
    "财通证券杭州上塘路": "上塘路",
    "财通证券股份有限公司杭州上塘路证券营业部": "上塘路",
    "中国银河证券北京中关村大街": "中关村",
    "中国银河证券股份有限公司北京中关村大街证券营业部": "中关村",
    "国盛证券宁波桑田路": "桑田路",
    "国盛证券有限责任公司宁波桑田路证券营业部": "桑田路",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_trade_date(date_str: str) -> str:
    """Normalize a trade date string to YYYYMMDD format."""
    cleaned = date_str.replace("-", "").replace("/", "")
    if len(cleaned) == 8:
        return cleaned
    # Try to parse and format.
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    return cleaned


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


def fetch_daily_lhb_list(trade_date: str) -> pd.DataFrame:
    """Fetch the full daily dragon-tiger board list.

    Uses ``ak.stock_lhb_detail_em(start_date=..., end_date=...)``.

    Args:
        trade_date: Date in YYYY-MM-DD or YYYYMMDD format.

    Returns:
        DataFrame with columns typically including: 代码, 名称, 上榜原因,
        净买额, 买入额, 卖出额, etc.  Empty DataFrame on failure.
    """
    date_fmt = _format_trade_date(trade_date)
    logger.info("Fetching daily LHB list for %s", date_fmt)

    if not _ak_available:
        logger.warning("AKShare unavailable — cannot fetch LHB list")
        return _empty_df(["code", "name", "reason", "net_amount"])

    try:
        df = _ak.stock_lhb_detail_em(start_date=date_fmt, end_date=date_fmt)
    except TypeError:
        # Some AKShare versions use date= instead of start_date/end_date.
        try:
            df = _ak.stock_lhb_detail_em(date=date_fmt)
        except Exception:
            df = _ak.stock_lhb_detail_em()
    except Exception as e:
        logger.error("stock_lhb_detail_em(%s) failed: %s", date_fmt, e)
        return _empty_df(["code", "name", "reason", "net_amount"])

    if df is None or df.empty:
        logger.info("No LHB entries for %s", date_fmt)
        return _empty_df(["code", "name", "reason", "net_amount"])

    logger.info("Fetched %d LHB entries for %s", len(df), date_fmt)

    # Attach trader tags where the seat column exists.
    df = identify_famous_traders(df)
    return df


def fetch_lhb_stock_detail(stock_code: str, trade_date: str) -> pd.DataFrame:
    """Fetch individual stock dragon-tiger board detail.

    Uses ``ak.stock_lhb_stock_detail_em(symbol=code, date=...)`` with
    fallback to filtering the daily list.

    Args:
        stock_code: 6-digit stock code.
        trade_date: Date in YYYY-MM-DD or YYYYMMDD format.

    Returns:
        DataFrame with seat-level buy/sell breakdown. Empty on failure.
    """
    date_fmt = _format_trade_date(trade_date)
    logger.info("Fetching LHB detail for %s on %s", stock_code, date_fmt)

    if not _ak_available:
        return _empty_df(["seat", "buy_amount", "sell_amount"])

    # Strategy 1: Use the single-stock detail function.
    try:
        df = _ak.stock_lhb_stock_detail_em(symbol=stock_code, date=date_fmt)
        if df is not None and not df.empty:
            logger.info("Fetched %d detail rows for %s", len(df), stock_code)
            df = identify_famous_traders(df)
            return df
    except Exception as e:
        logger.debug("stock_lhb_stock_detail_em(%s, %s) failed: %s", stock_code, date_fmt, e)

    # Strategy 2: Fall back to filtering the daily list.
    daily_list = fetch_daily_lhb_list(trade_date)
    if daily_list.empty:
        return _empty_df(["seat", "buy_amount", "sell_amount"])

    # Find the stock in the list and get its detail.
    code_col = _find_column(daily_list, ["代码", "code", "symbol"])
    if code_col:
        matches = daily_list[daily_list[code_col].astype(str).str.contains(stock_code)]
        if not matches.empty:
            logger.info("Found %s in daily LHB list via fallback", stock_code)
            return matches

    logger.warning("No LHB detail found for %s on %s", stock_code, date_fmt)
    return _empty_df(["seat", "buy_amount", "sell_amount"])


def fetch_institutional_lhb(trade_date: str) -> pd.DataFrame:
    """Fetch institutional dragon-tiger board data.

    Uses ``ak.stock_lhb_jgmm(trade_date=...)``.

    Args:
        trade_date: Date in YYYY-MM-DD or YYYYMMDD format.

    Returns:
        DataFrame with institutional buy/sell activity. Empty on failure.
    """
    date_fmt = _format_trade_date(trade_date)
    logger.info("Fetching institutional LHB for %s", date_fmt)

    if not _ak_available:
        return _empty_df(["code", "name", "buy_amount", "sell_amount"])

    try:
        df = _ak.stock_lhb_jgmm(trade_date=date_fmt)
    except TypeError:
        try:
            df = _ak.stock_lhb_jgmm(date=date_fmt)
        except Exception:
            df = _ak.stock_lhb_jgmm()
    except Exception as e:
        logger.error("stock_lhb_jgmm(%s) failed: %s", date_fmt, e)
        return _empty_df(["code", "name", "buy_amount", "sell_amount"])

    if df is None or df.empty:
        logger.info("No institutional LHB data for %s", date_fmt)
        return _empty_df(["code", "name", "buy_amount", "sell_amount"])

    logger.info("Fetched %d institutional LHB rows for %s", len(df), date_fmt)
    return df


# ---------------------------------------------------------------------------
# Trader seat identification
# ---------------------------------------------------------------------------


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first candidate column name that exists in *df*."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def identify_famous_traders(df: pd.DataFrame) -> pd.DataFrame:
    """Tag rows with identified famous trader names (游资).

    Scans seat-related columns for known trader seat keywords and
    adds a ``trader_tag`` column.

    Args:
        df: LHB DataFrame with seat name columns.

    Returns:
        Same DataFrame with an added ``trader_tag`` column.
    """
    if df.empty:
        df = df.copy()
        df["trader_tag"] = ""
        return df

    df = df.copy()

    # Find seat name columns.
    seat_cols = [c for c in df.columns if "席位" in str(c) or "营业部" in str(c) or "seat" in str(c).lower()]
    if not seat_cols:
        # Try common Chinese column names.
        seat_cols = [c for c in df.columns if "买方" in str(c) or "卖方" in str(c) or "交易营业部" in str(c)]

    tags: List[str] = []
    for _, row in df.iterrows():
        tag = ""
        for col in seat_cols:
            seat_name = str(row.get(col, ""))
            if not seat_name or seat_name == "nan":
                continue
            for keyword, trader in FAMOUS_TRADER_SEATS.items():
                if keyword in seat_name:
                    if tag:
                        tag += ";" + trader
                    else:
                        tag = trader
                    break  # First match per seat column
        tags.append(tag)

    df["trader_tag"] = tags

    tagged = sum(1 for t in tags if t)
    if tagged:
        logger.info("Tagged %d row(s) with famous trader labels", tagged)

    return df


# ---------------------------------------------------------------------------
# Batch fetch
# ---------------------------------------------------------------------------


def fetch_all_lhb(watchlist: List[str], trade_date: str) -> Dict[str, Any]:
    """Batch-fetch LHB data for a watchlist.

    Args:
        watchlist: List of 6-digit stock codes.
        trade_date: Trade date in YYYY-MM-DD or YYYYMMDD format.

    Returns:
        Dict with keys:
            - "daily_list": pd.DataFrame — full LHB list for the date
            - "stock_details": Dict[str, pd.DataFrame] — per-stock detail
            - "institutional": pd.DataFrame — institutional activity
            - "on_board": List[str] — codes from watchlist that appear on LHB
            - "trader_summary": Dict[str, int] — trader tag -> appearance count
            - "fetched_at": ISO-format timestamp
    """
    logger.info("Batch fetching LHB for %d symbols on %s", len(watchlist), trade_date)

    daily_list = fetch_daily_lhb_list(trade_date)

    # Determine which watchlist stocks are on the board.
    on_board: List[str] = []
    stock_details: Dict[str, pd.DataFrame] = {}

    if not daily_list.empty:
        code_col = _find_column(daily_list, ["代码", "code", "symbol"])
        if code_col:
            board_codes = set(daily_list[code_col].astype(str).str.zfill(6))
            if watchlist:
                watch_set = {str(c).zfill(6) for c in watchlist}
                on_board = sorted(watch_set & board_codes)
            else:
                # Empty watchlist = return ALL on-board stocks
                on_board = sorted(board_codes)
        else:
            # Try to find by partial matching.
            board_str = daily_list.to_string()
            for code in watchlist:
                if code in board_str:
                    on_board.append(code)

    # Parallel fetch LHB detail for top N stocks
    top_codes = on_board[:30]
    if top_codes:
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(fetch_lhb_stock_detail, code, trade_date): code for code in top_codes}
            for future in as_completed(futures):
                code = futures[future]
                try:
                    stock_details[code] = future.result()
                except Exception as exc:
                    logger.debug("LHB detail fetch failed for %s: %s", code, exc)

    institutional = fetch_institutional_lhb(trade_date)

    # Summarize trader appearances.
    trader_summary: Dict[str, int] = {}
    if "trader_tag" in daily_list.columns:
        for tag in daily_list["trader_tag"].dropna():
            tag_str = str(tag)
            if tag_str:
                for t in tag_str.split(";"):
                    t = t.strip()
                    if t:
                        trader_summary[t] = trader_summary.get(t, 0) + 1

    result: Dict[str, Any] = {
        "daily_list": daily_list,
        "stock_details": stock_details,
        "institutional": institutional,
        "on_board": on_board,
        "trader_summary": trader_summary,
        "fetched_at": datetime.now().isoformat(),
    }

    logger.info(
        "LHB batch complete: %d stocks on board, %d with details, %d institutional rows",
        len(on_board),
        len(stock_details),
        len(institutional),
    )

    return result


# ---------------------------------------------------------------------------
# Test block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=== Dragon-Tiger Board Fetcher — Self Test ===\n")

    if not _ak_available:
        print("AKShare is NOT available. Tests will return empty DataFrames.\n")
    else:
        print(f"AKShare available (version: {getattr(_ak, '__version__', 'unknown')})\n")

    today = datetime.now()
    # Use previous trading day if today is a weekend.
    if today.weekday() >= 5:
        offset = today.weekday() - 4
        today = today - timedelta(days=offset)
    trade_date = today.strftime("%Y-%m-%d")
    print(f"Trade date: {trade_date}\n")

    # --- 1. Daily LHB list ---
    print("--- Test 1: Daily LHB List ---")
    daily = fetch_daily_lhb_list(trade_date)
    if not daily.empty:
        print(f"Rows: {len(daily)}")
        print(f"Columns: {list(daily.columns)}")
        if "trader_tag" in daily.columns:
            tagged = daily[daily["trader_tag"] != ""]
            if not tagged.empty:
                print(f"Tagged rows: {len(tagged)}")
            else:
                print("No famous trader tags found")
        print(daily.head(5).to_string())
    else:
        print("[empty]")
    print()

    # --- 2. Institutional LHB ---
    print("--- Test 2: Institutional LHB ---")
    inst = fetch_institutional_lhb(trade_date)
    if not inst.empty:
        print(f"Rows: {len(inst)}")
        print(inst.head(3).to_string())
    else:
        print("[empty]")
    print()

    # --- 3. Trader identification smoke test ---
    print("--- Test 3: Trader Identification (smoke test) ---")
    sample_df = pd.DataFrame({
        "买方席位": [
            "中信证券股份有限公司上海分公司",
            "国泰君安上海分公司",
            "华泰证券总部",
            "未知营业部",
        ],
        "卖方席位": [
            "东方证券上海分公司",
            "中信建投证券股份有限公司上海分公司",
            "招商证券深圳分公司",
            "华鑫证券上海分公司",
        ],
    })
    tagged_df = identify_famous_traders(sample_df)
    print(tagged_df.to_string())
    print()

    # --- 4. Batch fetch ---
    print("--- Test 4: Batch fetch_all_lhb ---")
    batch = fetch_all_lhb(["600519", "000858", "300750"], trade_date)
    print(f"  Daily list:    {len(batch['daily_list'])} rows")
    print(f"  On board:      {batch['on_board']}")
    print(f"  Stock details: {list(batch['stock_details'].keys())}")
    print(f"  Institutional: {len(batch['institutional'])} rows")
    print(f"  Trader summary: {batch['trader_summary']}")
    print(f"  Fetched at:    {batch['fetched_at']}")

    print("\n=== Test complete ===")
