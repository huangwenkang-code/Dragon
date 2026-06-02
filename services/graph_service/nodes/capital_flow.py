"""capital_flow node — market-wide capital activity auto-discovery.

LIVE mode (today): uses tx_finance (qt.gtimg.cn) realtime API.
HISTORICAL mode (backfill): reads stock_daily_bars_v2 — deterministic, no API bias.
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import date as dt_date, datetime, timedelta

import httpx

from shared.schemas.agent_state import AgentState
from shared.utils.logging import get_logger

logger = get_logger(__name__)

TX_BASE = "http://qt.gtimg.cn/q="
BATCH_SIZE = 500
TOP_ACTIVE_N = 40


async def capital_flow(state: AgentState) -> dict:
    """Auto-discover active stocks.

    LIVE: tx_finance realtime. HISTORICAL: stock_daily_bars_v2 (deterministic).
    """
    from shared.utils.trading_day import get_trading_date_str

    trade_date = state.get("trade_date") or get_trading_date_str()
    today_str = dt_date.today().isoformat()
    is_live = (trade_date[:10] == today_str)

    _t_total = time.time()
    logger.info("[capital_flow] %s mode for %s", "LIVE" if is_live else "HISTORICAL", trade_date)

    if is_live:
        fundamentals = await _fetch_live()
    else:
        fundamentals = await _fetch_historical(trade_date[:10])

    if not fundamentals:
        logger.warning("[capital_flow] no fundamentals for %s", trade_date)
        return _empty_result()

    # Compute volume ratio (today_vol / 20d_avg) from DB
    try:
        td_obj = dt_date.fromisoformat(trade_date[:10])
        lookback = td_obj - timedelta(days=35)
        avg_vol_map = await _fetch_avg_volume(trade_date[:10], lookback.isoformat())
    except Exception as exc:
        logger.warning("[capital_flow] avg volume query failed: %s", exc)
        avg_vol_map = {}

    for s in fundamentals:
        sym = s["symbol"]
        today_vol = s.get("volume", 0) or 0
        avg_vol = avg_vol_map.get(sym, 0) or 0
        if avg_vol > 0 and today_vol > 0:
            s["volume_ratio"] = round(today_vol / avg_vol, 2)
        else:
            s["volume_ratio"] = 1.0

    # Rank: BEAR uses volume activity; BULL/CHOPPY use momentum
    from shared.market_regime import detect_regime
    regime = await detect_regime(trade_date[:10])
    if regime == "BEAR":
        # BEAR: 不看涨跌看活跃度（量比×换手率）
        fundamentals.sort(key=lambda r: r.get("volume_ratio", 1.0) * max(r.get("turnover_pct", 0) / 100, 0.01), reverse=True)
    else:
        fundamentals.sort(key=lambda r: abs(r["change_pct"]) * r.get("volume_ratio", 1.0), reverse=True)
    top_stocks = fundamentals[:TOP_ACTIVE_N]
    logger.info("[capital_flow] regime=%s top %d: %s", regime, len(top_stocks),
                [(s["symbol"], f'{s.get("volume_ratio",0):.1f}x') for s in top_stocks[:10]])

    logger.info("[capital_flow] top %d: %s",
                len(top_stocks), [(s["symbol"], f'{s.get("volume_ratio", 0):.1f}x') for s in top_stocks[:10]])

    # Build flow records
    stock_flow: list[dict] = []
    for s in top_stocks:
        amount = s.get("amount", 0)
        mcap = s.get("market_cap", 0)
        stock_flow.append({
            "symbol": s["symbol"],
            "stock_name": s.get("name", s["symbol"]),
            "price": s.get("price", 0),
            "change_pct": s.get("change_pct", 0),
            "amount": amount,
            "amount_wan": round(amount / 10000, 2),
            "main_force_net": 0.0, "main_force_ratio": 0.0,
            "super_large_net": 0.0, "large_net": 0.0, "mid_net": 0.0, "small_net": 0.0,
            "total_net": 0.0, "northbound_net": 0.0,
            "flow_ratio": round(amount / (mcap + 1), 6) if mcap > 0 else 0.0,
            "flow_score": 0.0, "sector_flow": 0.0,
            "turnover_pct": s.get("turnover_pct", 0),
            "pe": s.get("pe", 0), "pb": s.get("pb", 0), "market_cap": mcap,
            "_source": "db_bars" if not is_live else "tx_finance",
        })

    if stock_flow:
        max_vr = max(r.get("volume_ratio", 1.0) for r in stock_flow) or 1.0
        for r in stock_flow:
            r["flow_score"] = round(r.get("volume_ratio", 1.0) / max_vr, 3)

    elapsed = time.time() - _t_total
    logger.info("[capital_flow] complete: %d stocks in %.1fs", len(stock_flow), elapsed)

    return {
        "capital_flow_records": stock_flow,
        "capital_flow_summary": {
            "total_main_inflow": sum(r.get("main_force_net", 0) for r in stock_flow if r.get("main_force_net", 0) > 0),
            "top_sectors": [],
            "scanned_stocks": len(stock_flow),
            "scan_time": datetime.now().isoformat(),
        },
    }


# ---------------------------------------------------------------------------
# LIVE: tx_finance realtime API
# ---------------------------------------------------------------------------

async def _fetch_live() -> list[dict]:
    """Fetch fundamentals from tx_finance for ALL A-shares."""
    all_codes = _get_all_stock_codes()
    if not all_codes:
        return []
    return await _batch_fetch_fundamentals(all_codes)


# ---------------------------------------------------------------------------
# HISTORICAL: stock_daily_bars_v2 (deterministic!)
# ---------------------------------------------------------------------------

async def _fetch_historical(trade_date: str) -> list[dict]:
    """Read fundamentals from stock_daily_bars_v2 for a historical date."""
    from db.connection import async_session_factory
    from sqlalchemy import text as sa_text

    async with async_session_factory() as session:
        # Get bars for the trade_date, with prev close for change_pct
        r = await session.execute(
            sa_text("""
                SELECT symbol, open, high, low, close, volume, amount, change_pct, turnover_pct
                FROM stock_daily_bars_v2
                WHERE trade_date = :td AND close > 0 AND volume > 0
            """),
            {"td": dt_date.fromisoformat(trade_date)},
        )
        rows = r.fetchall()
        result = []
        for row in rows:
            sym, o, h, l, c, v, amt, chg, turnover = row
            result.append({
                "symbol": sym,
                "name": sym,
                "price": float(c or 0),
                "change_pct": float(chg or 0),
                "volume": float(v or 0),
                "amount": float(amt or 0),
                "turnover_pct": float(turnover or 0),
                "pe": 0, "pb": 0, "market_cap": 0,
            })
        return result


# ---------------------------------------------------------------------------
# Stage 1: stock code list (AKShare, cached)
# ---------------------------------------------------------------------------

_stock_codes_cache: list[str] | None = None


def _get_all_stock_codes() -> list[str]:
    global _stock_codes_cache
    if _stock_codes_cache is not None:
        return _stock_codes_cache
    try:
        import akshare as ak
        df = ak.stock_info_a_code_name()
        codes = [str(c).zfill(6) for c in df["code"].tolist()]
        _stock_codes_cache = codes
        logger.info("[capital_flow] cached %d A-share codes", len(codes))
        return codes
    except Exception as exc:
        logger.warning("[capital_flow] stock_info_a_code_name failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Stage 2: tx_finance batch query (LIVE only)
# ---------------------------------------------------------------------------

async def _batch_fetch_fundamentals(all_codes: list[str]) -> list[dict]:
    batches = [all_codes[i:i + BATCH_SIZE] for i in range(0, len(all_codes), BATCH_SIZE)]
    async with httpx.AsyncClient(timeout=15.0) as client:
        results = await asyncio.gather(*(_fetch_one_batch(client, b) for b in batches))
    all_rows = []
    for rows in results:
        all_rows.extend(rows)
    return all_rows


async def _fetch_one_batch(client: httpx.AsyncClient, codes: list[str]) -> list[dict]:
    code_str = _build_tx_codes(codes)
    url = f"{TX_BASE}{code_str}"
    try:
        resp = await client.get(url)
        if resp.status_code != 200:
            return []
        return _parse_tx_fundamentals(resp.text)
    except Exception:
        return []


def _build_tx_codes(symbols: list[str]) -> str:
    codes = []
    for s in symbols:
        code = s.strip()
        if code.startswith(("6", "68", "9")):
            codes.append(f"sh{code}")
        else:
            codes.append(f"sz{code}")
    return ",".join(codes)


def _parse_tx_fundamentals(raw: str) -> list[dict]:
    rows = []
    for match in re.finditer(r'v_(\w+)="([^"]*)"', raw):
        code = match.group(1)
        fields = match.group(2).split("~")
        if len(fields) < 50:
            continue
        try:
            symbol = code[2:] if len(code) > 2 else code
            price = float(fields[3]) if fields[3] else 0.0
            change_pct = float(fields[32]) if len(fields) > 32 and fields[32] else 0.0
            amount = float(fields[37]) if len(fields) > 37 and fields[37] else 0.0
            volume = float(fields[6]) if len(fields) > 6 and fields[6] else 0.0
            if price <= 0:
                continue
            rows.append({
                "symbol": symbol,
                "name": fields[1] if len(fields) > 1 else "",
                "price": price,
                "change_pct": round(change_pct, 2),
                "volume": volume,
                "amount": amount,
                "turnover_pct": float(fields[38]) if len(fields) > 38 and fields[38] else 0.0,
                "pe": float(fields[39]) if len(fields) > 39 and fields[39] else 0.0,
                "pb": float(fields[47]) if len(fields) > 47 and fields[47] else 0.0,
                "market_cap": float(fields[45]) if len(fields) > 45 and fields[45] else 0.0,
            })
        except (ValueError, IndexError):
            continue
    return rows


# ---------------------------------------------------------------------------
# 20-day average volume helper
# ---------------------------------------------------------------------------

async def _fetch_avg_volume(today: str, lookback: str) -> dict[str, float]:
    from db.connection import async_session_factory
    from sqlalchemy import text as sa_text

    td_obj = dt_date.fromisoformat(today)
    lb_obj = dt_date.fromisoformat(lookback)

    async with async_session_factory() as session:
        result = await session.execute(
            sa_text(
                "SELECT symbol, AVG(volume)::float as avg_vol "
                "FROM stock_daily_bars_v2 "
                "WHERE trade_date >= :lb AND trade_date < :td AND volume > 0 "
                "GROUP BY symbol"
            ),
            {"lb": lb_obj, "td": td_obj},
        )
        rows = result.fetchall()
        return {row[0]: row[1] for row in rows if row[1]}


# ---------------------------------------------------------------------------

async def _detect_regime(trade_date: str) -> str:
    """BULL/CHOPPY/BEAR from index MA20."""
    from db.connection import async_session_factory
    from sqlalchemy import text as sa_text

    try:
        td = dt_date.fromisoformat(trade_date)
        lookback = td - timedelta(days=60)
        async with async_session_factory() as session:
            r = await session.execute(sa_text(
                "SELECT close FROM stock_daily_bars_v2 "
                "WHERE symbol='000001.SH' AND trade_date >= :lb AND trade_date <= :td "
                "ORDER BY trade_date"
            ), {"lb": lookback, "td": td})
            closes = [float(row[0] or 0) for row in r.fetchall() if row[0]]
            if len(closes) < 26:
                return "CHOPPY"
            today_close = closes[-1]
            ma20 = sum(closes[-21:-1]) / 20
            ma20_5d = sum(closes[-26:-6]) / 20
            pma = (today_close - ma20) / ma20 if ma20 > 0 else 0
            slope = (ma20 - ma20_5d) / ma20_5d if ma20_5d > 0 else 0
            if pma > 0.02 and slope > 0.005: return "BULL"
            if pma < -0.02 and slope < -0.005: return "BEAR"
            return "CHOPPY"
    except Exception:
        return "CHOPPY"


def _empty_result() -> dict:
    return {
        "capital_flow_records": [],
        "capital_flow_summary": {
            "total_main_inflow": 0,
            "top_sectors": [],
            "scanned_stocks": 0,
            "scan_time": datetime.now().isoformat(),
        },
    }
