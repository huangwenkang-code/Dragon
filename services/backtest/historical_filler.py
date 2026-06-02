"""Historical data filler — backfill pipeline runs for the last N trading days.

Uses external APIs: 同花顺 hot stocks, 腾讯行情 quote, 东财龙虎榜 datacenter.
Each trading day creates a simplified pipeline run and persists it to DB.
"""

from __future__ import annotations

import asyncio
import re
import urllib.request
from datetime import date, timedelta

import requests

from shared.utils.logging import get_logger

logger = get_logger(__name__)


async def backfill_trading_days(start_date: str, end_date: str):
    """Backfill pipeline data for all trading days between start and end.

    For each trading day:
    1. Fetch hot stocks + reasons from 同花顺
    2. Fetch fundamental data via tencent_quote
    3. Fetch dragon tiger board via 东财 datacenter
    4. Build simplified pipeline result
    5. Persist via db.persist.persist_run()
    """
    from db.persist import persist_run

    trade_dates = _get_trading_days(start_date, end_date)
    logger.info("Backfilling %d trading days: %s → %s", len(trade_dates), start_date, end_date)

    for td in trade_dates:
        logger.info("[backfill] processing %s", td)
        try:
            result = await _run_one_day(td)
            run_id = await persist_run(td, result)
            logger.info("[backfill] %s persisted as %s", td, run_id)
        except Exception as e:
            logger.error("[backfill] %s failed: %s", td, e)


async def _run_one_day(trade_date: str) -> dict:
    """Run a simplified pipeline for one historical trading day."""
    # 1. Fetch hot stocks from 同花顺
    hot_stocks = []
    try:
        url = f"http://zx.10jqka.com.cn/event/api/getharden/date/{trade_date}/orderby/date/orderway/desc/charset/GBK/"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/117.0.0.0"}
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()
        if data.get("errocode", 0) == 0:
            for row in (data.get("data") or []):
                hot_stocks.append({
                    "symbol": row.get("code", ""),
                    "stock_name": row.get("name", ""),
                    "reason": row.get("reason", ""),
                    "change_pct": float(row.get("zhangfu", 0)),
                    "turnover_pct": float(row.get("huanshou", 0)),
                    "ddejingliang": float(row.get("ddejingliang", 0)) if row.get("ddejingliang") else 0,
                    "close": float(row.get("close", 0)),
                })
        logger.info("[backfill] %s: %d hot stocks", trade_date, len(hot_stocks))
    except Exception as e:
        logger.warning("[backfill] 同花顺 hot stocks failed: %s", e)

    # 2. Fetch fundamentals for hot stocks via tencent_quote
    codes = [s["symbol"] for s in hot_stocks[:40]]
    quotes = {}
    if codes:
        prefixed = []
        for c in codes:
            if c.startswith(("6", "9")):
                prefixed.append(f"sh{c}")
            elif c.startswith("8"):
                prefixed.append(f"bj{c}")
            else:
                prefixed.append(f"sz{c}")
        q_url = "https://qt.gtimg.cn/q=" + ",".join(prefixed)
        req = urllib.request.Request(q_url)
        req.add_header("User-Agent", "Mozilla/5.0")
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            raw = resp.read().decode("gbk")
            for match in re.finditer(r'v_(\w+)="([^"]*)"', raw):
                code_key = match.group(1)[2:]
                fields = match.group(2).split("~")
                if len(fields) < 50:
                    continue
                quotes[code_key] = {
                    "name": fields[1],
                    "price": float(fields[3]) if fields[3] else 0,
                    "change_pct": float(fields[32]) if fields[32] else 0,
                    "amount": float(fields[37]) if fields[37] else 0,
                    "turnover_pct": float(fields[38]) if fields[38] else 0,
                    "pe": float(fields[39]) if fields[39] else 0,
                    "mcap_yi": float(fields[45]) if fields[45] else 0,
                }
        except Exception as e:
            logger.warning("[backfill] tencent quote failed: %s", e)

    # 3. Build capital_flow_records
    capital_flow_records = []
    for s in hot_stocks[:40]:
        q = quotes.get(s["symbol"], {})
        amount = q.get("amount", 0)
        mcap = q.get("mcap_yi", 0)
        capital_flow_records.append({
            "symbol": s["symbol"],
            "stock_name": s["stock_name"],
            "price": q.get("price", 0),
            "change_pct": s.get("change_pct", 0),
            "amount": amount,
            "amount_wan": round(amount / 10000, 2) if amount else 0,
            "turnover_pct": s.get("turnover_pct", q.get("turnover_pct", 0)),
            "pe": q.get("pe", 0),
            "market_cap": mcap,
            "main_force_net": 0,
            "flow_score": 0,
            "_source": "historical_backfill",
        })

    # 4. Build dragon_tiger_records
    dragon_tiger_records = []
    try:
        dt_url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
        dt_params = {
            "reportName": "RPT_DAILYBILLBOARD_DETAILSNEW",
            "columns": "ALL",
            "filter": f"(TRADE_DATE>='{trade_date}')(TRADE_DATE<='{trade_date}')",
            "pageNumber": "1",
            "pageSize": "500",
            "sortTypes": "-1",
            "sortColumns": "BILLBOARD_NET_AMT",
            "source": "WEB",
            "client": "WEB",
        }
        dt_headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"}
        dt_r = requests.get(dt_url, params=dt_params, headers=dt_headers, timeout=15)
        dt_d = dt_r.json()
        if dt_d.get("success") and dt_d.get("result", {}).get("data"):
            for row in dt_d["result"]["data"]:
                dragon_tiger_records.append({
                    "stock_code": row.get("SECURITY_CODE", ""),
                    "stock_name": row.get("SECURITY_NAME_ABBR", ""),
                    "trade_date": trade_date,
                    "reason": row.get("EXPLANATION", ""),
                    "net_amount": (row.get("BILLBOARD_NET_AMT") or 0) / 10000,
                    "lhb_score": min(abs((row.get("BILLBOARD_NET_AMT") or 0) / 10000) / 5000, 1.0),
                    "famous_traders": [],
                    "trader_signal": "",
                })
    except Exception as e:
        logger.warning("[backfill] dragon tiger failed: %s", e)

    logger.info("[backfill] %s: %d capital_flow, %d dragon_tiger",
                trade_date, len(capital_flow_records), len(dragon_tiger_records))

    # 5. Build leader_candidates (simplified: rank by hot stock order)
    leader_candidates = []
    for i, s in enumerate(hot_stocks[:10]):
        leader_candidates.append({
            "rank": i + 1,
            "stock_code": s["symbol"],
            "stock_name": s["stock_name"],
            "leader_score": round(min(1.0, (10 - i) / 10), 2),
            "monster_potential": 0,
            "limit_up_prob": 0,
            "reasoning": s.get("reason", ""),
            "sector": "",
            "sentiment_sub": 0,
            "flow_sub": 0,
            "lhb_sub": 0,
            "ml_sub": 0,
            "event_sub": 0,
            "sector_tag_sub": 0,
        })

    # 6. Build active_stocks
    active_stocks = []
    for i, s in enumerate(hot_stocks[:40]):
        active_stocks.append({
            "symbol": s["symbol"],
            "stock_name": s["stock_name"],
            "matched_concepts": [s.get("reason", "")],
            "active_score": round(min(1.0, (40 - i) / 40), 2),
        })

    return {
        "events": [],
        "sentiment_scores": [],
        "capital_flow_records": capital_flow_records,
        "sector_flow_records": [],
        "capital_flow_summary": {},
        "active_stocks": active_stocks,
        "dragon_tiger_records": dragon_tiger_records,
        "leader_candidates": leader_candidates,
        "risk_flags": [],
        "activated_memories": [],
        "watchlist": [],
        "top_n": 5,
        "metadata": {"started_at": f"{trade_date}T15:00:00", "backfill": True},
    }


def _get_trading_days(start_date: str, end_date: str) -> list[str]:
    """Generate list of trading days between start and end, excluding weekends."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Mon-Fri
            days.append(current.isoformat())
        current += timedelta(days=1)
    return days


if __name__ == "__main__":
    import sys

    start = sys.argv[1] if len(sys.argv) > 1 else (date.today() - timedelta(days=30)).isoformat()
    end = sys.argv[2] if len(sys.argv) > 2 else date.today().isoformat()
    asyncio.run(backfill_trading_days(start, end))
