"""dragon_tiger_board node — real LHB analysis via AKShare."""

from shared.schemas.agent_state import AgentState, DragonTigerRecord
from shared.utils.logging import get_logger

logger = get_logger(__name__)


async def dragon_tiger_board(state: AgentState) -> dict:
    """Fetch ALL daily dragon-tiger board data — no symbol filter needed.

    Fetches: daily LHB list, per-stock seat detail, institutional activity.
    Identifies famous traders and computes lhb_score per stock.
    The merge node cross-references with capital_flow active stocks.
    """
    trade_date = state.get("trade_date", "")

    logger.info("[dragon_tiger_board] fetching ALL daily LHB for %s", trade_date)

    try:
        import time as _time
        from services.dragon_tiger_service.fetcher import fetch_all_lhb

        _t0 = _time.time()
        # Fetch ALL LHB stocks for the day (pass empty list = no filter)
        batch = fetch_all_lhb([], trade_date)
        logger.info("[dragon_tiger_board] fetch_all_lhb took %.1fs, on_board=%d, detail_keys=%s, trader_summary_keys=%d",
                    _time.time() - _t0,
                    len(batch.get("on_board", [])),
                    list(batch.get("stock_details", {}).keys()),
                    len(batch.get("trader_summary", {})))
    except Exception as exc:
        logger.error("[dragon_tiger_board] fetch failed: %s", exc)
        return {"dragon_tiger_records": []}

    on_board = batch.get("on_board", [])
    stock_details = batch.get("stock_details", {})
    trader_summary = batch.get("trader_summary", {})

    records: list[dict] = []
    for sym in on_board:
        detail = stock_details.get(sym)
        record = _parse_lhb_detail(sym, trade_date, detail, trader_summary)
        records.append(record)

    logger.info(
        "[dragon_tiger_board] %d stocks on LHB, %d with details, %d unique traders",
        len(on_board), len(stock_details), len(trader_summary),
    )
    return {"dragon_tiger_records": records}


def _parse_lhb_detail(
    sym: str,
    trade_date: str,
    detail,  # pd.DataFrame or None
    trader_summary: dict,
) -> dict:
    """Parse LHB detail DataFrame into a structured record."""
    if detail is None or (hasattr(detail, 'empty') and detail.empty):
        return {
            "stock_code": sym,
            "stock_name": sym,
            "trade_date": trade_date,
            "reason": "",
            "buy_seats": [],
            "sell_seats": [],
            "total_buy": 0.0,
            "total_sell": 0.0,
            "net_amount": 0.0,
            "famous_traders": [],
            "trader_signal": "",
            "lhb_score": 0.0,
        }

    import pandas as pd
    if not isinstance(detail, pd.DataFrame):
        return {
            "stock_code": sym, "stock_name": sym, "trade_date": trade_date,
            "reason": "", "buy_seats": [], "sell_seats": [],
            "total_buy": 0.0, "total_sell": 0.0, "net_amount": 0.0,
            "famous_traders": [], "trader_signal": "", "lhb_score": 0.0,
        }

    # Extract buy seats
    buy_cols = [c for c in detail.columns if "买" in str(c) and ("席位" in str(c) or "营业部" in str(c))]
    sell_cols = [c for c in detail.columns if "卖" in str(c) and ("席位" in str(c) or "营业部" in str(c))]
    buy_amt_cols = [c for c in detail.columns if "买入金额" in str(c) or "买额" in str(c)]
    sell_amt_cols = [c for c in detail.columns if "卖出金额" in str(c) or "卖额" in str(c)]

    buy_seats = []
    sell_seats = []
    total_buy = 0.0
    total_sell = 0.0

    for i, (_, row) in enumerate(detail.head(5).iterrows()):
        if i < len(buy_cols):
            seat_name = str(row.get(buy_cols[i], ""))
            amt = _safe_float(row, [buy_amt_cols[i]] if i < len(buy_amt_cols) else [])
            if seat_name and seat_name != "nan":
                buy_seats.append({"seat": seat_name, "amount": round(amt, 2)})
                total_buy += amt

        if i < len(sell_cols):
            seat_name = str(row.get(sell_cols[i], ""))
            amt = _safe_float(row, [sell_amt_cols[i]] if i < len(sell_amt_cols) else [])
            if seat_name and seat_name != "nan":
                sell_seats.append({"seat": seat_name, "amount": round(amt, 2)})
                total_sell += amt

    # Also try total columns
    total_buy_col = _find_col(detail, ["总买入", "买入总计", "total_buy"])
    total_sell_col = _find_col(detail, ["总卖出", "卖出总计", "total_sell"])
    if total_buy_col:
        total_buy = _safe_float(detail.iloc[0], [total_buy_col]) if len(detail) > 0 else total_buy
    if total_sell_col:
        total_sell = _safe_float(detail.iloc[0], [total_sell_col]) if len(detail) > 0 else total_sell

    net_amount = total_buy - total_sell

    # Identify famous traders from the seats
    trader_tags = detail.get("trader_tag", detail.columns)
    if isinstance(trader_tags, str):
        pass  # single row
    famous_traders = []
    if "trader_tag" in detail.columns:
        for tag in detail["trader_tag"].dropna():
            tag_str = str(tag)
            if tag_str:
                for t in tag_str.split(";"):
                    t = t.strip()
                    if t and t not in famous_traders:
                        famous_traders.append(t)

    # Trader signal: 合力做多 / 分歧 / 出货
    buy_traders = set()
    sell_traders = set()
    for bs in buy_seats:
        for kw, name in _TRADER_MAP.items():
            if kw in bs["seat"]:
                buy_traders.add(name)
    for ss in sell_seats:
        for kw, name in _TRADER_MAP.items():
            if kw in ss["seat"]:
                sell_traders.add(name)

    common = buy_traders & sell_traders
    if len(buy_traders) >= 2 and not common:
        trader_signal = "合力做多"
    elif len(sell_traders) >= 2 and not common:
        trader_signal = "出货"
    elif common:
        trader_signal = "分歧"
    elif len(buy_traders) == 0 and len(sell_traders) == 0:
        trader_signal = ""
    else:
        trader_signal = "合力做多" if len(buy_traders) > len(sell_traders) else "出货"

    # LHB score: net positive + famous trader presence
    score = 0.0
    if net_amount > 0:
        score += 0.4
    elif net_amount > 5000:
        score += 0.6
    if famous_traders:
        score += min(0.3, len(famous_traders) * 0.1)
    if trader_signal == "合力做多":
        score += 0.3
    elif trader_signal == "分歧":
        score += 0.1
    score = min(1.0, score)

    return {
        "stock_code": sym,
        "stock_name": sym,
        "trade_date": trade_date,
        "reason": str(detail.iloc[0].get("上榜原因", "")) if len(detail) > 0 else "",
        "buy_seats": buy_seats,
        "sell_seats": sell_seats,
        "total_buy": round(total_buy, 2),
        "total_sell": round(total_sell, 2),
        "net_amount": round(net_amount, 2),
        "famous_traders": famous_traders,
        "trader_signal": trader_signal,
        "lhb_score": round(score, 3),
    }


def _safe_float(row, candidates: list[str]) -> float:
    for col in candidates:
        val = row.get(col)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                continue
    return 0.0


def _find_col(df, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    return ""


_TRADER_MAP = {
    "中信证券上海分公司": "章盟主",
    "国泰君安上海分公司": "赵老哥",
    "中信建投上海分公司": "作手新一",
    "东方证券上海分公司": "炒股养家",
    "招商证券深圳分公司": "活跃游资",
    "华鑫证券上海分公司": "量化打板",
    "财通证券": "上塘路",
    "中国银河证券": "中关村",
    "国盛证券": "桑田路",
    "华泰证券总部": "量化/机构混合",
}
