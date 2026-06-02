"""
V2 Backtest: quality filter + regime-adaptive selection, no pipeline re-run needed.

Reads stock_daily_bars_v2 directly, applies filter + regime logic each day,
feeds filtered candidates into the existing BacktestEngine.

Usage:
  python backtest_v2.py 2024   # 2024 Jan-Sep
  python backtest_v2.py 2025   # 2025 full year
"""
import asyncio
import sys
from datetime import date, timedelta
from collections import defaultdict

# Setup path
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from db.connection import async_session_factory
from sqlalchemy import text as sa_text
from shared.market_regime import detect_regime, clear_cache
from shared.filters import find_oversold_bounce
from shared.utils.logging import get_logger
from shared.utils.trading_day import is_trading_day, get_trading_date_str

logger = get_logger(__name__)


async def get_trading_dates(start: str, end: str) -> list[str]:
    """Get all trading days between start and end (inclusive)."""
    async with async_session_factory() as session:
        r = await session.execute(
            sa_text("""
                SELECT DISTINCT trade_date::text
                FROM stock_daily_bars_v2
                WHERE trade_date >= :s AND trade_date <= :e
                ORDER BY trade_date
            """),
            {"s": date.fromisoformat(start), "e": date.fromisoformat(end)},
        )
        return [row[0] for row in r.fetchall()]


async def get_top_stocks(trade_date: str, regime: str, top_n: int = 60) -> list[dict]:
    """Get top candidates using regime-configurable strategy.

    Reads from REGIME_CONFIG to determine selection mode + parameters.
    """
    from shared.regime_config import get_strategy, RegimeStrategy

    cfg = get_strategy(regime)
    td = date.fromisoformat(trade_date)

    async with async_session_factory() as session:
        candidates: list[dict] = []
        symbols_seen: set[str] = set()

        # ---- Concept leaders (BULL / CHOPPY_UP — theme-driven alpha) ----
        # Use PREVIOUS trading day's concept leaders: today's leaders already surged,
        # but yesterday's leaders are still in play and buyable today.
        concept_candidates: list[dict] = []
        if regime in ("BULL", "CHOPPY_UP"):
            from shared.filters import fetch_concept_leaders
            try:
                # Find previous trading day
                prev_r = await session.execute(
                    sa_text("""
                        SELECT trade_date::text FROM stock_daily_bars_v2
                        WHERE trade_date < :td AND close > 0
                        ORDER BY trade_date DESC LIMIT 1
                    """),
                    {"td": td},
                )
                prev_row = prev_r.fetchone()
                prev_td = prev_row[0] if prev_row else trade_date

                concept_leaders = await fetch_concept_leaders(prev_td, top_n=40)
                # Bulk-fetch OHLCV data for concept leaders
                cl_symbols = [cl["symbol"] for cl in concept_leaders]
                if cl_symbols:
                    r = await session.execute(
                        sa_text("""
                            SELECT symbol, close, change_pct, turnover_pct, amount, volume
                            FROM stock_daily_bars_v2
                            WHERE symbol = ANY(:syms) AND trade_date = :td
                        """),
                        {"syms": cl_symbols, "td": td},
                    )
                    ohlcv = {row[0]: row for row in r.fetchall()}

                    for cl in concept_leaders:
                        sym = cl["symbol"]
                        if sym not in symbols_seen:
                            bar = ohlcv.get(sym)
                            price = float(bar[1] or 0) if bar else 0
                            cl.update({
                                "stock_code": sym,
                                "price": price,
                                "change_pct": float(bar[2] or 0) if bar else cl.get("change_pct", 0),
                                "turnover_pct": float(bar[3] or 0) if bar else 0,
                                "amount": float(bar[4] or 0) if bar else 0,
                                "amount_wan": round(float(bar[4] or 0) / 10000, 2) if bar else 0,
                                "volume": float(bar[5] or 0) if bar else 0,
                            })
                            concept_candidates.append(cl)
                            symbols_seen.add(sym)
            except Exception as exc:
                logger.warning("concept leaders unavailable: %s", exc)

        # ---- Oversold bounce (BEAR primary, CHOPPY/CHOPPY_DOWN blend) ----
        if cfg.selection in ("oversold_bounce", "blend"):
            oversold_n = top_n * 2 if cfg.selection == "oversold_bounce" else top_n
            oversold = await find_oversold_bounce(trade_date, top_n=oversold_n)
            for o in oversold:
                if o["symbol"] not in symbols_seen:
                    o["stock_code"] = o["symbol"]
                    o["_source"] = "oversold_bounce"
                    o["leader_score"] = 0.0
                    candidates.append(o)
                    symbols_seen.add(o["symbol"])

        # ---- Momentum (BULL primary, CHOPPY blend, BEAR diversity) ----
        if cfg.selection in ("momentum", "momentum_raw", "blend"):
            momentum_n = top_n * 2 if cfg.selection == "momentum" else top_n
            r = await session.execute(
                sa_text("""
                    SELECT symbol, close, change_pct, turnover_pct, amount, volume
                    FROM stock_daily_bars_v2
                    WHERE trade_date = :td AND close > 0 AND volume > 0
                      AND close >= :price_min
                      AND amount >= :amt_min
                    ORDER BY amount DESC
                    LIMIT :n
                """),
                {
                    "td": td,
                    "n": momentum_n,
                    "price_min": cfg.filter_price_min,
                    "amt_min": cfg.filter_amount_min,
                },
            )
            rows = r.fetchall()

            if rows:
                for row in rows:
                    sym, close, chg, turnover, amt, vol = row
                    if sym in symbols_seen:
                        continue
                    amt_f = float(amt or 0)
                    chg_f = float(chg or 0)
                    to_f = float(turnover or 0)

                    candidates.append({
                        "stock_code": sym,
                        "symbol": sym,
                        "stock_name": sym,
                        "price": float(close or 0),
                        "change_pct": chg_f,
                        "turnover_pct": to_f,
                        "amount": amt_f,
                        "amount_wan": round(amt_f / 10000, 2),
                        "leader_score": 0.0,
                        "_source": "momentum",
                    })
                    symbols_seen.add(sym)

    # Merge concept leaders into main pool
    if concept_candidates:
        candidates.extend(concept_candidates)

    # ---- Score ALL candidates with the 5-factor model ----
    from shared.scorer import score_candidates
    candidates = score_candidates(candidates, trade_date, regime)
    return candidates[:top_n]


async def run_backtest(year: str):
    """Run V2 backtest for a given year."""
    if year == "2024":
        start_date = "2024-01-01"
        end_date = "2024-09-30"
    else:
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

    print(f"\n{'='*60}")
    print(f"V2 Backtest: {start_date} → {end_date}")
    print(f"{'='*60}")

    clear_cache()

    # Get trading dates
    trading_dates = await get_trading_dates(start_date, end_date)
    print(f"Trading days: {len(trading_dates)}")

    # Load all price data
    async with async_session_factory() as session:
        # Load all bars for the period
        r = await session.execute(
            sa_text("""
                SELECT symbol, trade_date::text, open, high, low, close, volume
                FROM stock_daily_bars_v2
                WHERE trade_date >= :s AND trade_date <= :e
                  AND close > 0 AND open > 0
                ORDER BY symbol, trade_date
            """),
            {"s": date.fromisoformat(start_date), "e": date.fromisoformat(end_date)},
        )
        all_rows = r.fetchall()
        print(f"Total bars: {len(all_rows)}")

    # Build bars_by_symbol + prices_by_date
    bars_by_symbol: dict[str, list[dict]] = defaultdict(list)
    prices_by_date: dict[str, dict[str, float]] = defaultdict(dict)
    close_by_date: dict[str, dict[str, float]] = defaultdict(dict)

    for sym, td_str, o, h, l, c, v in all_rows:
        bar = {
            "trade_date": td_str,
            "open": float(o or 0),
            "high": float(h or 0),
            "low": float(l or 0),
            "close": float(c or 0),
            "volume": float(v or 0),
        }
        bars_by_symbol[sym].append(bar)
        prices_by_date[td_str][sym] = float(o or 0)  # open price for entry
        close_by_date[td_str][sym] = float(c or 0)

    # Sort bars for each symbol
    for sym in bars_by_symbol:
        bars_by_symbol[sym].sort(key=lambda b: b["trade_date"])

    # Load index bars
    index_bars_by_date: dict[str, list[dict]] = defaultdict(list)
    idx_bars = bars_by_symbol.get("000001.SH", [])
    for b in idx_bars:
        index_bars_by_date[b["trade_date"]].append(b)

    # Build prev_close
    sorted_dates = sorted(close_by_date.keys())
    prev_close_by_date: dict[str, dict[str, float]] = {}
    for i, ds in enumerate(sorted_dates):
        prev_td = None
        for j in range(i - 1, -1, -1):
            if is_trading_day(sorted_dates[j]):
                prev_td = sorted_dates[j]
                break
        prev_close_by_date[ds] = close_by_date.get(prev_td, {}) if prev_td else {}

    # Generate daily candidates using V2 logic
    daily_runs = []
    for i, td in enumerate(trading_dates):
        if i % 30 == 0:
            print(f"  Processing {td} ({i+1}/{len(trading_dates)})...")

        regime = await detect_regime(td)
        candidates = await get_top_stocks(td, regime, top_n=60)

        # Build day_bars (progressive cursor)
        day_bars = {}
        for sym, sym_bars in bars_by_symbol.items():
            filtered = [b for b in sym_bars if b["trade_date"] <= td]
            if len(filtered) >= 21:
                day_bars[sym] = filtered

        # Index bars up to this date
        idx_filtered = [b for b in idx_bars if b["trade_date"] <= td]

        daily_runs.append({
            "trade_date": td,
            "leader_candidates": candidates,
            "prices": prices_by_date.get(td, {}),
            "prev_close": prev_close_by_date.get(td, {}),
            "bars": day_bars,
            "index_bars": idx_filtered,
            "prev_day_change": {},
            "sector_volume_pct": {},
            "avg_volume_20": {},
            "turnover_pct": {},
            "today_volume": {},
            "total_market_volume": 0,
            "breadth": 0.5,
            "volatility": 0.02,
            "limit_up_count": 0,
            "sector_concentration": 0,
            "lianban_height": 0,
            "score_sort_desc": True,
        })

    print(f"\nDaily runs built: {len(daily_runs)}")
    total_cands = sum(len(d["leader_candidates"]) for d in daily_runs)
    print(f"Total candidates: {total_cands}, avg {total_cands/max(len(daily_runs),1):.0f}/day")

    # Run backtest engine
    from services.backtest.registry import get_registry
    from services.backtest.strategies import TradingStrategy
    from services.backtest.engine import BacktestEngine

    registry = get_registry()
    strategy = registry.get("策略A-每日固定资金")
    if not strategy:
        print("ERROR: strategy not found! Creating default...")
        strategy = TradingStrategy(
            name="策略A-每日固定资金",
            description="Auto",
            entry_rules=[],
            exit_rules=[],
            max_positions=10,
            initial_capital=1000000.0,
        )

    engine = BacktestEngine(strategy)
    result = engine.run(daily_runs)

    print(f"\n{'='*60}")
    print(f"V2 RESULTS: {year}")
    print(f"{'='*60}")
    print(f"Initial capital:  {result.initial_capital:,.0f}")
    print(f"Final equity:     {result.final_equity:,.0f}")
    print(f"Total return:     {result.total_return_pct:+.2f}%")
    print(f"Max drawdown:     {result.max_drawdown_pct:.2f}%")
    print(f"Sharpe ratio:     {result.sharpe_ratio:.3f}")
    print(f"Win rate:         {result.win_rate*100:.1f}%")
    print(f"Total trades:     {result.total_trades}")

    # Regime stats
    regime_stats = defaultdict(lambda: {"trades": 0, "pnl": 0.0})
    for t in result.trades:
        # Approximate regime from entry date (re-use cached regime)
        entry_date = str(t.entry_date)
        regime_stats[entry_date[:7]]["trades"] += 1
        regime_stats[entry_date[:7]]["pnl"] += t.net_pnl

    return result


if __name__ == "__main__":
    year = sys.argv[1] if len(sys.argv) > 1 else "2024"
    asyncio.run(run_backtest(year))
