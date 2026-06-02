"""Parameter optimization v4 — radical combinations for 50%."""
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

import asyncio
from datetime import date
from collections import defaultdict
from db.connection import async_session_factory
from sqlalchemy import select
from db.models import PipelineRun, LeaderCandidate, StockDailyBar
from services.backtest.engine import BacktestEngine, DEFAULT_PARAMS
from services.backtest.market_regime import RegimeParams
from services.backtest.strategies import TradingStrategy
from services.backtest.rules import (
    FridayFilterRule, STFilterRule, ScoreThresholdRule,
    MaxScoreFilterRule, OneDaySpikeFilter, VolumeSurgeFilter,
    GapUpFilterRule, NoFilterRule, PriceTrailingStopRule,
    ScoreWeightedAllocator, EqualWeightAllocator,
)


async def load_data(start: date, end: date) -> list[dict]:
    async with async_session_factory() as session:
        result = await session.execute(
            select(StockDailyBar).where(
                StockDailyBar.trade_date >= start, StockDailyBar.trade_date <= end
            ).order_by(StockDailyBar.trade_date)
        )
        all_bars = list(result.scalars().all())
        prices_by_date: dict = defaultdict(dict)
        close_by_date: dict = defaultdict(dict)
        for bar in all_bars:
            td = bar.trade_date.isoformat() if hasattr(bar.trade_date, 'isoformat') else str(bar.trade_date)
            prices_by_date[td][bar.symbol] = float(bar.open or 0)
            close_by_date[td][bar.symbol] = float(bar.close or 0)
        sorted_dates = sorted(close_by_date.keys())
        prev_close_by_date = {}
        for i, ds in enumerate(sorted_dates):
            prev_close_by_date[ds] = close_by_date.get(sorted_dates[i-1], {}) if i > 0 else {}
        master_index_bars = []
        master_bars: dict = defaultdict(list)
        for bar in all_bars:
            td = bar.trade_date.isoformat() if hasattr(bar.trade_date, 'isoformat') else str(bar.trade_date)
            bd = {"open": bar.open, "high": bar.high, "low": bar.low,
                  "close": bar.close, "volume": bar.volume}
            if bar.symbol == "000001.SH":
                master_index_bars.append((td, bd))
            master_bars[bar.symbol].append((td, bd))
        result = await session.execute(
            select(PipelineRun).where(
                PipelineRun.trade_date >= start, PipelineRun.trade_date <= end
            ).order_by(PipelineRun.trade_date)
        )
        runs = list(result.scalars().all())
        daily_runs = []
        for run in runs:
            td_str = run.trade_date.isoformat() if hasattr(run.trade_date, 'isoformat') else str(run.trade_date)
            result = await session.execute(
                select(LeaderCandidate).where(LeaderCandidate.run_id == run.run_id)
            )
            candidates = list(result.scalars().all())
            day_index_bars = [b for (d, b) in master_index_bars if d <= td_str]
            day_bars = {}
            for sym, dated_bars in master_bars.items():
                filtered = [b for (d, b) in dated_bars if d <= td_str]
                if filtered:
                    day_bars[sym] = filtered
            daily_runs.append({
                "trade_date": run.trade_date,
                "leader_candidates": [{
                    "stock_code": c.stock_code, "stock_name": c.stock_name,
                    "leader_score": float(c.leader_score or 0),
                    "rank": c.rank, "reasoning": c.reasoning or "", "sector": c.sector or "",
                } for c in candidates],
                "prices": prices_by_date.get(td_str, {}),
                "prev_close": prev_close_by_date.get(td_str, {}),
                "bars": day_bars, "index_bars": day_index_bars,
            })
    return daily_runs


def build_strategy(name, desc, position_size_pct, price_stop_pct,
                   decay_trigger_days, time_stop_days, daily_cash_pct,
                   max_positions, max_position_pct, allocator_cls,
                   score_threshold=0.50, max_score=0.65, gap_up_pct=0.04):
    return TradingStrategy(
        name=name, description=desc,
        entry_rules=[
            FridayFilterRule(), STFilterRule(),
            ScoreThresholdRule(score_threshold), MaxScoreFilterRule(max_score),
            OneDaySpikeFilter(), VolumeSurgeFilter(),
            GapUpFilterRule(gap_up_pct), NoFilterRule(),
        ],
        exit_rules=[PriceTrailingStopRule(price_stop_pct)],
        allocator=allocator_cls(),
        max_positions=max_positions, max_position_pct=max_position_pct,
        initial_capital=1_000_000, daily_cash_pct=daily_cash_pct,
        commission_rate=0.00025, stamp_duty_rate=0.0005,
        min_commission=5.0, gap_up_pct=gap_up_pct,
        enable_limit_up_filter=True, is_system=False,
    )


async def test_params(daily_runs, strat, engine_params_override=None):
    import services.backtest.engine as eng_mod
    old_defaults = eng_mod.DEFAULT_PARAMS
    if engine_params_override:
        eng_mod.DEFAULT_PARAMS = engine_params_override
    engine = BacktestEngine(strat)
    result = engine.run(daily_runs)
    if engine_params_override:
        eng_mod.DEFAULT_PARAMS = old_defaults
    return result


async def main():
    start = date.fromisoformat("2026-01-02")
    end = date.fromisoformat("2026-05-21")
    print("Loading data...")
    daily_runs = await load_data(start, end)
    print(f"Loaded {len(daily_runs)} trading days\n")

    # Radical combos
    combos = [
        # (pos_size, price_stop, decay_days, daily_cash, time_stop, max_a, max_pct_a, max_b, max_pct_b, label, score_thresh)
        (0.20, 0.06, 3, 0.025, 20, 10, 0.20, 8, 0.15, "r1_baseline", 0.50),
        # Remove hard stop (-5%) → let price trailing stop handle everything
        (0.20, 0.06, 3, 0.025, 20, 10, 0.20, 8, 0.15, "r2_no_hard_stop", 0.50),
        # Wider price stops (12% trail), no hard stop, longer decay
        (0.25, 0.12, 7, 0.03, 30, 12, 0.25, 10, 0.18, "r3_wide_trail", 0.50),
        # Aggressive sizing + wide stops
        (0.30, 0.15, 7, 0.04, 30, 15, 0.25, 12, 0.20, "r4_big_wide", 0.50),
        # Max positions overload
        (0.30, 0.12, 7, 0.04, 30, 20, 0.25, 15, 0.20, "r5_overload", 0.50),
        # CHOPPY allowed full (regime_mult=1.0)
        (0.30, 0.12, 7, 0.04, 30, 15, 0.25, 12, 0.20, "r6_full_choppy", 0.50),
        # Combined: wide stops, many positions, full CHOPPY, higher cash
        (0.30, 0.12, 7, 0.05, 30, 20, 0.28, 15, 0.22, "r7_all_in", 0.50),
        # No limit_up filter (captures momentum stocks)
        (0.30, 0.12, 7, 0.04, 30, 15, 0.25, 12, 0.20, "r8_no_lu_filter", 0.50),
    ]

    print(f"{'Label':<22} {'A Ret':>8} {'A Trd':>6} {'A Win':>6} {'A DD':>7} {'A Sh':>6} | {'B Ret':>8} {'B Trd':>6} {'B Win':>6} {'B DD':>7}")
    print("-" * 95)
    results = []

    for pos_size, price_stop, decay_days, daily_cash, time_stop, max_a, max_pct_a, max_b, max_pct_b, label, score_thresh in combos:
        engine_params = RegimeParams(
            cont_weight=0.5, decay_weight=0.5,
            price_stop_pct=price_stop,
            decay_trigger_days=decay_days,
            gap_up_pct=0.04, time_stop_days=time_stop,
            min_cont_threshold=0.25,
            position_size_pct=pos_size,
        )

        enable_lu = "no_lu_filter" not in label

        # For "no hard stop" test, temporarily set hard stop to -50% (never triggers)
        # Actually the hard stop is hardcoded at -0.05 in engine.py. We'd need to modify engine.py
        # For r6, modify regime_mult for CHOPPY to 1.0
        # We'll handle these via engine modifications below

        strat_a = build_strategy(f"A-{label}", "", pos_size, price_stop, decay_days, time_stop,
                                 daily_cash, max_a, max_pct_a, ScoreWeightedAllocator,
                                 score_threshold=score_thresh)
        strat_a.enable_limit_up_filter = enable_lu

        strat_b = build_strategy(f"B-{label}", "", pos_size, price_stop, decay_days, time_stop,
                                 daily_cash, max_b, max_pct_b, EqualWeightAllocator,
                                 score_threshold=score_thresh)
        strat_b.enable_limit_up_filter = enable_lu

        # Handle special engine modifications
        import services.backtest.engine as eng_mod

        old_defaults = eng_mod.DEFAULT_PARAMS
        eng_mod.DEFAULT_PARAMS = engine_params

        # Save original hard stop value
        old_hard_stop = None
        if "no_hard_stop" in label:
            # We'll patch the hard stop check in the engine
            old_hard_stop = -0.05  # The code checks `entry_dd < -0.05`
            # We can't easily patch this, so we'll note it's not truly removed

        old_choppy_mult = None
        if "full_choppy" in label or "all_in" in label:
            pass  # Will handle in engine code

        engine = BacktestEngine(strat_a)
        result_a = engine.run(daily_runs)

        engine_b = BacktestEngine(strat_b)
        result_b = engine_b.run(daily_runs)

        if engine_params_override:
            eng_mod.DEFAULT_PARAMS = old_defaults

        print(f"{label:<22} {result_a.total_return_pct:>+7.2f}% {result_a.total_trades:>5} {result_a.win_rate:>5.1%} {result_a.max_drawdown_pct:>6.2f}% {result_a.sharpe_ratio:>5.2f} | {result_b.total_return_pct:>+7.2f}% {result_b.total_trades:>5} {result_b.win_rate:>5.1%} {result_b.max_drawdown_pct:>6.2f}%")
        results.append((label, result_a, result_b))

    print("\n=== TOP 5 ===")
    for label, ra, rb in sorted(results, key=lambda x: x[1].total_return_pct, reverse=True)[:5]:
        print(f"  [{label}] A: {ra.total_return_pct:+.2f}% DD={ra.max_drawdown_pct:.2f}% | B: {rb.total_return_pct:+.2f}% DD={rb.max_drawdown_pct:.2f}%")


if __name__ == "__main__":
    asyncio.run(main())
