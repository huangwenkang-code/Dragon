"""Auto-optimize regime parameters — grid search on recent data, save best to DB.

Usage:
  python scripts/optimize_regime.py              # auto-detect regime, optimize
  python scripts/optimize_regime.py --regime BULL --days 90
"""

import asyncio
import json
import random
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.connection import async_session_factory
from sqlalchemy import text as sa_text
from shared.market_regime import detect_regime
from shared.utils.logging import get_logger
from shared.utils.trading_day import is_trading_day, get_trading_date_str

logger = get_logger(__name__)

# Optimization config
N_SAMPLES = 150        # random parameter combos to test
LOOKBACK_DAYS = 90     # how many days of data to test on
TOP_N_KEEP = 3         # keep top N results in DB

# Parameter search space
PARAM_SPACE = {
    "max_positions":     list(range(3, 16)),        # 3-15
    "position_size_pct": [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30],
    "capital_multiplier": [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
    "time_stop_days":    [20, 25, 30, 35, 40, 50, 60],
    "filter_amount_min": [20_000_000, 30_000_000, 40_000_000, 50_000_000, 60_000_000, 80_000_000, 100_000_000],
}


def random_params() -> dict:
    """Generate random parameter combination."""
    return {
        k: random.choice(v) for k, v in PARAM_SPACE.items()
    }


def score_trades(trades: list, daily_snapshots: list) -> float:
    """Multi-dimensional score: return(40%) + sharpe(30%) + wr(15%) - dd(15%)."""
    if not trades:
        return -999.0

    # Total return
    total_ret = sum(t.get("pnl_pct", 0) for t in trades)
    ret_score = total_ret * 100  # convert to %

    # Win rate
    wins = sum(1 for t in trades if t.get("pnl_pct", 0) > 0)
    wr = wins / len(trades)

    # Max drawdown
    peak = 1_000_000
    max_dd = 0.0
    for s in daily_snapshots:
        eq = s.get("equity", peak)
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Sharpe (simplified)
    daily_rets = []
    for i in range(1, len(daily_snapshots)):
        prev = daily_snapshots[i - 1].get("equity", 1_000_000)
        curr = daily_snapshots[i].get("equity", prev)
        if prev > 0:
            daily_rets.append((curr - prev) / prev)
    avg_ret = sum(daily_rets) / len(daily_rets) if daily_rets else 0
    var_ret = sum((r - avg_ret) ** 2 for r in daily_rets) / len(daily_rets) if daily_rets else 1
    sharpe = avg_ret / (var_ret ** 0.5) * (252 ** 0.5) if var_ret > 0 else 0

    # Composite score
    dd_penalty = max(0, 1.0 - max_dd * 5)  # heavy penalty for large drawdowns
    score = ret_score * 0.40 + sharpe * 0.30 + wr * 100 * 0.15 + dd_penalty * 100 * 0.15
    return round(score, 2)


async def run_backtest_for_params(
    params: dict, start: str, end: str
) -> tuple[float, int]:
    """Run backtest with given params. Returns (score, trades_count)."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=300) as c:
            r = await c.post("http://localhost:8000/backtest/run", json={
                "strategy_name": "策略A-每日固定资金",
                "start_date": start, "end_date": end,
                "use_scanner": False, "use_v4": True,
            })
            if r.status_code != 200:
                return -999.0, 0
            d = r.json()
            trades = d.get("trades", [])
            snaps = d.get("daily_snapshots", [])
            return score_trades(trades, snaps), len(trades)
    except Exception as e:
        logger.warning("backtest failed: %s", e)
        return -999.0, 0


async def optimize(regime: str | None = None, lookback_days: int = LOOKBACK_DAYS):
    """Main optimization loop."""
    today_str = get_trading_date_str()
    if regime is None:
        regime = await detect_regime(today_str)
    logger.info("Optimizing for regime=%s, lookback=%d days, samples=%d", regime, lookback_days, N_SAMPLES)

    # Determine date range
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=lookback_days + 30)  # extra buffer
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()

    # Try cached backtests first (fast path: vary params via engine config only)
    # For now: run fresh backtests with each param set via the API
    # Future: inject params directly into engine to avoid HTTP overhead

    best = []
    for i in range(N_SAMPLES):
        params = random_params()
        # Update DB with these params temporarily
        async with async_session_factory() as s:
            # Deactivate current
            await s.execute(sa_text(
                "UPDATE regime_config SET is_active = FALSE WHERE regime = :r AND is_active = TRUE"
            ), {"r": regime})
            # Insert new
            await s.execute(sa_text(
                "INSERT INTO regime_config (regime, params, is_active, score) VALUES (:r, :p, TRUE, 0)"
            ), {"r": regime, "p": json.dumps(params)})
            await s.commit()

        # Reload config and run backtest
        from shared.config_loader import reload_config
        await reload_config()

        score, trades = await run_backtest_for_params(params, start_str, end_str)
        best.append((score, trades, dict(params)))

        if (i + 1) % 20 == 0:
            top = sorted(best, key=lambda x: -x[0])[:3]
            logger.info("[%d/%d] top3: %s", i + 1, N_SAMPLES,
                        [(s, t) for s, t, _ in top])

    # Sort and save top N
    best.sort(key=lambda x: -x[0])

    # Restore original params approach: keep best, discard rest
    async with async_session_factory() as s:
        # Deactivate ALL for this regime
        await s.execute(sa_text(
            "UPDATE regime_config SET is_active = FALSE WHERE regime = :r"
        ), {"r": regime})
        await s.commit()

        # Insert top N with proper scores
        for rank, (score, trades, params) in enumerate(best[:TOP_N_KEEP]):
            await s.execute(sa_text(
                "INSERT INTO regime_config (regime, params, is_active, score) VALUES (:r, :p, :a, :s)"
            ), {"r": regime, "p": json.dumps(params), "a": rank == 0, "s": score})
        await s.commit()

    # Reload config to pick up changes
    from shared.config_loader import reload_config
    await reload_config()

    print(f"\n{'='*60}")
    print(f"Optimization complete for {regime}")
    print(f"Tested: {N_SAMPLES} combos on {lookback_days} days")
    print(f"\nTop {TOP_N_KEEP}:")
    for rank, (score, trades, params) in enumerate(best[:TOP_N_KEEP]):
        status = "✅ ACTIVE" if rank == 0 else ""
        print(f"  #{rank+1} score={score:.1f} trades={trades} {status}")
        print(f"      {json.dumps(params)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", type=str, default=None, help="Target regime (auto-detect if not set)")
    ap.add_argument("--days", type=int, default=LOOKBACK_DAYS, help="Lookback days")
    ap.add_argument("--samples", type=int, default=N_SAMPLES, help="Random samples")
    args = ap.parse_args()
    N_SAMPLES = args.samples
    asyncio.run(optimize(args.regime, args.days))
