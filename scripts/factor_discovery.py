"""
Factor discovery: compute pre-entry factors for all backtest trades,
then rank which factors best separate hard-stops from winners.

Uses stock_daily_bars_v2 via SQL for the pre-entry window.
"""
import asyncio, json, sys
from datetime import date, timedelta
from pathlib import Path
from collections import defaultdict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

from db.connection import async_session_factory
from sqlalchemy import select, and_
from db.models import StockDailyBar

# Load backtest trades
BACKTEST_FILE = sys.argv[1] if len(sys.argv) > 1 else 'D:/K/temp_bt_api_b2.json'
with open(BACKTEST_FILE, encoding='utf-8') as f:
    trades = json.load(f)['trades']


def compute_factors(bars: list[dict], entry_date_str: str) -> dict:
    """
    bars: list of bar dicts with open/high/low/close/volume, sorted by trade_date ASC.
    All bars are BEFORE entry_date (no look-ahead).
    Returns dict of factor_name -> value.
    """
    if len(bars) < 5:
        return {}  # Not enough history

    closes = [b['close'] for b in bars]
    highs = [b['high'] for b in bars]
    lows = [b['low'] for b in bars]
    opens = [b['open'] for b in bars]
    volumes = [b['volume'] or 0 for b in bars]

    factors = {}

    # ── Returns ──
    if len(closes) >= 21:
        # 1-month return (20 trading days)
        factors['ret_20d'] = (closes[-1] - closes[-21]) / closes[-21] if closes[-21] > 0 else 0
    if len(closes) >= 11:
        factors['ret_10d'] = (closes[-1] - closes[-11]) / closes[-11] if closes[-11] > 0 else 0
    if len(closes) >= 6:
        factors['ret_5d'] = (closes[-1] - closes[-6]) / closes[-6] if closes[-6] > 0 else 0
    factors['ret_1d'] = (closes[-1] - closes[-2]) / closes[-2] if len(closes) >= 2 and closes[-2] > 0 else 0

    # ── Momentum quality: Sharpe-like ratio (avg return / std of returns) ──
    if len(closes) >= 6:
        daily_rets = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(-5, 0) if closes[i-1] > 0]
        if daily_rets and len(daily_rets) >= 3:
            avg_ret = sum(daily_rets) / len(daily_rets)
            var = sum((r - avg_ret)**2 for r in daily_rets) / len(daily_rets)
            factors['momentum_quality_5d'] = avg_ret / (var**0.5) if var > 0 else 0

    # ── Trend smoothness: R^2 of linear fit over past 10 days ──
    if len(closes) >= 11:
        n = 10
        y = closes[-n:]
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        xy_cov = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        x_var = sum((xi - x_mean)**2 for xi in x)
        y_var = sum((yi - y_mean)**2 for yi in y)
        if x_var > 0 and y_var > 0:
            slope = xy_cov / x_var
            r2 = (xy_cov**2) / (x_var * y_var)
            factors['trend_r2_10d'] = r2 if slope > 0 else -r2  # positive = uptrend fit
            factors['slope_10d'] = slope / y_mean  # normalized slope

    # ── Max drawdown during run-up (lower = stronger trend) ──
    if len(closes) >= 21:
        peak = closes[-21]
        max_dd = 0.0
        for c in closes[-20:]:
            if c > peak:
                peak = c
            dd = (peak - c) / peak
            if dd > max_dd:
                max_dd = dd
        factors['max_dd_20d'] = -max_dd  # negative so higher=better

    # ── Distance from N-day high ──
    if len(highs) >= 20:
        h20 = max(highs[-20:])
        factors['dist_from_20d_high'] = (closes[-1] - h20) / h20 if h20 > 0 else 0
    if len(highs) >= 5:
        h5 = max(highs[-5:])
        factors['dist_from_5d_high'] = (closes[-1] - h5) / h5 if h5 > 0 else 0

    # ── Volume: 5-day avg volume / 20-day avg volume (volume expansion) ──
    if len(volumes) >= 20:
        vol_5d = sum(v for v in volumes[-5:] if v) / max(1, sum(1 for v in volumes[-5:] if v))
        vol_20d = sum(v for v in volumes[-20:] if v) / max(1, sum(1 for v in volumes[-20:] if v))
        if vol_20d > 0:
            factors['vol_ratio_5_20'] = vol_5d / vol_20d

    # ── Volume trend: is volume increasing on up days? ──
    if len(volumes) >= 6:
        up_day_vols = []
        down_day_vols = []
        for i in range(-5, 0):
            if closes[i] > closes[i-1] and closes[i-1] > 0:
                up_day_vols.append(volumes[i])
            elif closes[i] < closes[i-1]:
                down_day_vols.append(volumes[i])
        if up_day_vols and down_day_vols:
            avg_up = sum(up_day_vols) / len(up_day_vols)
            avg_down = sum(down_day_vols) / len(down_day_vols)
            factors['vol_up_down_ratio'] = avg_up / avg_down if avg_down > 0 else 2.0

    # ── Consecutive up/down days ──
    if len(closes) >= 2:
        cons_up = 0
        for i in range(len(closes)-1, max(len(closes)-10, 0), -1):
            if closes[i] > closes[i-1]:
                cons_up += 1
            else:
                break
        factors['consecutive_up_days'] = cons_up

    # ── Price position vs MAs ──
    if len(closes) >= 5:
        ma5 = sum(closes[-5:]) / 5
        factors['pct_from_ma5'] = (closes[-1] - ma5) / ma5 if ma5 > 0 else 0
    if len(closes) >= 10:
        ma10 = sum(closes[-10:]) / 10
        factors['pct_from_ma10'] = (closes[-1] - ma10) / ma10 if ma10 > 0 else 0

    # ── Daily amplitude (volatility proxy) ──
    if len(bars) >= 5:
        amps = []
        for b in bars[-5:]:
            if b['open'] and b['close'] and b['open'] > 0:
                amp = abs(b['high'] - b['low']) / b['open']
                amps.append(amp)
        if amps:
            factors['avg_amplitude_5d'] = sum(amps) / len(amps)

    # ── Limit-up count in past 5 days ──
    if len(bars) >= 6:
        limit_ups = 0
        for b in bars[-6:-1]:  # exclude last day (could be the entry day signal)
            if b['open'] and b['close'] and b['open'] > 0:
                chg = (b['close'] - b['open']) / b['open']
                if chg > 0.095:  # ~limit up
                    limit_ups += 1
        factors['limit_up_count_5d'] = limit_ups

    # ── Gap-up count in past 5 days ──
    if len(bars) >= 6:
        gap_ups = 0
        for i in range(-5, 0):
            if opens[i] and opens[i-1] and opens[i-1] > 0:
                gap = (opens[i] - closes[i-1]) / closes[i-1]
                if gap > 0.02:
                    gap_ups += 1
        factors['gap_up_count_5d'] = gap_ups

    return factors


async def main():
    # Get unique (stock_code, entry_date) pairs
    entries = [(t['stock_code'], t['entry_date']) for t in trades]
    symbols = list(set(s[0] for s in entries))

    # Date range: from 20 days before earliest entry to latest entry
    entry_dates = sorted(set(s[1] for s in entries))
    min_date = date.fromisoformat(entry_dates[0]) - timedelta(days=40)
    max_date = date.fromisoformat(entry_dates[-1])

    print(f"Loading bars for {len(symbols)} symbols from {min_date} to {max_date}...")

    async with async_session_factory() as session:
        result = await session.execute(
            select(StockDailyBar).where(
                and_(
                    StockDailyBar.trade_date >= min_date,
                    StockDailyBar.trade_date <= max_date,
                    StockDailyBar.symbol.in_(symbols),
                )
            ).order_by(StockDailyBar.trade_date.asc())
        )
        rows = result.scalars().all()

    # Organize bars by symbol
    bars_by_symbol: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        bars_by_symbol[r.symbol].append({
            'trade_date': str(r.trade_date),
            'open': float(r.open or 0),
            'high': float(r.high or 0),
            'low': float(r.low or 0),
            'close': float(r.close or 0),
            'volume': float(r.volume or 0),
        })

    # Compute factors for each trade
    print(f"Computing factors for {len(entries)} entries...")
    factor_data = []  # list of {trade info + factor dict}
    errors = 0

    for t in trades:
        sym = t['stock_code']
        entry_d = t['entry_date']
        all_bars = bars_by_symbol.get(sym, [])

        # Filter bars before entry_date
        pre_bars = [b for b in all_bars if b['trade_date'] < entry_d]

        if len(pre_bars) < 10:
            errors += 1
            continue

        factors = compute_factors(pre_bars, entry_d)
        if not factors:
            errors += 1
            continue

        factor_data.append({
            'stock_code': sym,
            'stock_name': t['stock_name'],
            'entry_date': entry_d,
            'entry_score': t['entry_score'],
            'holding_days': t['holding_days'],
            'pnl_pct': t['pnl_pct'],
            'net_pnl': t['net_pnl'],
            'exit_reason': t['exit_reason'],
            'is_winner': t['net_pnl'] > 0,
            'is_hard_stop': '硬止损' in t['exit_reason'],
            **factors,
        })

    print(f"Computed factors for {len(factor_data)} trades ({errors} skipped due to insufficient data)")

    # ── STATISTICAL ANALYSIS ──
    hard_stops = [f for f in factor_data if f['is_hard_stop']]
    winners = [f for f in factor_data if f['is_winner']]
    losers = [f for f in factor_data if not f['is_winner']]

    print(f"\nHard stops: {len(hard_stops)}, Winners: {len(winners)}, Losers: {len(losers)}")

    # For each factor, compute mean in hard-stop group vs winner group
    factor_names = [k for k in factor_data[0].keys()
                    if k.startswith(('ret_', 'momentum_', 'trend_', 'max_dd_', 'dist_',
                                     'vol_', 'consecutive_', 'pct_from_', 'avg_amplitude',
                                     'limit_up_', 'gap_up_', 'slope_'))]

    print(f"\n{'='*90}")
    print(f"  FACTOR ANALYSIS: Hard Stop vs Winner")
    print(f"{'='*90}")
    print(f"  {'Factor':<25} {'HS Mean':>10} {'Win Mean':>10} {'Diff':>10} {'T-stat':>8} {'Direction'}")
    print(f"  {'-'*75}")

    significant = []
    for fn in sorted(factor_names):
        hs_vals = [f[fn] for f in hard_stops if fn in f]
        w_vals = [f[fn] for f in winners if fn in f]

        if len(hs_vals) < 5 or len(w_vals) < 5:
            continue

        hs_mean = sum(hs_vals) / len(hs_vals)
        w_mean = sum(w_vals) / len(w_vals)
        diff = w_mean - hs_mean

        # Simple t-stat (Welch's approximation)
        hs_var = sum((v - hs_mean)**2 for v in hs_vals) / (len(hs_vals) - 1) if len(hs_vals) > 1 else 0
        w_var = sum((v - w_mean)**2 for v in w_vals) / (len(w_vals) - 1) if len(w_vals) > 1 else 0
        se = ((hs_var / len(hs_vals)) + (w_var / len(w_vals))) ** 0.5
        t_stat = diff / se if se > 0 else 0

        # Interpret direction
        if abs(t_stat) > 1.0:
            direction = "WIN higher" if diff > 0 else "HS higher"
            significant.append((fn, hs_mean, w_mean, diff, t_stat, direction))

        print(f"  {fn:<25} {hs_mean:>10.4f} {w_mean:>10.4f} {diff:>+10.4f} {t_stat:>+8.2f} {'***' if abs(t_stat) > 1.5 else ''}")

    print(f"\n{'='*90}")
    print(f"  MOST SIGNIFICANT FACTORS (|t-stat| > 1.0)")
    print(f"{'='*90}")
    significant.sort(key=lambda x: abs(x[4]), reverse=True)
    for fn, hs_m, w_m, diff, t_stat, direction in significant[:15]:
        print(f"  {fn:<25} t={t_stat:>+7.2f}  HS={hs_m:.4f}  Win={w_m:.4f}  [{direction}]")

    # ── CORRELATION WITH PNL ──
    print(f"\n{'='*90}")
    print(f"  FACTOR vs PNL% CORRELATION")
    print(f"{'='*90}")
    cors = []
    for fn in sorted(factor_names):
        vals = [(f[fn], f['pnl_pct']) for f in factor_data if fn in f]
        if len(vals) < 10:
            continue
        xs = [v[0] for v in vals]
        ys = [v[1] for v in vals]
        n = len(xs)
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        cov = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
        x_std = (sum((x - x_mean)**2 for x in xs) / n) ** 0.5
        y_std = (sum((y - y_mean)**2 for y in ys) / n) ** 0.5
        corr = cov / (n * x_std * y_std) if x_std > 0 and y_std > 0 else 0
        cors.append((fn, corr))

    cors.sort(key=lambda x: abs(x[1]), reverse=True)
    for fn, corr in cors[:15]:
        print(f"  {fn:<25} corr={corr:>+7.3f} {'***' if abs(corr) > 0.15 else ''}")

    # ── PRACTICAL FILTER SUGGESTIONS ──
    print(f"\n{'='*90}")
    print(f"  PROPOSED PRE-ENTRY FILTERS")
    print(f"{'='*90}")

    # Test combined filters
    for label, filter_fn in [
        ("ret_5d > 0 (positive 5-day momentum)", lambda f: f.get('ret_5d', -999) > 0),
        ("ret_5d < 0.15 (not overextended)", lambda f: 0 < f.get('ret_5d', 0) < 0.15),
        ("vol_up_down_ratio > 1.0 (bullish volume)", lambda f: f.get('vol_up_down_ratio', 0) > 1.0),
        ("dist_from_5d_high > -0.03 (not at peak)", lambda f: f.get('dist_from_5d_high', -999) > -0.03),
        ("slope_10d > 0 (uptrend)", lambda f: f.get('slope_10d', -999) > 0),
        ("max_dd_20d > -0.10 (shallow pullback)", lambda f: f.get('max_dd_20d', -999) > -0.10),
        ("pct_from_ma5 > -0.02 (above MA5)", lambda f: f.get('pct_from_ma5', -999) > -0.02),
        ("consecutive_up_days <= 3 (not exhausted)", lambda f: 0 < f.get('consecutive_up_days', 99) <= 3),
        ("avg_amplitude_5d < 0.06 (stable)", lambda f: f.get('avg_amplitude_5d', 999) < 0.06),
    ]:
        kept = [f for f in factor_data if filter_fn(f)]
        filtered = len(factor_data) - len(kept)
        if kept:
            wr = sum(1 for f in kept if f['is_winner']) / len(kept) * 100
            pnl = sum(f['net_pnl'] for f in kept)
            hs = sum(1 for f in kept if f['is_hard_stop'])
            hs_rate = hs / len(kept) * 100
            print(f"  {label}")
            print(f"    Keep {len(kept)}/{len(factor_data)} trades, WR={wr:.1f}%, PnL={pnl:,.0f}, HS_rate={hs_rate:.0f}%")


if __name__ == "__main__":
    asyncio.run(main())
