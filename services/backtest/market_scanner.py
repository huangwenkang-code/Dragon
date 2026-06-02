"""
Full-market momentum scanner — computes dragon_score for ALL stocks
and returns top-N candidates, replacing the 同花顺 hot-stock pipeline.

Dragon Score formula:
  trend_r2_10d  * 0.25  — R² of 10-day linear fit (trend quality)
  slope_10d     * 0.20  — normalized slope (trend strength)
  mom_quality   * 0.20  — Sharpe of 5-day returns (risk-adjusted momentum)
  ret_5d        * 0.15  — 5-day return, sigmoid clamped [0, 0.25]
  vol_health    * 0.10  — up/down volume ratio, capped
  position      * 0.10  — distance from 5-day high (closer = continuation)
"""


def compute_dragon_score(bars: list[dict]) -> float:
    """
    Compute dragon_score from OHLCV bars (all bars up to trade_date).
    Uses pre-entry bars only (excludes the current day's bar).
    Returns score in [0, 1] range.
    """
    if len(bars) < 21:
        return 0.0

    pre_bars = bars[:-1]  # exclude today's bar (entry day)
    if len(pre_bars) < 20:
        return 0.0

    closes = [b.get("close", 0) for b in pre_bars]
    highs = [b.get("high", 0) for b in pre_bars]
    volumes = [b.get("volume", 0) or 0 for b in pre_bars]

    # Filter out stocks with zero/negative prices
    if any(c <= 0 for c in closes[-15:]):
        return 0.0

    # ── 1. trend_r2_10d (0.25 weight) ──
    n = 10
    y = closes[-n:]
    x = list(range(n))
    x_m = sum(x) / n
    y_m = sum(y) / n
    xy_c = sum((x[i] - x_m) * (y[i] - y_m) for i in range(n))
    x_v = sum((xi - x_m) ** 2 for xi in x)
    y_v = sum((yi - y_m) ** 2 for yi in y)
    if x_v > 0 and y_v > 0:
        slope = xy_c / x_v
        r2 = (xy_c ** 2) / (x_v * y_v)
        if slope < 0:
            r2 = -r2  # negative for downtrend
    else:
        r2 = 0.0
        slope = 0.0

    # Normalize: r2 ∈ [-1, 1] → map [-0.3, 0.8] to [0, 1]
    trend_score = max(0.0, min(1.0, (r2 + 0.3) / 1.1))

    # ── 2. slope_10d (0.20 weight) ──
    # slope is absolute daily return; normalize by price level
    slope_norm = slope / y_m if y_m > 0 else 0.0
    # Typical range: -0.02 to +0.03; map [0, 0.02] to [0, 1]
    slope_score = max(0.0, min(1.0, slope_norm * 50.0))

    # ── 3. momentum_quality_5d (0.20 weight) ──
    daily_rets = []
    for i in range(-5, 0):
        if closes[i - 1] > 0:
            daily_rets.append((closes[i] - closes[i - 1]) / closes[i - 1])
    if len(daily_rets) >= 3:
        avg_ret = sum(daily_rets) / len(daily_rets)
        var_ret = sum((r - avg_ret) ** 2 for r in daily_rets) / len(daily_rets)
        mom_q = avg_ret / (var_ret ** 0.5) if var_ret > 0 else 0.0
    else:
        mom_q = 0.0
    # Map [0, 2.0] to [0, 1]
    mom_score = max(0.0, min(1.0, mom_q / 2.0))

    # ── 4. ret_5d (0.15 weight) ──
    ret_5d = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 and closes[-6] > 0 else 0.0
    # Sigmoid-like: map [-0.05, 0.25] to [0, 1], with peak at ~0.15
    if ret_5d < 0:
        ret_score = 0.0
    elif ret_5d > 0.25:
        ret_score = 0.3  # penalty for overextended (>25% in 5 days)
    else:
        ret_score = ret_5d / 0.25  # linear [0, 0.25] → [0, 1]

    # ── 5. vol_health (0.10 weight) ──
    up_vols, down_vols = [], []
    for i in range(-5, 0):
        if closes[i] > closes[i - 1]:
            up_vols.append(volumes[i])
        elif closes[i] < closes[i - 1]:
            down_vols.append(volumes[i])
    if up_vols and down_vols:
        avg_up = sum(up_vols) / len(up_vols)
        avg_down = sum(down_vols) / len(down_vols)
        vol_ratio = avg_up / avg_down if avg_down > 0 else 1.5
    else:
        vol_ratio = 1.0
    # Map [0.5, 3.0] to [0, 1], peak at ~1.5
    if vol_ratio < 0.5:
        vol_score = 0.0
    elif vol_ratio > 3.0:
        vol_score = 0.4  # penalty for extreme volume (distribution)
    else:
        vol_score = min(1.0, vol_ratio / 1.5)

    # ── 6. position vs 5d high (0.10 weight) ──
    h5 = max(highs[-5:])
    dist_5h = (closes[-1] - h5) / h5 if h5 > 0 else 0.0
    # Map [-0.08, 0.0] to [0, 1]; below -0.08 is too far from high (falling)
    if dist_5h < -0.10:
        pos_score = 0.0
    elif dist_5h > 0.0:
        pos_score = 0.9  # at new high
    else:
        pos_score = 1.0 + dist_5h / 0.10  # linear [-0.10, 0] → [0, 1]

    # ── Composite ──
    score = (
        trend_score * 0.25
        + slope_score * 0.20
        + mom_score * 0.20
        + ret_score * 0.15
        + vol_score * 0.10
        + pos_score * 0.10
    )
    return round(score, 4)


def scan_market(bars_by_symbol: dict[str, list[dict]],
                trade_date: str,
                prices: dict[str, float],
                top_n: int = 80) -> list[dict]:
    """
    Scan all stocks and return top-N candidates ranked by dragon_score.

    Args:
        bars_by_symbol: {symbol: [bar_dict, ...]} — bars UP TO trade_date
        trade_date: current trading date string
        prices: {symbol: today's open price}
        top_n: max candidates to return

    Returns:
        List of candidate dicts: {stock_code, dragon_score, ...}
    """
    candidates = []
    for sym, bars in bars_by_symbol.items():
        if len(bars) < 21:
            continue
        price = prices.get(sym, 0)
        if price <= 0 or price > 2000:  # skip invalid/too expensive
            continue

        score = compute_dragon_score(bars)
        if score <= 0:
            continue

        candidates.append({
            "stock_code": sym,
            "stock_name": sym,
            "dragon_score": score,
            "leader_score": score,  # used by engine for sorting & filtering
        })

    # Sort by abs(score - 0.58) — candidates closest to sweet spot first
    candidates.sort(key=lambda c: abs(c["dragon_score"] - 0.58))
    return candidates[:top_n]
