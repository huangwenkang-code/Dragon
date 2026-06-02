"""Holding scorer — continuation_score, decay_score, and ATR utility.

Pure price-driven from OHLCV bars.
"""


def compute_atr(stock_bars: list[dict], period: int = 14) -> float | None:
    """Average True Range over `period` bars. Returns None if insufficient data."""
    if len(stock_bars) < period + 1:
        return None
    trs = []
    for i in range(1, len(stock_bars)):
        h = stock_bars[i].get("high", 0) or 0
        l = stock_bars[i].get("low", 0) or 0
        prev_c = stock_bars[i - 1].get("close", 0) or 0
        if h <= 0 or l <= 0 or prev_c <= 0:
            continue
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
    if len(trs) < period:
        return None
    return sum(trs[-period:]) / period


def compute_continuation(
    stock_bars: list[dict],
    index_bars: list[dict],
    regime: str = "NORMAL",
) -> float:
    """Continuation score 0-1. 5 signals equal weight."""
    if not stock_bars or len(stock_bars) < 5:
        return 0.5

    signals = [
        _relative_strength(stock_bars, index_bars),
        _price_vs_ma5(stock_bars),
        _consecutive_up_days(stock_bars),
        _volume_trend(stock_bars),
        _new_high_frequency(stock_bars),
    ]
    return sum(s * w for s, w in zip(signals, [0.25, 0.20, 0.15, 0.20, 0.20]))


def compute_decay(
    stock_bars: list[dict],
    regime: str = "NORMAL",
) -> float:
    """Decay score 0-1. 5 signals equal weight."""
    if not stock_bars or len(stock_bars) < 5:
        return 0.5

    signals = [
        _gap_up_selloff(stock_bars),
        _volume_stagnation(stock_bars),
        _consecutive_down_days(stock_bars),
        _pullback_from_high(stock_bars),
        _ma_death_cross(stock_bars),
    ]
    return sum(s * w for s, w in zip(signals, [0.25, 0.20, 0.15, 0.25, 0.15]))


# ── Continuation signals ──

def _relative_strength(stock_bars, index_bars):
    """Relative strength: stock return vs index return."""
    if len(stock_bars) < 2 or len(index_bars) < 2:
        return 0.5
    stock_chg = (stock_bars[-1]["close"] - stock_bars[-2]["close"]) / max(stock_bars[-2]["close"], 0.01)
    index_chg = (index_bars[-1]["close"] - index_bars[-2]["close"]) / max(index_bars[-2]["close"], 0.01)
    diff = stock_chg - index_chg
    return min(1.0, max(0.0, 0.5 + diff * 10))


def _price_vs_ma5(stock_bars):
    """Price vs MA5."""
    closes = [b["close"] for b in stock_bars[-5:]]
    if not closes:
        return 0.5
    ma5 = sum(closes) / len(closes)
    if ma5 <= 0:
        return 0.5
    ratio = closes[-1] / ma5
    return min(1.0, max(0.0, ratio - 0.9))


def _consecutive_up_days(stock_bars):
    """Consecutive up days / 5."""
    recent = stock_bars[-5:]
    count = 0
    for i in range(1, len(recent)):
        if recent[i]["close"] > recent[i-1]["close"]:
            count += 1
    return count / 4 if len(recent) >= 5 else count / max(len(recent) - 1, 1)


def _volume_trend(stock_bars):
    """Volume trend: today's vol / 5-day avg, 1.0-1.5x optimal."""
    vols = [b.get("volume", 0) or 0 for b in stock_bars[-6:]]
    if not vols:
        return 0.5
    avg5 = sum(vols[:5]) / 5 if len(vols) >= 5 else sum(vols) / len(vols)
    if avg5 <= 0:
        return 0.5
    ratio = vols[-1] / avg5
    if 1.0 <= ratio <= 1.5:
        return 1.0
    elif ratio < 0.5:
        return 0.2
    elif ratio > 3.0:
        return 0.3
    else:
        return 0.6


def _new_high_frequency(stock_bars):
    """New high frequency in last 5 days / 5."""
    closes = [b["close"] for b in stock_bars[-10:]]
    if len(closes) < 5:
        return 0.0
    recent5 = closes[-5:]
    count = 0
    for i, c in enumerate(recent5):
        prev_high = max(closes[:5+i]) if closes[:5+i] else 0
        if c > prev_high:
            count += 1
    return count / 5


# ── Decay signals ──

def _gap_up_selloff(stock_bars):
    """Gap-up selloff intensity."""
    bar = stock_bars[-1]
    open_p = bar.get("open", 0) or 0
    close_p = bar.get("close", 0) or 0
    if open_p <= 0:
        return 0.0
    ratio = (open_p - close_p) / open_p
    return min(1.0, max(0.0, ratio / 0.05))


def _volume_stagnation(stock_bars):
    """High volume but price stagnant."""
    if len(stock_bars) < 2:
        return 0.0
    vol_now = stock_bars[-1].get("volume", 0) or 0
    vol_prev = stock_bars[-2].get("volume", 0) or 0
    price_now = stock_bars[-1].get("close", 0) or 0
    price_prev = stock_bars[-2].get("close", 0) or 0
    if vol_prev <= 0 or price_prev <= 0:
        return 0.0
    vol_chg = (vol_now - vol_prev) / vol_prev
    price_chg = abs(price_now - price_prev) / price_prev
    if vol_chg > 0.03 and price_chg < 0.01:
        return 0.8
    elif vol_chg > 0.03 and price_chg < 0.02:
        return 0.4
    return 0.0


def _consecutive_down_days(stock_bars):
    """Consecutive down days / 5."""
    recent = stock_bars[-5:]
    count = 0
    for i in range(1, len(recent)):
        if recent[i]["close"] < recent[i-1]["close"]:
            count += 1
    return count / 4 if len(recent) >= 5 else count / max(len(recent) - 1, 1)


def _pullback_from_high(stock_bars):
    """Pullback from 5-day high."""
    closes = [b["close"] for b in stock_bars[-5:]]
    high5 = max(closes)
    if high5 <= 0:
        return 0.0
    pullback = (high5 - closes[-1]) / high5
    return min(1.0, pullback / 0.10)


def _ma_death_cross(stock_bars):
    """MA5 crossing below MA10."""
    closes = [b["close"] for b in stock_bars]
    if len(closes) < 10:
        return 0.0
    ma5_prev = sum(closes[-6:-1]) / 5
    ma10_prev = sum(closes[-11:-1]) / 10
    ma5_now = sum(closes[-5:]) / 5
    ma10_now = sum(closes[-10:]) / 10
    if ma5_prev > ma10_prev and ma5_now < ma10_now:
        return 0.8
    elif ma5_now < ma10_now:
        return 0.3
    return 0.0
