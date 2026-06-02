"""Market Regime Service — 判定市场周期并输出动态参数."""
from dataclasses import dataclass


REGIME_LIANBAN = "LIANBAN"
REGIME_MONSTER = "MONSTER"
REGIME_AI_TREND = "AI_TREND"
REGIME_ICE = "ICE"
REGIME_NORMAL = "NORMAL"


@dataclass
class MarketRegime:
    temperature: float
    speculation_index: float
    regime: str
    volatility: float
    breadth: float


@dataclass
class RegimeParams:
    """Regime 调节参数."""
    cont_weight: float
    decay_weight: float
    price_stop_pct: float
    decay_trigger_days: int
    gap_up_pct: float
    time_stop_days: int
    min_cont_threshold: float
    position_size_pct: float = 0.025  # % of capital per stock


REGIME_PARAMS: dict[str, RegimeParams] = {
    REGIME_LIANBAN:    RegimeParams(0.6, 0.4, 0.07, 3, 0.03, 10, 0.25, 0.100),
    REGIME_MONSTER:    RegimeParams(0.5, 0.5, 0.07, 4, 0.04, 15, 0.25, 0.100),
    REGIME_AI_TREND:   RegimeParams(0.7, 0.3, 0.07, 4, 0.04, 15, 0.25, 0.100),
    REGIME_ICE:        RegimeParams(0.3, 0.7, 0.07, 3, 0.02, 15, 0.2,  0.100),
    REGIME_NORMAL:     RegimeParams(0.5, 0.5, 0.07, 4, 0.04, 15, 0.25, 0.100),
}


def determine_regime(
    limit_up_count: int,
    total_stocks: int,
    speculation_index: float,
    sector_concentration: float,
    breadth: float,
    volatility: float,
    lianban_height: int,
    consecutive_days: int = 1,
) -> MarketRegime:
    """判定当前市场周期."""
    temperature = min(1.0, limit_up_count / max(total_stocks, 1) * 10)

    if lianban_height >= 5 and limit_up_count > 80 and speculation_index > 0.7:
        return MarketRegime(temperature, speculation_index, REGIME_LIANBAN, volatility, breadth)

    if speculation_index > 0.6 and lianban_height >= 7:
        return MarketRegime(temperature, speculation_index, REGIME_MONSTER, volatility, breadth)

    if sector_concentration > 0.5 and consecutive_days > 5:
        return MarketRegime(temperature, speculation_index, REGIME_AI_TREND, volatility, breadth)

    if limit_up_count < 30 and breadth < 0.3:
        return MarketRegime(temperature, speculation_index, REGIME_ICE, volatility, breadth)

    return MarketRegime(temperature, speculation_index, REGIME_NORMAL, volatility, breadth)


def compute_speculation_index(
    daily_open_prices: dict[str, float],
    prev_close_prices: dict[str, float],
    sector_volume: dict[str, float],
    total_market_volume: float,
) -> float:
    """计算投机指数 0-1."""
    if not daily_open_prices:
        return 0.5

    near_limit_up = 0
    for sym in daily_open_prices:
        if sym in prev_close_prices and prev_close_prices[sym] > 0:
            gap = (daily_open_prices[sym] - prev_close_prices[sym]) / prev_close_prices[sym]
            if gap > 0.07:
                near_limit_up += 1
    gap_score = min(1.0, near_limit_up / max(len(daily_open_prices), 1) * 10)

    max_sector_vol = max(sector_volume.values()) if sector_volume else 0
    concentration = max_sector_vol / max(total_market_volume, 1) if total_market_volume > 0 else 0
    concentration_score = min(1.0, concentration * 2)

    return (gap_score * 0.5 + concentration_score * 0.5)


def get_params(regime: str) -> RegimeParams:
    """获取指定 regime 的参数表."""
    return REGIME_PARAMS.get(regime, REGIME_PARAMS[REGIME_NORMAL])
