"""Regime-adaptive strategy configuration.

Single source of truth for per-regime parameters.
Used by both pipeline (generate_candidates) and backtest engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RegimeStrategy:
    """Parameters for a single market regime."""
    # Selection mode
    selection: str = "momentum"  # "momentum" | "oversold_bounce" | "blend"

    # For momentum mode: score = abs(change) * amount_percentile * w + ...
    momentum_change_weight: float = 0.35
    momentum_amount_weight: float = 0.40
    momentum_turnover_weight: float = 0.25

    # For oversold_bounce mode
    oversold_drawdown_min: float = 0.30   # min 60d drawdown
    oversold_amount_min: float = 50_000_000  # min daily turnover
    oversold_price_min: float = 3.0

    # For blend mode: ratio of momentum vs oversold in candidate pool
    blend_momentum_ratio: float = 0.5

    # Quality filter (applied in all modes)
    filter_price_min: float = 3.0
    filter_amount_min: float = 50_000_000
    filter_exclude_limit_down: bool = True

    # Position sizing
    max_positions: int = 10
    position_size_pct: float = 0.20
    capital_multiplier: float = 1.0  # regime-level leverage (BEAR=0.5 etc)

    # Entry threshold
    min_score: float = 0.0  # minimum leader_score to enter

    # Top N candidates per day
    top_n: int = 60


# ---------------------------------------------------------------------------
# Default regime configuration — calibrated from 2024-2026 backtests
# ---------------------------------------------------------------------------

REGIME_CONFIG: dict[str, RegimeStrategy] = {
    "BULL": RegimeStrategy(
        selection="momentum",
        # BULL: ride the wave — allow penny stocks to surge
        momentum_change_weight=0.35,
        momentum_amount_weight=0.40,
        momentum_turnover_weight=0.25,
        filter_price_min=1.0,          # NO price floor in BULL (penny stocks can 10x)
        filter_amount_min=30_000_000,  # relaxed
        max_positions=10,
        position_size_pct=0.20,
        capital_multiplier=1.0,
        min_score=0.0,
        top_n=60,
    ),
    "CHOPPY_UP": RegimeStrategy(
        selection="momentum",     # trending up → momentum, relaxed filters
        momentum_change_weight=0.30,
        momentum_amount_weight=0.45,
        momentum_turnover_weight=0.25,
        filter_price_min=1.0,          # relaxed — captures low-price surges
        filter_amount_min=30_000_000,
        max_positions=8,
        position_size_pct=0.18,
        capital_multiplier=0.90,
        min_score=0.0,
        top_n=60,
    ),
    "CHOPPY": RegimeStrategy(
        selection="blend",        # truly flat → half momentum, half oversold
        blend_momentum_ratio=0.5,
        momentum_change_weight=0.25,
        momentum_amount_weight=0.45,
        momentum_turnover_weight=0.30,
        oversold_drawdown_min=0.25,
        filter_price_min=3.0,
        filter_amount_min=50_000_000,
        max_positions=6,
        position_size_pct=0.12,
        capital_multiplier=0.70,
        min_score=0.0,
        top_n=60,
    ),
    "CHOPPY_DOWN": RegimeStrategy(
        selection="oversold_bounce",  # trending down → go defensive
        oversold_drawdown_min=0.25,
        oversold_amount_min=50_000_000,
        oversold_price_min=3.0,
        filter_price_min=3.0,
        filter_amount_min=50_000_000,
        max_positions=5,
        position_size_pct=0.10,
        capital_multiplier=0.50,
        min_score=0.0,
        top_n=60,
    ),
    "BEAR": RegimeStrategy(
        selection="oversold_bounce",
        oversold_drawdown_min=0.30,
        oversold_amount_min=50_000_000,
        oversold_price_min=3.0,
        filter_price_min=3.0,
        filter_amount_min=50_000_000,
        max_positions=5,
        position_size_pct=0.10,
        capital_multiplier=0.50,
        min_score=0.0,
        top_n=60,
    ),
}


def get_strategy(regime: str) -> RegimeStrategy:
    """Get the strategy config for a given regime. Falls back to CHOPPY."""
    if regime in REGIME_CONFIG:
        return REGIME_CONFIG[regime]
    # Handle legacy/unknown regimes
    if "UP" in regime:
        return REGIME_CONFIG["CHOPPY_UP"]
    if "DOWN" in regime:
        return REGIME_CONFIG["CHOPPY_DOWN"]
    return REGIME_CONFIG["CHOPPY"]


def print_config():
    """Print current regime configuration."""
    for regime, cfg in REGIME_CONFIG.items():
        print(f"\n[{regime}]")
        print(f"  selection:          {cfg.selection}")
        if cfg.selection == "momentum":
            print(f"  momentum weights:   chg={cfg.momentum_change_weight} amt={cfg.momentum_amount_weight} turnover={cfg.momentum_turnover_weight}")
        elif cfg.selection == "oversold_bounce":
            print(f"  oversold drawdown:  >= {cfg.oversold_drawdown_min*100:.0f}%")
        elif cfg.selection == "blend":
            print(f"  blend ratio:        {cfg.blend_momentum_ratio:.0%} momentum / {1-cfg.blend_momentum_ratio:.0%} oversold")
        print(f"  filter:             price>={cfg.filter_price_min} amt>={cfg.filter_amount_min/1e4:.0f}万")
        print(f"  positions:          max={cfg.max_positions} size={cfg.position_size_pct:.0%}")
        print(f"  capital multiplier: {cfg.capital_multiplier:.0%}")
        print(f"  candidates/day:     {cfg.top_n}")
