"""Trading rules — EntryRule, ExitRule, PositionAllocator abstract base classes + concrete implementations.

Extensible via rule_type / allocator_type string registration.
Add new rules by subclassing and adding to the registry dicts — no engine changes needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------

class EntryRule(ABC):
    """Whether to buy a candidate stock."""
    rule_type: str = ""
    params: dict = {}

    @abstractmethod
    def should_enter(self, candidate: dict, context: dict) -> bool:
        """Return True if this candidate should be bought."""
        ...


class ExitRule(ABC):
    """Whether to sell a held position."""
    rule_type: str = ""
    params: dict = {}

    @abstractmethod
    def should_exit(self, position, current_candidate: dict | None, context: dict) -> str | None:
        """Return exit reason string if should sell, None otherwise."""
        ...


class PositionAllocator(ABC):
    """How much capital to allocate to each candidate."""
    allocator_type: str = ""
    params: dict = {}

    @abstractmethod
    def allocate(self, candidates: list[dict], available_cash: float, current_prices: dict, context: dict) -> list[dict]:
        """Return list of order dicts: {stock_code, stock_name, score, allocated_cash, shares, entry_price}"""
        ...


# ---------------------------------------------------------------------------
# Entry Rules
# ---------------------------------------------------------------------------

class ScoreThresholdRule(EntryRule):
    """Only enter candidates with leader_score >= min_score.
    For scanner candidates (with dragon_score field), uses dragon_score instead."""
    rule_type = "score_threshold"

    def __init__(self, min_score: float = 0.5):
        self.params = {"min_score": min_score}

    def should_enter(self, candidate: dict, context: dict) -> bool:
        score = candidate.get("dragon_score") or candidate.get("leader_score") or 0
        return score >= self.params["min_score"]


class NoFilterRule(EntryRule):
    """Allow all candidates to enter."""
    rule_type = "no_filter"

    def __init__(self):
        self.params = {}

    def should_enter(self, candidate: dict, context: dict) -> bool:
        return True


# ---------------------------------------------------------------------------
# Exit Rules
# ---------------------------------------------------------------------------

class ScoreCliffRule(ExitRule):
    """Sell when current score drops below an absolute threshold."""
    rule_type = "score_cliff"

    def __init__(self, threshold: float = 0.3):
        self.params = {"threshold": threshold}

    def should_exit(self, position, current_candidate: dict | None, context: dict) -> str | None:
        score = current_candidate.get("leader_score") or 0 if current_candidate else position.current_score
        if score < self.params["threshold"]:
            return f"得分悬崖({score:.2f}<{self.params['threshold']})"
        return None


class TrailingStopRule(ExitRule):
    """Sell when score drops more than drawdown_pct from its peak since entry."""
    rule_type = "trailing_stop"

    def __init__(self, drawdown_pct: float = 0.15):
        self.params = {"drawdown_pct": drawdown_pct}

    def should_exit(self, position, current_candidate: dict | None, context: dict) -> str | None:
        if position.peak_score > 0:
            dd = (position.peak_score - position.current_score) / position.peak_score
            if dd > self.params["drawdown_pct"]:
                return f"回撤止损({dd:.1%}>{self.params['drawdown_pct']:.0%})"
        return None


class ScoreDeclineRule(ExitRule):
    """Sell when score has decreased for N consecutive days."""
    rule_type = "score_decline"

    def __init__(self, consecutive_days: int = 3):
        self.params = {"consecutive_days": consecutive_days}

    def should_exit(self, position, current_candidate: dict | None, context: dict) -> str | None:
        history = position.score_history
        n = self.params["consecutive_days"]
        if len(history) >= n:
            recent = history[-n:]
            if all(recent[i] > recent[i + 1] for i in range(len(recent) - 1)):
                return f"连续{n}天得分下滑"
        return None


class PriceTrailingStopRule(ExitRule):
    """Sell when current price drops below (1 - max_drawdown) * peak_price since entry."""
    rule_type = "price_trailing_stop"

    def __init__(self, max_drawdown: float = 0.10, drawdown_pct: float = None):
        self.params = {"max_drawdown": drawdown_pct if drawdown_pct is not None else max_drawdown}

    def should_exit(self, position, current_candidate: dict | None, context: dict) -> str | None:
        if position.peak_price > 0 and position.current_price > 0:
            dd = (position.peak_price - position.current_price) / position.peak_price
            if dd > self.params["max_drawdown"]:
                return f"价格回撤止损({dd:.1%}>{self.params['max_drawdown']:.0%})"
        return None


# ---------------------------------------------------------------------------
# Position Allocators
# ---------------------------------------------------------------------------

class ScoreWeightedAllocator(PositionAllocator):
    """Allocate cash proportional to each candidate's leader_score."""
    allocator_type = "score_weighted"

    def __init__(self):
        self.params = {}

    def allocate(self, candidates: list[dict], available_cash: float, current_prices: dict, context: dict) -> list[dict]:
        total_score = sum(c.get("leader_score") or 0 for c in candidates)
        if total_score == 0:
            return []
        orders = []
        for c in candidates:
            weight = (c.get("leader_score") or 0) / total_score
            cash = available_cash * weight
            price = current_prices.get(c["stock_code"], 0)
            if price <= 0:
                continue
            shares = int(cash / price / 100) * 100  # round to 100-share lots
            if shares >= 100:
                orders.append({
                    "stock_code": c["stock_code"],
                    "stock_name": c.get("stock_name", ""),
                    "score": c.get("leader_score") or 0,
                    "allocated_cash": shares * price,
                    "shares": shares,
                    "entry_price": price,
                })
        return orders


class EqualWeightAllocator(PositionAllocator):
    """Allocate equal cash to each candidate."""
    allocator_type = "equal_weight"

    def __init__(self):
        self.params = {}

    def allocate(self, candidates: list[dict], available_cash: float, current_prices: dict, context: dict) -> list[dict]:
        if not candidates:
            return []
        per_stock = available_cash / len(candidates)
        orders = []
        for c in candidates:
            price = current_prices.get(c["stock_code"], 0)
            if price <= 0:
                continue
            shares = int(per_stock / price / 100) * 100
            if shares >= 100:
                orders.append({
                    "stock_code": c["stock_code"],
                    "stock_name": c.get("stock_name", ""),
                    "score": c.get("leader_score") or 0,
                    "allocated_cash": shares * price,
                    "shares": shares,
                    "entry_price": price,
                })
        return orders

class LimitUpFilterRule(EntryRule):
    """Auto-reject candidates whose open price is at/near limit-up (>=9.8% gap).
    Always enforced — injected automatically by the engine.
    """
    rule_type = "limit_up_filter"

    def __init__(self):
        self.params = {}

    def should_enter(self, candidate: dict, context: dict) -> bool:
        prev_close = context.get("prev_close", {}).get(candidate.get("stock_code", ""))
        open_price = context.get("prices", {}).get(candidate.get("stock_code", ""))
        if prev_close and open_price and prev_close > 0:
            gap = (open_price - prev_close) / prev_close
            if gap >= 0.098:
                return False
        return True


class GapUpFilterRule(EntryRule):
    """Skip candidates that gap up more than max_gap_pct vs previous close.
    Optional — only active when strategy.gap_up_pct is set.
    """
    rule_type = "gap_up_filter"

    def __init__(self, max_gap_pct: float = 0.04):
        self.params = {"max_gap_pct": max_gap_pct}

    def should_enter(self, candidate: dict, context: dict) -> bool:
        prev_close = context.get("prev_close", {}).get(candidate.get("stock_code", ""))
        open_price = context.get("prices", {}).get(candidate.get("stock_code", ""))
        if prev_close and open_price and prev_close > 0:
            gap = (open_price - prev_close) / prev_close
            if gap > self.params["max_gap_pct"]:
                return False
        return True


class STFilterRule(EntryRule):
    """Reject ST / *ST stocks (5% limit, higher delisting risk)."""
    rule_type = "st_filter"

    def __init__(self):
        self.params = {}

    def should_enter(self, candidate: dict, context: dict) -> bool:
        name = candidate.get("stock_name", "")
        return "ST" not in name and "*ST" not in name


class OneDaySpikeFilter(EntryRule):
    """过滤一日游：昨日近涨停(>9.5%) + 今日高开(>3%) → 疑似出货。

    context 需提供:
      - context["prev_day_change"] — dict[symbol → float] 昨日涨跌幅
      - context["prices"] — dict[symbol → float] 今日开盘价
      - context["prev_close"] — dict[symbol → float] 昨日收盘价
      - context["sector_volume_pct"] — dict[symbol → float] 板块成交占比(可选)
    """
    rule_type = "one_day_spike"

    def __init__(self):
        self.params = {}

    def should_enter(self, candidate: dict, context: dict) -> bool:
        code = candidate.get("stock_code", "")
        prev_day_change = context.get("prev_day_change", {}).get(code)
        if prev_day_change is None:
            return True
        if prev_day_change <= 0.095:
            return True
        prev_close = context.get("prev_close", {}).get(code, 0)
        open_price = context.get("prices", {}).get(code, 0)
        if prev_close > 0 and open_price > 0:
            gap = (open_price - prev_close) / prev_close
            if gap <= 0.03:
                return True
        else:
            return True
        sector_pct = context.get("sector_volume_pct", {}).get(code)
        if sector_pct is not None and sector_pct < 0.20:
            return True
        return False


class VolumeSurgeFilter(EntryRule):
    """过滤异常放量：当日量 > 20日均量 3倍 且 换手率 > 15%.

    context 需提供:
      - context["avg_volume_20"] — dict[symbol → float] 20日均量
      - context["turnover_pct"] — dict[symbol → float] 换手率
      - context["today_volume"] — dict[symbol → float] 当日量
    """
    rule_type = "volume_surge"

    def __init__(self, max_vol_ratio: float = 3.0, max_turnover: float = 0.15):
        self.params = {"max_vol_ratio": max_vol_ratio, "max_turnover": max_turnover}

    def should_enter(self, candidate: dict, context: dict) -> bool:
        code = candidate.get("stock_code", "")
        avg_vol20 = context.get("avg_volume_20", {}).get(code)
        turnover = context.get("turnover_pct", {}).get(code)
        if avg_vol20 is None or turnover is None:
            return True
        today_vol = context.get("today_volume", {}).get(code, 0)
        if today_vol > 0 and avg_vol20 > 0:
            ratio = today_vol / avg_vol20
            if ratio > self.params["max_vol_ratio"] and turnover > self.params["max_turnover"]:
                return False
        return True


class MaxScoreFilterRule(EntryRule):
    """Reject candidates with leader_score above max_score.
    For both pipeline (leader_score > 0.65 = overheated) and scanner
    (dragon_score > 0.65 = overbought momentum about to reverse)."""
    rule_type = "max_score_filter"

    def __init__(self, max_score: float = 0.7):
        self.params = {"max_score": max_score}

    def should_enter(self, candidate: dict, context: dict) -> bool:
        score = candidate.get("dragon_score") or candidate.get("leader_score") or 0
        return score <= self.params["max_score"]


class PreEntryMomentumFilter(EntryRule):
    """Pre-entry momentum quality filter — computes composite momentum score
    from OHLCV bars BEFORE entry date and rejects candidates below threshold.

    Composite score components (all from pre-entry data only):
      - trend_r2_10d (0.35): R² of linear fit over past 10 days (positive=uptrend)
      - momentum_quality_5d (0.25): Sharpe-like ratio of past 5 days returns
      - ret_5d (0.25): 5-day return clamped to [0, 0.3]
      - dist_from_5d_high (0.15): distance from 5-day high (closer=better)

    context 需提供:
      - context["bars"] — dict[symbol → list[bar_dict]] OHLCV bars up to trade_date
    """

    rule_type = "pre_entry_momentum"

    def __init__(self, min_composite: float = 0.25):
        self.params = {"min_composite": min_composite}

    @staticmethod
    def _compute_composite(bars: list[dict]) -> float:
        """Compute composite momentum score from pre-entry bars (excludes last bar)."""
        if len(bars) < 16:
            return -999.0  # insufficient data

        pre_bars = bars[:-1]  # exclude today's bar
        if len(pre_bars) < 15:
            return -999.0

        closes = [b.get("close", 0) for b in pre_bars]
        highs = [b.get("high", 0) for b in pre_bars]

        # trend_r2_10d
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
            trend_r2 = (xy_c ** 2) / (x_v * y_v)
            if slope < 0:
                trend_r2 = -trend_r2
        else:
            trend_r2 = 0.0
            slope = 0.0

        # momentum_quality_5d
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

        # ret_5d
        ret_5d = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 and closes[-6] > 0 else 0.0
        ret_5d = max(0.0, min(0.3, ret_5d))

        # dist_from_5d_high
        h5 = max(highs[-5:]) if len(highs) >= 5 else closes[-1]
        dist_5h = (closes[-1] - h5) / h5 if h5 > 0 else 0.0

        composite = (
            trend_r2 * 0.35
            + min(mom_q / 3.0, 0.5) * 0.25
            + (ret_5d / 0.3) * 0.25
            + min(dist_5h * 15.0 + 0.5, 1.0) * 0.15
        )
        return composite

    def should_enter(self, candidate: dict, context: dict) -> bool:
        code = candidate.get("stock_code", "")
        bars_by_symbol = context.get("bars", {})
        bars = bars_by_symbol.get(code, [])
        if not bars or len(bars) < 16:
            return True  # pass if insufficient data (don't filter blindly)

        composite = self._compute_composite(bars)
        if composite == -999.0:
            return True  # insufficient data → pass

        return composite >= self.params["min_composite"]


class FridayFilterRule(EntryRule):
    """周五不建新仓：A股周末效应，周五买入周一平均表现差。

    context 需提供:
      - context["date"] — str 交易日期 "YYYY-MM-DD"
    """
    rule_type = "friday_filter"

    def __init__(self):
        self.params = {}

    def should_enter(self, candidate: dict, context: dict) -> bool:
        trade_date = context.get("date", "")
        if not trade_date:
            return True
        from datetime import date as dt_date
        try:
            d = dt_date.fromisoformat(str(trade_date)[:10])
            return d.weekday() != 4  # 4 = Friday
        except (ValueError, TypeError):
            return True


# ---------------------------------------------------------------------------
# Rule registries — add new rule classes here to make them discoverable
# ---------------------------------------------------------------------------

ENTRY_RULES: dict[str, type[EntryRule]] = {
    "score_threshold": ScoreThresholdRule,
    "no_filter": NoFilterRule,
    "limit_up_filter": LimitUpFilterRule,
    "gap_up_filter": GapUpFilterRule,
    "st_filter": STFilterRule,
    "one_day_spike": OneDaySpikeFilter,
    "volume_surge": VolumeSurgeFilter,
    "friday_filter": FridayFilterRule,
    "max_score_filter": MaxScoreFilterRule,
    "pre_entry_momentum": PreEntryMomentumFilter,
}

# DEPRECATED exit rules removed from registry:
#   score_cliff → replaced by exit matrix (cont < min_cont_threshold)
#   trailing_stop → replaced by decay > cont for N days
#   score_decline → replaced by state machine DECAYING state
EXIT_RULES: dict[str, type[ExitRule]] = {
    "price_trailing_stop": PriceTrailingStopRule,
}

ALLOCATORS: dict[str, type[PositionAllocator]] = {
    "score_weighted": ScoreWeightedAllocator,
    "equal_weight": EqualWeightAllocator,
}
