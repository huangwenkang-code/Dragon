"""TradingStrategy — combines entry/exit rules + allocator into a named strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from services.backtest.rules import (
    EntryRule, ExitRule, PositionAllocator,
    NoFilterRule,
    EqualWeightAllocator, ScoreWeightedAllocator, FridayFilterRule, STFilterRule,
    PriceTrailingStopRule,
    OneDaySpikeFilter, VolumeSurgeFilter, GapUpFilterRule,
    MaxScoreFilterRule, ScoreThresholdRule,
    ENTRY_RULES, EXIT_RULES, ALLOCATORS,
)


@dataclass
class TradingStrategy:
    """A complete trading strategy: entry rules + exit rules + position allocator."""
    name: str
    description: str = ""
    entry_rules: list = field(default_factory=list)
    exit_rules: list = field(default_factory=list)
    allocator: PositionAllocator | None = None
    max_positions: int = 999
    max_position_pct: float = 1.0
    initial_capital: float = 100000.0
    daily_cash_pct: float = 0.5
    # NEW: fee & filter settings
    commission_rate: float = 0.00025     # 万2.5 commission
    stamp_duty_rate: float = 0.0005      # 0.05% stamp duty (sell only)
    min_commission: float = 5.0          # ¥5 minimum commission
    gap_up_pct: float | None = None      # None = no gap-up filter; value = max allowed gap
    enable_limit_up_filter: bool = True  # limit-up filter (default on)
    is_system: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "entry_rules": [{"type": r.rule_type, "params": r.params} for r in self.entry_rules],
            "exit_rules": [{"type": r.rule_type, "params": r.params} for r in self.exit_rules],
            "allocator": {"type": self.allocator.allocator_type, "params": self.allocator.params} if self.allocator else None,
            "max_positions": self.max_positions,
            "max_position_pct": self.max_position_pct,
            "initial_capital": self.initial_capital,
            "daily_cash_pct": self.daily_cash_pct,
            "commission_rate": self.commission_rate,
            "stamp_duty_rate": self.stamp_duty_rate,
            "min_commission": self.min_commission,
            "gap_up_pct": self.gap_up_pct,
            "enable_limit_up_filter": self.enable_limit_up_filter,
            "is_system": self.is_system,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TradingStrategy":
        entry = []
        for er in d.get("entry_rules", []):
            rule_cls = ENTRY_RULES[er["type"]]
            entry.append(rule_cls(**er.get("params", {})))
        exit_ = []
        for xr in d.get("exit_rules", []):
            rule_cls = EXIT_RULES[xr["type"]]
            exit_.append(rule_cls(**xr.get("params", {})))
        alloc = None
        if d.get("allocator"):
            a = d["allocator"]
            alloc = ALLOCATORS[a["type"]](**a.get("params", {}))
        return cls(
            name=d["name"], description=d.get("description", ""),
            entry_rules=entry, exit_rules=exit_, allocator=alloc,
            max_positions=d.get("max_positions", 999),
            max_position_pct=d.get("max_position_pct", 1.0),
            initial_capital=d.get("initial_capital", 100000),
            daily_cash_pct=d.get("daily_cash_pct", 0.5),
            commission_rate=d.get("commission_rate", 0.00025),
            stamp_duty_rate=d.get("stamp_duty_rate", 0.0005),
            min_commission=d.get("min_commission", 5.0),
            gap_up_pct=d.get("gap_up_pct"),
            enable_limit_up_filter=d.get("enable_limit_up_filter", True),
            is_system=d.get("is_system", False),
        )


# ---------------------------------------------------------------------------
# Pre-built system strategies
# ---------------------------------------------------------------------------

STRATEGY_A = TradingStrategy(
    name="策略A-每日固定资金",
    description="资金1000w，每日2.5%仓位，score加权分配，最多10只，单只≤20%，-5%硬止损+7%价格止损，大盘过滤",
    entry_rules=[
        FridayFilterRule(),
        STFilterRule(),
        ScoreThresholdRule(0.50),
        MaxScoreFilterRule(0.65),
        OneDaySpikeFilter(),
        GapUpFilterRule(0.04),
        NoFilterRule(),
    ],
    exit_rules=[
        PriceTrailingStopRule(0.07),
    ],
    allocator=ScoreWeightedAllocator(),
    max_positions=10,
    max_position_pct=0.20,
    initial_capital=1_000_000,
    daily_cash_pct=0.025,
    commission_rate=0.00025,
    stamp_duty_rate=0.0005,
    min_commission=5.0,
    gap_up_pct=0.04,
    enable_limit_up_filter=True,
    is_system=True,
)

STRATEGY_B = TradingStrategy(
    name="策略B-仓位上限管理",
    description="资金1000w，每日2.5%仓位，等权分配，最多8只，单只≤15%，-5%硬止损+7%价格止损，大盘过滤",
    entry_rules=[
        FridayFilterRule(),
        STFilterRule(),
        ScoreThresholdRule(0.50),
        MaxScoreFilterRule(0.65),
        OneDaySpikeFilter(),
        GapUpFilterRule(0.04),
        NoFilterRule(),
    ],
    exit_rules=[
        PriceTrailingStopRule(0.07),
    ],
    allocator=EqualWeightAllocator(),
    max_positions=8,
    max_position_pct=0.15,
    initial_capital=1_000_000,
    daily_cash_pct=0.025,
    commission_rate=0.00025,
    stamp_duty_rate=0.0005,
    min_commission=5.0,
    gap_up_pct=0.04,
    enable_limit_up_filter=True,
    is_system=True,
)

STRATEGY_V4 = TradingStrategy(
    name="策略V4-五因子",
    description="V4 5因子分位数打分，去掉分数门槛(已内置在排序中)，保留技术过滤",
    entry_rules=[
        FridayFilterRule(),
        STFilterRule(),
        OneDaySpikeFilter(),
        GapUpFilterRule(0.04),
        NoFilterRule(),
    ],
    exit_rules=[
        PriceTrailingStopRule(0.07),
    ],
    allocator=ScoreWeightedAllocator(),
    max_positions=10,
    max_position_pct=0.20,
    initial_capital=1_000_000,
    daily_cash_pct=0.025,
    commission_rate=0.00025,
    stamp_duty_rate=0.0005,
    min_commission=5.0,
    gap_up_pct=0.04,
    enable_limit_up_filter=True,
    is_system=True,
)

SYSTEM_STRATEGIES = {STRATEGY_A.name: STRATEGY_A, STRATEGY_B.name: STRATEGY_B, STRATEGY_V4.name: STRATEGY_V4}
