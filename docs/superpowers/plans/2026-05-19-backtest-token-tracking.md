# 回测引擎 + Token 追踪 + 前端三页面 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) to implement this plan task-by-task.

**Goal:** 构建可插拔交易规则回测引擎、Token消耗追踪、前端三页面、历史数据回放

**Architecture:** 四个独立模块并行开发 — backtest engine (rules/strategies/engine), token tracker (LangChain callback), 前端三页面 (Vue3+ECharts), historical data filler (skill APIs)。DB 新增 3 张表 + PipelineRun 加 token_usage 字段。API 路由挂在 graph_service FastAPI 上。

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy async, Vue 3 + Element Plus + ECharts, LangChain callbacks, akshare

---

### Task 1: `services/backtest/models.py` — Pydantic 数据模型

**Files:**
- Create: `services/backtest/__init__.py`
- Create: `services/backtest/models.py`

**Context:** 回测引擎的核心数据结构。Position 表示持仓，Trade 表示一次完整买卖，Order 表示买入指令，BacktestResult 是完整回测结果。这些模型被 rules/strategies/engine 共同使用。

- [ ] **Step 1: Write models.py**

```python
"""Backtest data models — Position, Trade, Order, BacktestResult."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class Position:
    """A single holding during backtest."""
    stock_code: str
    stock_name: str
    entry_date: date
    entry_price: float
    entry_score: float           # leader_score at entry
    shares: int = 0
    cost: float = 0.0            # total cost (entry_price * shares)
    peak_score: float = 0.0      # highest leader_score since entry
    peak_price: float = 0.0      # highest price since entry
    days_held: int = 0
    score_history: list[float] = field(default_factory=list)  # daily scores since entry
    current_price: float = 0.0
    current_score: float = 0.0


@dataclass
class Trade:
    """A completed round-trip trade."""
    stock_code: str
    stock_name: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    shares: int
    cost: float
    proceeds: float
    pnl: float                   # absolute profit/loss
    pnl_pct: float               # percentage return
    exit_reason: str             # which exit rule triggered
    holding_days: int


@dataclass
class Order:
    """A buy instruction produced by PositionAllocator."""
    stock_code: str
    stock_name: str
    score: float
    allocated_cash: float
    shares: int                  # rounded down to 100-share lots
    entry_price: float


@dataclass
class DailySnapshot:
    """Portfolio state at end of a trading day."""
    date: date
    equity: float                # total portfolio value
    cash: float                  # unused cash
    positions: list[Position]
    daily_return: float = 0.0    # day-over-day return


@dataclass
class BacktestResult:
    """Full backtest output."""
    strategy_name: str
    start_date: date
    end_date: date
    initial_capital: float
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0        # % of profitable trades
    total_trades: int = 0
    trades: list[Trade] = field(default_factory=list)
    daily_snapshots: list[DailySnapshot] = field(default_factory=list)
    benchmark_return_pct: float = 0.0  # CSI 300 over same period
```

- [ ] **Step 2: Write `__init__.py`**

```python
"""Backtest engine — pluggable trading rules, position management, simulation."""
```

- [ ] **Step 3: Verify**

```bash
cd D:/K/dragon-engine && D:/K/dragon-engine/venv/Scripts/python.exe -c "
from services.backtest.models import Position, Trade, Order, BacktestResult
print('models OK')
"
```

---

### Task 2: `services/backtest/rules.py` — 入场/卖出/仓位规则

**Files:**
- Create: `services/backtest/rules.py`

**Context:** 规则基类和具体实现。EntryRule/ExitRule/PositionAllocator 是抽象基类，子类通过 rule_type 标识注册。必须遵循开闭原则 — 加新规则不改引擎代码。

**Key interfaces (must match models.py types):**

```python
from abc import ABC, abstractmethod

class EntryRule(ABC):
    rule_type: str = ""
    params: dict = {}
    @abstractmethod
    def should_enter(self, candidate: dict, context: dict) -> bool: ...

class ExitRule(ABC):
    rule_type: str = ""
    params: dict = {} 
    @abstractmethod
    def should_exit(self, position, current_candidate: dict | None, context: dict) -> str | None: ...
    # Returns exit reason string or None

class PositionAllocator(ABC):
    allocator_type: str = ""
    params: dict = {}
    @abstractmethod
    def allocate(self, candidates: list[dict], available_cash: float, current_prices: dict, context: dict) -> list[dict]: ...
    # Returns list of {stock_code, stock_name, score, allocated_cash, shares, entry_price}
```

- [ ] **Step 1: Implement concrete rules**

```python
class ScoreThresholdRule(EntryRule):
    rule_type = "score_threshold"
    def __init__(self, min_score: float = 0.5):
        self.params = {"min_score": min_score}
    def should_enter(self, candidate, context):
        return candidate.get("leader_score", 0) >= self.params["min_score"]

class NoFilterRule(EntryRule):
    rule_type = "no_filter"
    def should_enter(self, candidate, context):
        return True

class ScoreCliffRule(ExitRule):
    rule_type = "score_cliff"
    def __init__(self, threshold: float = 0.3):
        self.params = {"threshold": threshold}
    def should_exit(self, position, current_candidate, context):
        score = current_candidate.get("leader_score", 0) if current_candidate else position.current_score
        if score < self.params["threshold"]:
            return f"得分悬崖({score:.2f}<{self.params['threshold']})"
        return None

class TrailingStopRule(ExitRule):
    rule_type = "trailing_stop"
    def __init__(self, drawdown_pct: float = 0.15):
        self.params = {"drawdown_pct": drawdown_pct}
    def should_exit(self, position, current_candidate, context):
        if position.peak_score > 0:
            dd = (position.peak_score - position.current_score) / position.peak_score
            if dd > self.params["drawdown_pct"]:
                return f"回撤止损({dd:.1%}>{self.params['drawdown_pct']:.0%})"
        return None

class ScoreDeclineRule(ExitRule):
    rule_type = "score_decline"
    def __init__(self, consecutive_days: int = 3):
        self.params = {"consecutive_days": consecutive_days}
    def should_exit(self, position, current_candidate, context):
        history = position.score_history
        n = self.params["consecutive_days"]
        if len(history) >= n:
            recent = history[-n:]
            if all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
                return f"连续{n}天得分下滑"
        return None

class ScoreWeightedAllocator(PositionAllocator):
    allocator_type = "score_weighted"
    def allocate(self, candidates, available_cash, current_prices, context):
        total_score = sum(c.get("leader_score", 0) for c in candidates)
        if total_score == 0:
            return []
        orders = []
        for c in candidates:
            weight = c.get("leader_score", 0) / total_score
            cash = available_cash * weight
            price = current_prices.get(c["stock_code"], 0)
            if price <= 0:
                continue
            shares = int(cash / price / 100) * 100  # round to 100-share lots
            if shares >= 100:
                orders.append({
                    "stock_code": c["stock_code"],
                    "stock_name": c.get("stock_name", ""),
                    "score": c.get("leader_score", 0),
                    "allocated_cash": shares * price,
                    "shares": shares,
                    "entry_price": price,
                })
        return orders

class EqualWeightAllocator(PositionAllocator):
    allocator_type = "equal_weight"
    def allocate(self, candidates, available_cash, current_prices, context):
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
                    "score": c.get("leader_score", 0),
                    "allocated_cash": shares * price,
                    "shares": shares,
                    "entry_price": price,
                })
        return orders
```

- [ ] **Step 2: Add rule registry**

```python
# At bottom of rules.py
ENTRY_RULES = {cls.rule_type: cls for cls in [ScoreThresholdRule, NoFilterRule]}
EXIT_RULES = {cls.rule_type: cls for cls in [ScoreCliffRule, TrailingStopRule, ScoreDeclineRule]}
ALLOCATORS = {cls.allocator_type: cls for cls in [ScoreWeightedAllocator, EqualWeightAllocator]}
```

- [ ] **Step 3: Verify imports**

```bash
cd D:/K/dragon-engine && D:/K/dragon-engine/venv/Scripts/python.exe -c "
from services.backtest.rules import (
    ScoreThresholdRule, NoFilterRule, ScoreCliffRule,
    TrailingStopRule, ScoreDeclineRule,
    ScoreWeightedAllocator, EqualWeightAllocator,
    ENTRY_RULES, EXIT_RULES, ALLOCATORS,
)
print('rules OK, entry:', list(ENTRY_RULES.keys()), 'exit:', list(EXIT_RULES.keys()))
"
```

---

### Task 3: `services/backtest/strategies.py` + `registry.py` — 策略定义与注册

**Files:**
- Create: `services/backtest/strategies.py`
- Create: `services/backtest/registry.py`

**Context:** TradingStrategy 组合 rules，registry 负责策略的 CRUD 和 DB 持久化。策略序列化为 JSON 存入 DB。预置 Strategy A 和 Strategy B。

**Dependencies:** Task 1 (models), Task 2 (rules)

- [ ] **Step 1: Write strategies.py**

```python
"""TradingStrategy — combines entry/exit rules + allocator into a named strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from services.backtest.rules import (
    EntryRule, ExitRule, PositionAllocator,
    ScoreThresholdRule, NoFilterRule,
    ScoreCliffRule, TrailingStopRule, ScoreDeclineRule,
    ScoreWeightedAllocator,
    ENTRY_RULES, EXIT_RULES, ALLOCATORS,
)


@dataclass
class TradingStrategy:
    name: str
    description: str = ""
    entry_rules: list = field(default_factory=list)
    exit_rules: list = field(default_factory=list)
    allocator: PositionAllocator | None = None
    max_positions: int = 999
    max_position_pct: float = 1.0
    initial_capital: float = 100000.0
    daily_cash_pct: float = 0.5     # Strategy A uses this
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
            "is_system": self.is_system,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TradingStrategy:
        entry = []
        for er in d.get("entry_rules", []):
            rule_cls = ENTRY_RULES[er["type"]]
            entry.append(rule_cls(**er["params"]))
        exit_ = []
        for xr in d.get("exit_rules", []):
            rule_cls = EXIT_RULES[xr["type"]]
            exit_.append(rule_cls(**xr["params"]))
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
            is_system=d.get("is_system", False),
        )


# Pre-built system strategies
STRATEGY_A = TradingStrategy(
    name="策略A-每日固定资金",
    description="每天用总资金50%按score加权买入当天top-5，持仓无上限",
    entry_rules=[NoFilterRule()],
    exit_rules=[ScoreCliffRule(0.3), TrailingStopRule(0.15), ScoreDeclineRule(3)],
    allocator=ScoreWeightedAllocator(),
    max_positions=999,
    max_position_pct=1.0,
    daily_cash_pct=0.5,
    is_system=True,
)

STRATEGY_B = TradingStrategy(
    name="策略B-仓位上限管理",
    description="最多8只持仓，单只≤15%，有空位才买新的",
    entry_rules=[ScoreThresholdRule(0.5)],
    exit_rules=[ScoreCliffRule(0.3), TrailingStopRule(0.15), ScoreDeclineRule(3)],
    allocator=ScoreWeightedAllocator(),
    max_positions=8,
    max_position_pct=0.15,
    daily_cash_pct=1.0,
    is_system=True,
)

SYSTEM_STRATEGIES = {"A": STRATEGY_A, "B": STRATEGY_B}
```

- [ ] **Step 2: Write registry.py**

```python
"""Strategy registry — CRUD + DB persistence for trading strategies."""

from __future__ import annotations

from services.backtest.strategies import TradingStrategy, SYSTEM_STRATEGIES, STRATEGY_A, STRATEGY_B
from shared.utils.logging import get_logger

logger = get_logger(__name__)


class StrategyRegistry:
    """In-memory + DB-backed strategy store."""

    def __init__(self):
        self._strategies: dict[str, TradingStrategy] = dict(SYSTEM_STRATEGIES)

    def list_all(self) -> list[TradingStrategy]:
        return list(self._strategies.values())

    def get(self, name: str) -> TradingStrategy | None:
        return self._strategies.get(name)

    def add(self, s: TradingStrategy):
        if s.name in self._strategies and self._strategies[s.name].is_system:
            raise ValueError(f"Cannot overwrite system strategy: {s.name}")
        self._strategies[s.name] = s

    def remove(self, name: str):
        s = self._strategies.get(name)
        if s and s.is_system:
            raise ValueError(f"Cannot delete system strategy: {name}")
        self._strategies.pop(name, None)

    async def save_to_db(self, session):
        """Persist all non-system strategies to backtest_strategies table."""
        from db.models import BacktestStrategy
        for s in self._strategies.values():
            if s.is_system:
                continue
            row = BacktestStrategy(
                name=s.name, description=s.description,
                config_json=s.to_dict(),
            )
            session.add(row)
        await session.flush()

    async def load_from_db(self, session):
        """Load custom strategies from DB, merge with system defaults."""
        from sqlalchemy import select
        from db.models import BacktestStrategy
        result = await session.execute(select(BacktestStrategy))
        for row in result.scalars().all():
            s = TradingStrategy.from_dict(row.config_json)
            s.is_system = False
            self._strategies[s.name] = s


_registry: StrategyRegistry | None = None


def get_registry() -> StrategyRegistry:
    global _registry
    if _registry is None:
        _registry = StrategyRegistry()
    return _registry
```

- [ ] **Step 3: Verify**

```bash
cd D:/K/dragon-engine && D:/K/dragon-engine/venv/Scripts/python.exe -c "
from services.backtest.strategies import STRATEGY_A, STRATEGY_B, SYSTEM_STRATEGIES
from services.backtest.registry import get_registry
r = get_registry()
print('Strategy A:', r.get('A').name)
print('Strategy B:', r.get('B').name)
print('All:', [s.name for s in r.list_all()])
print('strategies OK')
"
```

---

### Task 4: `services/backtest/engine.py` — 回测引擎

**Files:**
- Create: `services/backtest/engine.py`

**Context:** 逐日迭代 PipelineRun，模拟持仓变化。每天：检查卖出 → 筛选入场 → 分配仓位 → 记录快照。

**Dependencies:** Task 1, 2, 3

- [ ] **Step 1: Write engine.py**

```python
"""BacktestEngine — iterate daily PipelineRuns, simulate portfolio."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from services.backtest.models import Position, Trade, Order, DailySnapshot, BacktestResult
from services.backtest.strategies import TradingStrategy
from shared.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestContext:
    strategy: TradingStrategy
    current_date: date
    positions: dict[str, Position]   # stock_code -> Position
    trades: list[Trade]
    snapshots: list[DailySnapshot]
    cash: float
    benchmark_value: float = 1.0     # CSI 300 normalized


class BacktestEngine:
    """Iterate daily PipelineRun data, simulate trading."""

    def __init__(self, strategy: TradingStrategy):
        self.strategy = strategy
        self.context = BacktestContext(
            strategy=strategy,
            current_date=date.today(),
            positions={},
            trades=[],
            snapshots=[],
            cash=strategy.initial_capital,
        )

    def run(self, daily_runs: list[dict]) -> BacktestResult:
        """
        daily_runs: [{trade_date, leader_candidates: [{stock_code, stock_name, leader_score, ...}]}, ...]
        Sorted by trade_date ascending.
        """
        if not daily_runs:
            return self._empty_result()

        runs_sorted = sorted(daily_runs, key=lambda r: r["trade_date"])
        start_date = runs_sorted[0]["trade_date"]
        end_date = runs_sorted[-1]["trade_date"]

        for run in runs_sorted:
            self._process_day(run)

        return self._build_result(start_date, end_date)

    def _process_day(self, run: dict):
        trade_date = run["trade_date"]
        candidates = run.get("leader_candidates", [])

        # Build candidate lookup by stock_code
        cand_by_code = {c.get("stock_code"): c for c in candidates}

        # Build price map (using change_pct to estimate price; in real data
        # this would come from capital_flow_records)
        prices = {}
        for c in candidates:
            # If we have price data, use it; otherwise default to 10
            prices[c.get("stock_code")] = c.get("price", 10.0)

        # Step 1: Update position current_score from today's candidates
        for code, pos in self.context.positions.items():
            pos.days_held += 1
            if code in cand_by_code:
                pos.current_score = cand_by_code[code].get("leader_score", 0)
                pos.score_history.append(pos.current_score)
                pos.current_price = prices.get(code, pos.entry_price)
                if pos.current_score > pos.peak_score:
                    pos.peak_score = pos.current_score
                if pos.current_price > pos.peak_price:
                    pos.peak_price = pos.current_price
            else:
                # Stock not in today's candidates — score drops to 0
                pos.current_score = 0.0
                pos.score_history.append(0.0)

        # Step 2: Check exit rules
        for code, pos in list(self.context.positions.items()):
            current_cand = cand_by_code.get(code)
            for rule in self.strategy.exit_rules:
                reason = rule.should_exit(pos, current_cand, {"date": trade_date})
                if reason:
                    self._close_position(code, pos, reason, trade_date)
                    break

        # Step 3: Check entry rules
        eligible = []
        for c in candidates:
            code = c.get("stock_code", "")
            if code in self.context.positions:
                continue  # already holding
            passed = all(r.should_enter(c, {"date": trade_date}) for r in self.strategy.entry_rules)
            if passed:
                eligible.append(c)

        # Step 4: Position limit enforcement
        current_count = len(self.context.positions)
        slots = max(0, self.strategy.max_positions - current_count)
        if slots == 0:
            eligible = []
        elif slots < len(eligible):
            # Keep top-N by score
            eligible.sort(key=lambda c: c.get("leader_score", 0), reverse=True)
            eligible = eligible[:slots]

        # Step 5: Single-position limit enforcement
        max_per_position = self.strategy.max_position_pct * self.strategy.initial_capital
        # (applied in allocator)

        # Step 6: Allocate
        if eligible:
            # Strategy A: fixed daily cash
            daily_cash = self.context.cash * self.strategy.daily_cash_pct
            alloc_cash = min(daily_cash, self.context.cash)
            orders = self.strategy.allocator.allocate(eligible, alloc_cash, prices, {"date": trade_date})
            for order in orders:
                self._open_position(order, trade_date)

        # Step 7: Record daily snapshot
        equity = self.context.cash + sum(
            p.current_price * p.shares for p in self.context.positions.values()
        )
        prev_equity = self.context.snapshots[-1].equity if self.context.snapshots else self.strategy.initial_capital
        daily_return = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0

        self.context.snapshots.append(DailySnapshot(
            date=trade_date,
            equity=equity,
            cash=self.context.cash,
            positions=list(self.context.positions.values()),
            daily_return=daily_return,
        ))
        self.context.current_date = trade_date

    def _open_position(self, order: dict, trade_date: date):
        pos = Position(
            stock_code=order["stock_code"],
            stock_name=order.get("stock_name", ""),
            entry_date=trade_date,
            entry_price=order["entry_price"],
            entry_score=order.get("score", 0),
            shares=order["shares"],
            cost=order["allocated_cash"],
            peak_score=order.get("score", 0),
            peak_price=order["entry_price"],
            current_price=order["entry_price"],
            current_score=order.get("score", 0),
        )
        self.context.positions[order["stock_code"]] = pos
        self.context.cash -= order["allocated_cash"]

    def _close_position(self, code: str, pos: Position, reason: str, trade_date: date):
        proceeds = pos.current_price * pos.shares
        pnl = proceeds - pos.cost
        pnl_pct = pnl / pos.cost if pos.cost > 0 else 0
        trade = Trade(
            stock_code=code,
            stock_name=pos.stock_name,
            entry_date=pos.entry_date,
            exit_date=trade_date,
            entry_price=pos.entry_price,
            exit_price=pos.current_price,
            shares=pos.shares,
            cost=pos.cost,
            proceeds=proceeds,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            holding_days=pos.days_held,
        )
        self.context.trades.append(trade)
        self.context.cash += proceeds
        del self.context.positions[code]

    def _build_result(self, start_date, end_date) -> BacktestResult:
        final_equity = self.context.snapshots[-1].equity if self.context.snapshots else self.strategy.initial_capital
        total_return = (final_equity - self.strategy.initial_capital) / self.strategy.initial_capital
        max_dd = self._calc_max_drawdown()
        sharpe = self._calc_sharpe()
        wins = sum(1 for t in self.context.trades if t.pnl > 0)
        wr = wins / len(self.context.trades) if self.context.trades else 0

        return BacktestResult(
            strategy_name=self.strategy.name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.strategy.initial_capital,
            final_equity=round(final_equity, 2),
            total_return_pct=round(total_return * 100, 2),
            max_drawdown_pct=round(max_dd * 100, 2),
            sharpe_ratio=round(sharpe, 3),
            win_rate=round(wr, 3),
            total_trades=len(self.context.trades),
            trades=self.context.trades,
            daily_snapshots=self.context.snapshots,
        )

    def _calc_max_drawdown(self) -> float:
        peak = self.strategy.initial_capital
        max_dd = 0.0
        for snap in self.context.snapshots:
            if snap.equity > peak:
                peak = snap.equity
            dd = (peak - snap.equity) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _calc_sharpe(self) -> float:
        returns = [s.daily_return for s in self.context.snapshots]
        if len(returns) < 2:
            return 0.0
        mean_ret = sum(returns) / len(returns)
        var = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1) if len(returns) > 1 else 1e-9
        std = var ** 0.5
        if std == 0:
            return 0.0
        return (mean_ret / std) * (252 ** 0.5)  # annualized
```

- [ ] **Step 2: Verify engine runs with mock data**

```bash
cd D:/K/dragon-engine && D:/K/dragon-engine/venv/Scripts/python.exe -c "
from services.backtest.engine import BacktestEngine
from services.backtest.strategies import STRATEGY_A
engine = BacktestEngine(STRATEGY_A)
mock_runs = [
    {'trade_date': '2026-05-01', 'leader_candidates': [
        {'stock_code': '000001', 'stock_name': '测试1', 'leader_score': 0.7, 'price': 10.0},
        {'stock_code': '000002', 'stock_name': '测试2', 'leader_score': 0.6, 'price': 20.0},
    ]},
    {'trade_date': '2026-05-02', 'leader_candidates': [
        {'stock_code': '000001', 'stock_name': '测试1', 'leader_score': 0.3, 'price': 9.0},
        {'stock_code': '000002', 'stock_name': '测试2', 'leader_score': 0.65, 'price': 21.0},
    ]},
]
result = engine.run(mock_runs)
print(f'Return: {result.total_return_pct}% Trades: {result.total_trades} MaxDD: {result.max_drawdown_pct}%')
print('engine OK')
"
```

---

### Task 5: `services/token_tracker/` — Token 消耗追踪

**Files:**
- Create: `services/token_tracker/__init__.py`
- Create: `services/token_tracker/pricing.py`
- Create: `services/token_tracker/models.py`
- Create: `services/token_tracker/tracker.py`

**Context:** 用 LangChain BaseCallbackHandler 自动捕获 LLM token 用量，写 DB pipeline_runs.token_usage 字段。不侵入现有 event_extractor/analyzer 代码。

- [ ] **Step 1: Write pricing.py**

```python
"""Model pricing table — ¥ per 1K tokens."""

PRICING = {
    "qwen-turbo": {"input": 0.0003, "output": 0.0006},
    "qwen-plus": {"input": 0.0008, "output": 0.002},
    "qwen-max": {"input": 0.02, "output": 0.06},
    "default": {"input": 0.0003, "output": 0.0006},
}

def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = PRICING.get(model, PRICING["default"])
    cost = (prompt_tokens * p["input"] + completion_tokens * p["output"]) / 1000
    return round(cost, 6)
```

- [ ] **Step 2: Write models.py**

```python
"""Token tracking data models."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TokenRecord:
    run_id: str = ""
    step: str = ""               # e.g. "event_extraction", "sentiment_enrichment"
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "step": self.step, "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "timestamp": self.timestamp,
        }
```

- [ ] **Step 3: Write tracker.py**

```python
"""TokenUsageTracker — singleton + LangChain callback handler."""

from __future__ import annotations

from datetime import datetime
from langchain.callbacks import BaseCallbackHandler
from services.token_tracker.models import TokenRecord
from services.token_tracker.pricing import estimate_cost
from shared.utils.logging import get_logger

logger = get_logger(__name__)


class TokenTrackingCallback(BaseCallbackHandler):
    """LangChain callback that captures token usage from every LLM call."""

    def __init__(self, tracker: TokenUsageTracker):
        self._tracker = tracker

    def on_llm_end(self, response, **kwargs):
        try:
            llm_output = response.llm_output or {}
            usage = llm_output.get("token_usage", {})
            if not usage:
                return
            model = llm_output.get("model_name", "")
            record = TokenRecord(
                run_id=self._tracker.current_run_id,
                step=kwargs.get("name", "unknown"),
                model=model,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                cost=estimate_cost(model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)),
                timestamp=datetime.now().isoformat(),
            )
            self._tracker.record(record)
        except Exception as e:
            logger.debug("Token tracking callback error: %s", e)


class TokenUsageTracker:
    """Singleton tracking token usage across a pipeline run."""

    _instance: TokenUsageTracker | None = None

    def __init__(self):
        self.current_run_id: str = ""
        self.records: list[TokenRecord] = []

    @classmethod
    def instance(cls) -> TokenUsageTracker:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start_run(self, run_id: str):
        self.current_run_id = run_id
        self.records = []

    def record(self, record: TokenRecord):
        if record.run_id == self.current_run_id or not record.run_id:
            record.run_id = self.current_run_id
        self.records.append(record)

    def summary(self) -> dict:
        total_input = sum(r.prompt_tokens for r in self.records)
        total_output = sum(r.completion_tokens for r in self.records)
        total_cost = sum(r.cost for r in self.records)
        return {
            "run_id": self.current_run_id,
            "total_prompt_tokens": total_input,
            "total_completion_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_cost": round(total_cost, 6),
            "records": [r.to_dict() for r in self.records],
        }

    def create_callback(self) -> TokenTrackingCallback:
        return TokenTrackingCallback(self)
```

- [ ] **Step 4: Write `__init__.py`**

```python
"""Token consumption tracker for LLM calls."""
from services.token_tracker.tracker import TokenUsageTracker, TokenTrackingCallback
```

- [ ] **Step 5: Inject callback into llm_provider.py**

Modify `services/llm_adapter/llm_provider.py`: in `create_quick_llm()` (or equivalent), after creating the LLM instance, attach the token callback:

```python
# Add to llm_provider.py, in the function that creates the LLM:
from services.token_tracker import TokenUsageTracker
tracker = TokenUsageTracker.instance()
callback = tracker.create_callback()
llm.callbacks = llm.callbacks or []
llm.callbacks.append(callback)
```

- [ ] **Step 6: Wire run_id in graph_service/main.py**

In the `/run` endpoint, after generating run_id, call:
```python
from services.token_tracker import TokenUsageTracker
TokenUsageTracker.instance().start_run(run_id)
```

- [ ] **Step 7: Verify tracker imports**

```bash
cd D:/K/dragon-engine && D:/K/dragon-engine/venv/Scripts/python.exe -c "
from services.token_tracker import TokenUsageTracker
from services.token_tracker.pricing import estimate_cost
print('Cost qwen-turbo 1K in + 1K out:', estimate_cost('qwen-turbo', 1000, 1000))
print('token_tracker OK')
"
```

---

### Task 6: DB Schema Changes

**Files:**
- Modify: `db/models.py` — add BacktestStrategy, BacktestResult, BacktestTrade ORM models; add token_usage to PipelineRun
- Modify: `db/persist.py` — save token_usage on persist
- Create: `db/migrate_backtest.py` — ALTER TABLE + CREATE TABLE migration

**Context:** 3 new tables + 1 new column on PipelineRun. Must work with existing async SQLAlchemy setup.

- [ ] **Step 1: Add ORM models to db/models.py**

```python
# Add after existing model classes:

class BacktestStrategy(Base):
    __tablename__ = "backtest_strategies"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text, default="")
    config_json = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(100), nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    initial_capital = Column(Float, default=100000)
    final_equity = Column(Float, default=0)
    total_return_pct = Column(Float, default=0)
    max_drawdown_pct = Column(Float, default=0)
    sharpe_ratio = Column(Float, default=0)
    win_rate = Column(Float, default=0)
    total_trades = Column(Integer, default=0)
    daily_snapshots = Column(JSONB, default=[])
    created_at = Column(DateTime, default=datetime.utcnow)


class BacktestTrade(Base):
    __tablename__ = "backtest_trades"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("backtest_runs.id", ondelete="CASCADE"), nullable=False)
    stock_code = Column(String(10), nullable=False)
    stock_name = Column(String(50), default="")
    entry_date = Column(Date, nullable=False)
    exit_date = Column(Date, nullable=False)
    entry_price = Column(Float, default=0)
    exit_price = Column(Float, default=0)
    shares = Column(Integer, default=0)
    cost = Column(Float, default=0)
    proceeds = Column(Float, default=0)
    pnl = Column(Float, default=0)
    pnl_pct = Column(Float, default=0)
    exit_reason = Column(String(200), default="")
    holding_days = Column(Integer, default=0)
```

Add to PipelineRun model:
```python
token_usage = Column(JSONB, default={})
```

- [ ] **Step 2: Write migration script `db/migrate_backtest.py`**

```python
"""Migration: add backtest tables + token_usage column."""

from db.connection import engine
from db.models import Base, PipelineRun
from sqlalchemy import text

async def migrate():
    async with engine.begin() as conn:
        # Create new tables
        await conn.run_sync(Base.metadata.create_all, tables=[
            Base.metadata.tables["backtest_strategies"],
            Base.metadata.tables["backtest_runs"],
            Base.metadata.tables["backtest_trades"],
        ])
        # Add token_usage column if not exists
        try:
            await conn.execute(text(
                "ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS token_usage JSONB DEFAULT '{}'"
            ))
        except Exception:
            pass  # column already exists
    print("Migration complete.")
```

- [ ] **Step 3: Update persist.py — save token_usage**

In `persist_run()`, before commit, add:
```python
from services.token_tracker import TokenUsageTracker
tracker = TokenUsageTracker.instance()
summary = tracker.summary()
run.token_usage = {"cost": summary["total_cost"], "tokens": summary["total_tokens"],
                    "prompt_tokens": summary["total_prompt_tokens"],
                    "completion_tokens": summary["total_completion_tokens"],
                    "records": summary["records"]}
```

- [ ] **Step 4: Run migration + verify**

```bash
cd D:/K/dragon-engine && D:/K/dragon-engine/venv/Scripts/python.exe -c "
import asyncio
from db.migrate_backtest import migrate
asyncio.run(migrate())
"
```

---

### Task 7: API Routes — backtest + token-usage

**Files:**
- Modify: `services/graph_service/main.py` — add 4 new endpoints

**Context:** Add REST endpoints for: list/get/create strategies, run backtest, get token usage. All under existing FastAPI app.

- [ ] **Step 1: Add imports + Pydantic schemas to main.py**

```python
# New imports
from services.backtest.registry import get_registry
from services.backtest.engine import BacktestEngine
from services.backtest.strategies import TradingStrategy

# New request/response schemas
class StrategyConfig(BaseModel):
    name: str
    description: str = ""
    entry_rules: list = []
    exit_rules: list = []
    allocator: dict = {}
    max_positions: int = 999
    max_position_pct: float = 1.0
    initial_capital: float = 100000.0
    daily_cash_pct: float = 0.5

class BacktestRequest(BaseModel):
    strategy_name: str = "A"
    start_date: str = ""
    end_date: str = ""
```

- [ ] **Step 2: Add routes to main.py**

```python
@app.get("/backtest/strategies")
async def list_strategies():
    """List all available trading strategies."""
    registry = get_registry()
    return [s.to_dict() for s in registry.list_all()]

@app.post("/backtest/strategies")
async def create_strategy(config: StrategyConfig):
    """Create a new custom trading strategy."""
    registry = get_registry()
    s = TradingStrategy.from_dict(config.model_dump())
    registry.add(s)
    return {"status": "ok", "name": s.name}

@app.delete("/backtest/strategies/{name}")
async def delete_strategy(name: str):
    """Delete a custom strategy (system strategies cannot be deleted)."""
    registry = get_registry()
    try:
        registry.remove(name)
        return {"status": "ok"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/backtest/run")
async def run_backtest(req: BacktestRequest):
    """Run a backtest using a specified strategy over a date range."""
    from db.connection import async_session_factory
    from sqlalchemy import select
    from db.models import PipelineRun, LeaderCandidate

    registry = get_registry()
    strategy = registry.get(req.strategy_name)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Strategy '{req.strategy_name}' not found")

    # Load pipeline runs for the date range
    async with async_session_factory() as session:
        result = await session.execute(
            select(PipelineRun).where(
                PipelineRun.trade_date >= req.start_date,
                PipelineRun.trade_date <= req.end_date,
            ).order_by(PipelineRun.trade_date.asc())
        )
        runs = result.scalars().all()

        daily_runs = []
        for run in runs:
            cands_result = await session.execute(
                select(LeaderCandidate).where(
                    LeaderCandidate.run_id == run.run_id
                ).order_by(LeaderCandidate.rank)
            )
            cands = cands_result.scalars().all()
            daily_runs.append({
                "trade_date": run.trade_date,
                "leader_candidates": [
                    {
                        "stock_code": c.stock_code,
                        "stock_name": c.stock_name,
                        "leader_score": c.leader_score,
                        "sector": c.sector or "",
                    }
                    for c in cands
                ],
            })

    engine = BacktestEngine(strategy)
    result = engine.run(daily_runs)
    return {
        "strategy_name": result.strategy_name,
        "start_date": str(result.start_date),
        "end_date": str(result.end_date),
        "initial_capital": result.initial_capital,
        "final_equity": result.final_equity,
        "total_return_pct": result.total_return_pct,
        "max_drawdown_pct": result.max_drawdown_pct,
        "sharpe_ratio": result.sharpe_ratio,
        "win_rate": result.win_rate,
        "total_trades": result.total_trades,
        "trades": [
            {
                "stock_code": t.stock_code,
                "stock_name": t.stock_name,
                "entry_date": str(t.entry_date),
                "exit_date": str(t.exit_date),
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl": round(t.pnl, 2),
                "pnl_pct": round(t.pnl_pct, 4),
                "exit_reason": t.exit_reason,
                "holding_days": t.holding_days,
            }
            for t in result.trades
        ],
        "daily_snapshots": [
            {"date": str(s.date), "equity": round(s.equity, 2), "cash": round(s.cash, 2)}
            for s in result.daily_snapshots
        ],
    }

@app.get("/token-usage")
async def get_token_usage(days: int = 30):
    """Get token usage summary for recent pipeline runs."""
    from db.connection import async_session_factory
    from sqlalchemy import select
    from db.models import PipelineRun
    from datetime import date, timedelta

    cutoff = date.today() - timedelta(days=days)
    async with async_session_factory() as session:
        result = await session.execute(
            select(PipelineRun).where(
                PipelineRun.trade_date >= cutoff
            ).order_by(PipelineRun.trade_date.desc())
        )
        runs = result.scalars().all()
        return [
            {
                "run_id": r.run_id,
                "trade_date": str(r.trade_date),
                "token_usage": r.token_usage or {},
            }
            for r in runs
        ]
```

- [ ] **Step 3: Verify API starts**

```bash
cd D:/K/dragon-engine && timeout 5 D:/K/dragon-engine/venv/Scripts/python.exe -c "
from services.graph_service.main import app
print('Routes:', [r.path for r in app.routes])
print('API routes OK')
" 2>&1 || true
```

---

### Task 8: Historical Data Filler

**Files:**
- Create: `services/backtest/historical_filler.py`

**Context:** 利用 skill APIs 回填最近 1 个月管线数据。逐交易日运行简化版管线：同花顺热点 → 腾讯行情/akshare K线 → 龙虎榜 → 候选股筛选 → persist。

- [ ] **Step 1: Write historical_filler.py**

```python
"""Historical data filler — backfill pipeline runs for the last N trading days.

Uses skill APIs: ths_hot_reason(date), daily_dragon_tiger(date), 
tencent_quote(codes), akshare stock_zh_a_hist.
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from shared.utils.logging import get_logger

logger = get_logger(__name__)


async def backfill_trading_days(start_date: str, end_date: str):
    """Backfill pipeline data for all trading days between start and end.

    For each trading day:
    1. Fetch hot stocks + reasons from 同花顺 (ths_hot_reason)
    2. Fetch all-market dragon tiger board (daily_dragon_tiger)  
    3. Fetch fundamental data for hot stocks via tencent_quote
    4. Build capital_flow_records + dragon_tiger_records + leader_candidates
    5. Persist via db.persist.persist_run()
    """
    import requests
    import pandas as pd
    from datetime import datetime
    from db.persist import persist_run

    trade_dates = _get_trading_days(start_date, end_date)
    logger.info("Backfilling %d trading days: %s → %s", len(trade_dates), start_date, end_date)

    for td in trade_dates:
        logger.info("[backfill] processing %s", td)
        try:
            result = await _run_one_day(td)
            run_id = await persist_run(td, result)
            logger.info("[backfill] %s persisted as %s", td, run_id)
        except Exception as e:
            logger.error("[backfill] %s failed: %s", td, e)


async def _run_one_day(trade_date: str) -> dict:
    """Run a simplified pipeline for one historical trading day."""
    import requests
    import urllib.request
    import akshare as ak

    # 1. Fetch hot stocks from 同花顺
    url = f"http://zx.10jqka.com.cn/event/api/getharden/date/{trade_date}/orderby/date/orderway/desc/charset/GBK/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/117.0.0.0"}
    r = requests.get(url, headers=headers, timeout=10)
    data = r.json()
    hot_stocks = []
    if data.get("errocode", 0) == 0:
        for row in (data.get("data") or []):
            hot_stocks.append({
                "symbol": row.get("code", ""),
                "stock_name": row.get("name", ""),
                "reason": row.get("reason", ""),
                "change_pct": float(row.get("zhangfu", 0)),
                "turnover_pct": float(row.get("huanshou", 0)),
                "ddejingliang": float(row.get("ddejingliang", 0)) if row.get("ddejingliang") else 0,
                "close": float(row.get("close", 0)),
            })
    logger.info("[backfill] %s: %d hot stocks", trade_date, len(hot_stocks))

    # 2. Fetch fundamentals for hot stocks via tencent_quote
    codes = [s["symbol"] for s in hot_stocks[:40]]
    quotes = {}
    if codes:
        prefixed = []
        for c in codes:
            if c.startswith(("6", "9")): prefixed.append(f"sh{c}")
            elif c.startswith("8"): prefixed.append(f"bj{c}")
            else: prefixed.append(f"sz{c}")
        q_url = "https://qt.gtimg.cn/q=" + ",".join(prefixed)
        req = urllib.request.Request(q_url)
        req.add_header("User-Agent", "Mozilla/5.0")
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            raw = resp.read().decode("gbk")
            import re
            for match in re.finditer(r'v_(\w+)="([^"]*)"', raw):
                code_key = match.group(1)[2:]
                fields = match.group(2).split("~")
                if len(fields) < 50:
                    continue
                quotes[code_key] = {
                    "name": fields[1],
                    "price": float(fields[3]) if fields[3] else 0,
                    "change_pct": float(fields[32]) if fields[32] else 0,
                    "amount": float(fields[37]) if fields[37] else 0,
                    "turnover_pct": float(fields[38]) if fields[38] else 0,
                    "pe": float(fields[39]) if fields[39] else 0,
                    "mcap_yi": float(fields[45]) if fields[45] else 0,
                }
        except Exception as e:
            logger.warning("[backfill] tencent quote failed: %s", e)

    # 3. Build capital_flow_records
    capital_flow_records = []
    for s in hot_stocks[:40]:
        q = quotes.get(s["symbol"], {})
        amount = q.get("amount", 0)
        mcap = q.get("mcap_yi", 0)
        capital_flow_records.append({
            "symbol": s["symbol"],
            "stock_name": s["stock_name"],
            "price": q.get("price", 0),
            "change_pct": s.get("change_pct", 0),
            "amount": amount,
            "amount_wan": round(amount / 10000, 2) if amount else 0,
            "turnover_pct": s.get("turnover_pct", q.get("turnover_pct", 0)),
            "pe": q.get("pe", 0),
            "market_cap": mcap,
            "main_force_net": 0,
            "flow_score": 0,
            "_source": "historical_backfill",
        })

    # 4. Build dragon_tiger_records (from daily_dragon_tiger)
    dragon_tiger_records = []
    try:
        dt_url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
        dt_params = {
            "reportName": "RPT_DAILYBILLBOARD_DETAILSNEW",
            "columns": "ALL",
            "filter": f"(TRADE_DATE>='{trade_date}')(TRADE_DATE<='{trade_date}')",
            "pageNumber": "1", "pageSize": "500",
            "sortTypes": "-1", "sortColumns": "BILLBOARD_NET_AMT",
            "source": "WEB", "client": "WEB",
        }
        dt_headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"}
        dt_r = requests.get(dt_url, params=dt_params, headers=dt_headers, timeout=15)
        dt_d = dt_r.json()
        if dt_d.get("success") and dt_d.get("result", {}).get("data"):
            for row in dt_d["result"]["data"]:
                dragon_tiger_records.append({
                    "stock_code": row.get("SECURITY_CODE", ""),
                    "stock_name": row.get("SECURITY_NAME_ABBR", ""),
                    "trade_date": trade_date,
                    "reason": row.get("EXPLANATION", ""),
                    "net_amount": (row.get("BILLBOARD_NET_AMT") or 0) / 10000,
                    "lhb_score": min(abs((row.get("BILLBOARD_NET_AMT") or 0) / 10000) / 5000, 1.0),
                    "famous_traders": [],
                    "trader_signal": "",
                })
    except Exception as e:
        logger.warning("[backfill] dragon tiger failed: %s", e)

    logger.info("[backfill] %s: %d capital_flow, %d dragon_tiger",
                trade_date, len(capital_flow_records), len(dragon_tiger_records))

    # 5. Build leader_candidates (simplified: use 同花顺 hot stocks as candidates)
    leader_candidates = []
    for i, s in enumerate(hot_stocks[:10]):
        leader_candidates.append({
            "rank": i + 1,
            "stock_code": s["symbol"],
            "stock_name": s["stock_name"],
            "leader_score": min(1.0, (10 - i) / 10),
            "monster_potential": 0,
            "limit_up_prob": 0,
            "reasoning": s.get("reason", ""),
            "sector": "",
            "sentiment_sub": 0, "flow_sub": 0, "lhb_sub": 0,
            "ml_sub": 0, "event_sub": 0, "sector_tag_sub": 0,
        })

    return {
        "events": [],
        "sentiment_scores": [],
        "capital_flow_records": capital_flow_records,
        "sector_flow_records": [],
        "capital_flow_summary": {},
        "active_stocks": [
            {"symbol": s["symbol"], "stock_name": s["stock_name"],
             "matched_concepts": [s.get("reason", "")], "active_score": min(1.0, (40 - i) / 40)}
            for i, s in enumerate(hot_stocks[:40])
        ],
        "dragon_tiger_records": dragon_tiger_records,
        "leader_candidates": leader_candidates,
        "risk_flags": [],
        "activated_memories": [],
        "watchlist": [],
        "top_n": 5,
        "metadata": {"started_at": f"{trade_date}T15:00:00", "backfill": True},
    }


def _get_trading_days(start_date: str, end_date: str) -> list[str]:
    """Generate list of trading days between start and end, excluding weekends."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Mon-Fri
            days.append(current.isoformat())
        current += timedelta(days=1)
    return days
```

- [ ] **Step 2: Add script entry point**

At bottom of historical_filler.py:
```python
if __name__ == "__main__":
    import sys
    start = sys.argv[1] if len(sys.argv) > 1 else (date.today() - timedelta(days=30)).isoformat()
    end = sys.argv[2] if len(sys.argv) > 2 else date.today().isoformat()
    asyncio.run(backfill_trading_days(start, end))
```

- [ ] **Step 3: Verify import**

```bash
cd D:/K/dragon-engine && D:/K/dragon-engine/venv/Scripts/python.exe -c "
from services.backtest.historical_filler import _get_trading_days
days = _get_trading_days('2026-05-01', '2026-05-10')
print(f'Trading days: {len(days)}: {days[:3]}...')
print('historical_filler OK')
"
```

---

### Task 9: Frontend — 回测结果页 + 策略配置页 + Token消耗页

**Files:**
- Create: `dragon-engine-web/src/views/Backtest/Results.vue`
- Create: `dragon-engine-web/src/views/Backtest/Strategies.vue`
- Create: `dragon-engine-web/src/views/TokenUsage/index.vue`
- Modify: `dragon-engine-web/src/router/index.ts`
- Modify: `dragon-engine-web/src/components/layout/SidebarMenu.vue`
- Modify: `dragon-engine-web/src/types/api.ts`

**Context:** 三个新页面，Vue 3 + Element Plus + ECharts，复用现有 dark OLED 主题和通用组件。

- [ ] **Step 1: Results.vue** — backtest runner + results display

Key features:
- Date range picker + strategy selector
- "Run Backtest" button with loading state
- Summary cards: total return, max drawdown, Sharpe, win rate, trade count
- ECharts equity curve (line chart, X=date Y=equity)
- Trade detail table (el-table): entry date, exit date, stock, prices, P&L%, exit reason
- Use ScoreBadge component for P&L coloring
- Dark theme CSS matching existing dashboard

- [ ] **Step 2: Strategies.vue** — strategy CRUD

Key features:
- Strategy card list: name, description, rule summary chips
- System strategies (A/B) marked with "系统默认" badge, delete disabled
- "New Strategy" button → el-drawer form:
  - Name input
  - Entry rule selector (dropdown)
  - Exit rule multi-select with per-rule parameter inputs
  - Allocator selector
  - Max positions / max position pct / daily cash pct number inputs
- Save → POST /backtest/strategies → refresh list

- [ ] **Step 3: TokenUsage/index.vue** — token consumption display

Key features:
- Summary cards: monthly cost, total tokens, avg cost per run
- ECharts bar chart: X=date Y=tokens, dual bars for prompt vs completion
- Detail table: trade_date, model, prompt_tokens, completion_tokens, total_tokens, cost(¥)
- Click row to expand step-level breakdown
- Filter by date range

- [ ] **Step 4: Update router/index.ts**

```typescript
// Add new routes
{
  path: 'backtest/strategies',
  name: 'BacktestStrategies',
  component: () => import('@/views/Backtest/Strategies.vue'),
  meta: { title: '策略配置' },
},
{
  path: 'backtest/results',
  name: 'BacktestResults',
  component: () => import('@/views/Backtest/Results.vue'),
  meta: { title: '回测结果' },
},
{
  path: 'token-usage',
  name: 'TokenUsage',
  component: () => import('@/views/TokenUsage/index.vue'),
  meta: { title: 'Token 消耗' },
},
```

- [ ] **Step 5: Update SidebarMenu.vue**

Add after the existing "认知分析" sub-menu:
```html
<el-sub-menu index="/backtest">
  <template #title>
    <el-icon><TrendCharts /></el-icon>
    <span>回测中心</span>
  </template>
  <el-menu-item index="/backtest/strategies">策略配置</el-menu-item>
  <el-menu-item index="/backtest/results">回测结果</el-menu-item>
</el-sub-menu>

<el-menu-item index="/token-usage">
  <el-icon><Odometer /></el-icon>
  <template #title>Token 消耗</template>
</el-menu-item>
```

- [ ] **Step 6: Update types/api.ts**

```typescript
// Add new types
export interface BacktestTrade {
  stock_code: string
  stock_name: string
  entry_date: string
  exit_date: string
  entry_price: number
  exit_price: number
  pnl: number
  pnl_pct: number
  exit_reason: string
  holding_days: number
}

export interface BacktestResult {
  strategy_name: string
  start_date: string
  end_date: string
  initial_capital: number
  final_equity: number
  total_return_pct: number
  max_drawdown_pct: number
  sharpe_ratio: number
  win_rate: number
  total_trades: number
  trades: BacktestTrade[]
  daily_snapshots: { date: string; equity: number; cash: number }[]
}

export interface StrategyConfig {
  name: string
  description: string
  entry_rules: { type: string; params: Record<string, any> }[]
  exit_rules: { type: string; params: Record<string, any> }[]
  allocator: { type: string; params: Record<string, any> }
  max_positions: number
  max_position_pct: number
  initial_capital: number
  daily_cash_pct: number
  is_system?: boolean
}

export interface TokenUsageRow {
  run_id: string
  trade_date: string
  token_usage: {
    total_prompt_tokens: number
    total_completion_tokens: number
    total_tokens: number
    total_cost: number
    records: any[]
  }
}
```

- [ ] **Step 7: Verify frontend builds**

```bash
cd D:/K/dragon-engine/dragon-engine-web && npm run build 2>&1 | tail -5
```

---

### Task 10: Integration — wire token tracker callback into LLM provider

**Files:**
- Modify: `services/llm_adapter/llm_provider.py`

**Context:** Inject TokenTrackingCallback into the LLM instance so all LLM calls are automatically tracked. Must be done in the factory function that creates the LLM.

- [ ] **Step 1: Find and modify the LLM creation function**

In `services/llm_adapter/llm_provider.py`, find `create_quick_llm()` (or equivalent). After LLM object creation, add:

```python
from services.token_tracker import TokenUsageTracker

# Inside create_quick_llm() or equivalent:
tracker = TokenUsageTracker.instance()
callback = tracker.create_callback()
if hasattr(llm, 'callbacks'):
    llm.callbacks = (llm.callbacks or []) + [callback]
else:
    llm.callbacks = [callback]
```

- [ ] **Step 2: Verify callback injection**

```bash
cd D:/K/dragon-engine && D:/K/dragon-engine/venv/Scripts/python.exe -c "
from services.llm_adapter.llm_provider import create_quick_llm
llm = create_quick_llm()
print('LLM callbacks:', llm.callbacks)
print('Injection OK' if llm.callbacks else 'WARNING: no callbacks')
"
```

---

### Task 11: Run historical data backfill

**Context:** After all previous tasks complete, run the backfill to populate the DB with 1 month of pipeline data.

- [ ] **Step 1: Run backfill for last 22 trading days**

```bash
cd D:/K/dragon-engine && D:/K/dragon-engine/venv/Scripts/python.exe -c "
import asyncio
from services.backtest.historical_filler import backfill_trading_days
asyncio.run(backfill_trading_days('2026-04-15', '2026-05-19'))
"
```

- [ ] **Step 2: Verify data in DB**

```bash
cd D:/K/dragon-engine && D:/K/dragon-engine/venv/Scripts/python.exe -c "
import asyncio
from sqlalchemy import select, func
from db.connection import async_session_factory
from db.models import PipelineRun

async def check():
    async with async_session_factory() as s:
        cnt = (await s.execute(select(func.count(PipelineRun.run_id)))).scalar()
        print(f'Total pipeline runs in DB: {cnt}')

asyncio.run(check())
"
```

Expected: 20+ runs in DB.

---

### Task 12: End-to-end smoke test

- [ ] **Step 1: Start server and test API**

```bash
# Start server in background
cd D:/K/dragon-engine && D:/K/dragon-engine/venv/Scripts/python.exe -m uvicorn services.graph_service.main:app --host 0.0.0.0 --port 8000 &
sleep 3
# Test endpoints
curl -s http://localhost:8000/backtest/strategies | python -m json.tool | head -20
curl -s http://localhost:8000/token-usage | python -m json.tool | head -20
# Run a backtest
curl -s -X POST http://localhost:8000/backtest/run \
  -H "Content-Type: application/json" \
  -d '{"strategy_name":"A","start_date":"2026-05-01","end_date":"2026-05-19"}' | python -m json.tool | head -30
```

- [ ] **Step 2: Verify frontend**

Start dev server, navigate to:
- `/backtest/strategies` — see Strategy A and B
- `/backtest/results` — run backtest, see equity curve
- `/token-usage` — see token consumption data

---

## Execution Order

Parallel groups for sub-agent dispatch:

**Wave 1** (no deps, can run simultaneously):
- Task 1: models.py
- Task 2: rules.py  
- Task 5: token_tracker
- Task 9: Frontend three pages

**Wave 2** (depends on Wave 1):
- Task 3: strategies.py + registry.py (needs Task 1, 2)
- Task 4: engine.py (needs Task 1, 2, 3)
- Task 6: DB schema (needs Task 1)
- Task 10: llm_provider injection (needs Task 5)

**Wave 3** (depends on Wave 2):
- Task 7: API routes (needs Task 3, 4, 6)
- Task 8: historical_filler (needs Task 4)

**Wave 4** (integration):
- Task 11: Run backfill (needs Task 7, 8)
- Task 12: E2E test (needs all)
