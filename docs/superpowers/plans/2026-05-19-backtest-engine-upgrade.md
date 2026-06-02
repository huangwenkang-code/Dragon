# 回测引擎升级 + 价格数据 + 前端重设计 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 回测引擎从"零P&L"升级为真实盈亏 — 开盘价买卖 + 手续费/印花税 + 高开/涨停过滤 + 每日价格数据入库 + 前端 ECharts 重做。

**Architecture:** 三层改动 — 数据层新增 `stock_daily_bars` 表存每日 OHLCV，引擎层用开盘价替代默认 ¥10、注入费率计算、加两个 EntryRule，前端层用 ECharts 替代手写 SVG、加多策略切换、加主题适配。

**Tech Stack:** Python 3.12 + SQLAlchemy 2.0 async + PostgreSQL 18 + Vue 3 + TypeScript + ECharts 5 + Element Plus

---

## Wave 1 — 4 个并行子任务（无相互依赖）

### Task 1: Token 追踪 Bug 修复

**Files:**
- Modify: `services/token_tracker/tracker.py:20-39`
- Modify: `services/graph_service/main.py:115-134`
- Modify: `db/persist.py:63-77`

Token 追踪返回空 `{}` 的根因：`on_llm_end` 中 `response.llm_output` 可能为 `None`，且 `token_usage` 键名可能因 provider 而异。修复 callback 使其兼容多种 LLM 响应格式，并确保 `persist_run` 正确调用 `tracker.summary()`。

- [ ] **Step 1: 修复 TokenTrackingCallback.on_llm_end 兼容多种 token_usage 格式**

```python
# services/token_tracker/tracker.py — 替换 on_llm_end 方法

def on_llm_end(self, response, **kwargs):
    try:
        llm_output = response.llm_output or {}
        # 兼容多种 LLM provider 的 token_usage 格式
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if not usage:
            # 尝试从 response 直接获取 (langchain >= 0.3 某些版本)
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = {
                    "prompt_tokens": response.usage_metadata.get("input_tokens", 0),
                    "completion_tokens": response.usage_metadata.get("output_tokens", 0),
                    "total_tokens": response.usage_metadata.get("total_tokens", 0),
                }
            elif hasattr(response, 'response_metadata') and response.response_metadata:
                usage = response.response_metadata.get("token_usage", {})
        if not usage:
            # 最后一次尝试：AIMessage 的 usage_metadata 字段
            generations = getattr(response, 'generations', [])
            if generations and len(generations) > 0:
                gen0 = generations[0]
                if hasattr(gen0, 'message'):
                    msg = gen0[0] if isinstance(gen0, list) and len(gen0) > 0 else gen0
                    if hasattr(msg, 'usage_metadata') and msg.usage_metadata:
                        um = msg.usage_metadata
                        usage = {
                            "prompt_tokens": um.get("input_tokens", 0),
                            "completion_tokens": um.get("output_tokens", 0),
                            "total_tokens": um.get("total_tokens", 0),
                        }
            if not usage:
                logger.debug("No token_usage found in response. llm_output keys: %s, response type: %s",
                           list(llm_output.keys()), type(response).__name__)
                return
        model = llm_output.get("model_name", "") or getattr(response, 'model_name', "") or "unknown"
        record = TokenRecord(
            run_id=self._tracker.current_run_id,
            step=kwargs.get("name", "unknown"),
            model=model,
            prompt_tokens=usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0) or usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            cost=estimate_cost(model,
                usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0),
                usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)),
            timestamp=datetime.now().isoformat(),
        )
        self._tracker.record(record)
    except Exception as e:
        logger.debug("Token tracking callback error: %s", e)
```

- [ ] **Step 2: 确保持久化逻辑正确 — 检查 persist_run 中 token_usage 保存**

```python
# db/persist.py — 替换 token_usage 保存块 (行 63-77)
# 原代码已正确，但增加日志便于排查：

try:
    from services.token_tracker import TokenUsageTracker
    tracker = TokenUsageTracker.instance()
    summary = tracker.summary()
    logger.info("[persist] token tracker summary: tokens=%d cost=%.6f records=%d",
                summary["total_tokens"], summary["total_cost"], len(summary["records"]))
    if summary["total_tokens"] > 0:
        run.token_usage = {
            "total_cost": summary["total_cost"],
            "total_tokens": summary["total_tokens"],
            "total_prompt_tokens": summary["total_prompt_tokens"],
            "total_completion_tokens": summary["total_completion_tokens"],
            "records": summary["records"],
        }
    else:
        logger.warning("[persist] token tracker has 0 total_tokens — no LLM calls tracked?")
        run.token_usage = summary  # 至少保存空结构以便前端诊断
except Exception:
    logger.exception("[persist] token tracker unavailable")
```

- [ ] **Step 3: 验证 — 运行一次 /run 并检查 token_usage**

Run: `curl -X POST http://localhost:8000/run -H "Content-Type: application/json" -d '{"trade_date":"2026-05-19","top_n":5}'`
Then: `curl -s http://localhost:8000/token-usage | python -m json.tool`

Expected: token_usage 中 records 非空，total_tokens > 0（如果管线调用了 LLM）。

---

### Task 2: stock_daily_bars 表 + ORM + SQL

**Files:**
- Modify: `db/models.py` — 追加 `StockDailyBar` ORM 类
- Modify: `db/schema.sql` — 追加 CREATE TABLE + 索引
- Create: `db/migrate_005_bars.py` — 独立迁移脚本

- [ ] **Step 1: 在 db/models.py 追加 StockDailyBar ORM**

在文件末尾（`RiskFlag` 之后）追加：

```python
# ===========================================================================
# 16. Stock daily bars — historical OHLCV for backtest
# ===========================================================================

class StockDailyBar(Base):
    __tablename__ = "stock_daily_bars"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    symbol      = Column(String(10), nullable=False)
    trade_date  = Column(Date, nullable=False)
    open        = Column(Float, default=0)
    high        = Column(Float, default=0)
    low         = Column(Float, default=0)
    close       = Column(Float, default=0)
    volume      = Column(BigInteger, default=0)
    amount      = Column(Float, default=0)
    change_pct  = Column(Float, default=0)
    turnover_pct = Column(Float, default=0)
    created_at  = Column(DateTime(timezone=True), default=_utcnow)

    __table_args__ = (UniqueConstraint("symbol", "trade_date"),)
```

在文件顶部 import 中确认 `BigInteger` 和 `UniqueConstraint` 已导入。

- [ ] **Step 2: 在 db/schema.sql 追加建表语句**

```sql
-- 16. Stock daily bars (OHLCV for backtest)
CREATE TABLE IF NOT EXISTS stock_daily_bars (
    id           SERIAL PRIMARY KEY,
    symbol       VARCHAR(10) NOT NULL,
    trade_date   DATE NOT NULL,
    open         FLOAT DEFAULT 0,
    high         FLOAT DEFAULT 0,
    low          FLOAT DEFAULT 0,
    close        FLOAT DEFAULT 0,
    volume       BIGINT DEFAULT 0,
    amount       FLOAT DEFAULT 0,
    change_pct   FLOAT DEFAULT 0,
    turnover_pct FLOAT DEFAULT 0,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_bars_symbol ON stock_daily_bars(symbol);
CREATE INDEX IF NOT EXISTS idx_bars_date ON stock_daily_bars(trade_date);
```

- [ ] **Step 3: 创建迁移脚本 db/migrate_005_bars.py**

```python
"""Migration 005: Create stock_daily_bars table."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
from db.connection import engine
from sqlalchemy import text

SQL = """
CREATE TABLE IF NOT EXISTS stock_daily_bars (
    id           SERIAL PRIMARY KEY,
    symbol       VARCHAR(10) NOT NULL,
    trade_date   DATE NOT NULL,
    open         FLOAT DEFAULT 0,
    high         FLOAT DEFAULT 0,
    low          FLOAT DEFAULT 0,
    close        FLOAT DEFAULT 0,
    volume       BIGINT DEFAULT 0,
    amount       FLOAT DEFAULT 0,
    change_pct   FLOAT DEFAULT 0,
    turnover_pct FLOAT DEFAULT 0,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_bars_symbol ON stock_daily_bars(symbol);
CREATE INDEX IF NOT EXISTS idx_bars_date ON stock_daily_bars(trade_date);
"""

async def migrate():
    async with engine.begin() as conn:
        await conn.execute(text(SQL))
    print("[005] stock_daily_bars table created")

if __name__ == "__main__":
    asyncio.run(migrate())
```

- [ ] **Step 4: 运行迁移**

Run: `cd D:\K\dragon-engine && D:/K/dragon-engine/venv/Scripts/python.exe db/migrate_005_bars.py`

Expected: 输出 `[005] stock_daily_bars table created` 无报错。

---

### Task 3: TradingStrategy 新增字段 + 序列化

**Files:**
- Modify: `services/backtest/strategies.py` — 全量修改 TradingStrategy + 系统策略

- [ ] **Step 1: 更新 TradingStrategy dataclass 和 to_dict/from_dict**

```python
# services/backtest/strategies.py — 完整替换

"""TradingStrategy — combines entry/exit rules + allocator into a named strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from services.backtest.rules import (
    EntryRule, ExitRule, PositionAllocator,
    ScoreThresholdRule, NoFilterRule,
    ScoreCliffRule, TrailingStopRule, ScoreDeclineRule,
    ScoreWeightedAllocator,
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
    commission_rate: float = 0.00025     # 佣金费率 (万2.5)
    stamp_duty_rate: float = 0.0005      # 印花税 (卖出时收 0.05%)
    min_commission: float = 5.0          # 最低佣金 (不足5元按5元)
    gap_up_pct: float | None = None      # None=不过滤高开, 值=最大允许高开比例
    enable_limit_up_filter: bool = True  # 涨停过滤 (默认开启)
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
    description="每天用总资金50%按score加权买入当天top-5，持仓无上限，开盘价成交，费率万2.5+印花税0.05%",
    entry_rules=[NoFilterRule()],
    exit_rules=[ScoreCliffRule(0.3), TrailingStopRule(0.15), ScoreDeclineRule(3)],
    allocator=ScoreWeightedAllocator(),
    max_positions=999,
    max_position_pct=1.0,
    daily_cash_pct=0.5,
    commission_rate=0.00025,
    stamp_duty_rate=0.0005,
    min_commission=5.0,
    gap_up_pct=None,
    enable_limit_up_filter=True,
    is_system=True,
)

STRATEGY_B = TradingStrategy(
    name="策略B-仓位上限管理",
    description="最多8只持仓，单只≤15%，有空位才买新的，开盘价成交",
    entry_rules=[ScoreThresholdRule(0.5)],
    exit_rules=[ScoreCliffRule(0.3), TrailingStopRule(0.15), ScoreDeclineRule(3)],
    allocator=ScoreWeightedAllocator(),
    max_positions=8,
    max_position_pct=0.15,
    daily_cash_pct=1.0,
    commission_rate=0.00025,
    stamp_duty_rate=0.0005,
    min_commission=5.0,
    gap_up_pct=None,
    enable_limit_up_filter=True,
    is_system=True,
)

SYSTEM_STRATEGIES = {STRATEGY_A.name: STRATEGY_A, STRATEGY_B.name: STRATEGY_B}
```

- [ ] **Step 2: 验证序列化往返**

Run: `cd D:\K\dragon-engine && D:/K/dragon-engine/venv/Scripts/python.exe -c "
from services.backtest.strategies import STRATEGY_A
d = STRATEGY_A.to_dict()
print('gap_up_pct:', d.get('gap_up_pct'))
print('commission_rate:', d.get('commission_rate'))
from services.backtest.strategies import TradingStrategy
s2 = TradingStrategy.from_dict(d)
print('roundtrip OK:', s2.name, s2.commission_rate)
"`

Expected: gap_up_pct=None, commission_rate=0.00025, roundtrip OK.

---

### Task 4: Backtest Model 升级 + 手续费计算

**Files:**
- Modify: `services/backtest/models.py` — Trade/BacktestResult 加字段

- [ ] **Step 1: 更新 Trade 和 BacktestResult dataclass**

```python
# services/backtest/models.py — 替换 Trade 和 BacktestResult

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
    pnl: float                   # gross profit/loss (before fees)
    pnl_pct: float               # gross percentage return
    entry_commission: float = 0.0   # 买入佣金
    exit_commission: float = 0.0    # 卖出佣金
    stamp_duty: float = 0.0        # 卖出印花税
    net_pnl: float = 0.0           # net P&L after all fees
    entry_score: float = 0.0       # leader_score at entry
    exit_score: float = 0.0        # leader_score at exit
    exit_reason: str = ""
    holding_days: int = 0


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
    win_rate: float = 0.0
    total_trades: int = 0
    total_commission: float = 0.0    # 总佣金
    total_stamp_duty: float = 0.0   # 总印花税
    trades: list[Trade] = field(default_factory=list)
    daily_snapshots: list[DailySnapshot] = field(default_factory=list)
    benchmark_return_pct: float = 0.0
```

- [ ] **Step 2: 添加手续费计算函数**

在 `services/backtest/models.py` 顶部（dataclass imports 之后）追加：

```python
def calc_commission(amount: float, rate: float, min_fee: float = 5.0) -> float:
    """Calculate commission with minimum fee floor.
    A-share: 0.025% rate, min ¥5 per trade.
    """
    return max(amount * rate, min_fee)
```

- [ ] **Step 3: 验证 models 导入**

Run: `cd D:\K\dragon-engine && D:/K/dragon-engine/venv/Scripts/python.exe -c "
from services.backtest.models import Trade, BacktestResult, calc_commission
t = Trade(stock_code='000001', stock_name='test', entry_date='2026-01-01', exit_date='2026-01-02', entry_price=10.0, exit_price=11.0, shares=1000, cost=10000, proceeds=11000, pnl=1000, pnl_pct=0.1, net_pnl=1000-5-5-5.5)
print('Trade OK:', t.net_pnl)
print('Commission on 10000:', calc_commission(10000, 0.00025, 5.0))
print('Commission on 2000:', calc_commission(2000, 0.00025, 5.0))
"`

Expected: Trade OK: 984.5, Commission on 10000: 5.0 (2.5 < 5, so min 5), Commission on 2000: 5.0 (0.5 < 5, so min 5).

---

### Task 5: 引擎层升级 — 价格来源 + 手续费 + 涨停/高开过滤

**⚠️ 依赖 Task 2/3/4 的文件修改（但无逻辑依赖，可并行）**

**Files:**
- Modify: `services/backtest/rules.py` — 加 LimitUpFilterRule + GapUpFilterRule
- Modify: `services/backtest/engine.py` — 价格来源 + 手续费 + 自动注入 limit_up 规则

- [ ] **Step 1: 在 rules.py 追加两个新 EntryRule**

在 `rules.py` 的 Entry Rules 区域（`NoFilterRule` 之后）追加：

```python
class LimitUpFilterRule(EntryRule):
    """Auto-reject candidates whose open price is at/near limit-up (≥9.8% gap).
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
```

- [ ] **Step 2: 更新 ENTRY_RULES 注册表**

```python
ENTRY_RULES: dict[str, type[EntryRule]] = {
    "score_threshold": ScoreThresholdRule,
    "no_filter": NoFilterRule,
    "limit_up_filter": LimitUpFilterRule,
    "gap_up_filter": GapUpFilterRule,
}
```

- [ ] **Step 3: 升级 BacktestEngine 支持开盘价 + 手续费**

完整替换 `services/backtest/engine.py`：

```python
"""BacktestEngine — iterate daily PipelineRuns, simulate portfolio with pluggable rules."""

from __future__ import annotations

from datetime import date
from services.backtest.models import (
    Position, Trade, DailySnapshot, BacktestResult, calc_commission
)
from services.backtest.strategies import TradingStrategy
from shared.utils.logging import get_logger

logger = get_logger(__name__)


class BacktestContext:
    """Mutable state during a backtest run."""

    def __init__(self, strategy: TradingStrategy):
        self.strategy = strategy
        self.current_date: date | None = None
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.snapshots: list[DailySnapshot] = []
        self.cash: float = strategy.initial_capital
        self.benchmark_value: float = 1.0


class BacktestEngine:
    """Iterate daily PipelineRun data, simulate trading with pluggable rules.

    Each day:
    1. Update position scores from today's candidates
    2. Check exit rules → close positions that trigger
    3. Filter entry candidates through entry rules (+ auto limit_up_filter)
    4. Enforce position limits
    5. Allocate capital via PositionAllocator
    6. Record daily snapshot with real prices
    """

    def __init__(self, strategy: TradingStrategy):
        self.strategy = strategy
        self.context = BacktestContext(strategy)

    def run(self, daily_runs: list[dict]) -> BacktestResult:
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
        prices = run.get("prices", {})            # symbol → open_price
        prev_close = run.get("prev_close", {})    # symbol → yesterday_close

        cand_by_code = {c.get("stock_code"): c for c in candidates}

        # Step 1: Update position scores & prices from today's data
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
                pos.current_score = 0.0
                pos.score_history.append(0.0)
                # 查真实收盘价（不是默认价）
                if code in prices:
                    pos.current_price = prices.get(code, pos.current_price)

        # Step 2: Check exit rules
        for code, pos in list(self.context.positions.items()):
            current_cand = cand_by_code.get(code)
            ctx = {"date": trade_date, "prices": prices}
            for rule in self.strategy.exit_rules:
                reason = rule.should_exit(pos, current_cand, ctx)
                if reason:
                    self._close_position(code, pos, reason, trade_date)
                    break

        # Step 3: Filter entry candidates
        ctx = {
            "date": trade_date,
            "prices": prices,
            "prev_close": prev_close,
        }

        eligible = []
        for c in candidates:
            code = c.get("stock_code", "")
            if code in self.context.positions:
                continue
            # Auto-inject limit_up_filter check
            if self.strategy.enable_limit_up_filter:
                if not self._check_limit_up(c, ctx):
                    continue
            # Check strategy entry rules
            passed = all(r.should_enter(c, ctx) for r in self.strategy.entry_rules)
            if passed:
                eligible.append(c)

        # Step 4: Position limit enforcement
        current_count = len(self.context.positions)
        slots = max(0, self.strategy.max_positions - current_count)
        if slots == 0:
            eligible = []
        elif slots < len(eligible):
            eligible.sort(key=lambda c: c.get("leader_score", 0), reverse=True)
            eligible = eligible[:slots]

        # Step 5: Allocate with real prices
        if eligible:
            daily_cash = self.context.cash * self.strategy.daily_cash_pct
            alloc_cash = min(daily_cash, self.context.cash)
            orders = self.strategy.allocator.allocate(eligible, alloc_cash, prices, {"date": trade_date})
            for order in orders:
                self._open_position(order, trade_date, prices)

        # Step 6: Record daily snapshot
        equity = self.context.cash + sum(
            p.current_price * p.shares for p in self.context.positions.values()
        )
        prev_equity = self.context.snapshots[-1].equity if self.context.snapshots else self.strategy.initial_capital
        daily_return = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0

        self.context.snapshots.append(DailySnapshot(
            date=trade_date,
            equity=equity,
            cash=self.context.cash,
            positions=list(self.context.positions.values()),
            daily_return=daily_return,
        ))
        self.context.current_date = trade_date

    def _check_limit_up(self, candidate: dict, ctx: dict) -> bool:
        """Check if candidate's open price is at/near limit-up."""
        code = candidate.get("stock_code", "")
        prev_close = ctx.get("prev_close", {}).get(code)
        open_price = ctx.get("prices", {}).get(code)
        if prev_close and open_price and prev_close > 0:
            gap = (open_price - prev_close) / prev_close
            if gap >= 0.098:
                return False
        return True

    def _open_position(self, order: dict, trade_date: date, prices: dict):
        code = order["stock_code"]
        price = order["entry_price"]
        cost = order["allocated_cash"]
        entry_comm = calc_commission(cost, self.strategy.commission_rate, self.strategy.min_commission)

        pos = Position(
            stock_code=code,
            stock_name=order.get("stock_name", ""),
            entry_date=trade_date,
            entry_price=price,
            entry_score=order.get("score", 0),
            shares=order["shares"],
            cost=cost,
            peak_score=order.get("score", 0),
            peak_price=price,
            current_price=price,
            current_score=order.get("score", 0),
        )
        self.context.positions[code] = pos
        self.context.cash -= (cost + entry_comm)

    def _close_position(self, code: str, pos: Position, reason: str, trade_date: date):
        sell_price = pos.current_price
        proceeds = sell_price * pos.shares
        pnl = proceeds - pos.cost
        pnl_pct = pnl / pos.cost if pos.cost > 0 else 0.0

        entry_comm = calc_commission(pos.cost, self.strategy.commission_rate, self.strategy.min_commission)
        exit_comm = calc_commission(proceeds, self.strategy.commission_rate, self.strategy.min_commission)
        stamp = proceeds * self.strategy.stamp_duty_rate
        net_pnl = pnl - entry_comm - exit_comm - stamp

        trade = Trade(
            stock_code=code,
            stock_name=pos.stock_name,
            entry_date=pos.entry_date,
            exit_date=trade_date,
            entry_price=pos.entry_price,
            exit_price=sell_price,
            shares=pos.shares,
            cost=pos.cost,
            proceeds=proceeds,
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 4),
            entry_commission=round(entry_comm, 2),
            exit_commission=round(exit_comm, 2),
            stamp_duty=round(stamp, 2),
            net_pnl=round(net_pnl, 2),
            entry_score=pos.entry_score,
            exit_score=pos.current_score,
            exit_reason=reason,
            holding_days=pos.days_held,
        )
        self.context.trades.append(trade)
        self.context.cash += (proceeds - exit_comm - stamp)
        del self.context.positions[code]

    def _build_result(self, start_date: date, end_date: date) -> BacktestResult:
        final_equity = self.context.snapshots[-1].equity if self.context.snapshots else self.strategy.initial_capital
        total_return = (final_equity - self.strategy.initial_capital) / self.strategy.initial_capital
        max_dd = self._calc_max_drawdown()
        sharpe = self._calc_sharpe()
        wins = sum(1 for t in self.context.trades if t.net_pnl > 0)
        wr = wins / len(self.context.trades) if self.context.trades else 0.0

        # Totals
        total_comm = sum(t.entry_commission + t.exit_commission for t in self.context.trades)
        total_stamp = sum(t.stamp_duty for t in self.context.trades)

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
            total_commission=round(total_comm, 2),
            total_stamp_duty=round(total_stamp, 2),
            trades=self.context.trades,
            daily_snapshots=self.context.snapshots,
        )

    def _calc_max_drawdown(self) -> float:
        peak = self.strategy.initial_capital
        max_dd = 0.0
        for snap in self.context.snapshots:
            if snap.equity > peak:
                peak = snap.equity
            dd = (peak - snap.equity) / peak if peak > 0 else 0.0
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
        return (mean_ret / std) * (252 ** 0.5)

    def _empty_result(self) -> BacktestResult:
        return BacktestResult(
            strategy_name=self.strategy.name,
            start_date=date.today(),
            end_date=date.today(),
            initial_capital=self.strategy.initial_capital,
            final_equity=self.strategy.initial_capital,
            total_return_pct=0.0,
            max_drawdown_pct=0.0,
        )
```

- [ ] **Step 4: 验证引擎可导入且结构正确**

Run: `cd D:\K\dragon-engine && D:/K/dragon-engine/venv/Scripts/python.exe -c "
from services.backtest.engine import BacktestEngine
from services.backtest.strategies import STRATEGY_A
from services.backtest.models import calc_commission
e = BacktestEngine(STRATEGY_A)
print('Engine init OK, commission test:', calc_commission(10000, 0.00025, 5))
"`

Expected: `Engine init OK, commission test: 5.0`

---

## Wave 2 — 2 个并行子任务（依赖 Wave 1 全部完成）

### Task 6: 价格数据批量写入（persist + batch scripts）

**Files:**
- Modify: `db/persist.py` — 加 `upsert_daily_bars()`
- Modify: `run_enhanced_backfill.py` — 写入 daily bars
- Modify: `run_batch_30days.py` — 写入 daily bars

- [ ] **Step 1: 在 db/persist.py 添加 upsert_daily_bars()**

```python
# 在 persist.py 顶部 import 区加：
from sqlalchemy.dialects.postgresql import insert as pg_insert

# 在文件末尾（persist_run 之后）加：

async def upsert_daily_bars(session: AsyncSession, bars: list[dict]) -> int:
    """Upsert daily OHLCV bars. Returns count of rows inserted/updated."""
    if not bars:
        return 0
    from db.models import StockDailyBar

    rows = []
    for b in bars:
        rows.append(dict(
            symbol=b.get("symbol", ""),
            trade_date=b.get("trade_date"),
            open=b.get("open", 0),
            high=b.get("high", 0),
            low=b.get("low", 0),
            close=b.get("close", 0),
            volume=b.get("volume", 0),
            amount=b.get("amount", 0),
            change_pct=b.get("change_pct", 0),
            turnover_pct=b.get("turnover_pct", 0),
        ))

    stmt = pg_insert(StockDailyBar).values(rows).on_conflict_do_update(
        index_elements=["symbol", "trade_date"],
        set_={
            "open": pg_insert.excluded.open,
            "high": pg_insert.excluded.high,
            "low": pg_insert.excluded.low,
            "close": pg_insert.excluded.close,
            "volume": pg_insert.excluded.volume,
            "amount": pg_insert.excluded.amount,
            "change_pct": pg_insert.excluded.change_pct,
            "turnover_pct": pg_insert.excluded.turnover_pct,
        }
    )
    await session.execute(stmt)
    return len(rows)
```

- [ ] **Step 2: 修改 run_enhanced_backfill.py — 在 run_one_day() 中写入 bars**

在 `run_one_day()` 函数中，`fetch_akshare_daily()` 调用之后加：

```python
# 在 run_enhanced_backfill.py 的 run_one_day() 中，akshare_data 获取后：
from db.persist import upsert_daily_bars
from db.connection import async_session_factory

# 写入 daily bars
bars_to_upsert = []
for sym, data in akshare_data.items():
    bars_to_upsert.append({
        "symbol": sym,
        "trade_date": trade_date,
        "open": data.get("open", 0),
        "high": data.get("high", 0),
        "low": data.get("low", 0),
        "close": data.get("close", 0),
        "volume": data.get("volume", 0),
        "amount": data.get("amount", 0),
        "change_pct": data.get("change_pct", 0),
        "turnover_pct": data.get("turnover_pct", 0),
    })
if bars_to_upsert:
    async with async_session_factory() as session:
        async with session.begin():
            count = await upsert_daily_bars(session, bars_to_upsert)
            logger.info("[ENHANCED] %s: %d daily bars upserted", trade_date, count)
```

- [ ] **Step 3: 修改 run_batch_30days.py — 同样逻辑**

在 `run_one_day()` 函数中，调用了 `graph.ainvoke()` 之后，从返回的 `capital_flow_records` 中提取价格数据写入 bars。在 `run_one_day()` 函数末尾（persist_run 之后）加：

```python
# 在 run_batch_30days.py 的 run_one_day() 中，persist_run 之后：
try:
    from db.persist import upsert_daily_bars
    from db.connection import async_session_factory
    bars = []
    for r in result.get("capital_flow_records", []):
        bars.append({
            "symbol": r.get("symbol", ""),
            "trade_date": trade_date,
            "open": r.get("price", 0),
            "high": r.get("price", 0),
            "low": r.get("price", 0),
            "close": r.get("price", 0),
            "volume": 0,
            "amount": r.get("amount", 0),
            "change_pct": r.get("change_pct", 0),
            "turnover_pct": r.get("turnover_pct", 0),
        })
    if bars:
        async with async_session_factory() as session:
            async with session.begin():
                count = await upsert_daily_bars(session, bars)
                logger.info("[BATCH] %s: %d daily bars upserted", trade_date, count)
except Exception:
    logger.warning("[BATCH] %s: daily bars upsert failed", trade_date)
```

- [ ] **Step 4: 验证 — 运行一天 enhanced backfill 检查数据写入**

Run: `cd D:\K\dragon-engine && D:/K/dragon-engine/venv/Scripts/python.exe -c "
import sys; sys.path.insert(0, '.')
from db.connection import async_session_factory
from sqlalchemy import text
import asyncio
async def check():
    async with async_session_factory() as session:
        r = await session.execute(text('SELECT count(*) FROM stock_daily_bars'))
        print('Total bars:', r.scalar())
        r = await session.execute(text('SELECT symbol, trade_date, open, close, volume FROM stock_daily_bars LIMIT 5'))
        for row in r:
            print(f'  {row}')
asyncio.run(check())
"`

Expected: 有数据（如果之前跑过 backfill）或 0 行（新表）。

---

### Task 7: 升级 backtest API endpoint — prices 注入 + 多策略

**Files:**
- Modify: `services/graph_service/main.py` — POST `/backtest/run` 端点

- [ ] **Step 1: 在 main.py 添加 load_prices 辅助函数**

在 `run_backtest` 函数之前加：

```python
async def _load_prices(session, symbols: set[str], start_d: date, end_d: date) -> tuple[dict, dict]:
    """Load open prices and prev_close for a set of symbols over a date range.
    Returns (prices_by_date: dict[str, dict[str,float]], prev_close_by_date: dict[str, dict[str,float]])
    """
    from db.models import StockDailyBar
    from sqlalchemy import select, and_

    prices_by_date: dict[str, dict[str, float]] = {}
    prev_close_by_date: dict[str, dict[str, float]] = {}

    rows = (await session.execute(
        select(StockDailyBar).where(
            and_(
                StockDailyBar.trade_date >= start_d,
                StockDailyBar.trade_date <= end_d,
                StockDailyBar.symbol.in_(symbols),
            )
        ).order_by(StockDailyBar.trade_date.asc())
    )).scalars().all()

    for r in rows:
        ds = str(r.trade_date) if hasattr(r.trade_date, 'isoformat') else r.trade_date
        if ds not in prices_by_date:
            prices_by_date[ds] = {}
        prices_by_date[ds][r.symbol] = r.open if r.open > 0 else r.close

    return prices_by_date, prev_close_by_date  # prev_close derived during iteration
```

- [ ] **Step 2: 重写 POST /backtest/run 端点**

替换 `services/graph_service/main.py` 行 193-268：

```python
@app.post("/backtest/run")
async def run_backtest(req: BacktestRequest):
    """Run a backtest using a specified strategy over a date range."""
    from datetime import date as date_type
    from db.connection import async_session_factory
    from sqlalchemy import select
    from db.models import PipelineRun, LeaderCandidate

    registry = get_registry()
    strategy = registry.get(req.strategy_name)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Strategy '{req.strategy_name}' not found")

    try:
        start_d = date_type.fromisoformat(req.start_date) if req.start_date else date_type.today()
        end_d = date_type.fromisoformat(req.end_date) if req.end_date else date_type.today()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

    try:
        async with async_session_factory() as session:
            result = await session.execute(
                select(PipelineRun).where(
                    PipelineRun.trade_date >= start_d,
                    PipelineRun.trade_date <= end_d,
                ).order_by(PipelineRun.trade_date.asc())
            )
            runs = result.scalars().all()

            # Collect all symbols to load prices
            all_symbols = set()
            daily_runs = []
            for run in runs:
                cands_result = await session.execute(
                    select(LeaderCandidate).where(
                        LeaderCandidate.run_id == run.run_id
                    ).order_by(LeaderCandidate.rank)
                )
                cands = cands_result.scalars().all()
                for c in cands:
                    all_symbols.add(c.stock_code)
                daily_runs.append({
                    "trade_date": str(run.trade_date),
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

            # Load price data
            prices_by_date, _ = await _load_prices(session, all_symbols, start_d, end_d)

            # Build prev_close map: for each day, prev_close[symbol] = yesterday's close
            from db.models import StockDailyBar
            bar_rows = (await session.execute(
                select(StockDailyBar).where(
                    StockDailyBar.trade_date >= start_d,
                    StockDailyBar.trade_date <= end_d,
                    StockDailyBar.symbol.in_(all_symbols),
                ).order_by(StockDailyBar.trade_date.asc(), StockDailyBar.symbol.asc())
            )).scalars().all()

            close_by_date = {}
            for b in bar_rows:
                ds = str(b.trade_date)
                if ds not in close_by_date:
                    close_by_date[ds] = {}
                close_by_date[ds][b.symbol] = b.close

            # Inject prices + prev_close into each day
            prev_dates = sorted(close_by_date.keys())
            for i, dr in enumerate(daily_runs):
                td = dr["trade_date"]
                dr["prices"] = prices_by_date.get(td, {})
                dr["prev_close"] = {}
                if i > 0:
                    dr["prev_close"] = close_by_date.get(prev_dates[i - 1], {})
                # fallback: if no prev_date in bars, use prices from same day
                if not dr["prev_close"]:
                    dr["prev_close"] = {}

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
            "total_commission": result.total_commission,
            "total_stamp_duty": result.total_stamp_duty,
            "trades": [
                {
                    "stock_code": t.stock_code,
                    "stock_name": t.stock_name,
                    "entry_date": str(t.entry_date),
                    "exit_date": str(t.exit_date),
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "shares": t.shares,
                    "cost": t.cost,
                    "proceeds": t.proceeds,
                    "pnl": round(t.pnl, 2),
                    "pnl_pct": round(t.pnl_pct, 4),
                    "entry_commission": t.entry_commission,
                    "exit_commission": t.exit_commission,
                    "stamp_duty": t.stamp_duty,
                    "net_pnl": t.net_pnl,
                    "entry_score": t.entry_score,
                    "exit_score": t.exit_score,
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
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Backtest failed")
        raise HTTPException(status_code=500, detail=str(e))
```

---

## Wave 3 — 前端重设计（依赖 Wave 2 完成）

### Task 8: 前端类型更新 + Results.vue 重写

**Files:**
- Modify: `dragon-engine-web/src/types/api.ts` — BacktestTrade/BacktestResult 加字段
- Rewrite: `dragon-engine-web/src/views/Backtest/Results.vue`

- [ ] **Step 1: 更新 api.ts 类型**

```typescript
// 替换 BacktestTrade 和 BacktestResult 接口

export interface BacktestTrade {
  stock_code: string
  stock_name: string
  entry_date: string
  exit_date: string
  entry_price: number
  exit_price: number
  shares: number
  cost: number
  proceeds: number
  pnl: number
  pnl_pct: number
  entry_commission: number
  exit_commission: number
  stamp_duty: number
  net_pnl: number
  entry_score: number
  exit_score: number
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
  total_commission: number
  total_stamp_duty: number
  trades: BacktestTrade[]
  daily_snapshots: { date: string; equity: number; cash: number }[]
}
```

- [ ] **Step 2: 重写 Results.vue**

完整文件：

```vue
<script setup lang="ts">
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import * as echarts from 'echarts'
import client from '@/api/client'
import type { StrategyConfig, BacktestResult, BacktestTrade } from '@/types/api'

const loading = ref(false)
const strategies = ref<StrategyConfig[]>([])
const selectedStrategy = ref('')
const dateRange = ref<[string, string]>(['2026-04-15', '2026-05-19'])
const result = ref<BacktestResult | null>(null)
const expandedTrade = ref<string | null>(null)

// Chart refs
const equityChartRef = ref<HTMLDivElement | null>(null)
const ddChartRef = ref<HTMLDivElement | null>(null)
const heatmapRef = ref<HTMLDivElement | null>(null)

let equityChart: echarts.ECharts | null = null
let ddChart: echarts.ECharts | null = null
let heatmapChart: echarts.ECharts | null = null
let themeObserver: MutationObserver | null = null

// Theme-aware chart init
function initChart(el: HTMLElement): echarts.ECharts {
  const isDark = document.documentElement.classList.contains('dark')
  return echarts.init(el, isDark ? 'dark' : undefined)
}

// Watch theme changes
function setupThemeWatch() {
  themeObserver = new MutationObserver(() => {
    ;[equityChart, ddChart, heatmapChart].forEach(c => c?.dispose())
    equityChart = null; ddChart = null; heatmapChart = null
    if (result.value) {
      nextTick(() => {
        renderAllCharts()
      })
    }
  })
  themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
}

onMounted(async () => {
  try {
    const res = await client.get('/backtest/strategies')
    strategies.value = res.data?.data || res.data || []
    if (strategies.value.length > 0) {
      selectedStrategy.value = strategies.value[0].name
    }
  } catch (e) {
    console.error('Failed to load strategies', e)
  }
  setupThemeWatch()
})

onUnmounted(() => {
  themeObserver?.disconnect()
  ;[equityChart, ddChart, heatmapChart].forEach(c => c?.dispose())
})

function toggleTrade(stockCode: string) {
  expandedTrade.value = expandedTrade.value === stockCode ? null : stockCode
}

async function runBacktest() {
  loading.value = true
  try {
    const res = await client.post('/backtest/run', {
      strategy_name: selectedStrategy.value,
      start_date: dateRange.value[0],
      end_date: dateRange.value[1],
    })
    result.value = res.data?.data || res.data
    ElMessage.success('回测完成')
    nextTick(() => renderAllCharts())
  } catch (e: any) {
    ElMessage.error(e?.response?.data?.detail || '回测失败')
  } finally {
    loading.value = false
  }
}

function renderAllCharts() {
  renderEquityChart()
  renderDrawdownChart()
  renderHeatmap()
}

function renderEquityChart() {
  if (!equityChartRef.value || !result.value?.daily_snapshots?.length) return
  if (!equityChart) equityChart = initChart(equityChartRef.value)
  const snaps = result.value.daily_snapshots
  const trades = result.value.trades
  equityChart.setOption({
    tooltip: { trigger: 'axis' },
    grid: { left: '3%', right: '8%', bottom: '8%', containLabel: true },
    xAxis: { type: 'category', data: snaps.map(s => s.date), axisLabel: { rotate: 45, fontSize: 10 } },
    yAxis: { type: 'value', name: '净值 (¥)', axisLabel: { formatter: (v: number) => Math.round(v).toString() } },
    series: [{
      name: '权益曲线', type: 'line', data: snaps.map(s => s.equity),
      smooth: true, lineStyle: { color: '#409EFF', width: 2 },
      areaStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: 'rgba(64,158,255,0.3)' },
          { offset: 1, color: 'rgba(64,158,255,0.02)' },
        ]),
      },
      markPoint: {
        data: [
          ...trades.filter(t => t.net_pnl > 0).slice(0, 10).map(t => ({
            name: t.stock_name, coord: [t.entry_date, t.entry_price], symbol: 'triangle', symbolSize: 10,
            itemStyle: { color: '#67c23a' }, label: { show: false },
          })),
          ...trades.filter(t => t.net_pnl <= 0).slice(0, 10).map(t => ({
            name: t.stock_name, coord: [t.exit_date, t.exit_price], symbol: 'triangle', symbolSize: 10, symbolRotate: 180,
            itemStyle: { color: '#f56c6c' }, label: { show: false },
          })),
        ],
      },
    }],
  }, true)
}

function renderDrawdownChart() {
  if (!ddChartRef.value || !result.value?.daily_snapshots?.length) return
  if (!ddChart) ddChart = initChart(ddChartRef.value)
  const snaps = result.value.daily_snapshots
  let peak = result.value.initial_capital
  const dds: number[] = []
  for (const s of snaps) {
    if (s.equity > peak) peak = s.equity
    dds.push(peak > 0 ? -((peak - s.equity) / peak * 100) : 0)
  }
  ddChart.setOption({
    tooltip: { trigger: 'axis', valueFormatter: (v: number) => `${v.toFixed(2)}%` },
    grid: { left: '3%', right: '4%', bottom: '8%', containLabel: true },
    xAxis: { type: 'category', data: snaps.map(s => s.date), axisLabel: { rotate: 45, fontSize: 10 } },
    yAxis: { type: 'value', name: '回撤 %', axisLabel: { formatter: '{value}%' } },
    series: [{
      name: '回撤', type: 'line', data: dds,
      smooth: true, lineStyle: { color: '#f56c6c', width: 1.5 },
      areaStyle: { color: 'rgba(245,108,108,0.15)' },
    }],
  }, true)
}

function renderHeatmap() {
  if (!heatmapRef.value || !result.value?.trades?.length) return
  if (!heatmapChart) heatmapChart = initChart(heatmapRef.value)
  // Build heatmap data: [dateIdx, stockIdx, pnl_pct]
  const trades = result.value.trades
  const stocks = [...new Set(trades.map(t => t.stock_code))]
  const dates = [...new Set(result.value.daily_snapshots.map(s => s.date))]
  const stockIdx = Object.fromEntries(stocks.map((s, i) => [s, i]))
  const dateIdx = Object.fromEntries(dates.map((d, i) => [d, i]))
  const data: [number, number, number][] = []
  for (const t of trades) {
    for (const d of getDateRange(t.entry_date, t.exit_date, dates)) {
      const di = dateIdx[d]
      const si = stockIdx[t.stock_code]
      if (di !== undefined && si !== undefined) {
        data.push([di, si, t.pnl_pct])
      }
    }
  }
  heatmapChart.setOption({
    tooltip: { formatter: (p: any) => `${stocks[p.data[1]]} ${dates[p.data[0]]}: ${(p.data[2]*100).toFixed(1)}%` },
    grid: { left: '10%', right: '5%', bottom: '8%', top: '3%' },
    xAxis: { type: 'category', data: dates, axisLabel: { rotate: 45, fontSize: 9 } },
    yAxis: { type: 'category', data: stocks, axisLabel: { fontSize: 11 } },
    visualMap: { min: -0.1, max: 0.1, calculable: true, orient: 'horizontal', left: 'center', bottom: 0,
      inRange: { color: ['#f56c6c', '#222', '#67c23a'] } },
    series: [{ type: 'heatmap', data, label: { show: false }, emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' } } }],
  }, true)
}

function getDateRange(start: string, end: string, allDates: string[]): string[] {
  const si = allDates.indexOf(start), ei = allDates.indexOf(end)
  return si >= 0 && ei >= 0 ? allDates.slice(si, ei + 1) : []
}

function pnlColor(v: number): string {
  return v > 0 ? '#67c23a' : v < 0 ? '#f56c6c' : '#909399'
}
</script>

<template>
  <div class="page-container">
    <h2 class="page-title">回测结果</h2>

    <el-card class="control-card" shadow="never">
      <div class="control-row">
        <el-select v-model="selectedStrategy" placeholder="选择策略" style="width: 240px">
          <el-option v-for="s in strategies" :key="s.name" :label="s.name" :value="s.name" />
        </el-select>
        <el-date-picker
          v-model="dateRange" type="daterange"
          range-separator="至" start-placeholder="开始日期" end-placeholder="结束日期"
          value-format="YYYY-MM-DD" style="width: 280px"
        />
        <el-button type="primary" :loading="loading" @click="runBacktest">
          {{ loading ? '回测中...' : '运行回测' }}
        </el-button>
      </div>
    </el-card>

    <div v-if="result" class="result-content">
      <!-- Stats row -->
      <el-row :gutter="14" class="stats-row">
        <el-col :span="4">
          <el-statistic title="总收益率" :value="result.total_return_pct" suffix="%" :precision="2">
            <template #prefix><span :style="{color: result.total_return_pct >= 0 ? '#67c23a' : '#f56c6c'}">{{ result.total_return_pct >= 0 ? '▲' : '▼' }}</span></template>
          </el-statistic>
        </el-col>
        <el-col :span="4">
          <el-statistic title="最大回撤" :value="result.max_drawdown_pct" suffix="%" :precision="2" />
        </el-col>
        <el-col :span="4">
          <el-statistic title="夏普比率" :value="result.sharpe_ratio" :precision="3" />
        </el-col>
        <el-col :span="3">
          <el-statistic title="胜率" :value="(result.win_rate * 100).toFixed(1)" suffix="%" />
        </el-col>
        <el-col :span="3">
          <el-statistic title="交易次数" :value="result.total_trades" />
        </el-col>
        <el-col :span="6">
          <div class="stat-mini-row">
            <div class="stat-mini">
              <span class="stat-mini-label">最终净值</span>
              <span class="stat-mini-value">¥{{ result.final_equity.toLocaleString() }}</span>
            </div>
            <div class="stat-mini">
              <span class="stat-mini-label">手续费</span>
              <span class="stat-mini-value" style="color:#e6a23c">¥{{ (result.total_commission || 0).toFixed(0) }}</span>
            </div>
            <div class="stat-mini">
              <span class="stat-mini-label">印花税</span>
              <span class="stat-mini-value" style="color:#e6a23c">¥{{ (result.total_stamp_duty || 0).toFixed(0) }}</span>
            </div>
          </div>
        </el-col>
      </el-row>

      <!-- Charts row -->
      <el-row :gutter="14" class="chart-row">
        <el-col :span="16">
          <el-card shadow="never"><div ref="equityChartRef" style="height:380px"></div></el-card>
        </el-col>
        <el-col :span="8">
          <el-card shadow="never"><div ref="ddChartRef" style="height:380px"></div></el-card>
        </el-col>
      </el-row>

      <!-- Trade table -->
      <el-card class="table-card" shadow="never">
        <template #header>
          <span>交易明细 ({{ result.total_trades }}笔)</span>
          <span style="margin-left:12px;font-size:12px;color:#909399">
            已实现净盈亏: <span :style="{color: pnlColor(result.total_return_pct),fontWeight:'bold'}">¥{{ (result.final_equity - result.initial_capital).toFixed(0) }}</span>
          </span>
        </template>
        <el-table :data="result.trades" stripe max-height="600" row-key="stock_code">
          <el-table-column type="expand">
            <template #default="{ row }: { row: BacktestTrade }">
              <div class="trade-expand">
                <el-row :gutter="16">
                  <el-col :span="8">
                    <div class="trade-detail-block">
                      <h4>买入明细</h4>
                      <div class="trade-detail-row"><span>开盘价</span><span>¥{{ row.entry_price.toFixed(2) }}</span></div>
                      <div class="trade-detail-row"><span>买入股数</span><span>{{ row.shares }} 股</span></div>
                      <div class="trade-detail-row"><span>买入金额</span><span>¥{{ row.cost.toFixed(2) }}</span></div>
                      <div class="trade-detail-row"><span>佣金</span><span style="color:#f56c6c">-¥{{ row.entry_commission?.toFixed(2) || '5.00' }}</span></div>
                      <div class="trade-detail-row trade-detail-total"><span>龙头得分</span><span style="color:#409EFF;font-weight:600">{{ row.entry_score?.toFixed(3) || '-' }}</span></div>
                    </div>
                  </el-col>
                  <el-col :span="8">
                    <div class="trade-detail-block">
                      <h4>卖出明细</h4>
                      <div class="trade-detail-row"><span>开盘价</span><span>¥{{ row.exit_price.toFixed(2) }}</span></div>
                      <div class="trade-detail-row"><span>卖出金额</span><span>¥{{ row.proceeds.toFixed(2) }}</span></div>
                      <div class="trade-detail-row"><span>佣金</span><span style="color:#f56c6c">-¥{{ row.exit_commission?.toFixed(2) || '5.00' }}</span></div>
                      <div class="trade-detail-row"><span>印花税</span><span style="color:#f56c6c">-¥{{ row.stamp_duty?.toFixed(2) || '0.00' }}</span></div>
                      <div class="trade-detail-row"><span>卖出得分</span><span :style="{color: (row.exit_score||0) >= (row.entry_score||0) ? '#67c23a' : '#f56c6c',fontWeight:'600'}">{{ row.exit_score?.toFixed(3) || '-' }}</span></div>
                      <div class="trade-detail-row trade-detail-total"><span>净盈亏</span><span :style="{color: pnlColor(row.net_pnl),fontWeight:'700',fontSize:'15px'}">¥{{ row.net_pnl?.toFixed(2) || row.pnl.toFixed(2) }}</span></div>
                    </div>
                  </el-col>
                  <el-col :span="8">
                    <div class="trade-detail-block">
                      <h4>得分对比</h4>
                      <div style="display:flex;gap:24px;margin-bottom:8px">
                        <div><span style="color:#909399;font-size:11px">买入</span><br><span style="color:#409EFF;font-size:18px;font-weight:700">{{ (row.entry_score || 0).toFixed(3) }}</span></div>
                        <div><span style="color:#909399;font-size:11px">卖出</span><br><span style="font-size:18px;font-weight:700" :style="{color: (row.exit_score||0) >= (row.entry_score||0) ? '#67c23a' : '#f56c6c'}">{{ (row.exit_score || 0).toFixed(3) }}</span></div>
                        <div><span style="color:#909399;font-size:11px">变化</span><br><span style="font-size:18px;font-weight:700" :style="{color: ((row.exit_score||0) - (row.entry_score||0)) >= 0 ? '#67c23a' : '#f56c6c'}">{{ ((row.exit_score || 0) - (row.entry_score || 0) >= 0 ? '+' : '') + ((row.exit_score||0) - (row.entry_score||0)).toFixed(3) }}</span></div>
                      </div>
                    </div>
                  </el-col>
                </el-row>
              </div>
            </template>
          </el-table-column>
          <el-table-column prop="stock_code" label="代码" width="90" />
          <el-table-column prop="stock_name" label="名称" width="110" />
          <el-table-column label="日期" width="170">
            <template #default="{ row }">{{ row.entry_date }} → {{ row.exit_date }}</template>
          </el-table-column>
          <el-table-column label="价格" width="150">
            <template #default="{ row }">
              <span style="color:#909399">¥{{ row.entry_price.toFixed(2) }}</span>
              <span style="margin:0 4px;color:#666">→</span>
              <span :style="{color: row.exit_price >= row.entry_price ? '#67c23a' : '#f56c6c'}">¥{{ row.exit_price.toFixed(2) }}</span>
            </template>
          </el-table-column>
          <el-table-column label="净盈亏" width="110" align="right">
            <template #default="{ row }">
              <span :style="{color: pnlColor(row.net_pnl || row.pnl), fontWeight: 'bold'}">
                ¥{{ (row.net_pnl || row.pnl).toFixed(0) }}
              </span>
            </template>
          </el-table-column>
          <el-table-column label="收益率" width="90" align="right">
            <template #default="{ row }">
              <span :style="{color: pnlColor(row.pnl_pct), fontWeight: 'bold'}">
                {{ (row.pnl_pct * 100).toFixed(2) }}%
              </span>
            </template>
          </el-table-column>
          <el-table-column prop="holding_days" label="天数" width="55" align="center" />
          <el-table-column prop="exit_reason" label="卖出原因" min-width="180">
            <template #default="{ row }">
              <el-tag size="small" :type="row.exit_reason.includes('止损') ? 'danger' : row.exit_reason.includes('涨停') ? 'success' : 'warning'">
                {{ row.exit_reason }}
              </el-tag>
            </template>
          </el-table-column>
        </el-table>
      </el-card>

      <!-- Heatmap -->
      <el-card class="chart-card" shadow="never">
        <template #header>持仓热力图</template>
        <div ref="heatmapRef" style="height:300px"></div>
      </el-card>
    </div>

    <el-empty v-else description="选择策略和日期范围，点击「运行回测」开始" />
  </div>
</template>

<style lang="scss" scoped>
.page-container { padding: 24px; }
.page-title { margin: 0 0 20px; font-size: 20px; font-weight: 600; }
.control-card { margin-bottom: 20px; }
.control-row { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
.stats-row { margin-bottom: 20px; }
.chart-row { margin-bottom: 20px; }
.table-card { margin-bottom: 20px; }
.chart-card { margin-bottom: 20px; }
.result-content { animation: fadeIn .3s ease; }
.stat-mini-row { display: flex; gap: 16px; }
.stat-mini { text-align: center; flex: 1; }
.stat-mini-label { display: block; font-size: 12px; color: #909399; margin-bottom: 4px; }
.stat-mini-value { font-size: 14px; font-weight: 600; }
.trade-expand { padding: 12px 0; }
.trade-detail-block {
  h4 { margin: 0 0 8px; font-size: 13px; color: #909399; text-transform: uppercase; }
}
.trade-detail-row {
  display: flex; justify-content: space-between; padding: 3px 0; font-size: 13px;
  span:first-child { color: #909399; }
}
.trade-detail-total { border-top: 1px solid var(--el-border-color-lighter); margin-top: 4px; padding-top: 6px; }
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
```

- [ ] **Step 3: 验证前端编译**

Run: `cd dragon-engine-web && npm run build 2>&1 | tail -5`

Expected: Build succeeds with no errors.

---

## Wave 4 — 联调验证

### Task 9: E2E 验证

**Files:** 无新建，验证整个流程。

- [ ] **Step 1: 运行一次 enhanced backfill 写入真实价格数据**

Run: `cd D:\K\dragon-engine && D:/K/dragon-engine/venv/Scripts/python.exe -c "
import asyncio, sys; sys.path.insert(0, '.')
from run_enhanced_backfill import run_one_day
asyncio.run(run_one_day('2026-05-16'))
"`

Expected: 日志显示 `[ENHANCED] 2026-05-16: N daily bars upserted`。

- [ ] **Step 2: 测试回测 API 返回真实 P&L**

Run: Start backend server, then:
`curl -s -X POST http://localhost:8000/backtest/run -H 'Content-Type: application/json' -d '{"strategy_name":"策略A-每日固定资金","start_date":"2026-04-20","end_date":"2026-05-19"}' | python -c "import sys,json; d=json.load(sys.stdin); print(f\"Return: {d['total_return_pct']}% Trades: {d['total_trades']} Commission: {d.get('total_commission',0)} NetPNL sample: {d['trades'][0]['net_pnl'] if d['trades'] else 'none'}\")"`

Expected: Return != 0.0%, net_pnl != 0.

- [ ] **Step 3: 启动前端并打开回测页面验证**

Start Vite dev server, navigate to `/backtest/results`, run backtest, verify:
- ECharts charts render
- Expandable trade rows show commission/stamp/net_pnl/entry_score/exit_score
- Heatmap renders
- Dark/light theme switching works

---

## 执行顺序

```
Wave 1 (并行 4 agents):
  Agent A: Task 1 — Token tracking bug fix
  Agent B: Task 2 — stock_daily_bars table + ORM  
  Agent C: Task 3 — TradingStrategy new fields
  Agent D: Task 4 — Backtest model upgrades
  ↓
Wave 2 (并行 2 agents, after all Wave 1 done):
  Agent E: Task 5 — Engine upgrade (prices + commissions + rules)
  Agent F: Task 6 — Batch daily bars write
  ↓
Wave 3 (after Wave 2 done):
  Agent G: Task 7 — API endpoint upgrade (prices injection)
  Agent H: Task 8 — Frontend types + Results.vue rewrite
  ↓
Wave 4 (after Wave 3 done):
  Task 9 — E2E verification
```
