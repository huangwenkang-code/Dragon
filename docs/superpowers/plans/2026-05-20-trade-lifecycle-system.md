# Trade Lifecycle System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将回测引擎从"每日选股器"升级为"龙头生命周期管理系统"——每个持仓有状态机驱动的 TradeEpisode，不在候选池也有续航/退潮评分，受 Market Regime 动态调节。

**Architecture:** 四个新模块（episode, holding_scorer, market_regime, 新 EntryRule）插入 engine._process_day 的 8-step 流程。所有新模块通过 `stock_daily_bars` 获取 OHLCV 数据，不依赖 pipeline 上游。旧的得分卖出规则（ScoreCliffRule/TrailingStopRule/ScoreDeclineRule）废除。

**Tech Stack:** Python 3.12, dataclasses, PostgreSQL (stock_daily_bars), 现有 pipeline 不改

---

### Task 1: TradeEpisode + EpisodeRecord 数据模型

**Files:**
- Create: `services/backtest/episode.py`

- [ ] **Step 1: 创建 episode.py 及两个 dataclass**

```python
"""TradeEpisode — 一只票从发现到退出的完整生命周期."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

# State constants
STATE_DISCOVERED = "DISCOVERED"
STATE_HOLDING = "HOLDING"
STATE_ACCELERATING = "ACCELERATING"
STATE_DISTRIBUTING = "DISTRIBUTING"
STATE_DECAYING = "DECAYING"
STATE_EXITED = "EXITED"

VALID_STATES = {
    STATE_DISCOVERED, STATE_HOLDING, STATE_ACCELERATING,
    STATE_DISTRIBUTING, STATE_DECAYING, STATE_EXITED,
}

# Valid state transitions
VALID_TRANSITIONS: dict[str, set[str]] = {
    STATE_DISCOVERED:    {STATE_HOLDING, STATE_EXITED},
    STATE_HOLDING:       {STATE_ACCELERATING, STATE_DECAYING, STATE_DISTRIBUTING, STATE_EXITED},
    STATE_ACCELERATING:  {STATE_HOLDING, STATE_DISTRIBUTING, STATE_DECAYING, STATE_EXITED},
    STATE_DISTRIBUTING:  {STATE_DECAYING, STATE_EXITED},
    STATE_DECAYING:      {STATE_HOLDING, STATE_EXITED},
    STATE_EXITED:        set(),
}


@dataclass
class EpisodeRecord:
    """每日一条记录 — 替代旧 Position.score_history."""
    date: date
    state: str                          # 当日状态
    price: float                        # 当日价格(开盘价)
    full_score: float | None = None     # 候选池完整分(在池才有)
    continuation_score: float = 0.0     # 续航分(始终计算)
    decay_score: float = 0.0            # 退潮分(始终计算)
    volume: float = 0.0                 # 成交量
    pnl_pct: float = 0.0                # 浮动盈亏%
    is_in_candidate_pool: bool = False  # 当天是否在候选池


@dataclass
class TradeEpisode:
    """一只票从发现到退出的完整生命周期."""
    episode_id: str             # "EP20260520_600001"
    stock_code: str
    stock_name: str
    state: str                  # 当前状态
    entry_date: date
    entry_price: float
    entry_score: float          # 买入时的 leader_score
    shares: int = 0
    cost: float = 0.0

    daily_records: list[EpisodeRecord] = field(default_factory=list)
    peak_price: float = 0.0
    peak_score: float = 0.0
    max_floating_pnl: float = 0.0   # 历史最大浮盈(¥)
    max_drawdown: float = 0.0       # 历史最大回撤(%)

    exit_reason: str | None = None
    exit_date: date | None = None
    entry_reason: list[str] = field(default_factory=list)

    def add_record(self, record: EpisodeRecord):
        """追加一条日记录，同时更新 peak/max 字段."""
        self.daily_records.append(record)
        if record.price > self.peak_price:
            self.peak_price = record.price
        if record.continuation_score > self.peak_score:
            self.peak_score = record.continuation_score
        if self.cost > 0:
            floating = (record.price - self.entry_price) / self.entry_price
            if floating > self.max_floating_pnl:
                self.max_floating_pnl = floating
            dd = (self.peak_price - record.price) / self.peak_price if self.peak_price > 0 else 0.0
            if dd > self.max_drawdown:
                self.max_drawdown = dd

    def transition_to(self, new_state: str) -> bool:
        """尝试状态流转，成功返回 True."""
        if new_state in VALID_TRANSITIONS.get(self.state, set()):
            self.state = new_state
            return True
        return False
```

- [ ] **Step 2: 验证导入**

```bash
cd D:/K/dragon-engine && python -c "from services.backtest.episode import TradeEpisode, EpisodeRecord, VALID_STATES, VALID_TRANSITIONS; print('OK')"
```

Expected: `OK`

---

### Task 2: 状态机流转逻辑

**Files:**
- Modify: `services/backtest/episode.py` (追加)

- [ ] **Step 1: 追加 state_machine 模块到 episode.py**

```python
# 追加到 episode.py 末尾

def run_state_machine(
    episode: TradeEpisode,
    regime: str,          # MarketRegime.regime
    lookback_bars: list[dict],  # 近 20 天 OHLCV bars
) -> str:
    """评估 episode 当前状态，返回新状态（可能不变）。
    
    所有流转规则基于 spec 定义。
    """
    current = episode.state
    if current == STATE_EXITED:
        return STATE_EXITED

    if len(episode.daily_records) < 2:
        return current  # 第一天不做流转判断

    # 获取最近两天的续航/退潮分
    recent = episode.daily_records[-2:]
    cont_now = recent[-1].continuation_score
    decay_now = recent[-1].decay_score
    cont_prev = recent[0].continuation_score
    decay_prev = recent[0].decay_score

    # Regime 调节：DECAYING 触发天数
    # LIANBAN/ICE → 1天，其他 → 2天
    decay_trigger_days = 1 if regime in ("LIANBAN", "ICE") else 2

    # --- HOLDING 流转 ---
    if current == STATE_HOLDING:
        if decay_now > cont_now:
            # 检查持续天数
            if decay_trigger_days == 1:
                return STATE_DECAYING
            elif decay_prev > cont_prev:
                return STATE_DECAYING
        if _is_accelerating(episode, lookback_bars):
            return STATE_ACCELERATING

    # --- ACCELERATING 流转 ---
    elif current == STATE_ACCELERATING:
        # 日跌幅 > 5% → 分歧
        if len(lookback_bars) >= 2:
            today_close = lookback_bars[-1].get("close", 0)
            prev_close = lookback_bars[-2].get("close", 0)
            if prev_close > 0:
                day_change = (today_close - prev_close) / prev_close
                if day_change < -0.05:
                    return STATE_DISTRIBUTING
        # 高开低走 > 3% → 分歧
        if len(lookback_bars) >= 1:
            bar = lookback_bars[-1]
            open_p = bar.get("open", 0)
            close_p = bar.get("close", 0)
            if open_p > 0 and (open_p - close_p) / open_p > 0.03:
                return STATE_DISTRIBUTING

    # --- DISTRIBUTING 流转 ---
    elif current == STATE_DISTRIBUTING:
        if decay_now > cont_now:
            if decay_trigger_days == 1:
                return STATE_DECAYING
            elif decay_prev > cont_prev:
                return STATE_DECAYING

    # --- DECAYING 流转 (恢复) ---
    elif current == STATE_DECAYING:
        if cont_now > decay_now and _price_new_high(lookback_bars):
            return STATE_HOLDING

    return current  # 无变化


def _is_accelerating(episode: TradeEpisode, lookback_bars: list[dict]) -> bool:
    """判断是否加速：连续 3 天价格创新高 + 成交量放大."""
    records = episode.daily_records[-3:]
    bars = lookback_bars[-3:] if len(lookback_bars) >= 3 else lookback_bars
    if len(records) < 3 or len(bars) < 3:
        return False
    # 价格连续 3 天创新高
    prices = [r.price for r in records]
    if not (prices[0] < prices[1] < prices[2]):
        return False
    # 成交量放大 (今天 > 5日均量)
    avg_vol = sum(b.get("volume", 0) for b in bars) / len(bars) if bars else 0
    today_vol = bars[-1].get("volume", 0)
    return today_vol > avg_vol


def _price_new_high(lookback_bars: list[dict]) -> bool:
    """今日收盘是否创近 5 日新高."""
    if len(lookback_bars) < 5:
        return True
    today = lookback_bars[-1].get("close", 0)
    prev = [b.get("close", 0) for b in lookback_bars[-6:-1]]
    return today > max(prev)
```

- [ ] **Step 2: 验证导入**

```bash
cd D:/K/dragon-engine && python -c "from services.backtest.episode import run_state_machine, _is_accelerating, _price_new_high; print('OK')"```
Expected: `OK`

---

### Task 3: Holding Scorer — continuation + decay

**Files:**
- Create: `services/backtest/holding_scorer.py`

- [ ] **Step 1: 创建 holding_scorer.py**

```python
"""Holding scorer — 计算 continuation_score 和 decay_score.

纯价格驱动，不需要 pipeline 上游数据。
所有计算基于 stock_daily_bars (OHLCV) + 大盘指数数据。
"""


def compute_continuation(
    stock_bars: list[dict],     # 近 20 天 OHLCV bars
    index_bars: list[dict],    # 同期大盘指数 bars
    regime: str = "NORMAL",
) -> float:
    """计算续航分 (0-1)。5 个信号等权。"""
    if not stock_bars or len(stock_bars) < 5:
        return 0.5

    signals = [
        _relative_strength(stock_bars, index_bars),      # 25%
        _price_vs_ma5(stock_bars),                        # 20%
        _consecutive_up_days(stock_bars),                  # 15%
        _volume_trend(stock_bars),                         # 20%
        _new_high_frequency(stock_bars),                   # 20%
    ]
    return sum(s * w for s, w in zip(signals, [0.25, 0.20, 0.15, 0.20, 0.20]))


def compute_decay(
    stock_bars: list[dict],
    regime: str = "NORMAL",
) -> float:
    """计算退潮分 (0-1)。5 个信号等权。"""
    if not stock_bars or len(stock_bars) < 5:
        return 0.5

    signals = [
        _gap_up_selloff(stock_bars),        # 25%
        _volume_stagnation(stock_bars),     # 20%
        _consecutive_down_days(stock_bars), # 15%
        _pullback_from_high(stock_bars),    # 25%
        _ma_death_cross(stock_bars),        # 15%
    ]
    return sum(s * w for s, w in zip(signals, [0.25, 0.20, 0.15, 0.25, 0.15]))


# ── Continuation 子信号 ──

def _relative_strength(stock_bars, index_bars):
    """相对强度：股票涨幅 vs 大盘涨幅。跑赢→加分."""
    if len(stock_bars) < 2 or len(index_bars) < 2:
        return 0.5
    stock_chg = (stock_bars[-1]["close"] - stock_bars[-2]["close"]) / max(stock_bars[-2]["close"], 0.01)
    index_chg = (index_bars[-1]["close"] - index_bars[-2]["close"]) / max(index_bars[-2]["close"], 0.01)
    diff = stock_chg - index_chg
    return min(1.0, max(0.0, 0.5 + diff * 10))  # ±5% → full range


def _price_vs_ma5(stock_bars):
    """价格 vs 5日均线。均线以上→加分."""
    closes = [b["close"] for b in stock_bars[-5:]]
    if not closes:
        return 0.5
    ma5 = sum(closes) / len(closes)
    if ma5 <= 0:
        return 0.5
    ratio = closes[-1] / ma5
    return min(1.0, max(0.0, ratio - 0.9))  # 1.0=均线, 1.1+=满分


def _consecutive_up_days(stock_bars):
    """连涨天数 / 5."""
    recent = stock_bars[-5:]
    count = 0
    for i in range(1, len(recent)):
        if recent[i]["close"] > recent[i-1]["close"]:
            count += 1
    return count / 4 if len(recent) >= 5 else count / max(len(recent)-1, 1)


def _volume_trend(stock_bars):
    """成交量趋势：当日量/5日均量，1.0-1.5x最优."""
    vols = [b.get("volume", 0) for b in stock_bars[-6:]]
    if not vols:
        return 0.5
    avg5 = sum(vols[:5]) / 5 if len(vols) >= 5 else sum(vols) / len(vols)
    if avg5 <= 0:
        return 0.5
    ratio = vols[-1] / avg5
    if 1.0 <= ratio <= 1.5:
        return 1.0
    elif ratio < 0.5:
        return 0.2  # 缩量严重
    elif ratio > 3.0:
        return 0.3  # 放量过度
    else:
        return 0.6


def _new_high_frequency(stock_bars):
    """近 5 天创阶段新高天数 / 5."""
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


# ── Decay 子信号 ──

def _gap_up_selloff(stock_bars):
    """高开低走程度。>2% 开始计分."""
    bar = stock_bars[-1]
    open_p = bar.get("open", 0)
    close_p = bar.get("close", 0)
    if open_p <= 0:
        return 0.0
    ratio = (open_p - close_p) / open_p
    return min(1.0, max(0.0, ratio / 0.05))  # 5% gap-up selloff = 1.0


def _volume_stagnation(stock_bars):
    """放量滞涨：量增 > 3% + 涨跌 < 1%."""
    if len(stock_bars) < 2:
        return 0.0
    vol_now = stock_bars[-1].get("volume", 0)
    vol_prev = stock_bars[-2].get("volume", 0)
    price_now = stock_bars[-1].get("close", 0)
    price_prev = stock_bars[-2].get("close", 0)
    if vol_prev <= 0 or price_prev <= 0:
        return 0.0
    vol_chg = (vol_now - vol_prev) / vol_prev
    price_chg = abs(price_now - price_prev) / price_prev
    if vol_chg > 0.03 and price_chg < 0.01:
        return 0.8  # 典型滞涨
    elif vol_chg > 0.03 and price_chg < 0.02:
        return 0.4
    return 0.0


def _consecutive_down_days(stock_bars):
    """连跌天数 / 5."""
    recent = stock_bars[-5:]
    count = 0
    for i in range(1, len(recent)):
        if recent[i]["close"] < recent[i-1]["close"]:
            count += 1
    return count / 4 if len(recent) >= 5 else count / max(len(recent)-1, 1)


def _pullback_from_high(stock_bars):
    """距近 5 日高点回撤比例."""
    closes = [b["close"] for b in stock_bars[-5:]]
    high5 = max(closes)
    if high5 <= 0:
        return 0.0
    pullback = (high5 - closes[-1]) / high5
    return min(1.0, pullback / 0.10)  # 10% pullback = 1.0


def _ma_death_cross(stock_bars):
    """5日线是否下穿 10 日线."""
    closes = [b["close"] for b in stock_bars]
    if len(closes) < 10:
        return 0.0
    ma5_prev = sum(closes[-6:-1]) / 5
    ma10_prev = sum(closes[-11:-1]) / 10
    ma5_now = sum(closes[-5:]) / 5
    ma10_now = sum(closes[-10:]) / 10
    if ma5_prev > ma10_prev and ma5_now < ma10_now:
        return 0.8  # 死叉
    elif ma5_now < ma10_now:
        return 0.3  # 均线已空头
    return 0.0
```

- [ ] **Step 2: 快速单元验证（无需 pytest，直接 import）**

```bash
cd D:/K/dragon-engine && python -c "
from services.backtest.holding_scorer import compute_continuation, compute_decay
bars = [{'open': 10, 'close': 10+i, 'high': 11, 'low': 9, 'volume': 1e6} for i in range(10)]
cont = compute_continuation(bars, bars, 'NORMAL')
decay = compute_decay(bars, 'NORMAL')
print(f'cont={cont:.3f} decay={decay:.3f}')
# 上升趋势 → cont 高, decay 低
assert cont > 0.5, f'Expected cont>0.5 got {cont}'
assert decay < 0.5, f'Expected decay<0.5 got {decay}'
print('PASS')
"
```

Expected: `cont>0.5 decay<0.5 PASS`

---

### Task 4: Market Regime Service

**Files:**
- Create: `services/backtest/market_regime.py`

- [ ] **Step 1: 创建 market_regime.py**

```python
"""Market Regime Service — 判定市场周期并输出动态参数."""
from dataclasses import dataclass


REGIME_LIANBAN = "LIANBAN"
REGIME_MONSTER = "MONSTER"
REGIME_AI_TREND = "AI_TREND"
REGIME_ICE = "ICE"
REGIME_NORMAL = "NORMAL"


@dataclass
class MarketRegime:
    temperature: float        # 0-1 市场热度
    speculation_index: float  # 0-1 投机指数
    regime: str               # 当前周期
    volatility: float         # 最近5日涨跌幅标准差
    breadth: float            # 市场宽度 (上涨家数占比)


@dataclass
class RegimeParams:
    """Regime 调节参数."""
    cont_weight: float          # continuation 权重
    decay_weight: float         # decay 权重
    price_stop_pct: float       # 价格止损回撤阈值
    decay_trigger_days: int     # DECAYING 触发天数
    gap_up_pct: float           # 高开买入过滤阈值
    time_stop_days: int         # 时间止损天数
    min_cont_threshold: float   # 极端衰退阈值


# Regime → 参数映射表
REGIME_PARAMS: dict[str, RegimeParams] = {
    REGIME_LIANBAN:    RegimeParams(0.6, 0.4, 0.05, 1, 0.03, 10, 0.3),
    REGIME_MONSTER:    RegimeParams(0.5, 0.5, 0.07, 2, 0.04, 15, 0.3),
    REGIME_AI_TREND:   RegimeParams(0.7, 0.3, 0.07, 2, 0.04, 15, 0.3),
    REGIME_ICE:        RegimeParams(0.3, 0.7, 0.03, 1, 0.02, 15, 0.2),
    REGIME_NORMAL:     RegimeParams(0.5, 0.5, 0.07, 2, 0.04, 15, 0.3),
}


def determine_regime(
    limit_up_count: int,         # 当日涨停家数
    total_stocks: int,           # 全市场股票数
    speculation_index: float,    # 投机指数(妖股活跃度)
    sector_concentration: float, # 板块集中度(单板块成交占比)
    breadth: float,              # 市场宽度(上涨/总数)
    volatility: float,           # 近5日波动率
    lianban_height: int,         # 最高连板高度
    consecutive_days: int = 1,   # 当前条件持续天数
) -> MarketRegime:
    """判定当前市场周期。
    
    输入值可从 stock_daily_bars + 候选池数据提取。
    """
    temperature = min(1.0, limit_up_count / max(total_stocks, 1) * 10)

    # LIANBAN: 连板高度≥5 + 涨停>80 + 投机>0.7
    if lianban_height >= 5 and limit_up_count > 80 and speculation_index > 0.7:
        return MarketRegime(temperature, speculation_index, REGIME_LIANBAN, volatility, breadth)

    # MONSTER: 投机>0.6 + 连板≥7
    if speculation_index > 0.6 and lianban_height >= 7:
        return MarketRegime(temperature, speculation_index, REGIME_MONSTER, volatility, breadth)

    # AI_TREND: 板块集中度>0.5 + 持续>5天
    if sector_concentration > 0.5 and consecutive_days > 5:
        return MarketRegime(temperature, speculation_index, REGIME_AI_TREND, volatility, breadth)

    # ICE: 涨停<30 + 宽度<0.3
    if limit_up_count < 30 and breadth < 0.3:
        return MarketRegime(temperature, speculation_index, REGIME_ICE, volatility, breadth)

    return MarketRegime(temperature, speculation_index, REGIME_NORMAL, volatility, breadth)


def compute_speculation_index(
    daily_open_prices: dict[str, float],    # symbol → open_price
    prev_close_prices: dict[str, float],    # symbol → prev_close
    sector_volume: dict[str, float],        # 各板块成交额
    total_market_volume: float,             # 全市场成交额
) -> float:
    """计算投机指数 0-1。

    信号：
    - 涨停占比（接近涨停的股票比例）
    - 板块成交集中度（单一板块超越比例）
    """
    if not daily_open_prices:
        return 0.5

    # 接近涨停(>7%)的占比
    near_limit_up = 0
    for sym in daily_open_prices:
        if sym in prev_close_prices and prev_close_prices[sym] > 0:
            gap = (daily_open_prices[sym] - prev_close_prices[sym]) / prev_close_prices[sym]
            if gap > 0.07:
                near_limit_up += 1
    gap_score = min(1.0, near_limit_up / max(len(daily_open_prices), 1) * 10)

    # 板块集中度
    max_sector_vol = max(sector_volume.values()) if sector_volume else 0
    concentration = max_sector_vol / max(total_market_volume, 1) if total_market_volume > 0 else 0
    concentration_score = min(1.0, concentration * 2)

    return (gap_score * 0.5 + concentration_score * 0.5)


def get_params(regime: str) -> RegimeParams:
    """获取指定 regime 的参数表."""
    return REGIME_PARAMS.get(regime, REGIME_PARAMS[REGIME_NORMAL])
```

- [ ] **Step 2: 验证导入 + 基础逻辑**

```bash
cd D:/K/dragon-engine && python -c "
from services.backtest.market_regime import determine_regime, get_params, REGIME_PARAMS
# Test LIANBAN detection
r = determine_regime(100, 5000, 0.8, 0.3, 0.6, 0.03, 6)
print(f'Regime: {r.regime}')
assert r.regime == 'LIANBAN', f'Expected LIANBAN, got {r.regime}'
# Test NORMAL
r2 = determine_regime(50, 5000, 0.4, 0.2, 0.5, 0.02, 3)
print(f'Regime: {r2.regime}')
assert r2.regime == 'NORMAL', f'Expected NORMAL, got {r2.regime}'
# Test params
p = get_params('ICE')
print(f'ICE stop: {p.price_stop_pct}')
assert p.price_stop_pct == 0.03
print('PASS')
"
```

Expected: `Regime: LIANBAN | Regime: NORMAL | ICE stop: 0.03 | PASS`

---

### Task 5: 新 EntryRule + 废除旧 ExitRule

**Files:**
- Modify: `services/backtest/rules.py`

- [ ] **Step 1: 追加 OneDaySpikeFilter 和 VolumeSurgeFilter**

在 `STFilterRule` 类之后追加：

```python
class OneDaySpikeFilter(EntryRule):
    """过滤一日游：昨日近涨停(>9.5%) + 今日高开(>3%) → 疑似出货。

    引擎需在 context 中提供:
      - context["prev_day_change"] — dict[symbol → float] 昨日涨跌幅
      - context["prices"] — dict[symbol → float] 今日开盘价
      - context["prev_close"] — dict[symbol → float] 昨日收盘价
      - context["sector_volume_pct"] — dict[symbol → float] 板块成交占比(可选, 不传则跳过)
    """
    rule_type = "one_day_spike"

    def __init__(self):
        self.params = {}

    def should_enter(self, candidate: dict, context: dict) -> bool:
        code = candidate.get("stock_code", "")

        prev_day_change = context.get("prev_day_change", {}).get(code)
        if prev_day_change is None:
            return True  # 无数据，放行

        # 昨日涨幅 > 9.5% (接近涨停)
        if prev_day_change <= 0.095:
            return True

        # 今日高开 > 3%
        prev_close = context.get("prev_close", {}).get(code, 0)
        open_price = context.get("prices", {}).get(code, 0)
        if prev_close > 0 and open_price > 0:
            gap = (open_price - prev_close) / prev_close
            if gap <= 0.03:
                return True
        else:
            return True  # 无价格数据，放行

        # 板块成交集中度检查(可选，有数据才过滤)
        sector_pct = context.get("sector_volume_pct", {}).get(code)
        if sector_pct is not None and sector_pct < 0.20:
            return True  # 板块成交不集中，不触发

        return False  # 疑似一日游出货


class VolumeSurgeFilter(EntryRule):
    """过滤异常放量：当日量 > 20日均量 3倍 且 换手率 > 15%。

    引擎需在 context 中提供:
      - context["avg_volume_20"] — dict[symbol → float] 20日均量
      - context["turnover_pct"] — dict[symbol → float] 换手率
    """
    rule_type = "volume_surge"

    def __init__(self, max_vol_ratio: float = 3.0, max_turnover: float = 0.15):
        self.params = {"max_vol_ratio": max_vol_ratio, "max_turnover": max_turnover}

    def should_enter(self, candidate: dict, context: dict) -> bool:
        code = candidate.get("stock_code", "")

        avg_vol20 = context.get("avg_volume_20", {}).get(code)
        turnover = context.get("turnover_pct", {}).get(code)

        if avg_vol20 is None or turnover is None:
            return True  # 无数据，放行

        today_vol = context.get("today_volume", {}).get(code, 0)
        if today_vol > 0 and avg_vol20 > 0:
            ratio = today_vol / avg_vol20
            if ratio > self.params["max_vol_ratio"] and turnover > self.params["max_turnover"]:
                return False

        return True
```

- [ ] **Step 2: 追加到 ENTRY_RULES registry**

```python
# 在现有 ENTRY_RULES dict 内追加两条:
ENTRY_RULES: dict[str, type[EntryRule]] = {
    "score_threshold": ScoreThresholdRule,
    "no_filter": NoFilterRule,
    "limit_up_filter": LimitUpFilterRule,
    "gap_up_filter": GapUpFilterRule,
    "st_filter": STFilterRule,
    "one_day_spike": OneDaySpikeFilter,      # NEW
    "volume_surge": VolumeSurgeFilter,       # NEW
}
```

- [ ] **Step 3: 标记废除的旧退出规则（保留类定义以向后兼容，但从 registry 移除）**

修改 `EXIT_RULES` registry，移除 `score_cliff`, `trailing_stop`, `score_decline`:

```python
EXIT_RULES: dict[str, type[ExitRule]] = {
    # "score_cliff": ScoreCliffRule,           # DEPRECATED → exit decision matrix
    # "trailing_stop": TrailingStopRule,       # DEPRECATED → decay>cont 持续N天
    # "score_decline": ScoreDeclineRule,       # DEPRECATED → state machine DECAYING
    "price_trailing_stop": PriceTrailingStopRule,  # KEPT — 受 regime 调节阈值
}
```

- [ ] **Step 4: 验证**

```bash
cd D:/K/dragon-engine && python -c "
from services.backtest.rules import ENTRY_RULES, EXIT_RULES
print('ENTRY:', list(ENTRY_RULES.keys()))
print('EXIT:', list(EXIT_RULES.keys()))
assert 'one_day_spike' in ENTRY_RULES
assert 'volume_surge' in ENTRY_RULES
assert 'score_cliff' not in EXIT_RULES
assert 'price_trailing_stop' in EXIT_RULES
print('PASS')
"
```

Expected: `ENTRY: [7 keys] | EXIT: ['price_trailing_stop'] | PASS`

---

### Task 6: 更新策略配置

**Files:**
- Modify: `services/backtest/strategies.py`

- [ ] **Step 1: 更新 import（移除废除的规则引用，添加新引用）**

```python
from services.backtest.rules import (
    EntryRule, ExitRule, PositionAllocator,
    NoFilterRule,
    PriceTrailingStopRule,
    EqualWeightAllocator, STFilterRule,
    OneDaySpikeFilter, VolumeSurgeFilter, GapUpFilterRule,
    ENTRY_RULES, EXIT_RULES, ALLOCATORS,
)
```

- [ ] **Step 2: 更新 STRATEGY_A — 新的 entry/exit rules**

替换 STRATEGY_A 的 entry_rules 和 exit_rules：

```python
STRATEGY_A = TradingStrategy(
    name="策略A-每日固定资金",
    description="资金100w，每只2.5万等权，生命周期管理，7%价格止损",
    entry_rules=[
        STFilterRule(),
        OneDaySpikeFilter(),
        VolumeSurgeFilter(),
        GapUpFilterRule(0.04),   # 高开>4% 不买 (受 regime 动态调节)
        NoFilterRule(),
    ],
    exit_rules=[
        PriceTrailingStopRule(0.07),  # 价格止损 (阈值受 regime 动态调节)
    ],
    allocator=EqualWeightAllocator(),
    max_positions=999,
    max_position_pct=1.0,
    initial_capital=1_000_000,
    daily_cash_pct=0.025,
    commission_rate=0.00025,
    stamp_duty_rate=0.0005,
    min_commission=5.0,
    gap_up_pct=0.04,  # 启用高开过滤
    enable_limit_up_filter=True,
    is_system=True,
)
```

- [ ] **Step 3: 更新 STRATEGY_B — 同 A**

```python
STRATEGY_B = TradingStrategy(
    name="策略B-仓位上限管理",
    description="资金100w，每只2.5万等权，生命周期管理，7%价格止损",
    entry_rules=[
        STFilterRule(),
        OneDaySpikeFilter(),
        VolumeSurgeFilter(),
        GapUpFilterRule(0.04),
        NoFilterRule(),
    ],
    exit_rules=[
        PriceTrailingStopRule(0.07),
    ],
    allocator=EqualWeightAllocator(),
    max_positions=999,
    max_position_pct=1.0,
    initial_capital=1_000_000,
    daily_cash_pct=0.025,
    commission_rate=0.00025,
    stamp_duty_rate=0.0005,
    min_commission=5.0,
    gap_up_pct=0.04,
    enable_limit_up_filter=True,
    is_system=True,
)
```

- [ ] **Step 4: 验证导入**

```bash
cd D:/K/dragon-engine && python -c "
from services.backtest.strategies import STRATEGY_A, STRATEGY_B
print('A entry rules:', [r.rule_type for r in STRATEGY_A.entry_rules])
print('A exit rules:', [r.rule_type for r in STRATEGY_A.exit_rules])
assert len(STRATEGY_A.exit_rules) == 1  # only price_trailing_stop
assert STRATEGY_A.exit_rules[0].rule_type == 'price_trailing_stop'
assert STRATEGY_A.gap_up_pct == 0.04
print('PASS')
"
```

Expected: `A exit rules: ['price_trailing_stop'] | PASS`

---

### Task 7: Engine 核心重构 — TradeEpisode 替换 Position

**Files:**
- Modify: `services/backtest/engine.py`

这是最大的改动。分三步完成。

- [ ] **Step 7a: 更新 engine.py import 和 BacktestContext**

```python
from services.backtest.episode import (
    TradeEpisode, EpisodeRecord,
    STATE_HOLDING, STATE_EXITED,
    run_state_machine,
)
from services.backtest.holding_scorer import compute_continuation, compute_decay
from services.backtest.market_regime import (
    MarketRegime, determine_regime, compute_speculation_index, get_params,
    REGIME_NORMAL,
)

# BacktestContext 中 positions 改为 episodes:
class BacktestContext:
    def __init__(self, strategy: TradingStrategy):
        self.strategy = strategy
        self.current_date: date | None = None
        self.episodes: dict[str, TradeEpisode] = {}   # 改名: positions → episodes
        self.trades: list[Trade] = []
        self.snapshots: list[DailySnapshot] = []
        self.cash: float = strategy.initial_capital
        self.benchmark_value: float = 1.0
        self.current_regime: MarketRegime | None = None  # NEW
        self.episode_counter: int = 0                    # NEW: 用于生成 episode_id
```

- [ ] **Step 7b: 更新所有引用 positions 的地方**

全局替换规则（在整个 engine.py 中）：
- `self.context.positions` → `self.context.episodes`
- `self.context.positions.items()` → `self.context.episodes.items()`
- `self.context.positions[code]` → `self.context.episodes[code]`
- `code in self.context.positions` → `code in self.context.episodes`
- `len(self.context.positions)` → `len(self.context.episodes)`
- `list(self.context.positions.values())` → `list(self.context.episodes.values())`
- `list(self.context.positions.items())` → `list(self.context.episodes.items())`
- `del self.context.positions[code]` → `del self.context.episodes[code]`

- [ ] **Step 7c: 验证导入无报错（功能验证在后续任务中）**

```bash
cd D:/K/dragon-engine && python -c "
from services.backtest.engine import BacktestEngine, BacktestContext
from services.backtest.strategies import STRATEGY_A
ctx = BacktestContext(STRATEGY_A)
assert hasattr(ctx, 'episodes'), 'expected episodes attr'
assert not hasattr(ctx, 'positions'), 'positions should be gone'
print('PASS')
"
```

Expected: `PASS`

---

### Task 8: Engine 集成 — Step 1 价格更新 + Step 2 每日续评

**Files:**
- Modify: `services/backtest/engine.py` — `_process_day` method

- [ ] **Step 1: 重写 _process_day 的 Step 1 + Step 2**

替换 `_process_day` 中 Step 1 和 Step 2 的代码：

```python
def _process_day(self, run: dict, pending_candidates: list[dict]):
    trade_date = run["trade_date"]
    candidates = run.get("leader_candidates", [])
    prices = run.get("prices", {})
    prev_close = run.get("prev_close", {})

    cand_by_code = {c.get("stock_code"): c for c in candidates}

    # ── Step 1: 更新价格 ──
    for code, ep in list(self.context.episodes.items()):
        if code in prices and prices[code] > 0:
            price = prices[code]
            ep.daily_records[-1].price = price  # 更新最后一条记录的价格

    # ── Step 2: 每日续评 ──
    # 从 run 中获取价格数据（engine 不直接访问 DB，由 main.py 注入）
    ohlcv_data = run.get("bars", {})  # NEW: main.py 需要注入 bars
    index_bars = run.get("index_bars", [])  # NEW: 大盘指数 bars

    # 计算/更新 MarketRegime（每天一次）
    spec_idx = compute_speculation_index(
        daily_open_prices=prices,
        prev_close_prices=prev_close,
        sector_volume=run.get("sector_volume", {}),
        total_market_volume=run.get("total_market_volume", 0),
    )
    breadth = run.get("breadth", 0.5)
    volatility = run.get("volatility", 0.02)
    limit_up_count = run.get("limit_up_count", 0)
    self.context.current_regime = determine_regime(
        limit_up_count=limit_up_count,
        total_stocks=5000,
        speculation_index=spec_idx,
        sector_concentration=run.get("sector_concentration", 0),
        breadth=breadth,
        volatility=volatility,
        lianban_height=run.get("lianban_height", 0),
    )
    regime = self.context.current_regime.regime

    for code, ep in self.context.episodes.items():
        stock_bars = ohlcv_data.get(code, [])
        is_in_pool = code in cand_by_code

        full_score = None
        if is_in_pool:
            full_score = cand_by_code[code].get("leader_score", 0)

        cont = compute_continuation(stock_bars, index_bars, regime)
        decay = compute_decay(stock_bars, regime)

        rec = EpisodeRecord(
            date=trade_date,
            state=ep.state,
            price=prices.get(code, ep.daily_records[-1].price if ep.daily_records else ep.entry_price),
            full_score=full_score,
            continuation_score=cont,
            decay_score=decay,
            volume=stock_bars[-1].get("volume", 0) if stock_bars else 0,
            pnl_pct=(prices.get(code, 0) - ep.entry_price) / ep.entry_price if ep.entry_price > 0 else 0.0,
            is_in_candidate_pool=is_in_pool,
        )
        ep.add_record(rec)
```

- [ ] **Step 2: 验证导入无报错**

```bash
cd D:/K/dragon-engine && python -c "from services.backtest.engine import BacktestEngine; print('OK')"
```

---

### Task 9: Engine 集成 — Step 3 状态机 + Step 4 退出矩阵

**Files:**
- Modify: `services/backtest/engine.py` — `_process_day` method, 续写 Step 3-4

- [ ] **Step 1: 追加 Step 3 (状态流转) + Step 4 (退出检查)**

在 `_process_day` 中 Step 2 之后追加：

```python
    # ── Step 3: 状态流转 ──
    for code, ep in self.context.episodes.items():
        stock_bars = ohlcv_data.get(code, [])
        new_state = run_state_machine(ep, regime, stock_bars)
        if new_state != ep.state:
            ep.transition_to(new_state)
            # 更新当日 record 的状态
            if ep.daily_records:
                ep.daily_records[-1].state = new_state

    # ── Step 4: 退出检查 ──
    params = get_params(regime)
    for code, ep in list(self.context.episodes.items()):
        exit_reason: str | None = None

        # 4a: 价格止损 (受 regime 调节)
        if ep.peak_price > 0:
            current_price = prices.get(code, 0)
            if current_price > 0:
                dd = (ep.peak_price - current_price) / ep.peak_price
                if dd > params.price_stop_pct:
                    exit_reason = f"价格止损({dd:.1%}>{params.price_stop_pct:.0%})"

        # 4b: 状态退出 — DECAYING 持续
        if not exit_reason and ep.state == STATE_DECAYING:
            recent = ep.daily_records[-params.decay_trigger_days:]
            if len(recent) >= params.decay_trigger_days:
                if all(r.decay_score > r.continuation_score for r in recent):
                    exit_reason = f"退潮状态持续{params.decay_trigger_days}天"

        # 4c: 量价背离 — 成交量枯竭
        if not exit_reason and ep.daily_records:
            last_rec = ep.daily_records[-1]
            if last_rec.volume > 0:
                # 当日量 < 5日均量 30%
                recent_vols = [r.volume for r in ep.daily_records[-5:] if r.volume > 0]
                if len(recent_vols) >= 5:
                    avg_vol = sum(recent_vols) / len(recent_vols)
                    if last_rec.volume < avg_vol * 0.3:
                        exit_reason = "成交量枯竭"

        # 4d: 极端衰退 — cont < threshold 且不在候选池
        if not exit_reason and ep.daily_records:
            last_rec = ep.daily_records[-1]
            if last_rec.continuation_score < params.min_cont_threshold and not last_rec.is_in_candidate_pool:
                exit_reason = f"续航极低({last_rec.continuation_score:.2f}<{params.min_cont_threshold})"

        # 4e: 时间止损 — 未加速
        if not exit_reason and len(ep.daily_records) >= params.time_stop_days:
            has_acc = any(r.state == "ACCELERATING" for r in ep.daily_records)
            if not has_acc:
                # 减仓 50% (Phase 1 简单实现：全卖)
                exit_reason = f"时间止损({len(ep.daily_records)}天未加速)"

        if exit_reason:
            ep.exit_reason = exit_reason
            ep.exit_date = trade_date
            self._close_episode(code, ep, exit_reason, trade_date)
```

- [ ] **Step 2: 创建 _close_episode 方法（替代 _close_position）**

`_close_episode` 和现有 `_close_position` 逻辑相同，但参数从 Position 改为 TradeEpisode。直接复制 `_close_position` 的代码并改参数名。

- [ ] **Step 3: 更新 _open_position → _open_episode**

替换 `_open_position`，创建 TradeEpisode 而非 Position：

```python
def _open_episode(self, order: dict, trade_date: date, prices: dict):
    code = order["stock_code"]
    price = order["entry_price"]
    cost = order["allocated_cash"]
    entry_comm = calc_commission(cost, self.strategy.commission_rate, self.strategy.min_commission)

    self.context.episode_counter += 1
    ep = TradeEpisode(
        episode_id=f"EP{trade_date.strftime('%Y%m%d')}_{self.context.episode_counter:04d}",
        stock_code=code,
        stock_name=order.get("stock_name", ""),
        state=STATE_HOLDING,
        entry_date=trade_date,
        entry_price=price,
        entry_score=order.get("score", 0),
        shares=order["shares"],
        cost=cost,
    )
    self.context.episodes[code] = ep
    self.context.cash -= (cost + entry_comm)
```

- [ ] **Step 4: 更新 run() 尾部强制平仓循环**

将 `for code, pos in list(self.context.positions.items()):` 改为 `for code, ep in list(self.context.episodes.items()):`

---

### Task 10: Engine 集成 — Step 5 入口过滤 + 上下文扩展

**Files:**
- Modify: `services/backtest/engine.py` — `_process_day`, Step 5-8

- [ ] **Step 1: 追加 Step 5 入口过滤（扩展 context）**

```python
    # ── Step 5: 入口过滤 ──
    ctx = {
        "date": trade_date,
        "prices": prices,
        "prev_close": prev_close,
        # NEW: OneDaySpikeFilter 需要的字段
        "prev_day_change": run.get("prev_day_change", {}),
        "sector_volume_pct": run.get("sector_volume_pct", {}),
        # NEW: VolumeSurgeFilter 需要的字段
        "avg_volume_20": run.get("avg_volume_20", {}),
        "turnover_pct": run.get("turnover_pct", {}),
        "today_volume": run.get("today_volume", {}),
    }

    eligible = []
    for c in pending_candidates:
        code = c.get("stock_code", "")
        if code in self.context.episodes:
            continue
        if self.strategy.enable_limit_up_filter:
            if not self._check_limit_up(c, ctx):
                continue
        passed = all(r.should_enter(c, ctx) for r in self.strategy.entry_rules)
        if passed:
            eligible.append(c)

    # ── Step 6: 仓位分配 (不设上限, per-stock 等权) ──
    current_count = len(self.context.episodes)
    slots = max(0, self.strategy.max_positions - current_count)
    if slots == 0:
        eligible = []
    elif slots < len(eligible):
        eligible.sort(key=lambda c: c.get("leader_score", 0), reverse=True)
        eligible = eligible[:slots]

    # Step 7: 分配
    if eligible:
        per_stock = self.strategy.initial_capital * self.strategy.daily_cash_pct
        alloc_cash = min(per_stock * len(eligible), self.context.cash)
        orders = self.strategy.allocator.allocate(eligible, alloc_cash, prices, {"date": trade_date})
        for order in orders:
            self._open_episode(order, trade_date, prices)

    # ── Step 8: 记录快照 ──
    equity = self.context.cash + sum(
        ep.daily_records[-1].price * ep.shares
        for ep in self.context.episodes.values()
        if ep.daily_records
    )
    prev_equity = self.context.snapshots[-1].equity if self.context.snapshots else self.strategy.initial_capital
    daily_return = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0

    self.context.snapshots.append(DailySnapshot(
        date=trade_date,
        equity=equity,
        cash=self.context.cash,
        positions=[
            Position(
                stock_code=ep.stock_code,
                stock_name=ep.stock_name,
                entry_date=ep.entry_date,
                entry_price=ep.entry_price,
                entry_score=ep.entry_score,
                shares=ep.shares,
                cost=ep.cost,
                peak_score=ep.peak_score,
                peak_price=ep.peak_price,
                days_held=len(ep.daily_records),
                score_history=[r.continuation_score for r in ep.daily_records],
                current_price=ep.daily_records[-1].price if ep.daily_records else ep.entry_price,
                current_score=ep.daily_records[-1].continuation_score if ep.daily_records else ep.entry_score,
            )
            for ep in self.context.episodes.values()
        ],
        daily_return=daily_return,
    ))
    self.context.current_date = trade_date
```

- [ ] **Step 2: 更新 _build_result 中 trades 的 exit_score**

将 `exit_score=pos.current_score` 改为从 episode 获取：

```python
# 在 _close_episode 中，trade 的 exit_score 设为:
episode.daily_records[-1].continuation_score if episode.daily_records else 0
```

---

### Task 11: API 端点扩展 — 注入 bars 和新上下文

**Files:**
- Modify: `services/graph_service/main.py` — `run_backtest` 端点

- [ ] **Step 1: 在 run_backtest 中注入 OHLCV bars 数据**

在 `_load_prices` 调用之后，追加 `_load_bars` 调用：

```python
# 加载每个 symbol 的 OHLCV bars（近 20 天）
bars_by_symbol = await _load_bars(session, all_symbols, start_d, end_d)

# 加载大盘指数 bars
index_bars = await _load_index_bars(session, start_d, end_d)

# 注入到每个 daily run
for dr in daily_runs:
    td = dr["trade_date"]
    dr["prices"] = prices_by_date.get(td, {})
    dr["prev_close"] = prev_close_by_date.get(td, {})
    dr["bars"] = {sym: bars_by_symbol.get(sym, []) for sym in all_symbols}
    dr["index_bars"] = index_bars
    # 计算 prev_day_change（用于 OneDaySpikeFilter）
    if i > 0:
        prev_td = sorted_dates[td_index - 1] if td_index > 0 else None  # ...需要索引映射
```

由于上下文注入较复杂，这里需要重构 daily_runs 的构建逻辑。完整代码如下：

```python
# 构建 sorted_dates 和索引
sorted_dates = sorted(prices_by_date.keys())
date_idx = {d: i for i, d in enumerate(sorted_dates)}

for dr in daily_runs:
    td = dr["trade_date"]
    i = date_idx.get(td, 0)

    dr["prices"] = prices_by_date.get(td, {})
    dr["prev_close"] = prev_close_by_date.get(td, {})

    # Bars for each held+watched symbol (引擎会从中按 symbol 提取)
    dr["bars"] = {sym: _get_bars_for_date(bars_by_symbol.get(sym, []), td)
                   for sym in all_symbols}
    dr["index_bars"] = _get_bars_for_date(index_bars, td)

    # prev_day_change (昨日涨跌幅)
    if i > 0:
        prev_td = sorted_dates[i - 1]
        dr["prev_day_change"] = {
            sym: ((close_by_date.get(td, {}).get(sym, 0) -
                   close_by_date.get(prev_td, {}).get(sym, 0)) /
                  max(close_by_date.get(prev_td, {}).get(sym, 0), 0.01))
            for sym in all_symbols
            if sym in close_by_date.get(td, {}) and sym in close_by_date.get(prev_td, {})
        }
    else:
        dr["prev_day_change"] = {}

    # 占位：暂时用默认值，后续可扩展
    dr["sector_volume_pct"] = {}
    dr["avg_volume_20"] = {}
    dr["turnover_pct"] = {}
    dr["today_volume"] = {}
    dr["sector_volume"] = {}
    dr["total_market_volume"] = 0
    dr["breadth"] = 0.5
    dr["volatility"] = 0.02
    dr["limit_up_count"] = 0
    dr["sector_concentration"] = 0
    dr["lianban_height"] = 0
```

- [ ] **Step 2: 创建辅助函数 _load_bars 和 _get_bars_for_date**

```python
async def _load_bars(session, symbols: set[str], start_d, end_d) -> dict[str, list[dict]]:
    """Load OHLCV bars for each symbol, returning {symbol: [bar_dict, ...]}."""
    from sqlalchemy import select
    from db.models import StockDailyBar

    result = {}
    for sym in symbols:
        r = await session.execute(
            select(StockDailyBar).where(
                StockDailyBar.symbol == sym,
                StockDailyBar.trade_date >= start_d - timedelta(days=20),  # 多取20天
                StockDailyBar.trade_date <= end_d,
            ).order_by(StockDailyBar.trade_date.asc())
        )
        rows = r.scalars().all()
        result[sym] = [
            {"open": row.open, "high": row.high, "low": row.low,
             "close": row.close, "volume": row.volume}
            for row in rows
        ]
    return result


async def _load_index_bars(session, start_d, end_d) -> list[dict]:
    """Load 上证指数 bars, returning [bar_dict, ...]."""
    from sqlalchemy import select
    from db.models import StockDailyBar

    r = await session.execute(
        select(StockDailyBar).where(
            StockDailyBar.symbol == "000001.SH",  # 上证指数
            StockDailyBar.trade_date >= start_d - timedelta(days=20),
            StockDailyBar.trade_date <= end_d,
        ).order_by(StockDailyBar.trade_date.asc())
    )
    rows = r.scalars().all()
    return [
        {"open": row.open, "high": row.high, "low": row.low,
         "close": row.close, "volume": row.volume}
        for row in rows
    ]


def _get_bars_for_date(all_bars: list[dict] | list, td: str) -> list[dict]:
    """从全量 bars 中筛选 <= td 的最多 20 条."""
    if isinstance(all_bars, list) and all_bars and isinstance(all_bars[0], dict):
        # 已是 dict list
        return all_bars[-20:]
    return []
```

- [ ] **Step 3: 验证启动**

```bash
cd D:/K/dragon-engine && python -c "from services.graph_service.main import app; print('OK')"
```

---

### Task 12: E2E 验证 — 回测运行

**Files:** 无

- [ ] **Step 1: 重启服务器**

```bash
# Kill existing
powershell -Command 'Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }'
sleep 2
cd D:/K/dragon-engine && nohup python -m uvicorn services.graph_service.main:app --host 0.0.0.0 --port 8000 > /tmp/uvicorn.log 2>&1 &
sleep 8
tail -3 /tmp/uvicorn.log
```

Expected: `Application startup complete.`

- [ ] **Step 2: 运行回测**

```bash
python -c "
import json
data = {'strategy_name': '策略A-每日固定资金', 'start_date': '2025-01-01', 'end_date': '2026-05-20'}
with open('C:/Users/17812/AppData/Local/Temp/bt_e2e.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)
" && curl -s -X POST 'http://localhost:8000/backtest/run' -H 'Content-Type: application/json' \
    -d @'C:/Users/17812/AppData/Local/Temp/bt_e2e.json' --max-time 300 \
    -o 'C:/Users/17812/AppData/Local/Temp/bt_e2e_out.json' && echo "Done"
```

- [ ] **Step 3: 验证结果关键指标**

```bash
python -c "
import json, sys, io
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open('C:/Users/17812/AppData/Local/Temp/bt_e2e_out.json', encoding='utf-8') as fh:
    d = json.load(fh)

reasons = Counter(t['exit_reason'] for t in d['trades'])
print('========== E2E 验收检查 ==========')
print(f'总收益: {d[\"total_return_pct\"]}%')
print(f'交易笔数: {d[\"total_trades\"]}')
print(f'胜率: {d[\"win_rate\"]}')
print()

# 验证1: 退出原因不再全是价格止损
price_stop = sum(1 for r in reasons if '价格止损' in r)
force_close = sum(1 for r in reasons if '强制平仓' in r)
other = len(d['trades']) - price_stop - force_close
print(f'退出原因分布:')
for reason, count in reasons.most_common():
    print(f'  {count:3d}  {reason}')
print()

# 验证2: 非价格止损退出应有退潮/续航/量价相关
score_related = sum(1 for r in reasons if '退潮' in r or '续航' in r or '枯竭' in r or '时间止损' in r)
print(f'新退出规则触发: {score_related} 笔')
if score_related > 0:
    print('>>> PASS: 新退出规则生效')
else:
    print('>>> WARN: 新退出规则未触发，检查数据')

# 验证3: 胜率改善
if d['win_rate'] > 0.37:
    print(f'>>> PASS: 胜率提升 ({d[\"win_rate\"]} > 0.37)')
else:
    print(f'>>> INFO: 胜率 {d[\"win_rate\"]} (基线 0.37)')
"
```

Expected: 退出原因出现 `退潮状态持续` / `续航极低` / `成交量枯竭` / `时间止损` 等新原因。非价格止损退出 > 0。

---

## 任务依赖

```
Task 1 (Episode model) ──┐
                          ├──→ Task 7 (Engine 核心重构)
Task 2 (State machine) ──┤
                          │
Task 3 (Holding scorer) ──┤
                          │
Task 4 (Market regime) ───┤
                          │
Task 5 (New EntryRules) ──┼──→ Task 6 (Strategy configs)
                          │
                          ├──→ Task 8 (Engine 续评)
                          │
                          ├──→ Task 9 (Engine 状态+退出)
                          │
                          ├──→ Task 10 (Engine 入口过滤)
                          │
Task 11 (API bars) ───────┘──→ Task 12 (E2E)
```

任务 1-6 可并行启动。任务 7-10 依序执行。任务 11 需要 DB 连接验证。任务 12 最后一环。
