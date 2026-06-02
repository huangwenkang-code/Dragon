# Trade Lifecycle System — 交易生命周期系统

> **Status:** Spec complete. Awaiting plan generation.

**Goal:** 将回测引擎从"每日选股器"升级为"龙头生命周期管理系统"，补上持有评分、状态机、市场环境适配、退出决策矩阵四个核心模块。

**Architecture:** 四层分离——发现层（选股）→ 持有层（续评+状态机）→ 退出层（决策矩阵）→ 反思层（进化），全部受 Market Regime 环境变量动态调节。

**Tech Stack:** Python 3.12, dataclasses, stock_daily_bars (PostgreSQL), 现有 pipeline 上游不变

---

## Layer 0: Market Regime Service（环境变量引擎）

**文件:** `services/backtest/market_regime.py` (新建)

### 输入

- `stock_daily_bars` — 全市场 OHLCV 数据
- 指数数据（如上证指数）— 也存储在 `stock_daily_bars` 中

### 输出

```python
@dataclass
class MarketRegime:
    temperature: float        # 0-1 市场热度 (涨停家数/全市场比例)
    speculation_index: float  # 0-1 投机指数 (妖股活跃度)
    regime: str               # "LIANBAN" | "AI_TREND" | "MONSTER" | "ICE" | "NORMAL"
    volatility: float         # 市场波动率 (近5日涨跌幅标准差)
    breadth: float            # 市场宽度 (上涨家数/总家数)
```

### 判定规则（纯规则，不用 ML）

| Regime | 条件 |
|--------|------|
| `LIANBAN` | 连板高度 ≥ 5 日、涨停家数 > 80、投机指数 > 0.7 |
| `MONSTER` | 妖股活跃度 > 0.6、连板高度 ≥ 7 日 |
| `AI_TREND` | 板块集中度 > 0.5（单一板块成交占比）、持续 > 5 天 |
| `ICE` | 涨停家数 < 30、市场宽度 < 0.3、波动率低 |
| `NORMAL` | 以上都不满足 |

### 调节作用

每种 regime 调整持有评分的权重和阈值：

| 参数 | LIANBAN | MONSTER | AI_TREND | ICE | NORMAL |
|------|---------|---------|----------|-----|--------|
| continuation 权重 | 0.6 | 0.5 | 0.7 | 0.3 | 0.5 |
| decay 权重 | 0.4 | 0.5 | 0.3 | 0.7 | 0.5 |
| 价格止损回撤阈值 | 5% | 7% | 7% | 3% | 7% |
| DECAYING 触发天数 | 1天 | 2天 | 2天 | 1天 | 2天 |
| gap_up 阈值 | 3% | 4% | 4% | 2% | 4% |

**核心逻辑:** 连板周期要更紧的止损和更快的退出（高波动高换手），冰点期止损要紧但仓位要极轻（低流动性没人接盘）。

---

## Layer 1: 发现层

### 保持现有

- pipeline 上游 6 信号不变（情绪/资金流/龙虎榜/妖股/事件/板块）
- `leader_score` 复合分不变
- `top_n` 候选池筛选不变

### 新增 EntryRule（`services/backtest/rules.py`）

**OneDaySpikeFilter**

```python
class OneDaySpikeFilter(EntryRule):
    """过滤一日游：昨日涨幅>9.5%(近涨停) + 今日高开>3% + 板块成交占比异常"""
    rule_type = "one_day_spike"

    # 依赖上下文提供:
    #   - context["prev_day_change"] — 昨日涨幅
    #   - context["sector_volume_pct"] — 板块内成交占比
    #   - context["prices"] — 今日开盘价
    #   - context["prev_close"] — 昨日收盘价
```

**VolumeSurgeFilter**

```python
class VolumeSurgeFilter(EntryRule):
    """过滤异常放量：当日量>20日均量3倍 且 换手率>15%"""
    rule_type = "volume_surge"

    # 依赖上下文提供:
    #   - context["avg_volume_20"] — 20日均量
    #   - context["turnover_pct"] — 换手率
```

**GapUpFilter 启用** — 已有代码，`strategies.py` 中 `gap_up_pct` 设有效值即可。默认 4%，受 regime 动态调节。

---

## Layer 2: 持有层

### TradeEpisode（`services/backtest/episode.py` 新建）

```python
@dataclass
class EpisodeRecord:
    date: date
    state: str
    price: float
    full_score: float | None        # 候选池内=leader_score，池外=None
    continuation_score: float       # 续航分(始终计算)
    decay_score: float              # 退潮分(始终计算)
    volume: float
    pnl_pct: float
    is_in_candidate_pool: bool


@dataclass
class TradeEpisode:
    episode_id: str                 # "EP20260520_600001"
    stock_code: str
    stock_name: str
    state: str                      # 当前状态
    entry_date: date
    entry_price: float
    entry_score: float
    entry_reason: list[str]         # ["政策催化", "板块共振"]
    shares: int
    cost: float

    daily_records: list[EpisodeRecord]
    peak_price: float
    peak_score: float
    max_floating_pnl: float
    max_drawdown: float

    exit_reason: str | None
    exit_date: date | None
```

### 状态机

```
DISCOVERED → HOLDING → ACCELERATING → EXITED
                │  ↑         │
                │  └─────────┘
                ↓              ↓
            DECAYING → DISTRIBUTING
                │            │
                └──→ EXITED ←┘
```

| 状态 | 含义 | 进入条件 | 触发流转的事件 |
|------|------|---------|----------------|
| DISCOVERED | 新发现 | 首次入候选池 | 买入→HOLDING |
| HOLDING | 正常持有 | 买入后默认 | — |
| ACCELERATING | 加速上涨 | HOLDING + 连续3天价格创新高+放量 | 日跌幅>5%→DISTRIBUTING |
| DISTRIBUTING | 高位分歧 | ACCELERATING + 放量滞涨/高开低走 | decay>cont 2天→DECAYING；触发出场→EXITED |
| DECAYING | 退潮萎缩 | decay>cont 持续2天(regime调节天数) | cont>decay+价格创新高→HOLDING |
| EXITED | 已退出 | 触发出场条件 | 终态 |

### 持有评分（`services/backtest/holding_scorer.py` 新建）

**continuation_score（续航分 0-1）:**

| 信号 | 权重 | 计算方式 | 数据源 |
|------|------|---------|--------|
| 相对强度 | 25% | 当日涨跌幅 vs 大盘涨跌幅 | stock_daily_bars |
| 价格位置 | 20% | 当前价 vs 5日均线 | stock_daily_bars |
| 连涨天数 | 15% | 近5天连续收阳天数/5 | stock_daily_bars |
| 成交量趋势 | 20% | 量/5日均量，1-1.5x最优 | stock_daily_bars |
| 新高频率 | 20% | 近5天创阶段新高天数/5 | stock_daily_bars |

**decay_score（退潮分 0-1）:**

| 信号 | 权重 | 计算方式 | 数据源 |
|------|------|---------|--------|
| 高开低走 | 25% | max(0, (open-close)/open)，>2%开始计分 | stock_daily_bars |
| 放量滞涨 | 20% | 量增>3%+涨跌<1% | stock_daily_bars |
| 连跌天数 | 15% | 近5天连续收阴天数/5 | stock_daily_bars |
| 距高点回撤 | 25% | (近5日高点-当前价)/近5日高点 | stock_daily_bars |
| 均线死叉 | 15% | 5日线是否下穿10日线 | stock_daily_bars |

**Regime 调节：** 权重和阈值按 MarketRegime 动态调整（见 Layer 0 表格）。

### 每日续评流程

```
Step 2: 每日续评
  for episode in active_episodes:
      if episode.stock_code in today_candidates:
          episode.daily_records.append(EpisodeRecord(
              full_score=candidates[code].leader_score,
              is_in_candidate_pool=True,
          ))
      else:
          episode.daily_records.append(EpisodeRecord(
              full_score=None,
              is_in_candidate_pool=False,
          ))
      # 无论如何都计算 continuation/decay
      ohlcv = get_bars(episode.stock_code, lookback=20)
      cont = compute_continuation(ohlcv, market_index_ohlcv, regime)
      decay = compute_decay(ohlcv, regime)
      record.continuation_score = cont
      record.decay_score = decay
```

---

## Layer 3: 退出层

### 退出决策矩阵

| 维度 | 条件 | 动作 | regime 调节 |
|------|------|------|-------------|
| 价格止损 | 距峰值回撤 > threshold | 立即退出 | threshold 见 regime 表 |
| 状态退出 | DECAYING 持续 N 天 | 退出 | N 见 regime 表 |
| 量价背离 | 当日量 < 5日均量 30% | 退出 | 不变 |
| 极端衰退 | cont < 0.3 且不在候选池 | 退出 | ICE 期调至 0.2 |
| 时间止损 | 持有 > 15 天未进入 ACCELERATING | 减仓 50% | LIANBAN→10天 |
| 强制平仓 | 回测结束 | 全部退出 | 不变 |

### 废除的旧规则

- `ScoreCliffRule` → 被 `cont < 0.3` 替代
- `TrailingStopRule` → 被 `decay > cont 持续N天` 替代
- `ScoreDeclineRule` → 被状态机 DECAYING 替代

### 保留的旧规则

- `PriceTrailingStopRule` → 保留，阈值受 regime 调节

---

## Layer 4: 反思层（Phase 2+）

**Phase 1 只做数据记录，不做自动学习。**

每笔交易结束时，TradeEpisode 的完整 `daily_records` + 入场特征 + 退出原因写入输出，供后续分析。Phase 2 实现模式自动统计和参数自适应。

### 预留数据结构

```python
@dataclass
class TradeReflection:
    episode_id: str
    entry_pattern: str          # "高开追涨" | "正常入选" | "低吸"
    failure_category: str       # "一日游" | "高位接力" | "板块退潮" | "事件兑现" | "资金不持续" | None
    max_floating_pnl: float
    max_drawdown: float
    holding_regime_changes: list[str]  # 持有期间经历的 regime 变化
    key_lessons: list[str]
```

---

## Engine 集成（`services/backtest/engine.py` 重写）

### _process_day 新流程

```
Step 1: 更新价格 (stock_daily_bars → pos.current_price, 保留现有逻辑)
Step 2: 计算 MarketRegime (如果当日未计算)
Step 3: 每日续评 — 每个 TradeEpisode:
        ├── 在候选池 → full_score = leader_score
        └── 不在池 → full_score = None
        两者都计算 continuation/decay (via holding_scorer + regime)
Step 4: 状态流转 — state_machine.transition(episode, regime)
Step 5: 退出检查 — exit_decision_matrix(episode, regime)
Step 6: 入口过滤 — EntryRule chain (含新 OneDaySpike/VolumeSurge) + GapUpFilter
Step 7: 仓位分配 — per-stock 等权 ¥25,000 (保持不变，Phase 2 加 dynamic sizing)
Step 8: 记录快照 — episode.daily_records + DailySnapshot
```

### 文件改动清单

| 文件 | 操作 | 内容 |
|------|------|------|
| `services/backtest/episode.py` | **新建** | TradeEpisode, EpisodeRecord, 状态机 |
| `services/backtest/holding_scorer.py` | **新建** | compute_continuation(), compute_decay() |
| `services/backtest/market_regime.py` | **新建** | MarketRegime 计算 + 参数调节表 |
| `services/backtest/engine.py` | **重写** | 集成三模块，TradeEpisode 替代 Position 核心逻辑 |
| `services/backtest/rules.py` | **修改** | 新增 OneDaySpikeFilter, VolumeSurgeFilter; 标记废除的得分卖出规则 |
| `services/backtest/strategies.py` | **修改** | 更新策略配置（新规则链 + gap_up_pct 默认值 + 移除旧退出规则） |
| `services/backtest/models.py` | **修改** | EpisodeReflection 预留；Position 保留给快照 |

---

## 验证清单

- [ ] MarketRegime 每天正确判定且日志可见
- [ ] TradeEpisode.daily_records 每天增长，不在候选池也有 continuation/decay
- [ ] 状态流转日志可见（HOLDING→DECAYING 等）
- [ ] 得分卖出规则重新生效（不再全部是价格止损）
- [ ] 一日游过滤器减少高开追涨买入
- [ ] 回测结果：胜率提升，止损亏损减少
- [ ] 无报错，所有卡片正常渲染
- [ ] regime 切换时参数变化日志可见
