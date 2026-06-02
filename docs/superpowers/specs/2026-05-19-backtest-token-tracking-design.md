# 回测引擎 + Token 追踪 + 交易规则配置 设计

2026-05-19

## 目标

三个独立模块 + 前端三页面：

1. **交易规则引擎** — 可插拔的入场/卖出/仓位策略，回测引擎逐日模拟
2. **Token 消耗追踪** — 自动捕获 LLM 调用 token 用量和费用，前端展示
3. **前端三页面** — 回测结果、Token 消耗、策略配置
4. **历史数据填充** — 跑最近 1 个月管线数据进 DB 供回测

---

## 第一部分：交易规则引擎

### 文件：`services/backtest/` (NEW package)

```
services/backtest/
├── __init__.py
├── rules.py          # EntryRule / ExitRule / Allocator 抽象基类 + 具体实现
├── strategies.py      # TradingStrategy 组合 rules
├── engine.py          # BacktestEngine 逐日迭代
├── models.py          # Pydantic: Position, Trade, Order, BacktestResult
└── registry.py        # 策略注册表 + DB 持久化
```

### 核心抽象

```python
# rules.py

class EntryRule(ABC):
    """是否买入候选股"""
    rule_type: str           # 规则类型标识
    params: dict             # 可配置参数
    @abstractmethod
    def should_enter(self, candidate: LeaderCandidate, context: BacktestContext) -> bool: ...

class ExitRule(ABC):
    """是否卖出持仓"""
    rule_type: str
    params: dict
    @abstractmethod
    def should_exit(self, position: Position, current_day: PipelineRun, context: BacktestContext) -> ExitSignal | None: ...

class PositionAllocator(ABC):
    """分配多少资金"""
    allocator_type: str
    params: dict
    @abstractmethod
    def allocate(self, candidates: list[LeaderCandidate], available_cash: float, context: BacktestContext) -> list[Order]: ...
```

### 具体规则实现

| 类型 | 规则名 | 参数 | 逻辑 |
|------|--------|------|------|
| Entry | `ScoreThresholdRule` | `min_score: float` | leader_score >= min_score |
| Entry | `NoFilterRule` | — | 全部通过 |
| Exit | `ScoreCliffRule` | `threshold: float` | 当日 score < 阈值 → 卖出 |
| Exit | `TrailingStopRule` | `drawdown_pct: float` | 从持仓期间最高 score 回撤 > drawdown_pct → 卖出 |
| Exit | `ScoreDeclineRule` | `consecutive_days: int` | 连续 N 天 score 递减 → 卖出 |
| Allocator | `ScoreWeightedAllocator` | — | 按 leader_score 占比分配资金 |
| Allocator | `EqualWeightAllocator` | — | 等权分配 |

### TradingStrategy 组合

```python
@dataclass
class TradingStrategy:
    name: str
    description: str
    entry_rules: list[EntryRule]
    exit_rules: list[ExitRule]
    allocator: PositionAllocator
    max_positions: int = 999       # 最大持仓数
    max_position_pct: float = 1.0  # 单只最大仓位比例
    initial_capital: float = 100000
    is_system: bool = False        # 系统默认不可删
```

### 预置两种策略

**策略 A — 每日固定资金**：
- Entry: NoFilterRule
- Exit: ScoreCliff(0.3) + TrailingStop(0.15) + ScoreDecline(3)
- Allocator: ScoreWeightedAllocator
- max_positions: 999, max_position_pct: 1.0
- 逻辑：每天用总资金的 50% 按 score 加权买入当天 top-5

**策略 B — 仓位上限管理**：
- Entry: ScoreThresholdRule(0.5)
- Exit: ScoreCliff(0.3) + TrailingStop(0.15) + ScoreDecline(3)
- Allocator: ScoreWeightedAllocator
- max_positions: 8, max_position_pct: 0.15
- 逻辑：有空余仓位时才买新的，单只 ≤15%

### BacktestEngine 逐日迭代

按 trade_date 排序遍历 PipelineRun：
1. 检查所有持仓 → 触发卖出规则 → 释放现金
2. 筛选当天 LeaderCandidate（入场规则）
3. 分配仓位（分配器）
4. 记录当日净值（持仓 × 当日涨跌幅）
5. 输出 BacktestResult（交易列表 + 每日净值）

### DB 存储

策略存 `backtest_strategies` 表（JSON 序列化规则列表），回测结果存 `backtest_results` + `backtest_trades` 表。

---

## 第二部分：Token 消耗追踪

### 文件：`services/token_tracker/` (NEW package)

```
services/token_tracker/
├── __init__.py
├── tracker.py       # TokenUsageTracker 单例 + LangChain Callback
├── models.py         # TokenRecord Pydantic
└── pricing.py        # 模型单价表
```

### 核心方案 — LangChain Callback Handler

不侵入现有代码，用 `BaseCallbackHandler` 自动捕获：

```python
from langchain.callbacks import BaseCallbackHandler

class TokenTrackingCallback(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        usage = response.llm_output.get("token_usage", {})
        record = TokenRecord(
            run_id=_current_run_id(),
            step=kwargs.get("name", "unknown"),
            model=response.llm_output.get("model_name", ""),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            cost=_estimate_cost(usage),
            timestamp=datetime.now(),
        )
        TokenUsageTracker.instance().record(record)
```

在 `llm_adapter` 创建 LLM 时注入 callback，覆盖现有所有 LLM 调用点。

### DB：PipelineRun 加字段

```sql
ALTER TABLE pipeline_runs ADD COLUMN token_usage JSONB DEFAULT '{}';
```

### 模型定价表（可配置）

| 模型 | 输入 ¥/1K tokens | 输出 ¥/1K tokens |
|------|-----------------|-----------------|
| qwen-turbo | 0.0003 | 0.0006 |
| qwen-plus | 0.0008 | 0.002 |
| qwen-max | 0.02 | 0.06 |

### API 端点

- `GET /api/token-usage?days=30` — 返回近 N 天的 token 消耗汇总
- `GET /api/token-usage/{run_id}` — 单次运行的详细 token 分解

---

## 第三部分：前端页面

### 路由新增

```
/backtest/strategies    → views/Backtest/Strategies.vue
/backtest/results       → views/Backtest/Results.vue
/token-usage            → views/TokenUsage/index.vue
```

### 侧边栏新增

```
回测中心    (el-sub-menu)
├── 策略配置  /backtest/strategies
├── 回测结果  /backtest/results
Token 消耗   /token-usage
```

### 页面 1: 回测结果 (`/backtest/results`)

- **操作栏**: 策略下拉 + 日期范围选择器 + 「运行回测」按钮（loading 态）
- **统计卡片**: 总收益率 / 夏普比率 / 最大回撤 / 胜率 / 交易次数
- **权益曲线**: ECharts 折线图，X=日期 Y=净值，叠加沪深300基准线
- **交易明细表**: 买入日/卖出日/股票/代码/买入价/卖出价/收益率/触发规则
- 复用现有 dark OLED 主题，ScoreBadge、ReasoningCard 等通用组件

### 页面 2: Token 消耗 (`/token-usage`)

- **汇总卡片**: 本月总费用(¥) / 总 token / 平均每次费用
- **消耗趋势图**: ECharts 柱状图，X=日期 Y=token数/费用
- **明细表**: 日期/模型/输入token/输出token/总token/费用(¥)
- 点击某行展开该次运行的步骤级分解

### 页面 3: 策略配置 (`/backtest/strategies`)

- **策略卡片列表**: 名称 + 描述 + 规则摘要 + 「编辑」「删除」「设为默认」
- 策略 A/B 标记"系统默认"，删除按钮 disabled
- **新建/编辑抽屉** (el-drawer):
  - 策略名输入
  - 入场规则：下拉选类型 → 动态表单显示参数
  - 卖出规则：多选 + 每个规则的参数
  - 仓位分配：下拉选分配器 + 最大持仓数 + 单只最大仓位
  - 保存 → 写 DB → 刷新列表

---

## 第四部分：历史数据填充

### 跑最近 1 个月管线数据

用子 agent 并行执行：从今天往前推 22 个交易日，逐日跑 `capital_flow` → `generate_candidates` → `persist`。

数据源限制：
- `capital_flow` 依赖腾讯实时行情（qt.gtimg.cn），不支持历史查询 → 降级用 AKShare `stock_zh_a_hist` 获取历史日线
- `events`/`sentiment` 也需要历史新闻 → 降级用 AKShare 新闻接口

**降级方案**：当前管线从实时数据切到历史数据时，部分维度缺失（如主力净流入、龙虎榜），用当日涨跌幅+成交额+换手率作为核心维度，其余降级。

可选方案：
- **方案 A**：只跑能跑的部分（capital_flow + 候选股筛选），忽略实时维度 → 快速但有信息缺失
- **方案 B**：实时管线照常跑，每天存数据，积累 1 个月后再回测 → 不需要历史数据回放

---

## 不做什么

- 不做 tick 级数据回放
- 不接券商 API 做实盘交易
- 不回测超过 1 个月（数据不足）
- 不添加滑点/手续费模拟（V1 阶段不需要）
- 前端不做实时 websocket 推送

---

## 文件清单

| 文件 | 操作 |
|------|------|
| `services/backtest/__init__.py` | 新建 |
| `services/backtest/rules.py` | 新建 |
| `services/backtest/strategies.py` | 新建 |
| `services/backtest/engine.py` | 新建 |
| `services/backtest/models.py` | 新建 |
| `services/backtest/registry.py` | 新建 |
| `services/token_tracker/__init__.py` | 新建 |
| `services/token_tracker/tracker.py` | 新建 |
| `services/token_tracker/models.py` | 新建 |
| `services/token_tracker/pricing.py` | 新建 |
| `services/llm_adapter/llm_provider.py` | 修改 — 注入 callback |
| `db/models.py` | 修改 — 加 backtest 表 + PipelineRun.token_usage |
| `db/persist.py` | 修改 — 存 token_usage |
| `api/routes/token_usage.py` | 新建 |
| `api/routes/backtest.py` | 新建 |
| `dragon-engine-web/src/views/Backtest/Results.vue` | 新建 |
| `dragon-engine-web/src/views/Backtest/Strategies.vue` | 新建 |
| `dragon-engine-web/src/views/TokenUsage/index.vue` | 新建 |
| `dragon-engine-web/src/router/index.ts` | 修改 |
| `dragon-engine-web/src/components/layout/SidebarMenu.vue` | 修改 |
| `dragon-engine-web/src/types/api.ts` | 修改 — 加类型 |

---

## 验证清单

- [ ] 策略 A/B 回测 — 产生交易记录和权益曲线数据
- [ ] Token 追踪 — 管线运行后在 DB token_usage 字段有数据
- [ ] Token 消耗页 — 显示费用和 token 趋势图
- [ ] 策略配置页 — 新建/编辑/删除策略
- [ ] 回测结果页 — 权益曲线 + 交易明细表
- [ ] 历史数据 — 最近 1 个月管线数据在 DB 中
- [ ] 策略注册表 — 加新规则不改引擎代码
