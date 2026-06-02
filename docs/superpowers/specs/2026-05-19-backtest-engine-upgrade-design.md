# 回测引擎升级 + 价格数据 + 前端重设计

> 目标：让回测产生真实 P&L，以开盘价竞价买卖，加手续费/印花税/高开过滤，前端用 ECharts 重做。

## 架构概览

三层改动：
1. **数据层** — 新表 `stock_daily_bars` 存 OHLCV，每日批次写入
2. **引擎层** — 开盘价买卖、涨停过滤、高开过滤、手续费
3. **前端层** — ECharts 权益曲线+回撤图、可展开交易明细、多策略切换、持仓热力图

所有前端图表用 ECharts（已在项目依赖中），页面遵循项目现有 Dark OLED 主题（`html.dark` / `:root` 两套变量）。

---

## 数据层

### stock_daily_bars 表

```sql
CREATE TABLE IF NOT EXISTS stock_daily_bars (
    id          SERIAL PRIMARY KEY,
    symbol      VARCHAR(10) NOT NULL,
    trade_date  DATE NOT NULL,
    open        FLOAT DEFAULT 0,
    high        FLOAT DEFAULT 0,
    low         FLOAT DEFAULT 0,
    close       FLOAT DEFAULT 0,
    volume      BIGINT DEFAULT 0,
    amount      FLOAT DEFAULT 0,
    change_pct  FLOAT DEFAULT 0,
    turnover_pct FLOAT DEFAULT 0,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_bars_symbol ON stock_daily_bars(symbol);
CREATE INDEX IF NOT EXISTS idx_bars_date ON stock_daily_bars(trade_date);
```

### 数据写入时机

在 `run_batch_30days.py` 和 `run_enhanced_backfill.py` 中，akshare 获取到 OHLCV 后写入此表。`ON CONFLICT (symbol, trade_date) DO UPDATE` 保证幂等。

```python
# 伪代码：在 run_one_day 中
akshare_data = fetch_akshare_daily(trade_date, symbols)  # 已有
await upsert_daily_bars(session, akshare_data)
```

`upsert_daily_bars()` 在 `db/persist.py` 中实现，独立函数，与 `persist_run()` 平级。

### ORM Model

`db/models.py` 追加 `StockDailyBar` 类。

---

## 引擎层

### 价格来源

BacktestEngine._process_day() 中，从 `stock_daily_bars` 查询当天所有持仓股 + 候选股的开盘价。LOAD 阶段注入 `daily_runs[i]["prices"]` 字段。

```python
# main.py run_backtest endpoint 中
# 加载 daily_runs 后，批量查询 stock_daily_bars
prices = await load_prices(session, symbols, dates)
# 注入到每个 day: run["prices"] = {symbol: open_price}
```

回测引擎使用 prices 而非默认 ¥10：
```python
price = prices.get(c["stock_code"], c.get("price", 10.0))
```

### EntryRule: 涨停过滤（内置，总是生效）

```python
class LimitUpFilterRule(EntryRule):
    """过滤开盘即涨停的票（买不到）。总是生效。"""
    rule_type = "limit_up_filter"

    def __init__(self):
        self.params = {}

    def should_enter(self, candidate, context):
        prev_close = context.get("prev_close", {}).get(candidate["stock_code"])
        open_price = context.get("prices", {}).get(candidate["stock_code"])
        if prev_close and open_price and prev_close > 0:
            gap = (open_price - prev_close) / prev_close
            if gap >= 0.098:  # >= 9.8% 即涨停
                return False
        return True
```

此规则在引擎初始化时自动注入到所有策略的 entry_rules 首位。

### EntryRule: 高开过滤（可选）

```python
class GapUpFilterRule(EntryRule):
    """开盘高开超过阈值不买入。可选启用。"""
    rule_type = "gap_up_filter"

    def __init__(self, max_gap_pct: float = 0.04):
        self.params = {"max_gap_pct": max_gap_pct}

    def should_enter(self, candidate, context):
        prev_close = context.get("prev_close", {}).get(candidate["stock_code"])
        open_price = context.get("prices", {}).get(candidate["stock_code"])
        if prev_close and open_price and prev_close > 0:
            gap = (open_price - prev_close) / prev_close
            if gap > self.params["max_gap_pct"]:
                return False
        return True
```

### TradingStrategy 新增字段

```python
@dataclass
class TradingStrategy:
    # ... 现有字段 ...
    commission_rate: float = 0.00025     # 万2.5
    stamp_duty_rate: float = 0.0005      # 卖出印花税 0.05%
    min_commission: float = 5.0          # 最低佣金
    gap_up_pct: float | None = None      # None=不过滤高开, 值=阈值如0.04
    enable_limit_up_filter: bool = True  # 涨停过滤（默认开启）
```

to_dict/from_dict 同步更新序列化。

### 手续费计算

在 `_open_position()` 和 `_close_position()` 中计算：

```python
def calc_commission(amount: float, rate: float, min_fee: float) -> float:
    return max(amount * rate, min_fee)

# 买入
entry_commission = calc_commission(order_cost, strategy.commission_rate, strategy.min_commission)
# 卖出
exit_commission = calc_commission(proceeds, strategy.commission_rate, strategy.min_commission)
stamp_duty = proceeds * strategy.stamp_duty_rate

# Trade 中新增字段
net_pnl = pnl - entry_commission - exit_commission - stamp_duty
```

### BacktestResult/Trade 新增

Trade dataclass 追加：`entry_commission`, `exit_commission`, `stamp_duty`, `net_pnl`, `entry_score`, `exit_score`
BacktestResult 追加：`total_commission`, `total_stamp_duty`

---

## 前端层

### 技术选型

- **ECharts**（已在 `dragon-engine-web/package.json` 中）— 权益曲线、回撤图、得分走势
- **Element Plus** — 表格、卡片、下拉框（已有）
- **主题** — 读取 `html.dark` 类名，ECharts 初始化时传 `'dark'` 或 `undefined`

### 回测结果页面布局

```
┌─────────────────────────────────────────────┐
│  策略选择 [下拉框 ▼]  日期范围 [选择器]  [运行回测]  │
├─────────────────────────────────────────────┤
│  总收益率  │ 最大回撤 │ 夏普 │ 胜率 │ 交易次数 │ 最终净值 │
│   -4.32%   │  8.75%  │-0.85│42.9%│   49    │ ¥95,679 │
├─────────────────────────────────────────────┤
│  权益曲线 (ECharts)          │ 回撤深度 (ECharts)   │
│  带买卖标记的折线图            │  面积图               │
├─────────────────────────────────────────────┤
│  交易明细 — 可展开行                           │
│  ▶ 002686 亿利达 04-20→04-23 ¥8.52→9.15 +7.39%  │
│  ▼ 603912 佳力图 04-20→04-22 ¥10.20→9.58 -6.08% │
│    买入明细 │ 卖出明细 │ 龙头得分走势(ECharts迷你图) │
│    开盘 10.20│开盘 9.58 │  ┌────────┐           │
│    1000股    │佣金 -5.00│  │ ╲得分  │           │
│    佣金 -5.00│印花 -4.79│  │   ╲    │           │
│    得分 0.598│净亏 -629 │  └────────┘           │
├─────────────────────────────────────────────┤
│  持仓热力图 (ECharts heatmap)                  │
│  股票 × 日期矩阵，颜色=当日收益率                 │
└─────────────────────────────────────────────┘
```

### 策略下拉框

回测结果接口改为接收参数 `?strategy_name=xxx`，或后端缓存最近一次回测的多策略结果。前端加载策略列表后，下拉框切换时重新请求对应策略的回测结果。

简化方案：后端 `/backtest/run` 返回结果时带 `strategy_name`。前端在 Results.vue 中维护一个 `Map<strategy_name, BacktestResult>`，切换策略时从缓存取或重新请求。

### ECharts 主题适配

```typescript
function initChart(el: HTMLElement): echarts.ECharts {
  const isDark = document.documentElement.classList.contains('dark')
  return echarts.init(el, isDark ? 'dark' : undefined)
}

// 监听主题切换
const observer = new MutationObserver(() => {
  chartInstance?.dispose()
  chartInstance = initChart(chartRef.value!)
  renderChart()
})
observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
```

---

## Token 追踪 Bug 修复

### 问题

`GET /token-usage` 返回的 `token_usage` 全是空 `{}`。

### 排查方向

1. 检查 `TokenTrackingCallback.on_llm_end()` 是否正确接收 `llm_output`
2. 检查 LLM provider 返回的 response 中是否包含 `token_usage`
3. 检查 `persist_run()` 中 `tracker.summary()` 是否在 `tracker.start_run()` 之后、`graph.ainvoke()` 之后被调用

### 预期修复

在 `main.py` 的 `/run` 端点中确保：
```python
tracker.start_run(trade_date)
result = await graph.ainvoke(initial_state)
await persist_run(trade_date, result)  # 内部读取 tracker.summary()
```

检查 `token_tracker/tracker.py` 中 `summary()` 的 `records` 是否正确累积。

---

## 实现顺序

| 任务 | 依赖 |
|------|------|
| 1. Token 追踪 Bug 修复 | 无 |
| 2. `stock_daily_bars` 表 + ORM + SQL | 无 |
| 3. 批次写入 daily bars | 2 |
| 4. TradingStrategy 新增字段 + 序列化 | 无 |
| 5. 手续费计算 + Trade 新增字段 | 4 |
| 6. LimitUpFilterRule + GapUpFilterRule | 4 |
| 7. 引擎价格来源改造 | 2, 5 |
| 8. 引擎集成 testing | 2-7 |
| 9. 前端重设计（ECharts + 多策略 + 主题） | 8 |
| 10. 前端联调验证 | 9 |

---

## 文件清单

| 文件 | 操作 |
|------|------|
| `db/models.py` | 修改 — 加 StockDailyBar ORM |
| `db/schema.sql` | 修改 — 加 CREATE TABLE |
| `db/persist.py` | 修改 — 加 upsert_daily_bars() |
| `services/backtest/models.py` | 修改 — Trade/BacktestResult 加字段 |
| `services/backtest/rules.py` | 修改 — 加 LimitUpFilterRule, GapUpFilterRule |
| `services/backtest/strategies.py` | 修改 — TradingStrategy 加字段 + 序列化 |
| `services/backtest/engine.py` | 修改 — 开盘价来源 + 手续费计算 |
| `services/token_tracker/tracker.py` | 修改 — Bug 修复 |
| `services/graph_service/main.py` | 修改 — prices 注入 + 多策略缓存 |
| `run_batch_30days.py` | 修改 — 写入 daily bars |
| `run_enhanced_backfill.py` | 修改 — 写入 daily bars |
| `dragon-engine-web/src/views/Backtest/Results.vue` | 重写 |
| `dragon-engine-web/src/types/api.ts` | 修改 — 加新字段 |
