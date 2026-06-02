<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import * as echarts from 'echarts'
import client from '@/api/client'
import type { StrategyConfig, BacktestResult, BacktestTrade } from '@/types/api'

const loading = ref(false)
const strategies = ref<StrategyConfig[]>([])
const selectedStrategy = ref('')
const dateRange = ref<[string, string]>(['2026-04-15', '2026-05-19'])
const result = ref<BacktestResult | null>(null)
const useScanner = ref(false)
const scannerTopN = ref(60)

// Sort state
const sortProp = ref<string>('')
const sortOrder = ref<string>('')

const sortedTrades = computed(() => {
  if (!result.value?.trades) return []
  const trades = [...result.value.trades]
  if (!sortProp.value || !sortOrder.value) return trades
  const order = sortOrder.value === 'ascending' ? 1 : -1
  return trades.sort((a, b) => {
    let va: number, vb: number
    if (sortProp.value === 'net_pnl') {
      va = a.net_pnl || a.pnl || 0
      vb = b.net_pnl || b.pnl || 0
    } else if (sortProp.value === 'pnl_pct') {
      va = a.pnl_pct || 0
      vb = b.pnl_pct || 0
    } else {
      va = (a as any)[sortProp.value] || 0
      vb = (b as any)[sortProp.value] || 0
    }
    return (va - vb) * order
  })
})

function handleSortChange({ prop, order }: { prop: string; order: string }) {
  sortProp.value = prop
  sortOrder.value = order
}

// Chart refs
const equityChartRef = ref<HTMLDivElement | null>(null)
const ddChartRef = ref<HTMLDivElement | null>(null)
const heatmapRef = ref<HTMLDivElement | null>(null)

let equityChart: echarts.ECharts | null = null
let ddChart: echarts.ECharts | null = null
let heatmapChart: echarts.ECharts | null = null
let themeObserver: MutationObserver | null = null

function isDark(): boolean {
  return document.documentElement.classList.contains('dark')
}

function initChart(el: HTMLElement): echarts.ECharts {
  return echarts.init(el, isDark() ? 'dark' : undefined)
}

function setupThemeWatch() {
  themeObserver = new MutationObserver(() => {
    ;[equityChart, ddChart, heatmapChart].forEach(c => c?.dispose())
    equityChart = null; ddChart = null; heatmapChart = null
    if (result.value) {
      nextTick(() => renderAllCharts())
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

async function runBacktest() {
  loading.value = true
  try {
    const res = await client.post('/backtest/run', {
      strategy_name: selectedStrategy.value,
      start_date: dateRange.value[0],
      end_date: dateRange.value[1],
      use_scanner: useScanner.value,
      scanner_top_n: scannerTopN.value,
      use_v4: true,
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
  equityChart.setOption({
    tooltip: { trigger: 'axis' },
    grid: { left: '3%', right: '4%', bottom: '8%', containLabel: true },
    xAxis: { type: 'category', data: snaps.map((s: any) => s.date), axisLabel: { rotate: 45, fontSize: 10 } },
    yAxis: { type: 'value', name: '净值 (¥)', axisLabel: { formatter: (v: number) => (v / 10000).toFixed(0) + '万' } },
    series: [{
      name: '权益曲线', type: 'line', data: snaps.map((s: any) => s.equity),
      smooth: true, lineStyle: { color: '#409EFF', width: 2 },
      areaStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: 'rgba(64,158,255,0.3)' },
          { offset: 1, color: 'rgba(64,158,255,0.02)' },
        ]),
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
    tooltip: { trigger: 'axis', valueFormatter: (v: number) => v.toFixed(2) + '%' },
    grid: { left: '3%', right: '4%', bottom: '8%', containLabel: true },
    xAxis: { type: 'category', data: snaps.map((s: any) => s.date), axisLabel: { rotate: 45, fontSize: 10 } },
    yAxis: { type: 'value', name: '回撤 %', axisLabel: { formatter: '{value}%' } },
    series: [{
      name: '回撤', type: 'line', data: dds,
      smooth: true, lineStyle: { color: '#67c23a', width: 1.5 },
      areaStyle: { color: 'rgba(103,194,58,0.15)' },
    }],
  }, true)
}

function renderHeatmap() {
  if (!heatmapRef.value || !result.value?.trades?.length) return
  if (!heatmapChart) heatmapChart = initChart(heatmapRef.value)
  const trades = result.value.trades
  const stocks = [...new Set(trades.map(t => t.stock_code))]
  const dates = [...new Set(result.value.daily_snapshots.map((s: any) => s.date))]
  const stockIdx: Record<string, number> = {}
  stocks.forEach((s, i) => { stockIdx[s] = i })
  const dateIdx: Record<string, number> = {}
  dates.forEach((d, i) => { dateIdx[d] = i })
  const data: [number, number, number][] = []
  for (const t of trades) {
    for (const d of getDateRange(t.entry_date, t.exit_date, dates)) {
      const di = dateIdx[d], si = stockIdx[t.stock_code]
      if (di !== undefined && si !== undefined) {
        data.push([di, si, t.pnl_pct * 100])
      }
    }
  }
  heatmapChart.setOption({
    tooltip: {
      formatter: (p: any) => {
        const stock = stocks[p.data[1]], date = dates[p.data[0]]
        return stock + ' ' + date + ': ' + p.data[2].toFixed(2) + '%'
      }
    },
    grid: { left: '10%', right: '5%', bottom: '12%', top: '3%' },
    xAxis: { type: 'category', data: dates, axisLabel: { rotate: 45, fontSize: 9 } },
    yAxis: { type: 'category', data: stocks, axisLabel: { fontSize: 11 } },
    visualMap: {
      min: -10, max: 10, calculable: true, orient: 'horizontal',
      left: 'center', bottom: 0,
      inRange: { color: ['#67c23a', '#222', '#f56c6c'] }
    },
    series: [{
      type: 'heatmap', data,
      label: { show: false },
      emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' } }
    }],
  }, true)
}

function getDateRange(start: string, end: string, allDates: string[]): string[] {
  const si = allDates.indexOf(start), ei = allDates.indexOf(end)
  return si >= 0 && ei >= 0 ? allDates.slice(si, ei + 1) : []
}

function pnlColor(v: number): string {
  // A-share convention: red=涨(gain), green=跌(loss)
  return v > 0 ? '#f56c6c' : v < 0 ? '#67c23a' : '#909399'
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
        <el-tooltip content="全市场 dragon_score 扫描替代同花顺热门股" placement="top">
          <el-switch v-model="useScanner" active-text="全市场扫描" inactive-text="管道模式" />
        </el-tooltip>
        <el-input-number
          v-if="useScanner" v-model="scannerTopN" :min="20" :max="200" :step="10" size="small"
          style="width:110px" title="每日候选股数量"
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
            <template #prefix>
              <span :style="{color: result.total_return_pct >= 0 ? '#f56c6c' : '#67c23a'}">
                {{ result.total_return_pct >= 0 ? '▲' : '▼' }}
              </span>
            </template>
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
          <el-card shadow="never">
            <div ref="equityChartRef" style="height:380px"></div>
          </el-card>
        </el-col>
        <el-col :span="8">
          <el-card shadow="never">
            <div ref="ddChartRef" style="height:380px"></div>
          </el-card>
        </el-col>
      </el-row>

      <!-- Trade table with expandable rows -->
      <el-card class="table-card" shadow="never">
        <template #header>
          <span>交易明细 ({{ result.total_trades }}笔)</span>
          <span style="margin-left:12px;font-size:12px;color:#909399">
            已实现净盈亏:
            <span :style="{color: pnlColor(result.total_return_pct),fontWeight:'bold'}">
              ¥{{ (result.final_equity - result.initial_capital).toFixed(0) }}
            </span>
          </span>
        </template>
        <el-table :data="sortedTrades" stripe max-height="600" row-key="stock_code" @sort-change="handleSortChange">
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
                      <div class="trade-detail-row"><span>佣金</span><span style="color:#67c23a">-¥{{ (row.entry_commission || 5).toFixed(2) }}</span></div>
                      <div class="trade-detail-row trade-detail-total"><span>龙头得分</span><span style="color:#409EFF;font-weight:600">{{ (row.entry_score || 0).toFixed(3) }}</span></div>
                    </div>
                  </el-col>
                  <el-col :span="8">
                    <div class="trade-detail-block">
                      <h4>卖出明细</h4>
                      <div class="trade-detail-row"><span>开盘价</span><span>¥{{ row.exit_price.toFixed(2) }}</span></div>
                      <div class="trade-detail-row"><span>卖出金额</span><span>¥{{ row.proceeds.toFixed(2) }}</span></div>
                      <div class="trade-detail-row"><span>佣金</span><span style="color:#67c23a">-¥{{ (row.exit_commission || 5).toFixed(2) }}</span></div>
                      <div class="trade-detail-row"><span>印花税</span><span style="color:#67c23a">-¥{{ (row.stamp_duty || 0).toFixed(2) }}</span></div>
                      <div class="trade-detail-row"><span>卖出得分</span><span :style="{color: (row.exit_score||0) >= (row.entry_score||0) ? '#f56c6c' : '#67c23a',fontWeight:'600'}">{{ (row.exit_score || 0).toFixed(3) }}</span></div>
                      <div class="trade-detail-row"><span>剩余现金</span><span style="color:#409EFF;font-weight:600">¥{{ (row.cash_after_trade || 0).toLocaleString() }}</span></div>
                      <div class="trade-detail-row trade-detail-total"><span>净盈亏</span><span :style="{color: pnlColor(row.net_pnl || row.pnl),fontWeight:'700',fontSize:'15px'}">¥{{ (row.net_pnl || row.pnl).toFixed(2) }}</span></div>
                    </div>
                  </el-col>
                  <el-col :span="8">
                    <div class="trade-detail-block">
                      <h4>得分对比</h4>
                      <div style="display:flex;gap:24px;margin-bottom:8px">
                        <div>
                          <span style="color:#909399;font-size:11px">买入</span><br>
                          <span style="color:#409EFF;font-size:18px;font-weight:700">{{ (row.entry_score || 0).toFixed(3) }}</span>
                        </div>
                        <div>
                          <span style="color:#909399;font-size:11px">卖出</span><br>
                          <span style="font-size:18px;font-weight:700" :style="{color: (row.exit_score||0) >= (row.entry_score||0) ? '#f56c6c' : '#67c23a'}">{{ (row.exit_score || 0).toFixed(3) }}</span>
                        </div>
                        <div>
                          <span style="color:#909399;font-size:11px">变化</span><br>
                          <span style="font-size:18px;font-weight:700" :style="{color: ((row.exit_score||0) - (row.entry_score||0)) >= 0 ? '#f56c6c' : '#67c23a'}">{{ ((row.exit_score || 0) - (row.entry_score || 0) >= 0 ? '+' : '') + ((row.exit_score||0) - (row.entry_score||0)).toFixed(3) }}</span>
                        </div>
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
              <span :style="{color: row.exit_price >= row.entry_price ? '#f56c6c' : '#67c23a'}">¥{{ row.exit_price.toFixed(2) }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="net_pnl" label="净盈亏" width="110" align="right" sortable="custom">
            <template #default="{ row }">
              <span :style="{color: pnlColor(row.net_pnl || row.pnl), fontWeight: 'bold'}">
                ¥{{ (row.net_pnl || row.pnl).toFixed(0) }}
              </span>
            </template>
          </el-table-column>
          <el-table-column prop="pnl_pct" label="收益率" width="90" align="right" sortable="custom">
            <template #default="{ row }">
              <span :style="{color: pnlColor(row.pnl_pct), fontWeight: 'bold'}">
                {{ (row.pnl_pct * 100).toFixed(2) }}%
              </span>
            </template>
          </el-table-column>
          <el-table-column prop="holding_days" label="天数" width="55" align="center" sortable="custom" />
          <el-table-column label="可用现金" width="120" align="right">
            <template #default="{ row }: { row: BacktestTrade }">
              <span style="font-size:12px;color:#909399">
                ¥{{ (row.cash_after_trade || 0).toLocaleString() }}
              </span>
            </template>
          </el-table-column>
          <el-table-column prop="exit_reason" label="卖出原因" min-width="180">
            <template #default="{ row }">
              <el-tag size="small" :type="row.exit_reason.includes('止损') ? 'success' : row.exit_reason.includes('涨停') ? 'danger' : 'warning'">
                {{ row.exit_reason }}
              </el-tag>
            </template>
          </el-table-column>
        </el-table>
      </el-card>

      <!-- Heatmap -->
      <el-card class="chart-card" shadow="never">
        <template #header>持仓热力图 (持仓期间每日收益率)</template>
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
.result-content { animation: fadeIn 0.3s ease; }
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
