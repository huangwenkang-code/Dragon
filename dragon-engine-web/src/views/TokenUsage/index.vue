<script setup lang="ts">
import { ref, onMounted } from 'vue'
import * as echarts from 'echarts'
import client from '@/api/client'
import type { TokenUsageRow } from '@/types/api'

const loading = ref(false)
const days = ref(30)
const rows = ref<TokenUsageRow[]>([])
const chartRef = ref<HTMLDivElement | null>(null)
const tableRef = ref<any>(null)
let chartInstance: echarts.ECharts | null = null

onMounted(loadData)

async function loadData() {
  loading.value = true
  try {
    const res = await client.get(`/token-usage?days=${days.value}`)
    rows.value = res.data?.data || res.data || []
    setTimeout(renderChart, 100)
  } catch (e) {
    console.error('Failed to load token usage', e)
  } finally {
    loading.value = false
  }
}

const totalCost = () => rows.value.reduce((s, r) => s + (r.token_usage?.total_cost || 0), 0)
const totalTokens = () => rows.value.reduce((s, r) => s + (r.token_usage?.total_tokens || 0), 0)
const avgCost = () => rows.value.length ? (totalCost() / rows.value.length).toFixed(4) : '0'

function renderChart() {
  if (!chartRef.value || !rows.value.length) return
  if (!chartInstance) chartInstance = echarts.init(chartRef.value, 'dark')
  const sorted = [...rows.value].sort((a, b) => a.trade_date.localeCompare(b.trade_date))
  chartInstance.setOption({
    tooltip: { trigger: 'axis' },
    legend: { data: ['输入Token', '输出Token', '费用(¥)'] },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: { type: 'category', data: sorted.map(r => r.trade_date), axisLabel: { rotate: 45 } },
    yAxis: [
      { type: 'value', name: 'Token 数' },
      { type: 'value', name: '费用 (¥)' },
    ],
    series: [
      { name: '输入Token', type: 'bar', data: sorted.map(r => r.token_usage?.total_prompt_tokens || 0), itemStyle: { color: '#409EFF' } },
      { name: '输出Token', type: 'bar', data: sorted.map(r => r.token_usage?.total_completion_tokens || 0), itemStyle: { color: '#67c23a' } },
      { name: '费用(¥)', type: 'line', yAxisIndex: 1, data: sorted.map(r => r.token_usage?.total_cost || 0), itemStyle: { color: '#e6a23c' } },
    ],
  }, true)
}

function toggleDetail(row: any) {
  if (!tableRef.value) return
  // Check if this row is currently expanded
  const isExpanded = tableRef.value.states?.expandRows?.has(row) ?? false
  tableRef.value.toggleRowExpansion(row, !isExpanded)
}
</script>

<template>
  <div class="page-container">
    <h2 class="page-title">Token 消耗</h2>

    <div class="control-row">
      <span>最近</span>
      <el-input-number v-model="days" :min="1" :max="365" size="small" style="width:120px" />
      <span>天</span>
      <el-button type="primary" size="small" @click="loadData" :loading="loading">刷新</el-button>
    </div>

    <el-row :gutter="16" class="stats-row">
      <el-col :span="8">
        <el-statistic title="总费用" :value="totalCost().toFixed(4)" prefix="¥" />
      </el-col>
      <el-col :span="8">
        <el-statistic title="总 Token" :value="totalTokens().toLocaleString()" />
      </el-col>
      <el-col :span="8">
        <el-statistic title="平均每次" :value="avgCost()" prefix="¥" />
      </el-col>
    </el-row>

    <el-card class="chart-card" shadow="never">
      <div ref="chartRef" style="width: 100%; height: 360px"></div>
    </el-card>

    <el-card class="table-card" shadow="never">
      <el-table ref="tableRef" :data="rows" stripe v-loading="loading" style="width: 100%" max-height="450">
        <el-table-column prop="trade_date" label="日期" width="120" />
        <el-table-column prop="run_id" label="Run ID" min-width="200" />
        <el-table-column label="输入Token" width="110">
          <template #default="{ row }">{{ (row.token_usage?.total_prompt_tokens || 0).toLocaleString() }}</template>
        </el-table-column>
        <el-table-column label="输出Token" width="110">
          <template #default="{ row }">{{ (row.token_usage?.total_completion_tokens || 0).toLocaleString() }}</template>
        </el-table-column>
        <el-table-column label="总Token" width="110">
          <template #default="{ row }">{{ (row.token_usage?.total_tokens || 0).toLocaleString() }}</template>
        </el-table-column>
        <el-table-column label="费用(¥)" width="110">
          <template #default="{ row }">
            <span style="color: #e6a23c; font-weight: bold">{{ (row.token_usage?.total_cost || 0).toFixed(4) }}</span>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="80" fixed="right">
          <template #default="{ row }">
            <el-button size="small" text @click="toggleDetail(row)">
              详情
            </el-button>
          </template>
        </el-table-column>
        <el-table-column type="expand" width="1">
          <template #default="{ row }">
            <div class="expand-content">
              <div v-if="!row.token_usage?.records?.length" class="empty-records">
                该次运行无 LLM token 消耗记录（可能未使用 LLM 或追踪未生效）
              </div>
              <el-table v-else :data="row.token_usage.records" size="small" stripe>
                <el-table-column prop="step" label="步骤" width="200" />
                <el-table-column prop="model" label="模型" width="150" />
                <el-table-column prop="prompt_tokens" label="输入Token" width="110" />
                <el-table-column prop="completion_tokens" label="输出Token" width="110" />
                <el-table-column prop="total_tokens" label="总Token" width="100" />
                <el-table-column label="费用(¥)" width="100">
                  <template #default="{ row: r }">{{ (r.cost || 0).toFixed(6) }}</template>
                </el-table-column>
              </el-table>
            </div>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<style lang="scss" scoped>
.page-container { padding: 24px; }
.page-title { margin: 0 0 20px; font-size: 20px; font-weight: 600; }
.control-row { display: flex; gap: 8px; align-items: center; margin-bottom: 20px; }
.stats-row { margin-bottom: 20px; }
.chart-card, .table-card { margin-bottom: 20px; }
.expand-content { padding: 12px 24px; background: rgba(0,0,0,0.02); }
.empty-records { padding: 20px; text-align: center; color: #909399; font-size: 13px; }
</style>
