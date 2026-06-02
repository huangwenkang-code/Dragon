<script setup lang="ts">
import { ref } from 'vue'
import { runPipeline } from '@/api/run'
import { backfillBars } from '@/api/backfill'
import { usePipelineStore } from '@/stores/pipeline'
import { useLoadingStore } from '@/stores/loading'
import { useAIThinkingStore } from '@/stores/aiThinking'
import type { RunRequest } from '@/types/api'
import AppHeader from '@/components/layout/AppHeader.vue'
import MetricsBar from '@/components/dashboard/MetricsBar.vue'
import LeaderRanking from '@/components/dashboard/LeaderRanking.vue'
import PipelineTabs from '@/components/pipeline/PipelineTabs.vue'
import DagCanvas from '@/components/graph/DagCanvas.vue'
import AIStatusPanel from '@/components/ai/AIStatusPanel.vue'

const pipeline = usePipelineStore()
const loading = useLoadingStore()
const ai = useAIThinkingStore()

const topN = ref(60)
const runDate = ref('')
const forceRun = ref(true)
const batchEnd = ref('')
const batchRunning = ref(false)
const batchProgress = ref('')
const barStatus = ref('')
const barRunning = ref(false)

async function run() {
  const req: RunRequest = {
    trade_date: runDate.value || '',
    top_n: topN.value,
    force: forceRun.value,
  }

  loading.startRun()
  ai.update({ active_stage: 'capital_flow', message: '扫描全市场资金流向...', progress: 10 })

  try {
    const { data } = await runPipeline(req)
    pipeline.setResults(data)

    loading.setStage('capital_flow', 'done')
    loading.setStage('ths_hot', 'done')
    loading.setStage('dragon_tiger_board', 'done')
    loading.setStage('merge_active_stocks', 'done')
    loading.setStage('find_news_double_layer', 'done')
    loading.setStage('analyze_sentiment', 'done')
    loading.setStage('generate_candidates', 'done')

    ai.update({ active_stage: '', message: '分析完成', progress: 100, reasoning: `${data.leader_candidates.length} 个候选` })
  } catch (err: any) {
    loading.setStage(loading.currentStage ?? 'capital_flow', 'error')
    ai.update({ active_stage: '', message: err.message ?? '请求失败', progress: 0 })
  }
}

async function runBatch() {
  if (!batchEnd.value) return
  batchRunning.value = true
  const start = '2026-01-01'
  const end = batchEnd.value

  // generate date list
  const d = new Date(start)
  const endD = new Date(end)
  const all: string[] = []
  while (d <= endD) {
    all.push(d.toISOString().slice(0, 10))
    d.setDate(d.getDate() + 1)
  }

  batchProgress.value = `0/${all.length}`
  let ok = 0
  for (let i = 0; i < all.length; i++) {
    batchProgress.value = `${i + 1}/${all.length}: ${all[i]}`
    try {
      await runPipeline({ trade_date: all[i], top_n: topN.value, force: true })
      ok++
    } catch { /* skip weekends */ }
  }
  batchProgress.value = `${ok}/${all.length} done`
  batchRunning.value = false
}

async function checkBars() {
  barRunning.value = true
  barStatus.value = '检测中...'
  try {
    const { data } = await backfillBars(1)
    if (data.symbols === 0) {
      barStatus.value = '日线完整 ✓'
    } else {
      barStatus.value = `缺 ${data.symbols} 只候选`
    }
  } catch (e: any) {
    barStatus.value = `检测失败: ${e.message}`
  }
  barRunning.value = false
}

async function doBackfillBars() {
  barRunning.value = true
  barStatus.value = '补全中...'
  try {
    const { data } = await backfillBars(7)
    barStatus.value = `补全 ${data.symbols} 只, ${data.rows} 行`
    if (data.failed > 0) barStatus.value += `, ${data.failed} 失败`
  } catch (e: any) {
    barStatus.value = `补全失败: ${e.message}`
  }
  barRunning.value = false
}
</script>

<template>
  <div class="dashboard">
    <AppHeader />

    <!-- Control bar -->
    <div class="control-bar">
      <!-- Left: Single day run (daily use) -->
      <div class="ctrl-group">
        <span class="ctrl-label">单日运行</span>
        <el-date-picker v-model="runDate" type="date" placeholder="留空=今天"
          value-format="YYYY-MM-DD" size="small" style="width:150px" />
        <el-switch v-model="forceRun" size="small" active-text="强制" title="跳过缓存重跑" />
        <el-button type="primary" @click="run" :loading="loading.isRunning" size="small">
          {{ loading.isRunning ? '运行中...' : '运行' }}
        </el-button>
      </div>

      <el-divider direction="vertical" />

      <!-- Middle: Batch backfill (occasional use) -->
      <div class="ctrl-group">
        <span class="ctrl-label">批量回填</span>
        <el-date-picker v-model="batchEnd" type="date" placeholder="截止日期"
          value-format="YYYY-MM-DD" size="small" style="width:140px" />
        <el-button @click="runBatch" :loading="batchRunning" :disabled="!batchEnd" size="small">
          {{ batchRunning ? '回填中...' : '开始回填' }}
        </el-button>
        <span v-if="batchProgress" class="progress-text">{{ batchProgress }}</span>
      </div>

      <el-divider direction="vertical" />

      <!-- Right: Bar data repair (rare use) -->
      <div class="ctrl-group">
        <span class="ctrl-label">日线修复</span>
        <el-button @click="doBackfillBars" :loading="barRunning" size="small">补全日线</el-button>
        <span v-if="barStatus" class="progress-text">{{ barStatus }}</span>
      </div>
    </div>

    <!-- Metrics -->
    <MetricsBar />

    <!-- Main content -->
    <div class="main-grid">
      <aside class="left-panel">
        <LeaderRanking />
        <AIStatusPanel />
      </aside>
      <main class="center-panel">
        <PipelineTabs />
      </main>
    </div>

    <DagCanvas />
  </div>
</template>

<style scoped>
.dashboard {
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
}
.control-bar {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 6px var(--space-lg);
  background: var(--color-bg-elevated);
  border-bottom: 1px solid var(--color-border);
}
.ctrl-group {
  display: flex;
  align-items: center;
  gap: 8px;
}
.ctrl-label {
  font-size: 11px;
  color: var(--color-text-tertiary);
  white-space: nowrap;
}
.progress-text {
  font-size: 11px;
  color: #06b6d4;
  margin-left: 8px;
}
.main-grid {
  display: grid;
  grid-template-columns: 320px 1fr;
  flex: 1;
  overflow: hidden;
}
.left-panel {
  display: flex;
  flex-direction: column;
  border-right: 1px solid var(--color-border);
  overflow: hidden;
}
.left-panel > :first-child {
  flex: 1;
  overflow-y: auto;
}
.center-panel {
  overflow: hidden;
}
</style>
