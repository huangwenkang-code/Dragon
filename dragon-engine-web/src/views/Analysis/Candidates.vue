<script setup lang="ts">
import { ref, computed } from 'vue'
import { usePipelineStore } from '@/stores/pipeline'
import { useLoadingStore } from '@/stores/loading'
import { runPipeline } from '@/api/run'

const pipeline = usePipelineStore()
const loading = useLoadingStore()
const topN = ref(60)
const runDate = ref('')
const forceRun = ref(true)

const candidates = computed(() => pipeline.leaderCandidates)
const hasData = computed(() => candidates.value.length > 0)

async function run() {
  loading.startRun()
  try {
    const { data } = await runPipeline({
      trade_date: runDate.value || '',
      top_n: topN.value,
      force: forceRun.value,
    })
    pipeline.setResults(data)
  } catch (e: any) {
    console.error(e)
  }
  loading.setStage('done', 'done')
}

function scoreColor(s: number) {
  if (s >= 0.65) return '#f56c6c'
  if (s >= 0.55) return '#e6a23c'
  if (s >= 0.45) return '#67c23a'
  return '#909399'
}
</script>

<template>
  <div class="candidates-page">
    <div class="toolbar">
      <el-date-picker v-model="runDate" type="date" placeholder="留空=最近交易日"
        value-format="YYYY-MM-DD" size="small" style="width:160px" />
      <el-switch v-model="forceRun" size="small" active-text="强制" />
      <el-button type="primary" @click="run" :loading="loading.isRunning" size="small">
        {{ loading.isRunning ? '运行中...' : '获取候选' }}
      </el-button>
      <span v-if="hasData" class="info">
        共 {{ candidates.length }} 只龙头候选
        <template v-if="pipeline.metadata?.trade_date">
          | 日期: {{ pipeline.metadata.trade_date }}
        </template>
      </span>
    </div>

    <el-empty v-if="!hasData && !loading.isRunning" description="点击"获取候选"查看明日龙头股" :image-size="60" />

    <el-table v-if="hasData" :data="candidates" stripe size="small" max-height="calc(100vh - 140px)" highlight-current-row>
      <el-table-column prop="rank" label="#" width="50" align="center" />
      <el-table-column prop="stock_code" label="代码" width="95" />
      <el-table-column prop="stock_name" label="名称" width="100" />
      <el-table-column prop="leader_score" label="评分" width="90" align="center">
        <template #default="{ row }">
          <span :style="{ color: scoreColor(row.leader_score), fontWeight: 'bold' }">
            {{ row.leader_score?.toFixed(3) }}
          </span>
        </template>
      </el-table-column>
      <el-table-column prop="sector" label="板块" width="120" show-overflow-tooltip />
      <el-table-column prop="reasoning" label="因子分析" min-width="300" show-overflow-tooltip>
        <template #default="{ row }">
          <span style="font-size:12px;color:#909399">{{ row.reasoning || '综合均衡' }}</span>
        </template>
      </el-table-column>
      <el-table-column prop="monster_potential" label="妖股潜力" width="90" align="center">
        <template #default="{ row }">
          <el-progress :percentage="(row.monster_potential || 0) * 100" :stroke-width="6"
            :color="row.monster_potential > 0.5 ? '#f56c6c' : '#909399'" :show-text="false" />
        </template>
      </el-table-column>
    </el-table>
  </div>
</template>

<style scoped>
.candidates-page {
  padding: 12px 16px;
  height: 100vh;
  display: flex;
  flex-direction: column;
}
.toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
  flex-shrink: 0;
}
.info {
  font-size: 13px;
  color: #909399;
  margin-left: 12px;
}
</style>
