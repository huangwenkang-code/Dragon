<script setup lang="ts">
import { usePipelineStore } from '@/stores/pipeline'
import { useSelectionStore } from '@/stores/selection'
import ScoreBadge from '@/components/common/ScoreBadge.vue'

const pipeline = usePipelineStore()
const selection = useSelectionStore()

const columns = [
  { key: 'rank', label: '#', width: 50 },
  { key: 'stock_code', label: '代码', width: 100 },
  { key: 'leader_score', label: '龙头分', width: 100 },
  { key: 'monster_potential', label: '妖股潜力', width: 100 },
  { key: 'limit_up_prob', label: '涨停概率', width: 100 },
  { key: 'sector', label: '板块', width: 80 },
]

function scoreColor(v: number) {
  if (v >= 0.7) return 'var(--color-danger)'
  if (v >= 0.4) return 'var(--color-accent)'
  return 'var(--color-text-tertiary)'
}
</script>

<template>
  <div class="leader-ranking">
    <div class="section-title">龙头排名</div>
    <el-table
      :data="pipeline.leaderCandidates"
      size="small"
      stripe
      row-class-name="leader-row"
      @row-click="(row: any) => selection.selectStock(row.stock_code)"
      highlight-current-row
      style="width: 100%"
    >
      <el-table-column prop="rank" label="#" width="50" />
      <el-table-column prop="stock_code" label="代码" width="100">
        <template #default="{ row }">
          <span class="stock-code">{{ row.stock_code }}</span>
        </template>
      </el-table-column>
      <el-table-column prop="leader_score" label="龙头分" width="100">
        <template #default="{ row }">
          <ScoreBadge :value="row.leader_score" :digits="1" />
        </template>
      </el-table-column>
      <el-table-column prop="monster_potential" label="妖股" width="90">
        <template #default="{ row }">
          <span :style="{ color: scoreColor(row.monster_potential), fontFamily: 'var(--font-mono)', fontWeight: 600 }">
            {{ (row.monster_potential * 100).toFixed(1) }}%
          </span>
        </template>
      </el-table-column>
      <el-table-column prop="limit_up_prob" label="涨停" width="90">
        <template #default="{ row }">
          <span :style="{ color: scoreColor(row.limit_up_prob), fontFamily: 'var(--font-mono)', fontWeight: 600 }">
            {{ (row.limit_up_prob * 100).toFixed(1) }}%
          </span>
        </template>
      </el-table-column>
      <el-table-column prop="sector" label="板块" width="80" />
    </el-table>
  </div>
</template>

<style scoped>
.leader-ranking {
  height: 100%;
  overflow: auto;
}
.section-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--color-text-tertiary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  padding: var(--space-md) var(--space-md) var(--space-sm);
}
.stock-code {
  font-family: var(--font-mono);
  font-weight: 600;
}
</style>
