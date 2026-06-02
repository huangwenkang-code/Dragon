<script setup lang="ts">
import { computed } from 'vue'
import { usePipelineStore } from '@/stores/pipeline'
import { useSelectionStore } from '@/stores/selection'
import ReasoningCard from '@/components/common/ReasoningCard.vue'
import ScoreBadge from '@/components/common/ScoreBadge.vue'

const pipeline = usePipelineStore()
const selection = useSelectionStore()

const selected = computed(() => {
  if (!selection.selectedStock) return pipeline.leaderCandidates[0]
  return pipeline.leaderCandidates.find((c) => c.stock_code === selection.selectedStock) ?? null
})

const relatedEvents = computed(() =>
  pipeline.events.filter((e) =>
    !selected.value || e.symbol_list.includes(selected.value.stock_code),
  ),
)

const relatedSentiment = computed(() =>
  pipeline.sentimentScores.filter((s) =>
    !selected.value || s.symbol === selected.value.stock_code,
  ),
)
</script>

<template>
  <div class="leader-tab">
    <template v-if="selected">
      <!-- Candidate header -->
      <div class="candidate-header">
        <div class="top-line">
          <span class="rank">#{{ selected.rank }}</span>
          <span class="code">{{ selected.stock_code }}</span>
          <span class="name" v-if="selected.stock_name">{{ selected.stock_name }}</span>
          <el-tag v-if="selected.sector" size="small" effect="plain">{{ selected.sector }}</el-tag>
        </div>
      </div>

      <!-- Score breakdown -->
      <div class="score-grid">
        <div class="score-item">
          <span class="s-label">龙头概率</span>
          <ScoreBadge :value="selected.leader_score" :digits="1" />
        </div>
        <div class="score-item">
          <span class="s-label">妖股潜力</span>
          <ScoreBadge :value="selected.monster_potential" :digits="1" />
        </div>
        <div class="score-item">
          <span class="s-label">涨停概率</span>
          <ScoreBadge :value="selected.limit_up_prob" :digits="1" />
        </div>
      </div>

      <!-- LLM Reasoning -->
      <ReasoningCard
        v-if="selected.reasoning"
        title="AI 推理过程"
        :reasoning="selected.reasoning"
      />

      <!-- Related sentiment scores -->
      <div v-if="relatedSentiment.length" class="sub-section">
        <div class="sub-title">情绪维度明细</div>
        <div class="dim-grid">
          <div v-for="s in relatedSentiment" :key="s.target_id" class="dim-card">
            <div class="dim-row"><span>情绪</span><ScoreBadge :value="s.sentiment_score" /></div>
            <div class="dim-row"><span>叙事</span><span class="dim-num">{{ (s.narrative_score * 100).toFixed(0) }}%</span></div>
            <div class="dim-row"><span>炒作</span><span class="dim-num">{{ (s.hype_score * 100).toFixed(0) }}%</span></div>
            <div class="dim-row"><span>一致性</span><span class="dim-num">{{ (s.consistency_score * 100).toFixed(0) }}%</span></div>
          </div>
        </div>
      </div>

      <!-- Related events -->
      <div v-if="relatedEvents.length" class="sub-section">
        <div class="sub-title">关联事件 ({{ relatedEvents.length }})</div>
        <div v-for="ev in relatedEvents" :key="ev.event_id" class="mini-event">
          <el-tag size="small" effect="dark">{{ ev.event_type }}</el-tag>
          <span class="ev-title">{{ ev.title?.slice(0, 60) }}</span>
          <span class="ev-strength">强度 {{ (ev.event_strength * 100).toFixed(0) }}%</span>
        </div>
      </div>
    </template>

    <el-empty v-else description="选择一只股票查看详情" />
  </div>
</template>

<style scoped>
.leader-tab {
  height: 100%;
  overflow-y: auto;
  padding: var(--space-md);
}
.candidate-header {
  margin-bottom: var(--space-lg);
}
.top-line {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
}
.rank {
  font-size: 28px;
  font-weight: 800;
  font-family: var(--font-mono);
  color: var(--color-accent);
}
.code {
  font-size: 20px;
  font-weight: 700;
  font-family: var(--font-mono);
}
.name {
  font-size: 16px;
  color: var(--color-text-secondary);
}
.score-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--space-md);
  margin-bottom: var(--space-lg);
}
.score-item {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: var(--space-md);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}
.s-label {
  font-size: 12px;
  color: var(--color-text-tertiary);
  text-transform: uppercase;
}
.sub-section {
  margin-top: var(--space-lg);
}
.sub-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--color-text-secondary);
  margin-bottom: var(--space-sm);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.dim-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: var(--space-sm);
}
.dim-card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  padding: var(--space-sm) var(--space-md);
}
.dim-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 2px 0;
  font-size: 13px;
}
.dim-row > span:first-child {
  color: var(--color-text-tertiary);
}
.dim-num {
  font-family: var(--font-mono);
  font-weight: 600;
}
.mini-event {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  padding: 6px 0;
  border-bottom: 1px solid var(--color-border);
  font-size: 13px;
}
.ev-title {
  flex: 1;
  color: var(--color-text-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ev-strength {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--color-text-tertiary);
}
</style>
