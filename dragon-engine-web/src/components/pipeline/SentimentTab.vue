<script setup lang="ts">
import { computed } from 'vue'
import { usePipelineStore } from '@/stores/pipeline'
import ScoreBadge from '@/components/common/ScoreBadge.vue'

const pipeline = usePipelineStore()

interface ChartItem {
  label: string
  value: number
  color: string
}

const scoreList = computed(() =>
  [...pipeline.sentimentScores].sort((a, b) => b.sentiment_score - a.sentiment_score),
)

const distribution = computed(() => {
  const bullish = pipeline.sentimentScores.filter((s) => s.sentiment_score > 0.2).length
  const neutral = pipeline.sentimentScores.filter((s) => Math.abs(s.sentiment_score) <= 0.2).length
  const bearish = pipeline.sentimentScores.filter((s) => s.sentiment_score < -0.2).length
  const max = Math.max(bullish, neutral, bearish, 1)
  return { bullish, neutral, bearish, max }
})

function barWidth(count: number, max: number) {
  return `${(count / max) * 100}%`
}

const avgScores = computed<ChartItem[]>(() => {
  const items: ChartItem[] = [
    { label: '情绪', value: 0, color: 'var(--color-primary)' },
    { label: '叙事', value: 0, color: 'var(--color-accent)' },
    { label: '炒作', value: 0, color: 'var(--color-warning)' },
    { label: '风险', value: 0, color: 'var(--color-danger)' },
    { label: '置信', value: 0, color: 'var(--color-success)' },
  ]
  const n = pipeline.sentimentScores.length || 1
  pipeline.sentimentScores.forEach((s) => {
    items[0].value += s.sentiment_score
    items[1].value += s.narrative_score
    items[2].value += s.hype_score
    items[3].value += s.risk_score
    items[4].value += s.confidence
  })
  items.forEach((i) => (i.value /= n))
  return items
})
</script>

<template>
  <div class="sentiment-tab">
    <!-- Distribution -->
    <div class="dist-section">
      <div class="dist-bar">
        <div class="bar-seg bearish" :style="{ width: barWidth(distribution.bearish, distribution.max) }">
          {{ distribution.bearish }} 看空
        </div>
        <div class="bar-seg neutral" :style="{ width: barWidth(distribution.neutral, distribution.max) }">
          {{ distribution.neutral }} 中性
        </div>
        <div class="bar-seg bullish" :style="{ width: barWidth(distribution.bullish, distribution.max) }">
          {{ distribution.bullish }} 看多
        </div>
      </div>
    </div>

    <!-- Average dimension scores -->
    <div class="avg-scores">
      <div v-for="item in avgScores" :key="item.label" class="dim-row">
        <span class="dim-label">{{ item.label }}</span>
        <div class="dim-track">
          <div class="dim-fill" :style="{ width: `${(item.value * 100).toFixed(0)}%`, background: item.color }" />
        </div>
        <span class="dim-val" :style="{ color: item.color }">{{ (item.value * 100).toFixed(0) }}%</span>
      </div>
    </div>

    <!-- Per-stock scores -->
    <div class="score-list">
      <div v-for="s in scoreList" :key="s.target_id" class="score-row">
        <span class="symbol">{{ s.symbol }}</span>
        <ScoreBadge :value="s.sentiment_score" />
        <span class="hype">炒作 {{ (s.hype_score * 100).toFixed(0) }}%</span>
        <span class="risk">风险 {{ (s.risk_score * 100).toFixed(0) }}%</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.sentiment-tab {
  height: 100%;
  overflow-y: auto;
  padding: var(--space-md);
}
.dist-section {
  margin-bottom: var(--space-lg);
}
.dist-bar {
  display: flex;
  height: 28px;
  border-radius: var(--radius-md);
  overflow: hidden;
  font-size: 12px;
  font-weight: 600;
}
.bar-seg {
  display: flex;
  align-items: center;
  justify-content: center;
  transition: width var(--transition-normal);
}
.bearish { background: rgba(16, 185, 129, 0.3); color: #6EE7B7; }
.neutral { background: rgba(100, 116, 139, 0.3); color: #94A3B8; }
.bullish { background: rgba(220, 38, 38, 0.3); color: #FCA5A5; }
.avg-scores {
  margin-bottom: var(--space-lg);
}
.dim-row {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  margin-bottom: var(--space-sm);
}
.dim-label {
  width: 40px;
  font-size: 12px;
  color: var(--color-text-tertiary);
}
.dim-track {
  flex: 1;
  height: 6px;
  background: var(--color-bg-elevated);
  border-radius: 3px;
  overflow: hidden;
}
.dim-fill {
  height: 100%;
  border-radius: 3px;
  transition: width var(--transition-slow);
}
.dim-val {
  width: 36px;
  font-size: 12px;
  font-family: var(--font-mono);
  font-weight: 600;
  text-align: right;
}
.score-row {
  display: flex;
  align-items: center;
  gap: var(--space-md);
  padding: 8px 12px;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  margin-bottom: 4px;
}
.symbol {
  font-family: var(--font-mono);
  font-weight: 600;
  font-size: 14px;
  width: 70px;
}
.hype, .risk {
  font-size: 12px;
  color: var(--color-text-tertiary);
  font-family: var(--font-mono);
}
</style>
