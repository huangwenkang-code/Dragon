<script setup lang="ts">
import { usePipelineStore } from '@/stores/pipeline'

const pipeline = usePipelineStore()

const metrics = [
  { key: 'events', label: '事件数', value: pipeline.eventCount, color: 'var(--color-primary)' },
  { key: 'sentiment', label: '情绪评分', value: pipeline.sentimentCount, color: 'var(--color-accent)' },
  { key: 'leaders', label: '龙头候选', value: pipeline.leaderCount, color: 'var(--color-success)' },
  { key: 'risk', label: '风险信号', value: pipeline.riskCount, color: 'var(--color-danger)' },
]
</script>

<template>
  <div class="metrics-bar">
    <div v-for="m in metrics" :key="m.key" class="metric-card">
      <span class="value" :style="{ color: m.color }">{{ m.value }}</span>
      <span class="label">{{ m.label }}</span>
    </div>
    <div class="metric-card sentiment-chip" v-if="pipeline.sentimentCount">
      <span class="label">市场情绪</span>
      <span class="value" :class="pipeline.avgSentiment >= 0 ? 'up' : 'down'">
        {{ (pipeline.avgSentiment * 100).toFixed(1) }}%
      </span>
    </div>
  </div>
</template>

<style scoped>
.metrics-bar {
  display: flex;
  gap: var(--space-md);
  padding: var(--space-md) var(--space-lg);
}
.metric-card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: 12px 20px;
  display: flex;
  flex-direction: column;
  align-items: center;
  min-width: 100px;
}
.value {
  font-size: 28px;
  font-weight: 700;
  font-family: var(--font-mono);
  line-height: 1.2;
}
.label {
  font-size: 12px;
  color: var(--color-text-tertiary);
  margin-top: 4px;
}
.sentiment-chip {
  margin-left: auto;
  flex-direction: row;
  gap: 12px;
  align-items: center;
}
.sentiment-chip .value {
  font-size: 20px;
}
</style>
