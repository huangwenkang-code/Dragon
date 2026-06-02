<script setup lang="ts">
import { computed } from 'vue'
import { useLoadingStore } from '@/stores/loading'
import { useAIThinkingStore } from '@/stores/aiThinking'
import StageIndicator from '@/components/common/StageIndicator.vue'

const loading = useLoadingStore()
const ai = useAIThinkingStore()

const stageDefs = [
  { key: 'ingest_event', label: '新闻抓取' },
  { key: 'analyze_sentiment', label: '情绪分析' },
  { key: 'generate_candidates', label: '龙头推理' },
] as const

const isActive = computed(() => loading.isRunning || ai.state.active_stage !== '')
</script>

<template>
  <div class="ai-panel" :class="{ active: isActive }">
    <div class="panel-header">
      <span class="dot" :class="{ pulse: isActive }" />
      <span class="title">AI 引擎状态</span>
    </div>

    <div class="stages">
      <StageIndicator
        v-for="sd in stageDefs"
        :key="sd.key"
        :stage="sd.key"
        :label="sd.label"
        :status="loading.stages[sd.key] === 'loading' ? 'loading' : loading.stages[sd.key] === 'done' ? 'done' : 'pending'"
      />
    </div>

    <div v-if="ai.state.reasoning" class="reasoning">
      <span class="r-label">当前推理</span>
      <p class="r-text">{{ ai.state.reasoning.slice(0, 200) }}{{ ai.state.reasoning.length > 200 ? '...' : '' }}</p>
    </div>
  </div>
</template>

<style scoped>
.ai-panel {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: var(--space-md);
  transition: border-color var(--transition-normal);
}
.ai-panel.active {
  border-color: var(--color-primary);
  box-shadow: var(--shadow-glow);
}
.panel-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: var(--space-sm);
}
.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--color-text-tertiary);
}
.dot.pulse {
  background: var(--color-primary);
  animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.6); }
  50% { opacity: 0.6; box-shadow: 0 0 0 8px rgba(59, 130, 246, 0); }
}
.title {
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text-tertiary);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.stages {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.reasoning {
  margin-top: var(--space-md);
  background: var(--color-bg-elevated);
  border-radius: var(--radius-sm);
  padding: var(--space-sm) var(--space-md);
  border-left: 2px solid var(--color-primary);
}
.r-label {
  font-size: 10px;
  color: var(--color-primary);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.r-text {
  font-size: 13px;
  color: var(--color-text-secondary);
  line-height: 1.6;
  margin-top: 4px;
}
</style>
