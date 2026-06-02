<script setup lang="ts">
import { Search } from '@element-plus/icons-vue'
import { usePipelineStore } from '@/stores/pipeline'

const pipeline = usePipelineStore()
</script>

<template>
  <header class="app-header">
    <div class="brand">
      <span class="logo">&#x9F99;</span>
      <span class="name">Dragon Engine</span>
      <span class="subtitle">A股AI龙头识别</span>
    </div>
    <div class="meta" v-if="pipeline.runId">
      <span class="run-id">{{ pipeline.runId.slice(0, 19) }}</span>
    </div>
  </header>
</template>

<style scoped>
.app-header {
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 var(--space-lg);
  background: var(--color-bg-elevated);
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
}
.brand {
  display: flex;
  align-items: center;
  gap: 10px;
}
.logo {
  font-size: 22px;
}
.name {
  font-size: 16px;
  font-weight: 700;
  letter-spacing: -0.02em;
}
.subtitle {
  font-size: 12px;
  color: var(--color-text-tertiary);
}
.meta {
  font-size: 12px;
  color: var(--color-text-tertiary);
  font-family: var(--font-mono);
}
</style>
