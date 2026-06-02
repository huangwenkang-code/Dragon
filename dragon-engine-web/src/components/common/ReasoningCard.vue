<script setup lang="ts">
defineProps<{
  title: string
  reasoning: string
  source?: string
}>()
</script>

<template>
  <div class="reasoning-card">
    <div class="header">
      <span class="title">{{ title }}</span>
      <span v-if="source" class="source">{{ source }}</span>
    </div>
    <p class="content">{{ reasoning || '—' }}</p>
  </div>
</template>

<style scoped>
.reasoning-card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: var(--space-md);
}
.header {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  margin-bottom: var(--space-sm);
}
.title {
  font-size: 13px;
  font-weight: 600;
  color: var(--color-primary);
}
.source {
  font-size: 11px;
  color: var(--color-text-tertiary);
  background: var(--color-bg-elevated);
  padding: 1px 6px;
  border-radius: var(--radius-sm);
}
.content {
  font-size: 14px;
  color: var(--color-text-secondary);
  line-height: 1.7;
  white-space: pre-wrap;
}
</style>
