<script setup lang="ts">
defineProps<{
  value: number
  label?: string
  digits?: number
}>()
</script>

<template>
  <span class="score-badge" :class="value >= 0 ? 'up' : 'down'">
    <span v-if="label" class="label">{{ label }}</span>
    <span class="value">{{ (value * 100).toFixed(digits ?? 1) }}%</span>
  </span>
</template>

<style scoped>
.score-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: var(--radius-sm);
}
.score-badge.up {
  background: rgba(220, 38, 38, 0.12);
}
.score-badge.down {
  background: rgba(16, 185, 129, 0.12);
}
.label {
  font-size: 11px;
  opacity: 0.7;
  text-transform: uppercase;
}
</style>
