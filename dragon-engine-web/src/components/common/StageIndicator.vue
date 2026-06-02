<script setup lang="ts">
import { Loading } from '@element-plus/icons-vue'

defineProps<{
  stage: string
  label: string
  status: 'pending' | 'loading' | 'done' | 'error'
}>()
</script>

<template>
  <div class="stage-indicator" :class="`status--${status}`">
    <el-icon v-if="status === 'loading'" class="spin"><Loading /></el-icon>
    <span v-else-if="status === 'done'" class="check">&#x2713;</span>
    <span v-else-if="status === 'error'" class="cross">&#x2717;</span>
    <span v-else class="dot">&#x25CB;</span>
    <span class="label">{{ label }}</span>
  </div>
</template>

<style scoped>
.stage-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  padding: 4px 0;
}
.stage-indicator span {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  font-size: 14px;
}
.status--pending { color: var(--color-text-tertiary); }
.status--loading { color: var(--color-primary); }
.status--done { color: var(--color-success); }
.status--error { color: var(--color-danger); }
.spin { animation: spin 1s linear infinite; }
@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
</style>
