<script setup lang="ts">
// Placeholder for individual stock deep-dive page
import { useRoute } from 'vue-router'
const route = useRoute()
</script>

<template>
  <div class="stock-detail">
    <h2>{{ route.params.code }} 详情</h2>
    <p>股票深度分析页 — 后续扩展</p>
  </div>
</template>

<style scoped>
.stock-detail {
  padding: var(--space-xl);
  color: var(--color-text-secondary);
}
</style>
