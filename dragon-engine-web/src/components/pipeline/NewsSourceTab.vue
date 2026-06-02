<script setup lang="ts">
import { computed } from 'vue'
import { usePipelineStore } from '@/stores/pipeline'

const pipeline = usePipelineStore()

const sources = computed(() => {
  const map = new Map<string, number>()
  pipeline.events.forEach((e) => map.set(e.source, (map.get(e.source) ?? 0) + 1))
  return Array.from(map.entries()).sort((a, b) => b[1] - a[1])
})
</script>

<template>
  <div class="news-tab">
    <div class="tab-header">
      <span class="count">{{ pipeline.eventCount }} 条新闻</span>
      <div class="sources">
        <span v-for="[src, n] in sources" :key="src" class="source-tag">{{ src }} · {{ n }}</span>
      </div>
    </div>
    <div class="news-list">
      <div v-for="ev in pipeline.events" :key="ev.event_id" class="news-item">
        <div class="item-header">
          <el-tag size="small" type="primary" effect="dark">{{ ev.event_type }}</el-tag>
          <span class="source">{{ ev.source }}</span>
          <span class="time">{{ ev.publish_time?.slice(0, 16) }}</span>
        </div>
        <h4 class="title">{{ ev.title }}</h4>
        <p class="summary" v-if="ev.summary !== ev.title">{{ ev.summary?.slice(0, 200) }}</p>
        <div class="symbols" v-if="ev.symbol_list.length">
          <el-tag v-for="s in ev.symbol_list" :key="s" size="small" effect="plain">{{ s }}</el-tag>
        </div>
      </div>
    </div>
    <el-empty v-if="!pipeline.eventCount" description="暂无新闻数据" />
  </div>
</template>

<style scoped>
.news-tab {
  height: 100%;
  overflow-y: auto;
  padding: var(--space-md);
}
.tab-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--space-md);
}
.count {
  font-size: 13px;
  color: var(--color-text-secondary);
}
.sources {
  display: flex;
  gap: var(--space-sm);
}
.source-tag {
  font-size: 11px;
  color: var(--color-text-tertiary);
  background: var(--color-surface);
  padding: 2px 8px;
  border-radius: var(--radius-sm);
}
.news-item {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: var(--space-md);
  margin-bottom: var(--space-sm);
}
.item-header {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  margin-bottom: var(--space-xs);
}
.title {
  font-size: 14px;
  font-weight: 600;
  line-height: 1.5;
  margin: 4px 0;
}
.summary {
  font-size: 13px;
  color: var(--color-text-secondary);
  line-height: 1.6;
  margin-top: 4px;
}
.time {
  font-size: 11px;
  color: var(--color-text-tertiary);
  margin-left: auto;
}
.symbols {
  display: flex;
  gap: 4px;
  margin-top: var(--space-xs);
}
</style>
