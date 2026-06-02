<script setup lang="ts">
import { computed } from 'vue'
import { usePipelineStore } from '@/stores/pipeline'

const pipeline = usePipelineStore()

const byType = computed(() => {
  const map = new Map<string, number>()
  pipeline.events.forEach((e) => map.set(e.event_type, (map.get(e.event_type) ?? 0) + 1))
  return Array.from(map.entries())
})

const byStrength = computed(() =>
  [...pipeline.events].sort((a, b) => b.event_strength - a.event_strength),
)

function strengthColor(v: number) {
  if (v >= 0.7) return '#DC2626'
  if (v >= 0.4) return '#D97706'
  return '#64748B'
}
</script>

<template>
  <div class="event-tab">
    <div class="type-dist">
      <span v-for="[type, n] in byType" :key="type" class="type-chip">
        {{ type }} <strong>{{ n }}</strong>
      </span>
    </div>
    <div class="event-list">
      <div v-for="ev in byStrength" :key="ev.event_id" class="event-card">
        <div class="row">
          <el-tag size="small" type="primary" effect="dark">{{ ev.event_type }}</el-tag>
          <span class="scope">{{ ev.scope }}</span>
          <span class="strength" :style="{ color: strengthColor(ev.event_strength) }">
            强度 {{ (ev.event_strength * 100).toFixed(0) }}%
          </span>
        </div>
        <h4 class="title">{{ ev.title }}</h4>
        <div class="narrative" v-if="ev.narrative">
          <span class="n-label">LLM 叙事分析</span>
          <p>{{ ev.narrative }}</p>
        </div>
        <div class="meta-row">
          <span v-if="ev.sector_list.length" class="sectors">
            {{ ev.sector_list.join(' · ') }}
          </span>
          <span class="keywords" v-if="ev.keywords.length">
            {{ ev.keywords.slice(0, 5).join(', ') }}
          </span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.event-tab {
  height: 100%;
  overflow-y: auto;
  padding: var(--space-md);
}
.type-dist {
  display: flex;
  gap: var(--space-sm);
  margin-bottom: var(--space-md);
  flex-wrap: wrap;
}
.type-chip {
  font-size: 13px;
  color: var(--color-text-secondary);
  background: var(--color-surface);
  padding: 4px 12px;
  border-radius: var(--radius-md);
  border: 1px solid var(--color-border);
}
.event-card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: var(--space-md);
  margin-bottom: var(--space-sm);
}
.row {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  margin-bottom: var(--space-xs);
}
.scope {
  font-size: 11px;
  color: var(--color-text-tertiary);
}
.strength {
  font-size: 13px;
  font-weight: 700;
  font-family: var(--font-mono);
  margin-left: auto;
}
.title {
  font-size: 14px;
  font-weight: 600;
  line-height: 1.5;
  margin: 4px 0;
}
.narrative {
  background: var(--color-bg-elevated);
  border-left: 2px solid var(--color-primary);
  padding: var(--space-sm) var(--space-md);
  margin: var(--space-sm) 0;
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
}
.n-label {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--color-primary);
}
.narrative p {
  font-size: 13px;
  color: var(--color-text-secondary);
  line-height: 1.6;
  margin-top: 4px;
}
.meta-row {
  display: flex;
  gap: var(--space-md);
  font-size: 12px;
  color: var(--color-text-tertiary);
  margin-top: var(--space-sm);
}
.sectors {
  color: var(--color-accent);
}
</style>
