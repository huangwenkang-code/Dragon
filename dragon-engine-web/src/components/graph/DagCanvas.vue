<script setup lang="ts">
// Placeholder — will be upgraded to LangGraph DAG visualization
// Nodes: ingest_event → analyze_sentiment → (sector_diffusion) → (leader_competition) → (risk_intercept) → generate_candidates
import { useLoadingStore } from '@/stores/loading'

const loading = useLoadingStore()

const nodes = [
  { id: 'ingest_event', x: 40, y: 120, label: '新闻抓取', width: 130 },
  { id: 'analyze_sentiment', x: 220, y: 120, label: '事件抽取+情绪', width: 130 },
  { id: 'generate_candidates', x: 400, y: 120, label: '龙头识别', width: 130 },
]

const edges = [
  { from: 'ingest_event', to: 'analyze_sentiment' },
  { from: 'analyze_sentiment', to: 'generate_candidates' },
]

function nodeColor(id: string) {
  const status = loading.stages[id as keyof typeof loading.stages]
  if (status === 'loading') return 'var(--color-primary)'
  if (status === 'done') return 'var(--color-success)'
  return 'var(--color-border)'
}

function nodeTextColor(id: string) {
  const status = loading.stages[id as keyof typeof loading.stages]
  return status === 'idle' ? 'var(--color-text-tertiary)' : '#fff'
}
</script>

<template>
  <div class="dag-canvas">
    <div class="dag-label">Pipeline DAG</div>
    <svg viewBox="0 0 570 240" class="dag-svg">
      <defs>
        <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
          <polygon points="0 0, 8 3, 0 6" fill="var(--color-border)" />
        </marker>
      </defs>
      <!-- Edges -->
      <line
        v-for="e in edges"
        :key="`${e.from}-${e.to}`"
        :x1="(nodes.find((n) => n.id === e.from)?.x ?? 0) + 130"
        :y1="(nodes.find((n) => n.id === e.from)?.y ?? 0) + 20"
        :x2="nodes.find((n) => n.id === e.to)?.x ?? 0"
        :y2="(nodes.find((n) => n.id === e.to)?.y ?? 0) + 20"
        stroke="var(--color-border)"
        stroke-width="2"
        marker-end="url(#arrowhead)"
      />
      <!-- Nodes -->
      <g v-for="n in nodes" :key="n.id">
        <rect
          :x="n.x"
          :y="n.y"
          :width="n.width"
          height="40"
          rx="8"
          :fill="nodeColor(n.id)"
          class="dag-node"
        />
        <text
          :x="n.x + n.width / 2"
          :y="n.y + 24"
          text-anchor="middle"
          :fill="nodeTextColor(n.id)"
          font-size="12"
          font-weight="600"
          font-family="Inter, sans-serif"
        >
          {{ n.label }}
        </text>
      </g>
    </svg>
    <div class="dag-hint">▲ 后续升级: LangGraph 完整 DAG 可视化 (含板块扩散 / 龙头竞争 / 风险拦截节点)</div>
  </div>
</template>

<style scoped>
.dag-canvas {
  padding: var(--space-md);
  border-top: 1px solid var(--color-border);
}
.dag-label {
  font-size: 11px;
  font-weight: 600;
  color: var(--color-text-tertiary);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: var(--space-sm);
}
.dag-svg {
  width: 100%;
  height: auto;
}
.dag-node {
  transition: fill var(--transition-normal);
}
.dag-hint {
  font-size: 11px;
  color: var(--color-text-tertiary);
  margin-top: var(--space-sm);
  text-align: center;
}
</style>
