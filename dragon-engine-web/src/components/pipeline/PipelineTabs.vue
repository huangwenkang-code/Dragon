<script setup lang="ts">
import { useSelectionStore } from '@/stores/selection'
import NewsSourceTab from './NewsSourceTab.vue'
import EventAnalysisTab from './EventAnalysisTab.vue'
import SentimentTab from './SentimentTab.vue'
import LeaderDetailTab from './LeaderDetailTab.vue'

const selection = useSelectionStore()

const tabs = [
  { name: 'events', label: '事件分析', component: EventAnalysisTab },
  { name: 'news', label: '新闻源', component: NewsSourceTab },
  { name: 'sentiment', label: '情绪评分', component: SentimentTab },
  { name: 'leaders', label: '龙头结论', component: LeaderDetailTab },
]
</script>

<template>
  <div class="pipeline-tabs">
    <el-tabs v-model="selection.activeTab" type="border-card">
      <el-tab-pane v-for="tab in tabs" :key="tab.name" :name="tab.name" :label="tab.label">
        <component :is="tab.component" />
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<style scoped>
.pipeline-tabs {
  height: 100%;
  display: flex;
  flex-direction: column;
}
.pipeline-tabs :deep(.el-tabs) {
  flex: 1;
  display: flex;
  flex-direction: column;
}
.pipeline-tabs :deep(.el-tabs__content) {
  flex: 1;
  overflow: hidden;
}
.pipeline-tabs :deep(.el-tab-pane) {
  height: 100%;
}
</style>
