<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { useAppStore } from '@/stores/app'
import {
  Odometer, TrendCharts, List, Search, Star, Document, InfoFilled, Setting,
} from '@element-plus/icons-vue'

const route = useRoute()
const appStore = useAppStore()
const activeMenu = computed(() => route.path)
</script>

<template>
  <el-menu
    :default-active="activeMenu"
    :collapse="appStore.sidebarCollapsed"
    :unique-opened="true"
    router
    class="sidebar-menu"
  >
    <el-menu-item index="/dashboard">
      <el-icon><Odometer /></el-icon>
      <template #title>龙头仪表板</template>
    </el-menu-item>

    <el-sub-menu index="/analysis">
      <template #title>
        <el-icon><TrendCharts /></el-icon>
        <span>认知分析</span>
      </template>
      <el-menu-item index="/analysis/candidates">龙头候选</el-menu-item>
      <el-menu-item index="/analysis/events">事件分析</el-menu-item>
    </el-sub-menu>

    <el-sub-menu index="/backtest">
      <template #title>
        <el-icon><TrendCharts /></el-icon>
        <span>回测中心</span>
      </template>
      <el-menu-item index="/backtest/strategies">策略配置</el-menu-item>
      <el-menu-item index="/backtest/results">回测结果</el-menu-item>
    </el-sub-menu>

    <el-menu-item index="/token-usage">
      <el-icon><Odometer /></el-icon>
      <template #title>Token 消耗</template>
    </el-menu-item>

    <el-menu-item index="/tasks">
      <el-icon><List /></el-icon>
      <template #title>历史记录</template>
    </el-menu-item>

    <el-menu-item index="/reports">
      <el-icon><Document /></el-icon>
      <template #title>分析报告</template>
    </el-menu-item>

    <el-menu-item index="/screening">
      <el-icon><Search /></el-icon>
      <template #title>股票筛选</template>
    </el-menu-item>

    <el-menu-item index="/favorites">
      <el-icon><Star /></el-icon>
      <template #title>自选股</template>
    </el-menu-item>

    <el-sub-menu index="/settings">
      <template #title>
        <el-icon><Setting /></el-icon>
        <span>设置</span>
      </template>
      <el-menu-item index="/settings/config">配置管理</el-menu-item>
      <el-menu-item index="/settings/logs">系统日志</el-menu-item>
      <el-menu-item index="/settings/sync">数据源同步</el-menu-item>
    </el-sub-menu>

    <el-menu-item index="/about">
      <el-icon><InfoFilled /></el-icon>
      <template #title>关于</template>
    </el-menu-item>
  </el-menu>
</template>

<style lang="scss" scoped>
.sidebar-menu {
  border: none;
  height: 100%;
  :deep(.el-menu-item.is-active) {
    background-color: var(--el-color-primary-light-9);
    color: var(--el-color-primary);
  }
}
</style>
