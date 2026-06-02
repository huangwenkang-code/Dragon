import { createRouter, createWebHistory } from 'vue-router'
import BasicLayout from '@/layouts/BasicLayout.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      component: BasicLayout,
      redirect: '/dashboard',
      children: [
        {
          path: 'dashboard',
          name: 'Dashboard',
          component: () => import('@/views/Dashboard/index.vue'),
          meta: { title: '龙头仪表板' },
        },
        {
          path: 'analysis/candidates',
          name: 'Candidates',
          component: () => import('@/views/Analysis/Candidates.vue'),
          meta: { title: '龙头候选' },
        },
        {
          path: 'analysis/events',
          name: 'Events',
          component: () => import('@/views/Analysis/Events.vue'),
          meta: { title: '事件分析' },
        },
        {
          path: 'tasks',
          name: 'Tasks',
          component: () => import('@/views/Tasks/index.vue'),
          meta: { title: '历史记录' },
        },
        {
          path: 'reports',
          name: 'Reports',
          component: () => import('@/views/Reports/index.vue'),
          meta: { title: '分析报告' },
        },
        {
          path: 'screening',
          name: 'Screening',
          component: () => import('@/views/Screening/index.vue'),
          meta: { title: '股票筛选' },
        },
        {
          path: 'favorites',
          name: 'Favorites',
          component: () => import('@/views/Favorites/index.vue'),
          meta: { title: '自选股' },
        },
        {
          path: 'settings/config',
          name: 'SettingsConfig',
          component: () => import('@/views/Settings/Config.vue'),
          meta: { title: '配置管理' },
        },
        {
          path: 'settings/logs',
          name: 'SettingsLogs',
          component: () => import('@/views/Settings/Logs.vue'),
          meta: { title: '系统日志' },
        },
        {
          path: 'settings/sync',
          name: 'SettingsSync',
          component: () => import('@/views/Settings/Sync.vue'),
          meta: { title: '数据源同步' },
        },
        {
          path: 'about',
          name: 'About',
          component: () => import('@/views/About/index.vue'),
          meta: { title: '关于' },
        },
        {
          path: 'backtest/strategies',
          name: 'BacktestStrategies',
          component: () => import('@/views/Backtest/Strategies.vue'),
          meta: { title: '策略配置' },
        },
        {
          path: 'backtest/results',
          name: 'BacktestResults',
          component: () => import('@/views/Backtest/Results.vue'),
          meta: { title: '回测结果' },
        },
        {
          path: 'token-usage',
          name: 'TokenUsage',
          component: () => import('@/views/TokenUsage/index.vue'),
          meta: { title: 'Token 消耗' },
        },
        {
          path: 'stock/:code',
          name: 'StockDetail',
          component: () => import('@/views/StockDetail.vue'),
          meta: { title: '个股详情' },
        },
      ],
    },
  ],
})

export default router
