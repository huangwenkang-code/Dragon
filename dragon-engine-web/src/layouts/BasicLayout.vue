<script setup lang="ts">
import { computed, watch, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useWindowSize } from '@vueuse/core'
import { useAppStore } from '@/stores/app'
import { Expand, Fold } from '@element-plus/icons-vue'
import SidebarMenu from '@/components/layout/SidebarMenu.vue'
import UserProfile from '@/components/layout/UserProfile.vue'
import Breadcrumb from '@/components/layout/Breadcrumb.vue'
import HeaderActions from '@/components/layout/HeaderActions.vue'
import AppFooter from '@/components/layout/AppFooter.vue'

const appStore = useAppStore()
const route = useRoute()
const { width } = useWindowSize()
const isMobile = computed(() => width.value < 768)

onMounted(() => {
  appStore.applyTheme()
  appStore.checkApiConnection()
  setInterval(() => appStore.checkApiConnection(), 30000)
})

watch(() => route.fullPath, () => {
  if (isMobile.value) appStore.setSidebarCollapsed(true)
})
</script>

<template>
  <div class="basic-layout">
    <aside
      class="sidebar"
      :class="{ collapsed: appStore.sidebarCollapsed }"
      :style="{ width: appStore.actualSidebarWidth + 'px' }"
    >
      <div class="sidebar-header">
        <div class="logo">
          <span class="logo-icon">&#x9F99;</span>
          <span v-show="!appStore.sidebarCollapsed" class="logo-text">Dragon Engine</span>
        </div>
      </div>
      <nav class="sidebar-nav"><SidebarMenu /></nav>
      <div class="sidebar-footer"><UserProfile /></div>
    </aside>

    <div
      v-if="isMobile && !appStore.sidebarCollapsed"
      class="sidebar-overlay"
      @click="appStore.setSidebarCollapsed(true)"
    />

    <div class="main-container" :style="{ marginLeft: appStore.actualSidebarWidth + 'px' }">
      <header class="header">
        <div class="header-left">
          <el-button type="text" @click.stop="appStore.toggleSidebar()" class="sidebar-toggle">
            <el-icon><Expand v-if="appStore.sidebarCollapsed" /><Fold v-else /></el-icon>
          </el-button>
          <Breadcrumb />
        </div>
        <div class="header-right"><HeaderActions /></div>
      </header>

      <main class="main-content">
        <router-view v-slot="{ Component }">
          <transition name="fade" mode="out-in" appear>
            <component :is="Component" :key="route.fullPath" />
          </transition>
        </router-view>
      </main>

      <footer class="footer"><AppFooter /></footer>
    </div>

    <el-backtop :right="40" :bottom="40" />
  </div>
</template>

<style lang="scss" scoped>
.basic-layout {
  min-height: 100vh;
  background-color: var(--el-bg-color-page);
}
.sidebar-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.35);
  z-index: 950;
}
.sidebar {
  position: fixed;
  top: 0;
  left: 0;
  height: 100vh;
  background-color: var(--el-bg-color);
  border-right: 1px solid var(--el-border-color-light);
  transition: width 0.3s ease;
  z-index: 1000;
  display: flex;
  flex-direction: column;
  &.collapsed { width: 64px !important; }
  .sidebar-header {
    height: 60px;
    display: flex;
    align-items: center;
    padding: 0 16px;
    border-bottom: 1px solid var(--el-border-color-lighter);
    .logo {
      display: flex;
      align-items: center;
      gap: 12px;
      .logo-icon { font-size: 28px; }
      .logo-text { font-size: 18px; font-weight: 700; color: var(--el-text-color-primary); white-space: nowrap; }
    }
  }
  .sidebar-nav { flex: 1; overflow-y: auto; padding: 8px 0; }
  .sidebar-footer { border-top: 1px solid var(--el-border-color-lighter); padding: 8px; }
}
.main-container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  transition: margin-left 0.3s ease;
}
.header {
  height: 60px;
  background-color: var(--el-bg-color);
  border-bottom: 1px solid var(--el-border-color-light);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  position: sticky;
  top: 0;
  z-index: 999;
  .header-left { display: flex; align-items: center; gap: 16px; }
  .header-right { display: flex; align-items: center; gap: 16px; }
}
.main-content {
  flex: 1;
  padding: 24px;
  min-height: calc(100vh - 60px - 60px);
}
.footer {
  height: 60px;
  background-color: var(--el-bg-color);
  border-top: 1px solid var(--el-border-color-light);
  display: flex;
  align-items: center;
  justify-content: center;
}
.fade-enter-active, .fade-leave-active { transition: opacity 0.3s ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
@media (max-width: 768px) {
  .sidebar { transform: translateX(-100%); &:not(.collapsed) { transform: translateX(0); } }
  .main-container { margin-left: 0 !important; }
  .main-content { padding: 16px; }
  .header { padding: 0 16px; }
}
</style>
