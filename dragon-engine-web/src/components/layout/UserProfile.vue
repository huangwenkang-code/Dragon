<script setup lang="ts">
import { useAppStore } from '@/stores/app'
import { User } from '@element-plus/icons-vue'

const appStore = useAppStore()
</script>

<template>
  <div class="user-profile" :class="{ collapsed: appStore.sidebarCollapsed }">
    <div class="profile-info">
      <el-avatar :size="32">
        <el-icon><User /></el-icon>
      </el-avatar>
      <div v-if="!appStore.sidebarCollapsed" class="user-info">
        <div class="username">Dragon Engine</div>
        <div class="user-role">A股AI龙头识别</div>
      </div>
    </div>
  </div>
</template>

<style lang="scss" scoped>
.user-profile {
  padding: 12px;
  &.collapsed { padding: 8px; text-align: center; }
  .profile-info {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px;
    border-radius: 6px;
    .user-info {
      .username { font-size: 14px; font-weight: 500; color: var(--el-text-color-primary); }
      .user-role { font-size: 12px; color: var(--el-text-color-placeholder); }
    }
  }
}
</style>
