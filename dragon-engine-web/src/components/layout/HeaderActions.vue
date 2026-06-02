<script setup lang="ts">
import { computed } from 'vue'
import { useAppStore } from '@/stores/app'
import { Sunny, Moon, FullScreen } from '@element-plus/icons-vue'

const appStore = useAppStore()

function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen()
  else document.documentElement.requestFullscreen()
}

const connectionText = computed(() => appStore.apiConnected ? '已连接' : '未连接')
</script>

<template>
  <div class="header-actions">
    <span class="api-status" :class="{ ok: appStore.apiConnected }">
      <span class="dot" />
      {{ connectionText }}
    </span>
    <el-tooltip content="切换主题" placement="bottom">
      <el-button type="text" @click="appStore.toggleTheme()" class="action-btn">
        <el-icon><Sunny v-if="appStore.isDarkTheme" /><Moon v-else /></el-icon>
      </el-button>
    </el-tooltip>
    <el-tooltip content="全屏" placement="bottom">
      <el-button type="text" @click="toggleFullscreen" class="action-btn">
        <el-icon><FullScreen /></el-icon>
      </el-button>
    </el-tooltip>
  </div>
</template>

<style lang="scss" scoped>
.header-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  .action-btn {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    .el-icon { font-size: 18px; }
  }
  .api-status {
    font-size: 12px;
    color: var(--el-text-color-secondary);
    display: flex;
    align-items: center;
    gap: 6px;
    .dot { width: 6px; height: 6px; border-radius: 50%; background: #DC2626; }
    &.ok .dot { background: #10B981; }
  }
}
</style>
