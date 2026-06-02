<script setup lang="ts">
</script>

<template>
  <div class="app-footer">
    <div class="footer-content">
      <span>Dragon Engine v0.1.0 — A股AI龙头识别认知引擎</span>
      <p class="disclaimer">
        本系统基于AI模型自动分析，所有结果仅供研究参考，不构成任何投资建议。市场有风险，投资需谨慎。
      </p>
    </div>
  </div>
</template>

<style lang="scss" scoped>
.app-footer {
  .footer-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    color: var(--el-text-color-regular);
    font-size: 13px;
    padding: 12px 16px;
    text-align: center;
  }
  .disclaimer {
    color: var(--el-text-color-secondary);
    font-size: 11px;
    margin: 0;
    max-width: 800px;
  }
}
</style>
