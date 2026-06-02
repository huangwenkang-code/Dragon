<script setup lang="ts">
</script>

<template>
  <div class="placeholder-page">
    <el-card shadow="hover">
      <div style="text-align: center; padding: 40px 0;">
        <h2>Dragon Engine</h2>
        <p style="color: var(--el-text-color-secondary);">A股 AI 龙头识别认知引擎 v0.1.0</p>
        <p style="color: var(--el-text-color-secondary); max-width: 600px; margin: 16px auto;">
          基于 LangGraph 多智能体协作框架，结合 DeepSeek 大语言模型，
          实现对 A 股市场的新闻事件驱动龙头股识别与妖股潜力分析。
        </p>
        <el-divider />
        <el-descriptions :column="2" border size="small" style="max-width: 600px; margin: 0 auto;">
          <el-descriptions-item label="后端引擎">LangGraph + FastAPI</el-descriptions-item>
          <el-descriptions-item label="前端框架">Vue 3 + Element Plus</el-descriptions-item>
          <el-descriptions-item label="LLM 模型">DeepSeek</el-descriptions-item>
          <el-descriptions-item label="数据源">东方财富新闻</el-descriptions-item>
        </el-descriptions>
      </div>
    </el-card>
  </div>
</template>

<style lang="scss" scoped>
.placeholder-page { max-width: 1200px; margin: 0 auto; }
</style>
