<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const breadcrumbs = computed(() => {
  const items: { title: string; path: string }[] = [{ title: '首页', path: '/' }]
  if (route.meta.title) items.push({ title: route.meta.title as string, path: route.path })
  return items
})
</script>

<template>
  <el-breadcrumb separator="/">
    <el-breadcrumb-item v-for="b in breadcrumbs" :key="b.path" :to="b.path">
      {{ b.title }}
    </el-breadcrumb-item>
  </el-breadcrumb>
</template>
