import { defineStore } from 'pinia'
import type { RouteLocationNormalized } from 'vue-router'
import { useStorage } from '@vueuse/core'

export interface AppState {
  loading: boolean
  loadingProgress: number
  theme: 'light' | 'dark' | 'auto'
  sidebarCollapsed: boolean
  sidebarWidth: number
  currentRoute: RouteLocationNormalized | null
  apiConnected: boolean
}

export const useAppStore = defineStore('app', {
  state: (): AppState => ({
    loading: false,
    loadingProgress: 0,
    theme: (useStorage('app-theme', 'dark').value || 'dark') as 'light' | 'dark' | 'auto',
    sidebarCollapsed: useStorage('sidebar-collapsed', false).value || false,
    sidebarWidth: 240,
    currentRoute: null,
    apiConnected: false,
  }),

  getters: {
    isDarkTheme(): boolean {
      if (this.theme === 'auto') {
        return window.matchMedia('(prefers-color-scheme: dark)').matches
      }
      return this.theme !== 'light'
    },
    actualSidebarWidth(): number {
      return this.sidebarCollapsed ? 64 : this.sidebarWidth
    },
  },

  actions: {
    toggleTheme() {
      const themes: Array<'light' | 'dark' | 'auto'> = ['light', 'dark', 'auto']
      const i = themes.indexOf(this.theme)
      this.theme = themes[(i + 1) % themes.length]
      this.applyTheme()
    },
    applyTheme() {
      document.documentElement.classList.toggle('dark', this.isDarkTheme)
    },
    toggleSidebar() {
      this.sidebarCollapsed = !this.sidebarCollapsed
    },
    setSidebarCollapsed(collapsed: boolean) {
      this.sidebarCollapsed = collapsed
    },
    setCurrentRoute(route: RouteLocationNormalized) {
      this.currentRoute = route
    },
    checkApiConnection() {
      fetch('http://localhost:8000/health')
        .then((r) => r.json())
        .then(() => (this.apiConnected = true))
        .catch(() => (this.apiConnected = false))
    },
  },
})
