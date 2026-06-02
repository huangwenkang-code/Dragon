// selection store — currently focused stock/event/node
import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useSelectionStore = defineStore('selection', () => {
  const selectedStock = ref<string | null>(null)
  const selectedEventId = ref<string | null>(null)
  const activeTab = ref<string>('leaders') // which pipeline tab is active

  function selectStock(code: string | null) {
    selectedStock.value = code
    selectedEventId.value = null
  }

  function selectEvent(eventId: string | null) {
    selectedEventId.value = eventId
  }

  function setTab(tab: string) {
    activeTab.value = tab
  }

  return { selectedStock, selectedEventId, activeTab, selectStock, selectEvent, setTab }
})
