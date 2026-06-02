// stream store — SSE connection + incremental pipeline data
import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { StreamConnection } from '@/types/stream'

export const useStreamStore = defineStore('stream', () => {
  const connection = ref<StreamConnection>({ status: 'disconnected' })
  const eventSource = ref<EventSource | null>(null)

  function setStatus(status: StreamConnection['status'], errorMessage?: string) {
    connection.value = { status, error_message: errorMessage, last_event_at: new Date().toISOString() }
  }

  function disconnect() {
    eventSource.value?.close()
    eventSource.value = null
    connection.value = { status: 'disconnected' }
  }

  function setSource(es: EventSource) {
    eventSource.value = es
    connection.value = { status: 'connected', last_event_at: new Date().toISOString() }
  }

  return { connection, eventSource, setStatus, disconnect, setSource }
})
