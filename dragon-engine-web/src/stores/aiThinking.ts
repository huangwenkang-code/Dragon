// aiThinking store — tracks current AI reasoning stage
import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { AIThinkingState } from '@/types/stream'

export const useAIThinkingStore = defineStore('aiThinking', () => {
  const state = ref<AIThinkingState>({
    active_stage: '',
    message: '',
    progress: 0,
    reasoning: '',
    arrived_at: '',
  })

  const history = ref<AIThinkingState[]>([])

  function update(partial: Partial<AIThinkingState>) {
    state.value = { ...state.value, ...partial, arrived_at: new Date().toISOString() }
    if (partial.reasoning) {
      history.value.push({ ...state.value })
    }
  }

  function reset() {
    state.value = { active_stage: '', message: '', progress: 0, reasoning: '', arrived_at: '' }
    history.value = []
  }

  return { state, history, update, reset }
})
