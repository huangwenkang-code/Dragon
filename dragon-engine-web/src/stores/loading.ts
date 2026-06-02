import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { PipelineStageName } from '@/types/pipeline'

const allStages: PipelineStageName[] = [
  'capital_flow',
  'ths_hot',
  'dragon_tiger_board',
  'merge_active_stocks',
  'find_news_double_layer',
  'analyze_sentiment',
  'generate_candidates',
]

type StageStatus = 'idle' | 'loading' | 'done' | 'error'

function defaultStages(): Record<PipelineStageName, StageStatus> {
  return {
    capital_flow: 'idle',
    ths_hot: 'idle',
    dragon_tiger_board: 'idle',
    merge_active_stocks: 'idle',
    find_news_double_layer: 'idle',
    analyze_sentiment: 'idle',
    generate_candidates: 'idle',
  }
}

export const useLoadingStore = defineStore('loading', () => {
  const stages = ref<Record<PipelineStageName, StageStatus>>(defaultStages())

  const isRunning = computed(() =>
    Object.values(stages.value).some((s) => s === 'loading'),
  )
  const isDone = computed(() =>
    Object.values(stages.value).every((s) => s === 'done' || s === 'idle'),
  )
  const currentStage = computed(() =>
    (Object.entries(stages.value).find(([, s]) => s === 'loading')?.[0] as PipelineStageName) ?? null,
  )

  function setStage(stage: PipelineStageName, status: StageStatus) {
    stages.value[stage] = status
  }

  function startRun() {
    const s = defaultStages()
    s.capital_flow = 'loading'
    s.ths_hot = 'loading'
    s.dragon_tiger_board = 'loading'
    stages.value = s
  }

  function reset() {
    stages.value = defaultStages()
  }

  return { stages, allStages, isRunning, isDone, currentStage, setStage, startRun, reset }
})
