// pipeline store — core analysis results
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type {
  PipelineEvent, SentimentScore, LeaderCandidate, RiskFlag,
  CapitalFlowRecord, SectorFlowRecord, CapitalFlowSummary, ActiveStock,
  DragonTigerRecord,
} from '@/types/api'

export const usePipelineStore = defineStore('pipeline', () => {
  const runId = ref('')
  const events = ref<PipelineEvent[]>([])
  const sentimentScores = ref<SentimentScore[]>([])
  const capitalFlowRecords = ref<CapitalFlowRecord[]>([])
  const sectorFlowRecords = ref<SectorFlowRecord[]>([])
  const capitalFlowSummary = ref<CapitalFlowSummary | null>(null)
  const activeStocks = ref<ActiveStock[]>([])
  const dragonTigerRecords = ref<DragonTigerRecord[]>([])
  const leaderCandidates = ref<LeaderCandidate[]>([])
  const riskFlags = ref<RiskFlag[]>([])
  const metadata = ref<Record<string, unknown>>({})

  const eventCount = computed(() => events.value.length)
  const sentimentCount = computed(() => sentimentScores.value.length)
  const flowCount = computed(() => capitalFlowRecords.value.length)
  const activeCount = computed(() => activeStocks.value.length)
  const leaderCount = computed(() => leaderCandidates.value.length)
  const riskCount = computed(() => riskFlags.value.length)
  const totalMainInflow = computed(() => capitalFlowSummary.value?.total_main_inflow ?? 0)

  const topLeader = computed(() => leaderCandidates.value[0] ?? null)

  const avgSentiment = computed(() => {
    if (!sentimentScores.value.length) return 0
    const sum = sentimentScores.value.reduce((a, s) => a + s.sentiment_score, 0)
    return sum / sentimentScores.value.length
  })

  function setResults(data: {
    run_id: string
    events: PipelineEvent[]
    sentiment_scores: SentimentScore[]
    capital_flow_records?: CapitalFlowRecord[]
    sector_flow_records?: SectorFlowRecord[]
    capital_flow_summary?: CapitalFlowSummary
    active_stocks?: ActiveStock[]
    dragon_tiger_records?: DragonTigerRecord[]
    leader_candidates: LeaderCandidate[]
    risk_flags: RiskFlag[]
    metadata: Record<string, unknown>
  }) {
    runId.value = data.run_id
    events.value = data.events
    sentimentScores.value = data.sentiment_scores
    capitalFlowRecords.value = data.capital_flow_records ?? []
    sectorFlowRecords.value = data.sector_flow_records ?? []
    capitalFlowSummary.value = data.capital_flow_summary ?? null
    activeStocks.value = data.active_stocks ?? []
    dragonTigerRecords.value = data.dragon_tiger_records ?? []
    leaderCandidates.value = data.leader_candidates
    riskFlags.value = data.risk_flags
    metadata.value = data.metadata
  }

  function reset() {
    runId.value = ''
    events.value = []
    sentimentScores.value = []
    capitalFlowRecords.value = []
    sectorFlowRecords.value = []
    capitalFlowSummary.value = null
    activeStocks.value = []
    dragonTigerRecords.value = []
    leaderCandidates.value = []
    riskFlags.value = []
    metadata.value = {}
  }

  return {
    runId, events, sentimentScores, capitalFlowRecords, sectorFlowRecords,
    capitalFlowSummary, activeStocks, dragonTigerRecords, leaderCandidates, riskFlags, metadata,
    eventCount, sentimentCount, flowCount, activeCount, leaderCount, riskCount,
    totalMainInflow, topLeader, avgSentiment,
    setResults, reset,
  }
})
